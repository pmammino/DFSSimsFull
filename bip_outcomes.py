"""
bip_outcomes.py
===============
Python port of mlb_projections_statcast.R's XGBoost classifier.

Predicts the outcome of each BIP (out / single / double / triple / home_run)
given launch_speed, launch_angle, adjusted_angle, sprint_speed, stand, and
home_team. Trained on real (non-synthetic) BIP data only — the imputed
synthetic observations are only used as INPUTS during inference, not as
training data, because they're sampled from a Gaussian and don't carry the
true outcome signal.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pipeline_config import XGB_PARAMS, XGB_SAMPLE_SIZE, BIP_OUTCOMES

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Map raw Statcast `events` strings to our 5 outcome classes
# ─────────────────────────────────────────────────────────────────────────────

# These are the "out" outcomes from the R script's case_when (line 94-101)
OUT_EVENTS = {
    "double_play", "fan_interference", "field_error", "field_out",
    "fielders_choice", "fielders_choice_out", "force_out",
    "grounded_into_double_play", "interf_def", "null",
    "sac_bunt", "sac_bunt_double_play",
    "sac_fly", "sac_fly_double_play",
    "triple_play",
}


def map_event_to_outcome(event: str) -> str:
    """Returns one of {out, single, double, triple, home_run} or NaN."""
    if pd.isna(event):
        return np.nan
    if event in OUT_EVENTS:
        return "out"
    if event in ("single", "double", "triple", "home_run"):
        return event
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Model training + prediction
# ─────────────────────────────────────────────────────────────────────────────

# Categorical features need integer encoding for XGBoost
CAT_FEATURES = ["stand", "home_team"]
NUM_FEATURES = ["launch_speed", "launch_angle", "adjusted_angle", "sprint_speed"]
FEATURES     = NUM_FEATURES + CAT_FEATURES


class BIPOutcomeModel:
    """Wraps an XGBoost multi-class classifier for BIP outcomes."""

    def __init__(self, params: dict | None = None):
        self.params  = (params or XGB_PARAMS).copy()
        self.model   = None
        self.encoder = {}      # categorical level → int
        # IMPORTANT: LabelEncoder sorts alphabetically. We adopt its ordering
        # as the canonical class index → name mapping, so our predict_proba
        # column names match the integer columns XGBoost produces.
        self.label_encoder = LabelEncoder().fit(BIP_OUTCOMES)
        self.outcome_classes = list(self.label_encoder.classes_)

    def _encode_categorical(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = df[FEATURES].copy()
        for c in CAT_FEATURES:
            if fit:
                levels = sorted(X[c].dropna().unique().tolist())
                self.encoder[c] = {lv: i for i, lv in enumerate(levels)}
            mapping = self.encoder.get(c, {})
            X[c] = X[c].map(mapping).fillna(-1).astype(int)
        for c in NUM_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "BIPOutcomeModel":
        """Train on a dataframe with columns FEATURES + 'Result'."""
        data = df.dropna(subset=FEATURES + ["Result"]).copy()
        data = data[data["Result"].isin(self.outcome_classes)]
        if verbose:
            print(f"  Training rows available: {len(data):,}")
            print("  Outcome distribution:")
            print(data["Result"].value_counts().to_string().replace("\n", "\n    "))

        # Train/test split, then sample
        train_full, test = train_test_split(
            data, test_size=0.25, stratify=data["Result"], random_state=42)
        if XGB_SAMPLE_SIZE and len(train_full) > XGB_SAMPLE_SIZE:
            train, _ = train_test_split(
                train_full, train_size=XGB_SAMPLE_SIZE,
                stratify=train_full["Result"], random_state=42)
        else:
            train = train_full
        if verbose:
            print(f"  Train: {len(train):,}   Test: {len(test):,}")

        Xtr = self._encode_categorical(train, fit=True)
        Xte = self._encode_categorical(test,  fit=False)
        ytr = self.label_encoder.transform(train["Result"].values)
        yte = self.label_encoder.transform(test["Result"].values)

        # Build XGBClassifier-equivalent via raw API
        params = self.params.copy()
        n_estimators = params.pop("n_estimators", 200)
        params["num_class"] = len(self.outcome_classes)
        # Map sklearn-style params → xgboost native
        params["eta"]      = params.pop("learning_rate", 0.1)
        params["nthread"]  = params.pop("n_jobs", -1)
        params["seed"]     = params.pop("random_state", 42)
        params["verbosity"] = 0
        dtr = xgb.DMatrix(Xtr.values, label=ytr,
                          feature_names=list(Xtr.columns))
        dte = xgb.DMatrix(Xte.values, label=yte,
                          feature_names=list(Xte.columns))
        self.model = xgb.train(params, dtr, num_boost_round=n_estimators,
                               evals=[(dte, "test")], verbose_eval=False)

        # Evaluate
        proba = self.model.predict(dte)
        pred  = np.argmax(proba, axis=1)
        if verbose:
            acc = accuracy_score(yte, pred)
            ll  = log_loss(yte, proba, labels=list(range(len(self.outcome_classes))))
            print(f"  Test accuracy:  {acc:.4f}")
            print(f"  Test log-loss:  {ll:.4f}")
            # Per-class breakdown
            print("  Per-class precision/recall:")
            for ci, cls in enumerate(self.outcome_classes):
                mask = yte == ci
                if mask.sum() == 0:
                    continue
                rec = (pred[mask] == ci).mean()
                prec = ((pred == ci) & (yte == ci)).sum() / max(1, (pred == ci).sum())
                print(f"    {cls:9s}  precision={prec:.3f}  recall={rec:.3f}  "
                      f"n={mask.sum():,}")
            # Feature importance
            imp = self.model.get_score(importance_type="gain")
            print("  Feature importance (gain):")
            for f, v in sorted(imp.items(), key=lambda x: -x[1]):
                print(f"    {f:18s}  {v:8.1f}")
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe with one column per outcome class."""
        X = self._encode_categorical(df, fit=False)
        d = xgb.DMatrix(X.values, feature_names=list(X.columns))
        proba = self.model.predict(d)
        cols = [f"prob_{c}" for c in self.outcome_classes]
        return pd.DataFrame(proba, columns=cols, index=df.index)

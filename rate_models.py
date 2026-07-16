"""
rate_models.py
==============
Projects per-PA rates (K%, BB%, HBP%, SF%) for the target season, using
PA-shrunken career features as predictors.

Unlike the earlier FanGraphs-CSV pipeline, we don't have plate discipline
features (O-Swing%, Stuff+, etc.) here because statsapi doesn't expose them.
Instead we use:
    * Last-season rate (PA-shrunk)
    * Career PA-weighted rate (PA-shrunk)
    * log(prior PA), log(career PA)
    * Age

For K% and BB% this still significantly beats naive shrinkage because the
model learns the rate of mean-regression as a function of PA and age.

For HBP% and SF% — both very rare and very noisy — we use pure Bayesian
shrinkage against the league mean. Trying to fit ML models adds complexity
without lift since the year-to-year r² is dominated by sampling noise.

All projections come with SDs from quantile-regression bands, scaled to
68% empirical coverage via walk-forward calibration. For HBP/SF (closed-
form shrinkage), the SDs come from a beta-binomial posterior.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from pipeline_config import (
    SHRINK_K, SHRINK_K_PITCHER, DEFAULT_LEAGUE_RATES,
    RATE_MIN_PA_TRAIN, RATE_MIN_PA_ACTIVE, RATE_ACTIVE_LOOKBACK,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian shrinkage helper
# ─────────────────────────────────────────────────────────────────────────────

def shrink(p: float, n: float, mu: float, k: float) -> float:
    """(n·p + k·mu) / (n + k) — the canonical regression-to-mean estimator."""
    if pd.isna(p):
        return mu
    return (n * p + k * mu) / (n + k)


def league_mean(df: pd.DataFrame, target_year: int, rate_col: str,
                pa_col: str, min_pa: int = 50) -> float:
    """PA-weighted league mean for a rate, using only rows before target_year."""
    pool = df[(df["Season"] < target_year) & (df[pa_col] >= min_pa)]
    if pool.empty:
        return DEFAULT_LEAGUE_RATES.get(rate_col, 0.0)
    wm = np.average(pool[rate_col].values, weights=pool[pa_col].values)
    return float(wm) if not np.isnan(wm) else DEFAULT_LEAGUE_RATES.get(rate_col, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Build the per-(player, year) training/inference panel
# ─────────────────────────────────────────────────────────────────────────────

# For K% / BB% the four metrics we use as features are themselves rates
RATE_TARGETS = ["K%", "BB%", "HBP%", "SF%"]


def build_rate_panel(df: pd.DataFrame, pa_col: str, shrink_k: dict) -> pd.DataFrame:
    """For each (player, target_year) build features from priors-only data.

    Returns one row per player-season; rows with no prior history are dropped.
    """
    rows = []
    df = df.sort_values(["PlayerId", "Season"]).reset_index(drop=True)
    # Cache per-target-year league means
    lm_cache: dict[tuple[int, str], float] = {}

    def lm(ty: int, rate: str) -> float:
        key = (ty, rate)
        if key not in lm_cache:
            lm_cache[key] = league_mean(df, ty, rate, pa_col)
        return lm_cache[key]

    for pid, g in df.groupby("PlayerId"):
        g = g.sort_values("Season")
        for _, target_row in g.iterrows():
            ty = int(target_row["Season"])
            prior = g[g["Season"] < ty]
            if len(prior) == 0:
                continue
            last = prior.iloc[-1]
            n_prev   = float(last[pa_col])
            career_n = float(prior[pa_col].sum())

            rec = {
                "PlayerId": int(pid),
                "Name":     target_row["Name"],
                "Team":     target_row["Team"],
                "Season":   ty,
                "Age":      int(target_row.get("Age", 0) or 0),
                "PA":       float(target_row[pa_col]),
                "p1_PA":    n_prev,
                "career_PA": career_n,
                "n_prior_seasons": int(len(prior)),
                "p1_Season_gap":  ty - int(last["Season"]),
            }
            # Targets
            for r in RATE_TARGETS:
                rec[f"target_{r}"] = float(target_row[r])
            # Features
            for r in RATE_TARGETS:
                mu = lm(ty, r)
                # Shrunk most-recent
                rec[f"p1_{r}_shrunk"] = shrink(last[r], n_prev, mu, shrink_k[r])
                # PA-weighted career, shrunk
                if career_n > 0:
                    cm = np.average(prior[r].values, weights=prior[pa_col].values)
                    rec[f"career_{r}_shrunk"] = shrink(cm, career_n, mu, shrink_k[r])
                else:
                    rec[f"career_{r}_shrunk"] = mu
            rec["log_p1_PA"]     = float(np.log1p(rec["p1_PA"]))
            rec["log_career_PA"] = float(np.log1p(rec["career_PA"]))
            rows.append(rec)
    return pd.DataFrame(rows)


FEATURE_COLS = (
    ["Age", "log_p1_PA", "log_career_PA", "p1_Season_gap", "n_prior_seasons"]
    + [f"p1_{r}_shrunk" for r in RATE_TARGETS]
    + [f"career_{r}_shrunk" for r in RATE_TARGETS]
)


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting — ensemble of Ridge + HGB with quantile SDs
# ─────────────────────────────────────────────────────────────────────────────

def _fit_point(X, y, w):
    ridge = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    hgb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, max_depth=4,
        min_samples_leaf=30, l2_regularization=2.0, random_state=0,
    ).fit(X, y, sample_weight=w)
    return ridge, hgb


def _predict_point(ridge, hgb, X):
    return 0.5 * ridge.predict(X) + 0.5 * hgb.predict(X)


def _fit_quantile(X, y, w, q):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q,
        max_iter=300, learning_rate=0.04, max_depth=4,
        min_samples_leaf=30, l2_regularization=2.0, random_state=0,
    ).fit(X, y, sample_weight=w)


def _walk_forward_calibration(panel: pd.DataFrame, target_name: str,
                              min_pa: int, target_year: int):
    """Walk-forward validation. Returns (metrics_df, sd_scale)."""
    target_col = f"target_{target_name}"
    elig = panel[(panel["p1_PA"] >= min_pa) & (panel["PA"] >= min_pa)]
    years = sorted(elig["Season"].unique())
    eval_years = [y for y in years if y >= years[0] + 1 and y < target_year]
    if not eval_years:
        return pd.DataFrame(), 1.0

    rows, z_resid, weights = [], [], []
    for ty in eval_years:
        tr = elig[elig["Season"] < ty]
        te = elig[elig["Season"] == ty]
        if len(tr) < 100 or len(te) < 30:
            continue
        Xtr, ytr = tr[FEATURE_COLS].values, tr[target_col].values
        Xte, yte = te[FEATURE_COLS].values, te[target_col].values
        wtr, wte = np.sqrt(tr["PA"].values), np.sqrt(te["PA"].values)

        ridge, hgb = _fit_point(Xtr, ytr, wtr)
        p_point = _predict_point(ridge, hgb, Xte)
        m16 = _fit_quantile(Xtr, ytr, wtr, 0.16)
        m84 = _fit_quantile(Xtr, ytr, wtr, 0.84)
        p16, p84 = m16.predict(Xte), m84.predict(Xte)
        base = te[f"p1_{target_name}_shrunk"].values

        rows.append({
            "year": ty, "n": len(te),
            "wMAE_baseline": mean_absolute_error(yte, base, sample_weight=wte),
            "wMAE_model":    mean_absolute_error(yte, p_point, sample_weight=wte),
            "wR2_baseline":  r2_score(yte, base, sample_weight=wte),
            "wR2_model":     r2_score(yte, p_point, sample_weight=wte),
        })
        h = np.maximum((p84 - p16) / 2.0, 1e-5)
        z_resid.append(np.abs(yte - p_point) / h)
        weights.append(wte)

    if not z_resid:
        return pd.DataFrame(rows), 1.0
    z_all = np.concatenate(z_resid)
    w_all = np.concatenate(weights)
    order = np.argsort(z_all)
    zs, ws = z_all[order], w_all[order]
    cum = np.cumsum(ws)
    cum /= cum[-1]
    idx = min(int(np.searchsorted(cum, 0.68)), len(zs) - 1)
    return pd.DataFrame(rows), float(zs[idx])


def fit_and_predict_rate(panel: pd.DataFrame, target_name: str,
                        target_year: int, min_pa: int = RATE_MIN_PA_TRAIN
                        ) -> tuple[dict, pd.DataFrame, float]:
    """Train an ensemble + quantile pair for one rate target.

    Returns (models_dict, validation_metrics_df, sd_calibration_scale).
    """
    metrics, scale = _walk_forward_calibration(panel, target_name, min_pa, target_year)
    elig = panel[(panel["p1_PA"] >= min_pa) & (panel["PA"] >= min_pa)]
    X = elig[FEATURE_COLS].values
    y = elig[f"target_{target_name}"].values
    w = np.sqrt(elig["PA"].values)
    ridge, hgb = _fit_point(X, y, w)
    q16 = _fit_quantile(X, y, w, 0.16)
    q84 = _fit_quantile(X, y, w, 0.84)
    models = {"ridge": ridge, "hgb": hgb, "q16": q16, "q84": q84}
    return models, metrics, scale


# ─────────────────────────────────────────────────────────────────────────────
# Build the inference row for each active player and apply models
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_panel(df: pd.DataFrame, target_year: int, pa_col: str,
                          shrink_k: dict,
                          min_pa_active: int = RATE_MIN_PA_ACTIVE,
                          active_lookback: int = RATE_ACTIVE_LOOKBACK
                          ) -> pd.DataFrame:
    """Generate one row per "active" player to project for target_year."""
    df_sorted = df.sort_values(["PlayerId", "Season"])
    latest = df_sorted.groupby("PlayerId").tail(1)
    active = latest[
        (latest["Season"] >= target_year - active_lookback)
        & (latest[pa_col] >= min_pa_active)
    ]
    rows = []
    lm_cache: dict[str, float] = {r: league_mean(df, target_year, r, pa_col)
                                  for r in RATE_TARGETS}
    for pid in active["PlayerId"].unique():
        g = df_sorted[df_sorted["PlayerId"] == pid]
        prior = g[g["Season"] < target_year]
        if len(prior) == 0:
            continue
        last = prior.iloc[-1]
        # Project age forward
        last_age = last.get("Age", 0)
        if pd.isna(last_age) or last_age == 0:
            last_age = 28
        age_proj = int(last_age) + (target_year - int(last["Season"]))

        career_n = float(prior[pa_col].sum())
        rec = {
            "PlayerId": int(pid),
            "Name": last["Name"], "Team": last["Team"],
            "Age": age_proj, "PA": np.nan,
            "p1_PA": float(last[pa_col]),
            "career_PA": career_n,
            "n_prior_seasons": int(len(prior)),
            "p1_Season_gap": target_year - int(last["Season"]),
            "Last_Season": int(last["Season"]),
        }
        for r in RATE_TARGETS:
            mu = lm_cache[r]
            rec[f"p1_{r}_shrunk"] = shrink(last[r], last[pa_col], mu, shrink_k[r])
            if career_n > 0:
                cm = np.average(prior[r].values, weights=prior[pa_col].values)
                rec[f"career_{r}_shrunk"] = shrink(cm, career_n, mu, shrink_k[r])
            else:
                rec[f"career_{r}_shrunk"] = mu
            # Also keep the most recent observed rate for "Last_K%" etc display
            rec[f"Last_{r}"] = float(last[r])
        rec["log_p1_PA"]     = float(np.log1p(rec["p1_PA"]))
        rec["log_career_PA"] = float(np.log1p(rec["career_PA"]))
        rows.append(rec)
    return pd.DataFrame(rows)


def predict_rates(infer: pd.DataFrame, models: dict, scale: float,
                  target_name: str) -> pd.DataFrame:
    """Add Pred_<target> / SD_<target> columns to infer."""
    X = infer[FEATURE_COLS].values
    point = _predict_point(models["ridge"], models["hgb"], X)
    half  = (models["q84"].predict(X) - models["q16"].predict(X)) / 2.0
    sd    = np.maximum(scale * half, 0.001)
    infer = infer.copy()
    infer[f"Pred_{target_name}"] = np.clip(point, 0, 1)
    infer[f"SD_{target_name}"]   = sd
    return infer


# ─────────────────────────────────────────────────────────────────────────────
# HBP / SF — pure shrinkage with beta-binomial SD (no ML; sample too noisy)
# ─────────────────────────────────────────────────────────────────────────────

def project_simple_rate(df: pd.DataFrame, target_year: int, pa_col: str,
                        rate_col: str, prior_k: float, infer: pd.DataFrame,
                        fit_df: pd.DataFrame | None = None
                        ) -> pd.DataFrame:
    """Beta-binomial shrinkage. Returns infer with Pred_<rate>, SD_<rate>.

    The point estimate is (career_successes + k·mu) / (career_PA + k).
    The SD is the analytical posterior SD of a Beta(α, β) where α and β
    come from the observed events and the prior — this naturally inflates
    SD for players with few PA.

    fit_df, when given, is the frame used to compute the league mean μ (the
    beta-binomial prior). It defaults to ``df``. Pass a real-only frame here
    when ``df`` carries synthetic minor-league-translation rows so those rows
    still contribute to a player's own career total (via ``df``) but do not
    move the league prior.
    """
    df_sorted = df.sort_values(["PlayerId", "Season"]).copy()
    mu = league_mean(fit_df if fit_df is not None else df,
                     target_year, rate_col, pa_col)

    # Map each rate_col back to the underlying counting stat to get successes
    rate_to_count = {"K%": "K", "BB%": "BB", "HBP%": "HBP", "SF%": "SF"}
    count_col = rate_to_count[rate_col]

    # Aggregate career successes & trials per player from all priors
    priors = df_sorted[df_sorted["Season"] < target_year].copy()
    agg = priors.groupby("PlayerId").agg(
        s_career=(count_col, "sum"),
        n_career=(pa_col, "sum"),
    ).reset_index()

    out = infer.merge(agg, on="PlayerId", how="left")
    out["s_career"] = out["s_career"].fillna(0)
    out["n_career"] = out["n_career"].fillna(0)

    # Beta-binomial: prior is Beta(k·mu, k·(1-mu))
    alpha = out["s_career"] + prior_k * mu
    beta  = (out["n_career"] - out["s_career"]) + prior_k * (1 - mu)
    p_post = alpha / (alpha + beta)
    # Posterior SD of Beta(α, β) = sqrt(αβ / ((α+β)² (α+β+1)))
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    sd  = np.sqrt(var)

    out[f"Pred_{rate_col}"] = p_post.clip(0, 1)
    out[f"SD_{rate_col}"]   = sd
    out.drop(columns=["s_career", "n_career"], inplace=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# K% / BB% — PA-weighted recency-decayed projection with shrinkage to mean
# ─────────────────────────────────────────────────────────────────────────────

def project_rate_with_decay(df: pd.DataFrame, target_year: int, pa_col: str,
                            rate_col: str, prior_k: float, decay: float,
                            infer: pd.DataFrame,
                            max_history_years: int | None = None,
                            mu_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-player PA-weighted, recency-decayed projection.

    mu_df, when given, is the frame used to compute the league mean the model
    shrinks toward. It defaults to ``df``. Pass a real-only frame here when
    ``df`` also carries synthetic minor-league-translation rows, so those
    synthetic rows still act as a player's own prior (via ``df``) but do NOT
    shift the league mean.

    Replaces the Ridge+HGB approach for K% and BB% — empirically it has
    nearly identical MAE (within ~1 point) but substantially less bias on
    elite performers.

    Formula:
        weighted = Σ(decay^year_back · PA_t · rate_t) / Σ(decay^year_back · PA_t)
        pred    = (n_eff · weighted + prior_k · league_mean) / (n_eff + prior_k)
    where n_eff is the decay-weighted effective sample size.

    Parameters
    ----------
    max_history_years : int | None
        If set, only consider data from the last N seasons. Capping the
        window helps for players whose true skill has evolved over time
        (e.g., Aaron Judge developed into an elite walker around 2022-23
        and a 9-year history dilutes that signal too much). Combined with
        a lower prior_k, this preserves recent-era projections better.

    SD is derived from quantile residuals on a walk-forward, calibrated to
    ~68% empirical coverage.
    """
    df_sorted = df.sort_values(["PlayerId", "Season"]).copy()
    if max_history_years is not None:
        df_sorted = df_sorted[df_sorted["Season"] >= target_year - max_history_years]
    mu = league_mean(mu_df if mu_df is not None else df, target_year, rate_col, pa_col)
    priors = df_sorted[df_sorted["Season"] < target_year].copy()

    rows = []
    for pid, g in priors.groupby("PlayerId"):
        g = g.sort_values("Season")
        if len(g) == 0:
            rows.append({"PlayerId": pid, "_pred_decay": mu, "_n_eff": 0.0})
            continue
        years_back = (target_year - g["Season"].astype(int)).values
        decay_w    = decay ** years_back
        pa_vals    = g[pa_col].astype(float).values
        w          = decay_w * pa_vals
        if w.sum() <= 0:
            rows.append({"PlayerId": pid, "_pred_decay": mu, "_n_eff": 0.0})
            continue
        weighted_rate = float(np.average(g[rate_col].astype(float).values, weights=w))
        n_eff = float(w.sum())
        # Bayesian shrinkage to league mean
        pred = (n_eff * weighted_rate + prior_k * mu) / (n_eff + prior_k)
        rows.append({"PlayerId": pid, "_pred_decay": pred, "_n_eff": n_eff})
    pred_df = pd.DataFrame(rows)

    out = infer.merge(pred_df, on="PlayerId", how="left")
    out["_pred_decay"] = out["_pred_decay"].fillna(mu)
    out["_n_eff"] = out["_n_eff"].fillna(0.0)
    out[f"Pred_{rate_col}"] = out["_pred_decay"].clip(0, 1)
    out.drop(columns=["_pred_decay"], inplace=True)
    return out


def calibrate_decay_sd(df: pd.DataFrame, pa_col: str, rate_col: str,
                       prior_k: float, decay: float,
                       min_pa: int = 100,
                       max_history_years: int | None = None
                       ) -> tuple[pd.DataFrame, float]:
    """Walk-forward validation of the decay model. Returns (metrics, sd_scale).

    SD scale is empirically calibrated so the marginal residual / SE is ≈1.0
    at the 68th percentile, matching the rate model SD interpretation as a
    1-sigma interval covering ~68% of true values.
    """
    seasons = sorted(df["Season"].unique())
    if len(seasons) < 2:
        return pd.DataFrame(), 1.0

    z_resid_pool = []
    weights_pool = []
    rows = []
    for ty in seasons[1:]:
        # Project ty using only prior data, optionally limited to recent window
        prior_only = df[df["Season"] < ty].copy()
        if max_history_years is not None:
            prior_only = prior_only[prior_only["Season"] >= ty - max_history_years]
        if prior_only.empty:
            continue
        # Manually project for each player active in ty
        target_rows = df[(df["Season"] == ty) & (df[pa_col] >= min_pa)]
        if target_rows.empty:
            continue
        mu_ty = league_mean(df, ty, rate_col, pa_col)
        preds = []
        truths = []
        ses = []
        pas = []
        for _, tr in target_rows.iterrows():
            pid = tr["PlayerId"]
            g = prior_only[prior_only["PlayerId"] == pid].sort_values("Season")
            if g.empty:
                pred = mu_ty
                n_eff = 0.0
            else:
                yb = (ty - g["Season"].astype(int)).values
                pa = g[pa_col].astype(float).values
                w = (decay ** yb) * pa
                if w.sum() <= 0:
                    pred = mu_ty
                    n_eff = 0.0
                else:
                    wr = float(np.average(g[rate_col].astype(float).values, weights=w))
                    n_eff = float(w.sum())
                    pred = (n_eff * wr + prior_k * mu_ty) / (n_eff + prior_k)
            n_total = n_eff + prior_k
            se = np.sqrt(max(pred * (1 - pred), 1e-6) / max(n_total, 10))
            preds.append(pred); truths.append(tr[rate_col])
            ses.append(se); pas.append(tr[pa_col])
        preds = np.array(preds); truths = np.array(truths)
        ses = np.array(ses); pas = np.array(pas)
        w = np.sqrt(pas)
        rows.append({
            "year": ty,
            "n": len(preds),
            "wMAE_model": float(np.average(np.abs(preds - truths), weights=w)),
            "wR2_model":  1 - np.average((preds - truths) ** 2, weights=w) /
                            max(np.average((truths - np.average(truths, weights=w)) ** 2,
                                           weights=w), 1e-9),
        })
        # Z residuals
        z_resid_pool.append(np.abs(truths - preds) / np.maximum(ses, 1e-6))
        weights_pool.append(w)
    if not z_resid_pool:
        return pd.DataFrame(rows), 1.0
    z_all = np.concatenate(z_resid_pool)
    w_all = np.concatenate(weights_pool)
    order = np.argsort(z_all)
    zs, ws = z_all[order], w_all[order]
    cum = np.cumsum(ws) / ws.sum()
    idx = min(int(np.searchsorted(cum, 0.68)), len(zs) - 1)
    return pd.DataFrame(rows), float(zs[idx])


def fit_and_predict_decay_rate(df: pd.DataFrame, target_year: int, pa_col: str,
                                rate_col: str, prior_k: float, decay: float,
                                infer: pd.DataFrame,
                                max_history_years: int | None = None,
                                fit_df: pd.DataFrame | None = None
                                ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """End-to-end: validate via walk-forward, then project for target_year.

    Parameters
    ----------
    max_history_years : int | None
        Limits the per-player history considered. None means use all
        available history; setting it to e.g. 5 means only the last 5
        seasons contribute. Combined with a lower prior_k, this is
        empirically better for players whose true skill has been evolving.
    fit_df : DataFrame | None
        Frame used for walk-forward SD calibration and the league mean. Defaults
        to ``df``. Pass a real-only frame here when ``df`` carries synthetic
        minor-league-translation rows: the synthetic rows then serve only as a
        player's own prior in the projection and never leak into calibration or
        the league mean.

    Returns (infer_with_predictions, walk_forward_metrics, sd_scale).
    """
    fit = fit_df if fit_df is not None else df
    metrics, sd_scale = calibrate_decay_sd(fit, pa_col, rate_col, prior_k, decay,
                                            max_history_years=max_history_years)

    out = project_rate_with_decay(df, target_year, pa_col, rate_col,
                                   prior_k, decay, infer,
                                   max_history_years=max_history_years,
                                   mu_df=fit)
    # Standard error from binomial: sqrt(p(1-p)/N) with N = n_eff + prior_k
    p = out[f"Pred_{rate_col}"].values
    n_total = (out["_n_eff"].fillna(0).values + prior_k)
    se = np.sqrt(np.maximum(p * (1 - p), 1e-6) / np.maximum(n_total, 10))
    out[f"SD_{rate_col}"] = sd_scale * se
    out.drop(columns=["_n_eff"], inplace=True, errors="ignore")
    return out, metrics, sd_scale

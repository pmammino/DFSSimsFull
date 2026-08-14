#!/usr/bin/env python3
"""
eval_ownership_calibration.py — out-of-sample calibration check for the
projected-ownership model against real DK contest %Drafted.
=====================================================================

Unlike ``fit_ownership.py`` (which *refits* the conditional-logit coefficients),
this script *scores the shipped model as-is* (``ownership_params.json``) on a set
of contest-standings CSVs, so it answers the operational question: **is the
ownership we feed the sim/portfolio actually calibrated to the field?**

For every contest it

  1. maps the contest to a slate ``date`` by name-overlap with a feature log
     (``ownership_history`` schema: date,name,pos,proj,ceiling,team_total,order…),
  2. rebuilds the model's per-slot conditional-logit ownership from the logged
     features using the *shipped* betas / tau / chalk_k, at the contest's field
     size, scaling each slot to its realised total (so the comparison isolates
     the within-slot shape/level the model controls), and
  3. compares projected vs actual %Drafted: rank (Spearman), error (MAE/RMSE),
     top-decile hit-rate, a reliability curve, and a concentration index
     (does the field pile onto chalk more than the model says?).

Usage
-----
    FEATURES=ownership_history_features.csv \
    CONTEST_DIR=Contests \
    python3 scripts/eval_ownership_calibration.py [--drop-multislate CID ...]

The feature log needs one row per (date, player) with at least
``date,name,pos,proj,ceiling,team_total,order``. Salary is optional; when it is
absent the ``value`` term is dropped exactly as the production model does.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ownership_model import (  # noqa: E402
    norm, load_params, _z, size_beta, _reshape_slot,
)


def parse_contest(path: str):
    raw = pd.read_csv(path)
    n_entries = int(raw["EntryId"].notna().sum())
    df = raw.dropna(subset=["Player"]).copy()
    df["nname"] = df["Player"].map(norm)
    df["own_act"] = df["%Drafted"].astype(str).str.rstrip("%").astype(float)
    df["Pos"] = df["Roster Position"].astype(str)
    return n_entries, df[["nname", "Player", "Pos", "own_act"]]


def map_date(pset, feat_by_date):
    best, bov = None, -1.0
    for d, s in feat_by_date.items():
        ov = len(pset & s) / max(1, len(pset))
        if ov > bov:
            best, bov = d, ov
    return best, bov


def project_slot(g, betas, tau, n_entries, params):
    """Shipped conditional-logit ownership for one (contest, slot) group,
    scaled to the slot's realised total and reshaped for field size."""
    u = np.zeros(len(g))
    used = []
    for f in ("proj", "ceil_shape", "value", "team_total", "order_score"):
        b = betas.get(f, 0.0)
        if b == 0.0 or f not in g.columns:
            continue
        col = g[f].to_numpy(dtype=float)
        if not np.isfinite(col).any():
            continue
        u = u + b * _z(np.where(np.isfinite(col), col, np.nanmean(col)))
        used.append(f)
    u = u / max(tau, 1e-6)
    u = u - np.nanmax(u)
    ex = np.exp(np.clip(u, -50, 50))
    total = g["own_act"].sum()
    own = ex / ex.sum() * total
    beta = size_beta(n_entries, params.n_medium, params.chalk_k)
    own = _reshape_slot(own, beta)
    if own.sum() > 0:
        own = own * (total / own.sum())
    return own, used


def metrics(pred, act):
    sp = stats.spearmanr(pred, act).correlation
    mae = float(np.abs(pred - act).mean())
    rmse = float(np.sqrt(((pred - act) ** 2).mean()))
    k = max(1, int(0.10 * len(act)))
    hit = len(set(np.argsort(-act)[:k]) & set(np.argsort(-pred)[:k])) / k
    return sp, mae, rmse, hit


def build(features_csv, contest_dir, drop=()):
    P = load_params()
    feat = pd.read_csv(features_csv)
    feat["nname"] = feat["name"].map(norm)
    feat["ceil_shape"] = (feat["ceiling"] / feat["proj"]).clip(1.0, 6.0)
    feat_by_date = {d: set(feat[feat.date == d]["nname"]) for d in feat.date.unique()}

    rows = []
    for path in sorted(glob.glob(os.path.join(contest_dir, "*.csv"))):
        cid = re.search(r"(\d+)", os.path.basename(path)).group(1)
        if cid in drop:
            continue
        n_entries, cdf = parse_contest(path)
        date, ov = map_date(set(cdf["nname"]), feat_by_date)
        fmap = feat[feat.date == date].drop_duplicates("nname").set_index("nname")
        for pos, g in cdf.groupby("Pos"):
            g = g.drop_duplicates("nname")
            g = g[g["nname"].isin(fmap.index)].copy()
            if len(g) < 2:
                continue
            g["proj"] = fmap.loc[g["nname"], "proj"].to_numpy()
            g["ceil_shape"] = fmap.loc[g["nname"], "ceil_shape"].to_numpy()
            g["team_total"] = fmap.loc[g["nname"], "team_total"].to_numpy()
            if "salary" in fmap.columns and fmap["salary"].notna().any():
                sal = fmap.loc[g["nname"], "salary"].to_numpy(dtype=float)
                g["value"] = g["proj"] / (sal / 1000.0)
            o = fmap.loc[g["nname"], "order"].to_numpy(dtype=float)
            g["order_score"] = np.where((o >= 1) & (o <= 9), 10.0 - o, 0.0)
            is_p = pos == "P"
            pred, _ = project_slot(g, P.pit if is_p else P.hit,
                                   P.tau_for(is_p), n_entries, P)
            g["own_pred"] = pred
            g["contest"] = cid
            g["date"] = date
            g["overlap"] = round(ov, 2)
            g["n_entries"] = n_entries
            g["is_pitcher"] = is_p
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def report(R):
    def hhi(v):
        v = np.asarray(v, float)
        t = v.sum()
        return ((v / t) ** 2).sum() if t > 0 else np.nan

    print(f"\nEvaluated {len(R)} player-rows across {R.contest.nunique()} contests "
          f"/ {R.date.nunique()} slates\n")
    print("=== pooled ===")
    for lbl, sub in [("ALL", R), ("HITTERS", R[~R.is_pitcher]),
                     ("PITCHERS", R[R.is_pitcher])]:
        sp, mae, rmse, hit = metrics(sub.own_pred.to_numpy(), sub.own_act.to_numpy())
        print(f"  {lbl:9s} n={len(sub):4d}  Spearman={sp:.3f}  MAE={mae:4.2f}%  "
              f"RMSE={rmse:4.2f}%  top10hit={hit:.2f}")

    print("\n=== per contest ===")
    for cid, g in R.groupby("contest"):
        sp, mae, rmse, hit = metrics(g.own_pred.to_numpy(), g.own_act.to_numpy())
        print(f"  {cid} {g.date.iloc[0]} {g.n_entries.iloc[0]:5d}e ov={g.overlap.iloc[0]:.2f} "
              f"n={len(g):4d}  Spearman={sp:.3f}  MAE={mae:4.2f}%  top10hit={hit:.2f}")

    print("\n=== concentration (mean per contest+slot; HHI higher = chalkier) ===")
    for lbl, sub in [("HITTERS", R[~R.is_pitcher]), ("PITCHERS", R[R.is_pitcher])]:
        ah = sub.groupby(["contest", "Pos"])["own_act"].apply(hhi).mean()
        ph = sub.groupby(["contest", "Pos"])["own_pred"].apply(hhi).mean()
        print(f"  {lbl}: actual_HHI={ah:.3f}  model_HHI={ph:.3f}  "
              f"model/actual={ph/ah:.2f}")

    print("\n=== reliability — HITTERS (bin by predicted) ===")
    H = R[~R.is_pitcher].copy()
    H["bin"] = pd.cut(H.own_pred, [0, 2, 4, 6, 10, 15, 25, 100])
    t = H.groupby("bin", observed=True).agg(
        n=("own_act", "size"), pred=("own_pred", "mean"), act=("own_act", "mean"))
    t["gap"] = t["pred"] - t["act"]
    print(t.round(2).to_string())

    print("\n=== reliability — PITCHERS (bin by predicted) ===")
    Pt = R[R.is_pitcher].copy()
    Pt["bin"] = pd.cut(Pt.own_pred, [0, 5, 10, 20, 30, 100])
    t = Pt.groupby("bin", observed=True).agg(
        n=("own_act", "size"), pred=("own_pred", "mean"), act=("own_act", "mean"))
    t["gap"] = t["pred"] - t["act"]
    print(t.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.environ.get("FEATURES", ""))
    ap.add_argument("--contest-dir", default=os.environ.get("CONTEST_DIR", "Contests"))
    ap.add_argument("--drop-multislate", nargs="*", default=[],
                    help="contest ids to drop (e.g. mismatched multi-slate days)")
    args = ap.parse_args()
    if not args.features:
        sys.exit("set --features / FEATURES to the ownership feature log CSV")
    R = build(args.features, args.contest_dir, drop=set(args.drop_multislate))
    report(R)


if __name__ == "__main__":
    main()

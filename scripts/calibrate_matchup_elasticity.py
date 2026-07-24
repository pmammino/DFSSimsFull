#!/usr/bin/env python3
"""calibrate_matchup_elasticity.py — data-derived opponent-quality elasticities.

Backs the constants in ``matchup.OPP_MATCHUP_ELASTICITY``. A pitcher's allowed
rates depend on the QUALITY of the lineup he faces, combined via the log5 /
odds-ratio matchup:

    logit(rate) = logit(pitcher_rate) + gamma * (logit(batter_rate) - logit(league))

log5 theory says gamma = 1. This script MEASURES the batter-side gamma
out-of-sample on Statcast batted-ball logs so it isn't a guess:

  1. estimate each batter's & pitcher's contact rate on 2024 (empirical-Bayes
     shrinkage toward the league rate),
  2. on 2025 balls-in-play, fit  logit P(event) = b0 + bB*offdev + bP*pitdev
     with offdev/pitdev the 2024 log-odds deviations (so the predictor never
     sees the outcome it scores).

`bB` is the batter/opponent-side elasticity. Findings (see the module docstring
in matchup.py): HR ~1.0 (full log5), balls-in-play hits ~0.7 (DIPS — pitchers
have limited BABIP control). K/BB are absent from balls-in-play logs and use the
theoretical log5 value 1.0.

Run:  python scripts/calibrate_matchup_elasticity.py   (needs bip_inputs/*.csv)
"""
import numpy as np
import pandas as pd

HITS = {"single", "double", "triple", "home_run"}
XBH = {"double", "triple", "home_run"}


def _load(season, bip_dir="bip_inputs"):
    df = pd.read_csv(f"{bip_dir}/bip_{season}.csv",
                     usecols=["batter", "pitcher", "events"])
    return df[df["events"].notna()]


def _eb(k, n, L, prior):
    return (k + L * prior) / (n + prior)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _irls_logit(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ b)))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = X @ b + (y - mu) / W
        XtW = X.T * W
        nb = np.linalg.solve(XtW @ X + 1e-8 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(nb - b)) < 1e-8:
            b = nb
            break
        b = nb
    mu = 1 / (1 + np.exp(-(X @ b)))
    W = np.clip(mu * (1 - mu), 1e-9, None)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * W) @ X)))
    return b, se


def calibrate(event_fn, train_season=2024, test_season=2025,
              prior_bat=250, prior_pit=350, min_bip=120, bip_dir="bip_inputs"):
    tr, te = _load(train_season, bip_dir).copy(), _load(test_season, bip_dir).copy()
    tr["ev"] = event_fn(tr["events"]).astype(float)
    te["ev"] = event_fn(te["events"]).astype(float)
    L = tr["ev"].mean()

    def rate_map(df, col, prior):
        g = df.groupby(col)["ev"].agg(["sum", "count"])
        g = g[g["count"] >= min_bip]
        return (_logit(_eb(g["sum"], g["count"], L, prior)) - _logit(L)).to_dict()

    boff, poff = rate_map(tr, "batter", prior_bat), rate_map(tr, "pitcher", prior_pit)
    te = te[te["batter"].isin(boff) & te["pitcher"].isin(poff)]
    X = np.column_stack([np.ones(len(te)),
                         te["batter"].map(boff).to_numpy(),
                         te["pitcher"].map(poff).to_numpy()])
    b, se = _irls_logit(X, te["ev"].to_numpy())
    return dict(L=L, n_test=len(te), bB=b[1], seB=se[1], bP=b[2], seP=se[2])


if __name__ == "__main__":
    print(f"{'event':6s} {'L':>6} {'n_test':>8} {'bB(opp)':>14} {'bP(pitcher)':>14}")
    for name, fn in [("HR", lambda e: e.eq("home_run")),
                     ("hit", lambda e: e.isin(HITS)),
                     ("XBH", lambda e: e.isin(XBH))]:
        r = calibrate(fn)
        print(f"{name:6s} {r['L']:6.3f} {r['n_test']:8d} "
              f"{r['bB']:6.3f} ± {r['seB']:.3f} {r['bP']:6.3f} ± {r['seP']:.3f}")
    print("\nlog5 predicts bB=bP=1. bB (measured with error -> a lower bound) is the "
          "opponent-quality elasticity used in matchup.OPP_MATCHUP_ELASTICITY.")

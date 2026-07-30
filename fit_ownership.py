#!/usr/bin/env python3
"""
fit_ownership.py — calibrate & validate the ownership model on real contests
============================================================================

Fits the conditional-logit coefficients in ``ownership_model.OwnershipParams``
against actual DraftKings ``%Drafted`` from contest-standings CSVs, using the
sim history for the matching slate as the feature source, then reports
leave-one-slate-out validation.

Data (see deliverables/ownership_model/README.md)
-------------------------------------------------
  HIST_DIR     dir of  history_<date>_{hitter,pitcher}_dk_sims.npy   (the sims)
  CONTEST_DIR  dir of  contest-standings-*.csv                        (targets)

Each contest CSV's right-hand block (Player, Roster Position, %Drafted, FPTS)
is the realised field ownership. We map each contest to its slate date by
player-name overlap with that day's sims, build the sim features, and fit.

What is and isn't fitted here
-----------------------------
The calibration set (Jul 26–29 2026) has sims + ownership + position, but NOT
salary or Vegas totals for those days. So this harness fits the two purely
sim-derived coefficients — ``proj`` and ``ceil_shape`` — for hitters and
pitchers, and estimates ``chalk_k`` from the two same-slate size pairs. The
``value`` and ``team_total`` betas keep their domain-prior defaults; rerun this
with salary/Vegas columns joined in (``--with-cost``) to fit them too.

Usage
-----
    HIST_DIR=... CONTEST_DIR=... python3 fit_ownership.py            # fit+report
    HIST_DIR=... CONTEST_DIR=... python3 fit_ownership.py --write    # + save json
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
from scipy.optimize import minimize

from ownership_model import (
    OwnershipParams, SLOT_COUNT, norm, sim_features, size_beta, _z,
    project_ownership,
)

HIST_DIR = os.environ.get("HIST_DIR", "./History")
CONTEST_DIR = os.environ.get("CONTEST_DIR", "./Contests")
FIT_FEATURES = ["proj", "ceil_shape"]          # fittable from sims alone


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_sims_for_date(date: str) -> dict:
    out = {}
    for kind in ("hitter", "pitcher"):
        f = os.path.join(HIST_DIR, f"history_{date}_{kind}_dk_sims.npy")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True).item()
            out.update({norm(k): np.asarray(v, float) for k, v in d.items()})
    return out


def available_dates() -> list[str]:
    ds = set()
    for f in glob.glob(os.path.join(HIST_DIR, "history_*_hitter_dk_sims.npy")):
        m = re.search(r"history_(\d{4}-\d{2}-\d{2})_", f)
        if m:
            ds.add(m.group(1))
    return sorted(ds)


def parse_contest(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # entry count comes from the LEFT (standings) block — one row per lineup —
    # and must be read before the right-block Player dropna collapses the frame.
    n_entries = int(raw["EntryId"].notna().sum())
    df = raw.dropna(subset=["Player"]).copy()
    df["own"] = df["%Drafted"].astype(str).str.rstrip("%").astype(float)
    df["nname"] = df["Player"].map(norm)
    df["Pos"] = df["Roster Position"].astype(str)
    df["n_entries"] = n_entries
    return df[["Player", "nname", "Pos", "own", "n_entries"]]


def map_contest_to_date(df: pd.DataFrame, sims_by_date: dict) -> tuple[str, float]:
    pset = set(df["nname"])
    best, bov = None, -1.0
    for d, sims in sims_by_date.items():
        ov = len(pset & set(sims)) / max(1, len(pset))
        if ov > bov:
            best, bov = d, ov
    return best, bov


# ---------------------------------------------------------------------------
# assemble the (slate, slot) softmax problems
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    dates = available_dates()
    if not dates:
        sys.exit(f"no sim history in HIST_DIR={HIST_DIR}")
    sims_by_date = {d: load_sims_for_date(d) for d in dates}

    rows = []
    for path in sorted(glob.glob(os.path.join(CONTEST_DIR, "*.csv"))):
        cid = re.search(r"(\d+)", os.path.basename(path)).group(1)
        df = parse_contest(path)
        date, ov = map_contest_to_date(df, sims_by_date)
        sims = sims_by_date[date]
        for r in df.itertuples():
            sc = sims.get(r.nname)
            if sc is None:
                continue
            f = sim_features(sc)
            rows.append({
                "contest": cid, "date": date, "overlap": round(ov, 2),
                "n_entries": r.n_entries, "name": r.nname, "pos": r.Pos,
                "is_pitcher": r.Pos == "P", "own": r.own,
                "proj": f["proj"], "ceil_shape": f["ceil_shape"],
            })
    data = pd.DataFrame(rows)
    print(f"matched {len(data)} player-rows across "
          f"{data['contest'].nunique()} contests / {data['date'].nunique()} slates")
    return data


# ---------------------------------------------------------------------------
# conditional-logit fit (cross-entropy over per-slot softmax groups)
# ---------------------------------------------------------------------------
def _groups(data: pd.DataFrame):
    """Yield (feature matrix Z, target share p, slot_total) per (contest, slot)."""
    for (cid, pos), g in data.groupby(["contest", "pos"]):
        if len(g) < 2:
            continue
        slot_total = 100.0 * SLOT_COUNT.get(pos, 1)
        Z = np.column_stack([_z(g[f].to_numpy()) for f in FIT_FEATURES])
        p = g["own"].to_numpy() / g["own"].sum() if g["own"].sum() > 0 else \
            np.full(len(g), 1.0 / len(g))
        yield Z, p


def fit_betas(data: pd.DataFrame) -> dict:
    groups = list(_groups(data))

    def nll(beta):
        tot = 0.0
        for Z, p in groups:
            u = Z @ beta
            u -= u.max()
            q = np.exp(u)
            q /= q.sum()
            tot += -(p * np.log(np.clip(q, 1e-12, None))).sum()
        return tot

    # non-negative bounds: more projection / more relative ceiling can only
    # raise attractiveness, never lower it (guards against small-sample sign
    # flips like a spuriously negative pitcher ceil_shape).
    res = minimize(nll, np.full(len(FIT_FEATURES), 0.3), method="L-BFGS-B",
                   bounds=[(0.0, 5.0)] * len(FIT_FEATURES))
    return dict(zip(FIT_FEATURES, res.x))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def predict_group(g: pd.DataFrame, betas: dict) -> np.ndarray:
    u = np.zeros(len(g))
    for f in FIT_FEATURES:
        u = u + betas[f] * _z(g[f].to_numpy())
    u -= u.max()
    q = np.exp(u)
    q /= q.sum()
    return q * g["own"].sum()          # scale to the slot's realised total


def evaluate(data: pd.DataFrame, betas_by_kind: dict, label: str):
    print(f"\n===== {label} =====")
    for kind, isp in (("HIT", False), ("PIT", True)):
        sub = data[data["is_pitcher"] == isp]
        preds, acts = [], []
        for (cid, pos), g in sub.groupby(["contest", "pos"]):
            if len(g) < 2:
                continue
            pred = predict_group(g, betas_by_kind[kind])
            preds.append(pred)
            acts.append(g["own"].to_numpy())
        if not preds:
            continue
        pred = np.concatenate(preds)
        act = np.concatenate(acts)
        sp = stats.spearmanr(pred, act).correlation
        mae = np.abs(pred - act).mean()
        # top-decile identification (chalk hit-rate)
        k = max(1, int(0.10 * len(act)))
        top_act = set(np.argsort(-act)[:k])
        top_pred = set(np.argsort(-pred)[:k])
        hit = len(top_act & top_pred) / k
        print(f"  {kind}: n={len(act):4d}  Spearman={sp:.3f}  MAE={mae:4.2f}%  "
              f"top10%-hit={hit:.2f}  betas={ {f: round(betas_by_kind[kind][f],2) for f in FIT_FEATURES} }")


def cross_validate(data: pd.DataFrame):
    """Leave-one-slate-out: fit on 3 days, predict the held-out day."""
    print("\n===== leave-one-slate-out validation =====")
    for kind, isp in (("HIT", False), ("PIT", True)):
        sps, maes, hits, ns = [], [], [], []
        for d in sorted(data["date"].unique()):
            train = data[(data["date"] != d)]
            test = data[(data["date"] == d) & (data["is_pitcher"] == isp)]
            if len(test) < 3:
                continue
            betas = fit_betas(train[train["is_pitcher"] == isp])
            preds, acts = [], []
            for (cid, pos), g in test.groupby(["contest", "pos"]):
                if len(g) < 2:
                    continue
                preds.append(predict_group(g, betas))
                acts.append(g["own"].to_numpy())
            pred = np.concatenate(preds)
            act = np.concatenate(acts)
            sp = stats.spearmanr(pred, act).correlation
            k = max(1, int(0.10 * len(act)))
            hit = len(set(np.argsort(-act)[:k]) & set(np.argsort(-pred)[:k])) / k
            sps.append(sp)
            maes.append(np.abs(pred - act).mean())
            hits.append(hit)
            ns.append(len(act))
            print(f"  {kind} holdout {d}: n={len(act):4d}  Spearman={sp:.3f}  "
                  f"MAE={np.abs(pred-act).mean():4.2f}%  top10%-hit={hit:.2f}")
        if sps:
            w = np.array(ns)
            print(f"  {kind} OOS mean: Spearman={np.average(sps,weights=w):.3f}  "
                  f"MAE={np.average(maes,weights=w):.2f}%  top10%-hit={np.average(hits,weights=w):.2f}")


def estimate_chalk_k(data: pd.DataFrame, n_medium: int) -> float:
    """Estimate k in beta(N)=1-k*log10(N/n_medium) from same-slate size pairs.

    For a same-slate pair, own_large ≈ own_small^(beta_l/beta_s); fitting that
    exponent and attributing it to the size gap gives k.
    """
    ks, ws = [], []
    for date, g in data.groupby("date"):
        sizes = sorted(g["n_entries"].unique())
        if len(sizes) < 2:
            continue
        lo, hi = sizes[0], sizes[-1]
        a = g[(g.n_entries == lo) & (~g.is_pitcher)][["name", "own"]]
        b = g[(g.n_entries == hi) & (~g.is_pitcher)][["name", "own"]]
        m = a.merge(b, on="name", suffixes=("_s", "_l"))
        m = m[(m.own_s > 0.1) & (m.own_l > 0.1)]
        if len(m) < 6:
            continue
        expo = np.polyfit(np.log(m.own_s), np.log(m.own_l), 1)[0]
        # own_l = own_s^expo, and own(N)=base^beta(N) => expo=beta(hi)/beta(lo).
        # anchor beta(lo)=1 (small≈baseline chalk); k from the size gap.
        k = (1.0 - expo) / (np.log10(hi / n_medium) - np.log10(lo / n_medium)) \
            if hi != lo else 0.0
        ks.append(k)
        ws.append(len(m))     # trust a pair in proportion to its matched players
        print(f"  chalk pair {date}: {lo}->{hi} entries  own_l~own_s^{expo:.2f}  "
              f"k~{k:.2f}  (matched {len(m)})")
    if not ks:
        return 0.20
    k = float(np.average(ks, weights=ws))
    return float(np.clip(k, 0.05, 0.6))     # keep the reshape gentle (few pairs)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="save fitted params to ownership_params.json")
    ap.add_argument("--n-medium", type=int, default=3000)
    args = ap.parse_args()

    data = build_dataset()

    # in-sample fit
    betas = {"HIT": fit_betas(data[~data.is_pitcher]),
             "PIT": fit_betas(data[data.is_pitcher])}
    evaluate(data, betas, "in-sample fit (all 4 slates)")

    # honest generalisation
    cross_validate(data)

    print("\n===== contest-size chalk (k) =====")
    k = estimate_chalk_k(data, args.n_medium)
    print(f"  estimated chalk_k = {k:.3f}")

    if args.write:
        P = OwnershipParams()
        P.n_medium = args.n_medium
        P.chalk_k = round(k, 3)
        for f in FIT_FEATURES:
            P.hit[f] = round(betas["HIT"][f], 3)
            P.pit[f] = round(betas["PIT"][f], 3)
        P.to_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ownership_params.json"))
        print("\nwrote ownership_params.json:", P.hit, P.pit,
              "chalk_k=", P.chalk_k)


if __name__ == "__main__":
    main()

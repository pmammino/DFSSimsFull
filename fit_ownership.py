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
  DFF_DIR      (optional) dir of DailyFantasyFuel cheatsheets
               `*DFF_MLB_cheatsheet_YYYYMMDD.csv` supplying per-player DK
               `salary` + Vegas `implied_team_score` for the slate. When
               present, the `value` (proj/salary) and `team_total` features are
               fitted too; without it only the sim-derived features are fitted.

Each contest CSV's right-hand block (Player, Roster Position, %Drafted, FPTS)
is the realised field ownership. We map each contest to its slate date by
player-name overlap with that day's sims, build the features, and fit.

The fit
-------
Ownership within a roster slot is a softmax over player attractiveness
`u = Σ β·z(feature)` (features standardised within slate/slot). β is fit by
minimising cross-entropy between predicted and actual per-slot shares, pooled
across all slate/slot groups, separately for hitters and pitchers, with
non-negative coefficients. When DFF salary/Vegas are supplied the harness fits
BOTH a sim-only model and the full model on the salary-covered rows and reports
both, so the lift from cost/context is explicit.

Usage
-----
    HIST_DIR=... CONTEST_DIR=... [DFF_DIR=...] python3 fit_ownership.py
    HIST_DIR=... CONTEST_DIR=... [DFF_DIR=...] python3 fit_ownership.py --write
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
)

HIST_DIR = os.environ.get("HIST_DIR", "./History")
CONTEST_DIR = os.environ.get("CONTEST_DIR", "./Contests")
DFF_DIR = os.environ.get("DFF_DIR", "")

# feature sets per player kind. proj/ceil_shape are always available (sims);
# value/team_total require DFF salary + Vegas. Pitchers get no team_total.
SIM_FEATURES = ["proj", "ceil_shape"]
FULL_HIT = ["proj", "ceil_shape", "value", "team_total", "order_score"]
FULL_PIT = ["proj", "ceil_shape", "value"]


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


def load_dff() -> dict:
    """{date -> {norm(name): (salary, implied_team_score)}} from DFF sheets.

    A day can have several classic slates (e.g. a main and an early sheet,
    ``..._YYYYMMDD.csv`` and ``..._YYYYMMDD_1.csv``) over disjoint games. They
    are merged into one per-day salary map (first sheet wins on the rare
    duplicate name), so all of the day's players carry a salary.
    """
    out = {}
    if not DFF_DIR:
        return out
    for f in sorted(glob.glob(os.path.join(DFF_DIR, "*DFF_MLB_cheatsheet_*.csv"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        d = m.group(1)
        date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        x = pd.read_csv(f)
        nn = (x["first_name"].astype(str) + " " + x["last_name"].astype(str)).map(norm)
        sal = pd.to_numeric(x["salary"], errors="coerce")
        itt = pd.to_numeric(x["implied_team_score"], errors="coerce")
        order = pd.to_numeric(x.get("confirmed_order"), errors="coerce") \
            if "confirmed_order" in x else pd.Series(np.nan, index=x.index)
        day = out.setdefault(date, {})
        for n, s, t, o in zip(nn, sal, itt, order):
            day.setdefault(n, (s, t, o))
    return out


def _order_score(o) -> float:
    try:
        o = float(o)
    except (TypeError, ValueError):
        return 0.0
    return (10.0 - o) if 1.0 <= o <= 9.0 else 0.0


def parse_contest(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
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
    dff = load_dff()

    rows = []
    for path in sorted(glob.glob(os.path.join(CONTEST_DIR, "*.csv"))):
        cid = re.search(r"(\d+)", os.path.basename(path)).group(1)
        df = parse_contest(path)
        date, ov = map_contest_to_date(df, sims_by_date)
        sims = sims_by_date[date]
        sal_map = dff.get(date, {})
        for r in df.itertuples():
            sc = sims.get(r.nname)
            if sc is None:
                continue
            f = sim_features(sc)
            salary, itt, order = sal_map.get(r.nname, (np.nan, np.nan, np.nan))
            value = f["proj"] / (salary / 1000.0) if salary and salary > 0 else np.nan
            rows.append({
                "contest": cid, "date": date, "overlap": round(ov, 2),
                "n_entries": r.n_entries, "name": r.nname, "pos": r.Pos,
                "is_pitcher": r.Pos == "P", "own": r.own,
                "proj": f["proj"], "ceil_shape": f["ceil_shape"],
                "salary": salary, "value": value, "team_total": itt,
                "order_score": _order_score(order),
            })
    data = pd.DataFrame(rows)
    ncov = data["salary"].notna().sum()
    print(f"matched {len(data)} player-rows across "
          f"{data['contest'].nunique()} contests / {data['date'].nunique()} slates"
          f"  ({ncov} with DFF salary)" if dff else
          f"matched {len(data)} player-rows across "
          f"{data['contest'].nunique()} contests / {data['date'].nunique()} slates")
    return data


# ---------------------------------------------------------------------------
# conditional-logit fit (cross-entropy over per-slot softmax groups)
# ---------------------------------------------------------------------------
def _groups(data: pd.DataFrame, features: list[str]):
    """Yield (Z, target share p) per (contest, slot) with complete features."""
    need = data.dropna(subset=features)
    for (cid, pos), g in need.groupby(["contest", "pos"]):
        if len(g) < 2:
            continue
        Z = np.column_stack([_z(g[f].to_numpy()) for f in features])
        p = g["own"].to_numpy() / g["own"].sum() if g["own"].sum() > 0 else \
            np.full(len(g), 1.0 / len(g))
        yield Z, p


def fit_betas(data: pd.DataFrame, features: list[str]) -> dict:
    groups = list(_groups(data, features))

    def nll(beta):
        tot = 0.0
        for Z, p in groups:
            u = Z @ beta
            u -= u.max()
            q = np.exp(u)
            q /= q.sum()
            tot += -(p * np.log(np.clip(q, 1e-12, None))).sum()
        return tot

    res = minimize(nll, np.full(len(features), 0.3), method="L-BFGS-B",
                   bounds=[(0.0, 5.0)] * len(features))
    return dict(zip(features, res.x))


def predict_group(g: pd.DataFrame, betas: dict, features: list[str]) -> np.ndarray:
    u = np.zeros(len(g))
    for f in features:
        u = u + betas[f] * _z(g[f].to_numpy())
    u -= u.max()
    q = np.exp(u)
    q /= q.sum()
    return q * g["own"].sum()          # scale to the slot's realised total


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _metrics(pred, act):
    sp = stats.spearmanr(pred, act).correlation
    mae = np.abs(pred - act).mean()
    k = max(1, int(0.10 * len(act)))
    hit = len(set(np.argsort(-act)[:k]) & set(np.argsort(-pred)[:k])) / k
    return sp, mae, hit


def evaluate(data: pd.DataFrame, betas_by_kind: dict, feats_by_kind: dict, label: str):
    print(f"\n===== {label} =====")
    for kind, isp in (("HIT", False), ("PIT", True)):
        feats = feats_by_kind[kind]
        sub = data[data["is_pitcher"] == isp].dropna(subset=feats)
        preds, acts = [], []
        for (cid, pos), g in sub.groupby(["contest", "pos"]):
            if len(g) < 2:
                continue
            preds.append(predict_group(g, betas_by_kind[kind], feats))
            acts.append(g["own"].to_numpy())
        if not preds:
            continue
        pred, act = np.concatenate(preds), np.concatenate(acts)
        sp, mae, hit = _metrics(pred, act)
        bs = {f: round(betas_by_kind[kind][f], 2) for f in feats}
        print(f"  {kind}: n={len(act):4d}  Spearman={sp:.3f}  MAE={mae:4.2f}%  "
              f"top10%-hit={hit:.2f}  betas={bs}")


def cross_validate(data: pd.DataFrame, feats_by_kind: dict, label: str):
    print(f"\n===== leave-one-slate-out — {label} =====")
    out = {}
    for kind, isp in (("HIT", False), ("PIT", True)):
        feats = feats_by_kind[kind]
        sps, maes, hits, ns = [], [], [], []
        for d in sorted(data["date"].unique()):
            train = data[data["date"] != d]
            test = data[(data["date"] == d) & (data["is_pitcher"] == isp)].dropna(subset=feats)
            if len(test) < 3:
                continue
            betas = fit_betas(train[train["is_pitcher"] == isp], feats)
            preds, acts = [], []
            for (cid, pos), g in test.groupby(["contest", "pos"]):
                if len(g) < 2:
                    continue
                preds.append(predict_group(g, betas, feats))
                acts.append(g["own"].to_numpy())
            if not preds:
                continue
            pred, act = np.concatenate(preds), np.concatenate(acts)
            sp, mae, hit = _metrics(pred, act)
            sps.append(sp); maes.append(mae); hits.append(hit); ns.append(len(act))
            print(f"  {kind} holdout {d}: n={len(act):4d}  Spearman={sp:.3f}  "
                  f"MAE={mae:4.2f}%  top10%-hit={hit:.2f}")
        if sps:
            w = np.array(ns)
            m = (np.average(sps, weights=w), np.average(maes, weights=w),
                 np.average(hits, weights=w))
            out[kind] = m
            print(f"  {kind} OOS mean: Spearman={m[0]:.3f}  MAE={m[1]:.2f}%  "
                  f"top10%-hit={m[2]:.2f}")
    return out


def estimate_sigma(data: pd.DataFrame, feats: list[str]) -> tuple[float, float]:
    """Fit the heteroskedastic ownership-uncertainty model σ(own) ≈ a + b·own
    from hitter residuals (predicted vs actual %Drafted), pooled across slates.
    """
    sub = data[~data.is_pitcher].dropna(subset=feats + ["own"]).copy()
    betas = fit_betas(sub, feats)
    preds = []
    for (cid, pos), g in sub.groupby(["contest", "pos"]):
        if len(g) < 2:
            continue
        p = predict_group(g, betas, feats)
        preds.append(pd.DataFrame({"pred": p, "own": g["own"].to_numpy()},
                                  index=g.index))
    if not preds:
        return 1.7, 0.41
    r = pd.concat(preds)
    r["resid"] = r["own"] - r["pred"]
    r["bin"] = pd.cut(r["pred"], [0, 3, 6, 10, 15, 100])
    binned = r.groupby("bin", observed=True).agg(
        pred=("pred", "mean"), sd=("resid", "std")).dropna()
    print("  σ(own) by predicted-ownership bin:")
    for _, row in binned.iterrows():
        print(f"    pred~{row['pred']:5.1f}%  resid σ={row['sd']:.2f}")
    if len(binned) < 2:
        return 1.7, 0.41
    b, a = np.polyfit(binned["pred"], binned["sd"], 1)
    return float(max(0.5, a)), float(max(0.0, b))


def estimate_chalk_k(data: pd.DataFrame, n_medium: int) -> float:
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
        k = (1.0 - expo) / (np.log10(hi / n_medium) - np.log10(lo / n_medium)) \
            if hi != lo else 0.0
        ks.append(k); ws.append(len(m))
        print(f"  chalk pair {date}: {lo}->{hi} entries  own_l~own_s^{expo:.2f}  "
              f"k~{k:.2f}  (matched {len(m)})")
    if not ks:
        return 0.20
    return float(np.clip(np.average(ks, weights=ws), 0.05, 0.6))


# ---------------------------------------------------------------------------
def build_dataset_from_log(path: str) -> pd.DataFrame:
    """Prepare the fit frame from an accumulated ownership_history log CSV.

    Uses each slate/day as a softmax group ("contest") so the same fit/validate
    code applies. Chalk-k is not estimated here (the log carries no field size);
    it keeps its prior/params value.
    """
    from ownership_history import load_log
    df = load_log(path, labeled_only=True).copy()
    if df.empty:
        sys.exit(f"no labeled rows in history log {path}")
    df["contest"] = df["date"]
    df["is_pitcher"] = df["pos"] == "P"
    df["ceil_shape"] = (df["ceiling"] / df["proj"]).clip(1.0, 6.0)
    df["order_score"] = df["order"].map(_order_score) if "order" in df else 0.0
    df["n_entries"] = 3000
    print(f"history log: {len(df)} labeled rows across {df['date'].nunique()} slates"
          f"  ({df['salary'].notna().sum()} with salary)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="save fitted params to ownership_params.json")
    ap.add_argument("--n-medium", type=int, default=3000)
    ap.add_argument("--history-csv", default="",
                    help="train from an accumulated ownership_history log CSV "
                         "instead of raw sims+contests (scales to many slates)")
    args = ap.parse_args()

    if args.history_csv:
        data = build_dataset_from_log(args.history_csv)
    else:
        data = build_dataset()
    have_cost = data["salary"].notna().any()

    sim_feats = {"HIT": SIM_FEATURES, "PIT": SIM_FEATURES}
    sim_betas = {"HIT": fit_betas(data[~data.is_pitcher], SIM_FEATURES),
                 "PIT": fit_betas(data[data.is_pitcher], SIM_FEATURES)}
    evaluate(data, sim_betas, sim_feats, "in-sample — sim-only (proj, ceil_shape)")
    cross_validate(data, sim_feats, "sim-only")

    full_betas, full_feats = sim_betas, sim_feats
    if have_cost:
        full_feats = {"HIT": FULL_HIT, "PIT": FULL_PIT}
        # fit the full model on the salary-covered rows
        full_betas = {"HIT": fit_betas(data[~data.is_pitcher], FULL_HIT),
                      "PIT": fit_betas(data[data.is_pitcher], FULL_PIT)}
        # report sim-only ON THE SAME salary-covered subset, for a fair lift
        cov = data.dropna(subset=["value"])
        evaluate(cov, sim_betas, sim_feats, "in-sample — sim-only (cost-covered rows)")
        evaluate(cov, full_betas, full_feats, "in-sample — full (+value +team_total)")
        cross_validate(cov, sim_feats, "sim-only (cost-covered rows)")
        cross_validate(cov, full_feats, "full (+value +team_total)")

    if args.history_csv:
        # the log has no field-size pairs; keep whatever chalk_k is on disk
        k = OwnershipParams().chalk_k
        from ownership_model import load_params
        try:
            k = load_params().chalk_k
        except Exception:
            pass
        print(f"\n===== contest-size chalk (k) =====\n  kept existing chalk_k = {k:.3f} "
              "(history log carries no field size to re-estimate from)")
    else:
        print("\n===== contest-size chalk (k) =====")
        k = estimate_chalk_k(data, args.n_medium)
        print(f"  estimated chalk_k = {k:.3f}")

    if have_cost:
        print("\n===== ownership uncertainty σ(own) = a + b·own =====")
        sig_a, sig_b = estimate_sigma(data, full_feats["HIT"])
        print(f"  fitted sigma_a={sig_a:.2f}  sigma_b={sig_b:.2f}")
    else:
        sig_a, sig_b = OwnershipParams().sigma_a, OwnershipParams().sigma_b

    if args.write:
        P = OwnershipParams()
        P.n_medium = args.n_medium
        P.chalk_k = round(k, 3)
        P.sigma_a = round(sig_a, 3)
        P.sigma_b = round(sig_b, 3)
        for f in full_feats["HIT"]:
            P.hit[f] = round(full_betas["HIT"][f], 3)
        for f in full_feats["PIT"]:
            P.pit[f] = round(full_betas["PIT"][f], 3)
        # zero any prior term the fit didn't cover so the shipped model is honest
        for f in list(P.hit):
            if f not in full_feats["HIT"]:
                P.hit[f] = 0.0
        for f in list(P.pit):
            if f not in full_feats["PIT"]:
                P.pit[f] = 0.0
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ownership_params.json")
        P.to_json(path)
        print(f"\nwrote {os.path.basename(path)}: hit={P.hit} pit={P.pit} "
              f"chalk_k={P.chalk_k}")


if __name__ == "__main__":
    main()

"""
backtest_rate_age.py — does an age adjustment improve the K% / BB% projection?

K% and BB% are projected by `rate_models.project_rate_with_decay`: a
decay-weighted, PA-weighted history shrunk to the league mean. It has no age
term (the module header still describes the Ridge+HGB model that did). Against
FanGraphs the gap grew as players got younger (hitters <=23: K% +2.1, SLG
-0.037; 32+: K% +0.5, SLG 0.000) and the pitcher K9 gap had the opposite sign
at the two ends of the age range. This script asks whether that is real, using
real seasons the model has not seen.

Method (walk-forward, no look-ahead)
------------------------------------
For each TEST year T:
  1. Baseline = the production function, fed seasons < T only, with the
     production parameters (decay, window, prior strength).
  2. The age adjustment is FITTED ON EARLIER TEST-LIKE YEARS ONLY: the same
     baseline is run for each T' in [T-TRAIN_YEARS, T-1], its residuals
     (actual - predicted) are averaged by age bin (PA-weighted, shrunk toward
     zero), and that table is added to year T's baseline.
  3. Both are scored on year T: PA-weighted MAE and signed bias, overall and
     by age bin.

Result (Lahman 2008-2021; wMAE, lower is better; "shape" = age effect with the
league-level shift removed):

    hitters  K%   baseline 0.02511   shape-only 0.02511   level-only 0.02497
    hitters  BB%  baseline 0.01645   shape-only 0.01643   level-only 0.01648
    pitchers K%   baseline 0.03052   shape-only 0.03026   level-only 0.02991
    pitchers BB%  baseline 0.01516   shape-only 0.01515   level-only 0.01503

The full age table beats the baseline mainly through the LEVEL term: K% rose
every year of the window and the baseline's bias tracks the league trend
(pitchers +0.3pp in 2008 -> +2.1pp in 2019). Age shape alone adds <1%, so it
was NOT added to the model. Do not read the table's positive offsets as an age
effect, and do not apply them to a flat-K% league.

Data: Lahman / Chadwick "baseballdatabank" Batting.csv, Pitching.csv and
People.csv (through 2021 in the copy this was run on). The files are not
committed; pass their directory.

    python scripts/backtest_rate_age.py --data /path/to/lahman
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_config import (RATE_DECAY, RATE_MAX_HISTORY_YEARS,  # noqa: E402
                             RATE_SHRINK_K_HITTER, RATE_SHRINK_K_PITCHER)
from rate_models import project_rate_with_decay  # noqa: E402

AGE_BINS = [0, 23, 25, 27, 29, 31, 33, 99]
AGE_LABELS = ["<=23", "24-25", "26-27", "28-29", "30-31", "32-33", "34+"]
TRAIN_YEARS = 8
SHRINK_PA = 4000.0      # PA-equivalents of evidence needed to trust a bin
FIRST_PANEL_YEAR = 1995


def load(data: Path):
    people = pd.read_csv(data / "People.csv", usecols=["playerID", "birthYear"])
    bat = pd.read_csv(data / "Batting.csv")
    for c in ("HBP", "SF", "SH", "SO", "BB"):
        bat[c] = bat[c].fillna(0)
    bat["PA"] = bat["AB"] + bat["BB"] + bat["HBP"] + bat["SF"] + bat["SH"]
    hit = (bat.groupby(["playerID", "yearID"], as_index=False)
           [["PA", "SO", "BB"]].sum())
    pit = pd.read_csv(data / "Pitching.csv")
    pit = (pit.groupby(["playerID", "yearID"], as_index=False)
           [["BFP", "SO", "BB"]].sum().rename(columns={"BFP": "PA"}))
    out = []
    for df in (hit, pit):
        df = df[df.yearID >= FIRST_PANEL_YEAR].merge(people, on="playerID")
        df = df.rename(columns={"yearID": "Season", "playerID": "PlayerId"})
        df["K%"] = df["SO"] / df["PA"].where(df["PA"] > 0)
        df["BB%"] = df["BB"] / df["PA"].where(df["PA"] > 0)
        df["Age"] = df["Season"] - df["birthYear"]
        out.append(df.dropna(subset=["K%", "BB%", "Age"]))
    return out


def baseline(df: pd.DataFrame, year: int, rate: str, k: float) -> pd.DataFrame:
    """Production projection of `year` for everyone who played in year-1."""
    infer = df[df.Season == year - 1][["PlayerId"]].drop_duplicates()
    got = project_rate_with_decay(
        df[df.Season < year], year, "PA", rate, prior_k=k, decay=RATE_DECAY,
        infer=infer, max_history_years=RATE_MAX_HISTORY_YEARS)
    got = got[["PlayerId", f"Pred_{rate}"]].rename(
        columns={f"Pred_{rate}": "pred"})
    act = df[df.Season == year][["PlayerId", "PA", rate, "Age"]]
    m = got.merge(act, on="PlayerId").rename(columns={rate: "actual"})
    m["Season"] = year
    m["bin"] = pd.cut(m["Age"], AGE_BINS, labels=AGE_LABELS)
    m["resid"] = m["actual"] - m["pred"]
    return m


def age_table(train: pd.DataFrame, min_pa: float) -> pd.Series:
    t = train[train.PA >= min_pa]
    g = t.groupby("bin", observed=False).apply(
        lambda d: pd.Series({"w": d.PA.sum(),
                             "mean": np.average(d.resid, weights=d.PA)
                             if len(d) else 0.0}), include_groups=False)
    shrunk = g["mean"].fillna(0.0) * g["w"] / (g["w"] + SHRINK_PA)
    return shrunk


def run(df, rate, k, test_years, min_pa, label):
    need = range(min(test_years) - TRAIN_YEARS, max(test_years) + 1)
    preds = {y: baseline(df, y, rate, k) for y in need}
    rows, per_year = [], []
    for T in test_years:
        train = pd.concat([preds[y] for y in range(T - TRAIN_YEARS, T)])
        tab = age_table(train, min_pa)
        te = preds[T][preds[T].PA >= min_pa].copy()
        te["adj"] = te["pred"] + te["bin"].map(tab).astype(float).fillna(0.0)
        # Attribution. A table fitted on a rising-K era is mostly a LEVEL
        # shift, which is not an age effect, so also score the two parts.
        tr = train[train.PA >= min_pa]
        lvl = np.average(tr.resid, weights=tr.PA)
        te["lvl"] = te["pred"] + lvl
        te["shape"] = te["pred"] + te["bin"].map(tab - lvl).astype(float) \
            .fillna(0.0)
        rows.append(te)
        for nm, col in (("baseline", "pred"), ("age-adjusted", "adj")):
            e = te["actual"] - te[col]
            per_year.append({"year": T, "model": nm,
                             "wMAE": np.average(e.abs(), weights=te.PA)})
    allr = pd.concat(rows)
    print(f"\n=== {label} {rate}  (test {min(test_years)}-{max(test_years)}, "
          f"n={len(allr):,} player-seasons, PA >= {min_pa:.0f}) ===")
    summ = []
    for nm, col in (("baseline", "pred"), ("level-only", "lvl"),
                    ("age-shape-only", "shape"), ("age-adjusted", "adj")):
        e = allr["actual"] - allr[col]
        summ.append({"model": nm,
                     "wMAE": np.average(e.abs(), weights=allr.PA),
                     "wRMSE": np.sqrt(np.average(e ** 2, weights=allr.PA)),
                     "bias": np.average(e, weights=allr.PA)})
    print(pd.DataFrame(summ).to_string(index=False,
                                       float_format=lambda x: f"{x:.5f}"))
    by = []
    for b, d in allr.groupby("bin", observed=False):
        if d.empty:
            continue
        r = {"age": b, "n": len(d)}
        for nm, col in (("base", "pred"), ("adj", "adj")):
            e = d["actual"] - d[col]
            r[f"{nm}_bias"] = np.average(e, weights=d.PA)
            r[f"{nm}_wMAE"] = np.average(e.abs(), weights=d.PA)
        by.append(r)
    print(pd.DataFrame(by).to_string(index=False,
                                     float_format=lambda x: f"{x:+.4f}"))
    py = pd.DataFrame(per_year).pivot(index="year", columns="model",
                                      values="wMAE")
    wins = int((py["age-adjusted"] < py["baseline"]).sum())
    print(f"age-adjusted beats baseline in {wins} of {len(py)} test years "
          f"(wMAE)")
    tab = age_table(pd.concat([preds[y] for y in
                               range(max(test_years) - TRAIN_YEARS + 1,
                                     max(test_years) + 1)]), min_pa)
    print("adjustment table fitted on the latest window:",
          {str(i): round(float(v), 4) for i, v in tab.items()})
    return allr, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--first", type=int, default=2008)
    ap.add_argument("--last", type=int, default=2021)
    a = ap.parse_args()
    hit, pit = load(a.data)
    years = [y for y in range(a.first, a.last + 1)]
    for rate in ("K%", "BB%"):
        run(hit, rate, RATE_SHRINK_K_HITTER, years, 200, "HITTERS")
    for rate in ("K%", "BB%"):
        run(pit, rate, RATE_SHRINK_K_PITCHER, years, 150, "PITCHERS")


if __name__ == "__main__":
    main()

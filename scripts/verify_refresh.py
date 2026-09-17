#!/usr/bin/env python3
"""
verify_refresh.py — assert a freshly-built projection set is actually fixed.

    python scripts/verify_refresh.py --target-year 2027

Exits non-zero, with a specific message, when a regenerated projection set
still carries a bug the pipeline is supposed to have fixed. Built for the
one-off refresh workflow (.github/workflows/refresh-projections.yml), where a
green run has to mean more than "the pipeline did not crash".

Every check corresponds to a defect found in the baseline audit
(deliverables/projection_engine/CURRENT_STATE_ASSESSMENT.md). The committed
artifacts pre-date the fixes, so this script FAILS against them — that is the
point. It should pass only against a genuinely rebuilt `out/`.

Thresholds are deliberately loose. The goal is to catch a regression of a
known, large bug (a -19% extra-base haircut), not to police normal
year-to-year variation, so they compare against real batted-ball data in
`bip_inputs/` rather than hardcoded league constants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A regenerated set may suppress extra-base hits by at most this much relative
# to their real share of the batted-ball pool. The bug this guards produced
# 0.81x on home runs; normal projection regression is a few percent.
XBH_MIN_RATIO = 0.90
# Singles absorbed the deflected mass at 1.07x, so cap them too.
SINGLE_MAX_RATIO = 1.05

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))

    @property
    def failed(self) -> bool:
        return any(s == FAIL for s, _, _ in self.rows)

    def report(self) -> str:
        icon = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN"}
        width = max(len(n) for _, n, _ in self.rows) if self.rows else 10
        lines = [f"{'':<4}  {'check':<{width}}  detail",
                 f"{'-' * 4}  {'-' * width}  {'-' * 60}"]
        for status, name, detail in self.rows:
            lines.append(f"{icon[status]:<4}  {name:<{width}}  {detail}")
        return "\n".join(lines)


def check_team_identity(checks: Checks, h: pd.DataFrame,
                        p: pd.DataFrame) -> None:
    """All 30 franchises distinct, and numeric team ids present on both sides.

    The old `team["name"][:3]` fallback collapsed 30 clubs into 26 labels,
    merging Chi/Los/New/San. This also broke same-name resolution on the daily
    path, so it is not a cosmetic check.
    """
    for label, df in (("hitters", h), ("pitchers", p)):
        n = df["Team"].nunique() if "Team" in df.columns else 0
        checks.add(PASS if n == 30 else FAIL, f"team labels ({label})",
                   f"{n} distinct, expected 30"
                   + ("" if n == 30 else " — data_acquisition._team_code "
                                        "fallback regressed?"))
        has_id = "Pred_target_team_id" in df.columns
        missing = (int(df["Pred_target_team_id"].isna().sum()) if has_id
                   else len(df))
        checks.add(PASS if has_id else FAIL, f"team ids ({label})",
                   f"Pred_target_team_id present, {missing} missing"
                   if has_id else "Pred_target_team_id ABSENT")


def check_extra_base_hits(checks: Checks, h: pd.DataFrame) -> None:
    """Projected per-BIP outcome shares vs REAL batted balls.

    Ground truth comes from bip_inputs/ rather than remembered league rates, so
    the check stays honest if the source data changes.
    """
    bip = sorted(ROOT.glob("bip_inputs/bip_2*.csv"))
    if not bip:
        checks.add(WARN, "extra-base hits",
                   "no bip_inputs/bip_*.csv — cannot verify against real data")
        return
    newest = bip[-1]
    b = pd.read_csv(newest, usecols=["events"])
    label = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
    actual = b["events"].map(label).fillna("Out").value_counts(normalize=True)

    w = pd.to_numeric(h.get("Last_PA"), errors="coerce").fillna(0).clip(lower=0)
    if w.sum() <= 0:
        w = pd.Series(np.ones(len(h)))
    cols = {"HR": "P_HR", "3B": "P_3B", "2B": "P_2B", "1B": "P_1B",
            "Out": "P_BIPOut"}
    sf = np.average(h["P_SF"], weights=w)
    total = sum(np.average(h[c], weights=w) for c in cols.values()) + sf

    for ev, col in cols.items():
        proj = np.average(h[col], weights=w) / total
        if ev == "Out":
            proj += sf / total
        ratio = proj / actual[ev]
        if ev in ("HR", "2B", "3B"):
            status = PASS if ratio >= XBH_MIN_RATIO else FAIL
            note = ("" if status == PASS else
                    f" — still suppressed; EVENT_BLEND_WEIGHTS_* regressed?")
        elif ev == "1B":
            status = PASS if ratio <= SINGLE_MAX_RATIO else FAIL
            note = ("" if status == PASS else
                    " — singles inflated, the signature of deflected "
                    "extra-base mass")
        else:
            status = PASS
            note = ""
        checks.add(status, f"per-BIP {ev}",
                   f"{proj:.4f} vs {actual[ev]:.4f} real "
                   f"({newest.name}) = {ratio:.2f}x{note}")


def check_playing_time(checks: Checks, h: pd.DataFrame,
                       p: pd.DataFrame) -> None:
    """Tiers present, and floor-tier players carry exactly the floor."""
    from pipeline_config import PT_FLOOR_IP, PT_FLOOR_PA

    for label, df, col, floor in (("hitters", h, "Proj_PA", PT_FLOOR_PA),
                                  ("pitchers", p, "Proj_IP", PT_FLOOR_IP)):
        if "pt_tier" not in df.columns:
            checks.add(FAIL, f"pt_tier ({label})",
                       "absent — playing-time step did not run")
            continue
        counts = df["pt_tier"].astype(str).value_counts().to_dict()
        checks.add(PASS, f"pt_tier ({label})", f"{counts}")
        if col not in df.columns:
            checks.add(FAIL, f"{col}", "absent")
            continue
        floor_rows = df[df["pt_tier"].astype(str) == "floor"]
        bad = int((pd.to_numeric(floor_rows[col], errors="coerce")
                   != floor).sum()) if len(floor_rows) else 0
        checks.add(PASS if bad == 0 else FAIL, f"{col} floor",
                   f"{len(floor_rows)} floor-tier rows, {bad} not at {floor}")


def check_league_calibration(checks: Checks, h: pd.DataFrame,
                             p: pd.DataFrame) -> None:
    """Offense and defense must still agree with each other.

    Their agreement (+0.3% in the original audit) is the property that makes
    league closure possible; a refresh that breaks it is a real problem even if
    every other check passes.
    """
    from pitcher_outputs import LINEAR_WEIGHTS_RUNS as LW
    from pitcher_outputs import RUNS_INTERCEPT_DEFAULT as ICPT

    def weights(df):
        w = pd.to_numeric(df.get("Career_PA"), errors="coerce").fillna(0)
        return w if w.sum() > 0 else pd.Series(np.ones(len(df)))

    wh = weights(h)
    rpa_h = sum(lw * np.average(h[c], weights=wh)
                for c, lw in LW.items() if c in h.columns) + ICPT
    if "R_per_PA" not in p.columns:
        checks.add(WARN, "offense/defense", "pitcher R_per_PA absent")
        return
    wp = weights(p)
    rpa_p = float(np.average(pd.to_numeric(p["R_per_PA"], errors="coerce")
                             .fillna(0), weights=wp))
    gap = abs(rpa_h - rpa_p) / max(rpa_p, 1e-9)
    checks.add(PASS if gap <= 0.05 else FAIL, "offense/defense",
               f"hitter R/PA {rpa_h:.4f} vs pitcher {rpa_p:.4f} "
               f"({gap * 100:+.1f}%)")
    checks.add(PASS, "implied R/G", f"{rpa_h * 38:.2f} (MLB ~4.40-4.50)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2].strip())
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--warn-only", action="store_true",
                    help="report but always exit 0 (for an exploratory run)")
    a = ap.parse_args(argv)

    hp = a.out_dir / f"hitter_pa_projections_{a.target_year}.csv"
    pp = a.out_dir / f"pitcher_pa_projections_{a.target_year}.csv"
    for f in (hp, pp):
        if not f.exists():
            print(f"FAIL: missing {f}")
            return 1
    h, p = pd.read_csv(hp), pd.read_csv(pp)

    print("=" * 78)
    print(f"REFRESH VERIFICATION — {a.target_year}")
    print("=" * 78)
    print(f"  {len(h)} hitters, {len(p)} pitchers\n")

    checks = Checks()
    check_team_identity(checks, h, p)
    check_extra_base_hits(checks, h)
    check_playing_time(checks, h, p)
    check_league_calibration(checks, h, p)
    print(checks.report())

    if checks.failed:
        print("\nFAILED — the rebuilt projections still carry a known defect. "
              "See the rows marked FAIL above and\n"
              "deliverables/projection_engine/CURRENT_STATE_ASSESSMENT.md.")
        return 0 if a.warn_only else 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

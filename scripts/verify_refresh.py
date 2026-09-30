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
    from team_context import TEAM_ABBR_BY_ID

    mlb = set(TEAM_ABBR_BY_ID.values())
    for label, df in (("hitters", h), ("pitchers", p)):
        labels = (set(df["Team"].dropna().unique()) if "Team" in df.columns
                  else set())
        found = labels & mlb
        other = labels - mlb
        # Count the 30 MLB clubs, not the distinct label total. Once the
        # minor-league feed works, the output legitimately contains affiliates
        # too, so a bare `nunique() == 30` fails on a CORRECT refresh — it read
        # 150 (30 clubs + 120 affiliates) and blamed _team_code. What the
        # original defect actually broke was clubs COLLAPSING into each other
        # (Chi/Los/New/San merging 30 into 26), and that is what this asserts.
        checks.add(PASS if len(found) == 30 else FAIL,
                   f"MLB team labels ({label})",
                   f"{len(found)}/30 clubs present"
                   + ("" if len(found) == 30
                      else f", missing {sorted(mlb - found)} — "
                           "data_acquisition._team_code fallback regressed?"))
        # Non-MLB labels are reported, never failed on: with the minor-league
        # feed alive they are mostly affiliates, which is a sign of health.
        # Only worth reporting once the 30 clubs are actually present — when
        # they are not, the FAIL above already says what is wrong and these
        # labels are the same defect counted twice. No cause is asserted here,
        # because the same symptom has had two different causes: collapsed
        # `name[:3]` codes, and unresolved parent orgs.
        if other and len(found) == 30:
            ex = ", ".join(sorted(map(str, other))[:3])
            checks.add(WARN, f"non-MLB labels ({label})",
                       f"{len(other)} non-MLB labels (e.g. {ex}) — expected "
                       "for MiLB affiliates; each means no parent org resolved")

        has_id = "Pred_target_team_id" in df.columns
        missing = (int(df["Pred_target_team_id"].isna().sum()) if has_id
                   else len(df))
        share = missing / max(1, len(df))
        # Present-but-empty is not a pass. A missing team id costs the player
        # park factors AND team context, so >25% missing is a defect even
        # though the column exists.
        ok = has_id and share <= 0.25
        checks.add(PASS if ok else FAIL, f"team ids ({label})",
                   (f"Pred_target_team_id present, {missing} missing "
                    f"({share:.0%})"
                    + ("" if share <= 0.25
                       else " — these get no park factor and no team context"))
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


def check_fielding(checks: Checks, out_dir, target_year: int) -> None:
    """Validate the fielding fetch's response shape.

    This check exists because the fielding fetch was written against the
    documented statsapi shape but could not be exercised from the dev sandbox.
    It reports WARN rather than FAIL for a missing file, since fielding is
    additive to the projection — but FAILS on a file that is present and
    malformed, because a silently-wrong shape is worse than an absent one.
    """
    path = out_dir / f"fielding_history_{target_year}.csv"
    if not path.exists():
        # Distinguish "this ref has no fielding fetch" from "it has one and it
        # failed". Those need opposite responses, and conflating them cost a
        # whole refresh run: the workflow was dispatched from the default
        # branch, which did not yet carry the fielding code, so the fetch never
        # ran and a plain WARN read as if the endpoint had merely come back
        # empty.
        try:
            from data_acquisition import fetch_fielding_data  # noqa: F401
            has_fetch = True
        except Exception:
            has_fetch = False

        if has_fetch:
            checks.add(FAIL, "fielding fetch",
                       f"{path.name} absent although fetch_fielding_data "
                       "EXISTS on this ref — the fetch ran and produced "
                       "nothing, or the pipeline step did not execute")
        else:
            checks.add(WARN, "fielding fetch",
                       f"{path.name} absent and fetch_fielding_data is not on "
                       "this ref — nothing to validate. Re-run the workflow "
                       "with 'Use workflow from' set to the branch carrying "
                       "the fielding code.")
        return

    f = pd.read_csv(path)
    checks.add(PASS if len(f) else FAIL, "fielding rows",
               f"{len(f):,} (player, season, position) rows")
    if f.empty:
        return

    expected = {"PlayerId", "Season", "Pos", "Innings", "PO", "A", "E", "DP"}
    absent = expected - set(f.columns)
    checks.add(PASS if not absent else FAIL, "fielding columns",
               "all present" if not absent else f"MISSING {sorted(absent)}")
    if absent:
        return

    # Positions are the field the pipeline has never had; they arrive free with
    # this fetch, so confirm they actually came through.
    from fielding_model import ALIGNMENT
    seen = {str(p).upper() for p in f["Pos"].dropna().unique()}
    covered = set(ALIGNMENT) & seen
    checks.add(PASS if len(covered) >= 9 else FAIL, "positions",
               f"{len(covered)}/9 alignment positions present"
               + ("" if len(covered) >= 9 else f" — saw {sorted(seen)}"))

    innings = pd.to_numeric(f["Innings"], errors="coerce").fillna(0)
    checks.add(PASS if (innings > 0).any() else FAIL, "fielding innings",
               f"{int((innings > 0).sum()):,} rows with innings > 0"
               + ("" if (innings > 0).any() else
                  " — the 'X.Y' innings parse may have failed"))

    # The 27-putout identity, measured on real data. A league-wide PO per 9
    # defensive innings far from 27 means the innings parse or the putout field
    # is being read wrong.
    if (innings > 0).any():
        po = pd.to_numeric(f["PO"], errors="coerce").fillna(0).sum()
        team_innings = innings.sum() / len(ALIGNMENT)
        po_per_9 = po / (team_innings / 9.0)
        ok = 24.0 <= po_per_9 <= 30.0
        checks.add(PASS if ok else FAIL, "PO per 9",
                   f"{po_per_9:.2f} (expect ~27)"
                   + ("" if ok else " — innings or putout field misread?"))


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Always a numeric Series aligned to `df`, even for an absent column.

    `pd.to_numeric(df.get(col))` returns a SCALAR nan when the column is
    missing, and the next `.fillna(0)` raises AttributeError on a numpy float —
    crashing the gate, which reads as a broken verifier rather than as the
    clean run it actually is.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def volume_weights(df: pd.DataFrame) -> pd.Series:
    """Career-PA weights with FLOOR-TIER players zeroed out.

    League aggregates should describe the players who will actually play. The
    floor tier is everyone with no real MLB evidence inside
    PT_PROJECTED_LOOKBACK (2) years of the target, and it is where three
    populations land:

      - the retired. Widening RATE_ACTIVE_LOOKBACK to 4 admitted anyone with
        >= 1 PA since 2023, so Miguel Cabrera and Nelson Cruz re-entered the
        hitter pool carrying ~11,000 career PA apiece — the same aggregate
        weight as an active star, on decline-phase rates from three years ago.
      - pitchers in the HITTER pool. Adam Wainwright appeared among the
        hitters for the same reason: a pitcher with a handful of plate
        appearances clears a 1-PA bar.
      - MLE-translated minor leaguers, always floor by construction.

    Career_PA is still the weight for everyone projected, because Proj_PA is
    NaN until the playing-time model exists — it is 1.0 ONLY for the floor
    tier, so weighting by it directly would invert the bias and count nobody
    but the floor. Zeroing the floor tier gets the same answer without waiting
    for that model. With no pt_tier column this degrades to plain Career_PA,
    which is what it always was.
    """
    w = _num(df, "Career_PA").fillna(0)
    if "pt_tier" in df.columns:
        w = w.where(df["pt_tier"].astype(str) != "floor", 0.0)
    if w.sum() > 0:
        return w
    return pd.Series(np.ones(len(df)), index=df.index)


def check_pool_composition(checks: Checks, h: pd.DataFrame,
                           p: pd.DataFrame) -> None:
    """How much aggregate weight sits on players who will not play.

    Reported, never failed on: the floor tier existing is correct and
    intended — it is how organizational depth gets a baseline. What this makes
    visible is the SHARE, because that is what silently moved the league
    aggregates when the active-player bar dropped to 1 PA.
    """
    for label, df in (("hitters", h), ("pitchers", p)):
        if "pt_tier" not in df.columns or "Career_PA" not in df.columns:
            continue
        career = _num(df, "Career_PA").fillna(0)
        floor = df["pt_tier"].astype(str) == "floor"
        tot = float(career.sum())
        share = float(career[floor].sum()) / tot if tot > 0 else 0.0
        checks.add(PASS, f"floor-tier weight ({label})",
                   f"{share:.1%} of career volume is floor tier "
                   f"({int(floor.sum())} rows) — excluded from league "
                   "aggregates")


def check_physically_possible(checks: Checks, h: pd.DataFrame,
                              p: pd.DataFrame) -> None:
    """Per-player sanity bounds — no aggregate, no weighting, no excuses.

    Every other check here is an AGGREGATE, and the weighted ones are blind by
    construction to a player carrying almost no weight. That is how a run
    shipped with a 0.283 per-BIP home-run rate (170 HR per 600 PA) and a
    NEGATIVE ERA: the extra-base checks passed because they are PA-weighted and
    the offending players had ~2 effective PA, and the only check that noticed
    reported an 8% offense/defense gap, which names the symptom and not the
    cause. These bounds are per-row and generous — they cannot flag a merely
    optimistic projection, only an impossible one — so anything they catch is a
    genuine defect and the message says where to look.
    """
    if "P_HR" in h.columns:
        hr = _num(h, "P_HR")
        # The all-time single-season per-PA HR record is ~0.11 (Bonds 2001:
        # 73 HR in 664 PA = 0.110). P_HR is per-PA, so 0.12 is above anything
        # a real hitter has ever done and cannot flag a good projection.
        bad = hr > 0.12
        worst_name = ""
        if bad.any() and "Name" in h.columns:
            worst_name = f", worst {h.loc[hr.idxmax(), 'Name']}"
        mx = float(hr.max()) if hr.notna().any() else float("nan")
        checks.add(PASS if not bad.any() else FAIL, "P_HR plausible",
                   f"max {mx:.4f}" + ("" if not bad.any() else
                   f" — {int(bad.sum())} above 0.12/PA{worst_name}"
                   " (unshrunk MLE translation?)"))
    for col, lo in (("RA9", 0.0), ("ERA", 0.0), ("R_per_PA", 0.0)):
        if col not in p.columns:
            continue
        v = _num(p, col)
        n_neg = int((v < lo).sum())
        checks.add(PASS if not n_neg else FAIL, f"{col} non-negative",
                   f"min {float(v.min()):.3f}" + ("" if not n_neg else
                   f" — {n_neg} NEGATIVE; the linear-weights runs mapping "
                   "left its fitted domain (see MIN_RUNS_PER_PA)"))
    # League RA9 is the one aggregate worth a hard bound: it is the number a
    # season engine multiplies by innings, so a mean of 2.15 against a real
    # ~4.40 is a 50% error in every pitcher's run total.
    if "RA9" in p.columns:
        ra9 = _num(p, "RA9")
        ok = ra9.notna().to_numpy()
        if not ok.any():
            # No data is not a calibration failure. FAILing here would make an
            # absent column indistinguishable from a broken run environment.
            checks.add(WARN, "league RA9", "no RA9 values to average")
        else:
            # Same basis as the offense/defense check: a retired pitcher's
            # career volume must not steer the league number.
            wv = np.asarray(volume_weights(p), dtype=float)[ok]
            vals = ra9.to_numpy()[ok]
            m = (float(np.average(vals, weights=wv)) if wv.sum() > 0
                 else float(vals.mean()))
            checks.add(PASS if 3.5 <= m <= 5.5 else FAIL, "league RA9",
                       f"{m:.2f} (MLB ~4.20-4.60)")


def check_league_calibration(checks: Checks, h: pd.DataFrame,
                             p: pd.DataFrame) -> None:
    """Offense and defense must still agree with each other.

    Their agreement (+0.3% in the original audit) is the property that makes
    league closure possible; a refresh that breaks it is a real problem even if
    every other check passes.
    """
    from pitcher_outputs import LINEAR_WEIGHTS_RUNS as LW
    from pitcher_outputs import RUNS_INTERCEPT_DEFAULT as ICPT

    wh = volume_weights(h)
    rpa_h = sum(lw * np.average(h[c], weights=wh)
                for c, lw in LW.items() if c in h.columns) + ICPT
    if "R_per_PA" not in p.columns:
        checks.add(WARN, "offense/defense", "pitcher R_per_PA absent")
        return
    # Drop rows with no rate instead of .fillna(0) on the VALUE, which counted
    # every such pitcher as allowing ZERO runs per PA at full weight — a
    # fabricated defect in one direction and a mask for a real one in the other.
    rp = pd.to_numeric(p["R_per_PA"], errors="coerce")
    ok = rp.notna().to_numpy()
    wp = np.asarray(volume_weights(p), dtype=float)[ok]
    if wp.sum() <= 0:
        checks.add(WARN, "offense/defense",
                   f"no weighted pitcher R_per_PA ({int(ok.sum())} of "
                   f"{len(p)} rows have a rate)")
        return
    rpa_p = float(np.average(rp.to_numpy()[ok], weights=wp))
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
    check_pool_composition(checks, h, p)
    check_physically_possible(checks, h, p)
    check_league_calibration(checks, h, p)
    check_fielding(checks, a.out_dir, a.target_year)
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

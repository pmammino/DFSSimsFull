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
# And may not inflate them either. The Gaussian imputation sampler ran the
# other way — 1.083x on doubles while triples sat at 0.855x — so a one-sided
# floor let half of that pass unremarked.
XBH_MAX_RATIO = 1.10
# Singles absorbed the deflected mass at 1.07x, so cap them too.
SINGLE_MAX_RATIO = 1.05
# Total bases per batted ball, against the same real pool.
#
# The per-event checks have a blind spot: their extra-base floor is applied
# one event at a time, and singles have no floor at all, only a cap — the bug
# that check was written for INFLATED them. So a uniform suppression of every
# hit type passes all of it, with the lost mass going to the out share, which
# nothing bounds. TB per BIP is the weighted combination that does not
# survive that, and it is the term that drives SLG. It needs no PA
# denominator, so it is also immune to how complete the batted-ball scrape
# was.
#
# The band is a loose backstop, not a precision instrument: it is there to
# catch gross drift without crying wolf. Current refresh sits at 0.984.
TB_PER_BIP_BAND = (0.95, 1.05)

# Offense and defense are two views of one league, so their run rates are the
# same number counted twice. League reconciliation (step 13c) closes this to
# machine precision, which makes the band a check that the step RAN, not a
# tolerance for a real disagreement. It was +3.8% before reconciliation
# existed, and +4.3% on the run totals.
OFFENSE_DEFENSE_MAX_GAP = 0.01

# The same identity on the volume side: every plate appearance is one batter
# faced. This one is NOT closed by reconciliation — it is the joint
# consistency of TEAM_PA_BUDGET (162 x 38), TEAM_IP_BUDGET (162 x 9) and
# TBF_PER_IP_CALIBRATION, which together pin a required batters-faced-per-
# inning of 6156/1458 = 4.222 against a realized 4.200. Measured at -0.52%,
# so the band is set where it will catch a drift without failing on a
# residual we have named and not yet resolved.
VOLUME_MAX_GAP = 0.015

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
    from team_context import FREE_AGENT_TEAM_ID, TEAM_ABBR_BY_ID

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
    # Pool every available season rather than only the newest. The quantity
    # being estimated — what happens to a batted ball, given contact — moves
    # very little year to year, while a single season carries only ~670
    # triples, a 3.9% standard error on the rarest class the check has to
    # judge against a 10% tolerance. Pooling halves the check's own noise so
    # it is testing the projections and not the sample.
    b = pd.concat([pd.read_csv(f, usecols=["events"]) for f in bip],
                  ignore_index=True)
    source = f"{len(bip)} season{'s' if len(bip) > 1 else ''}, {len(b):,} BIP"
    label = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
    actual = b["events"].map(label).fillna("Out").value_counts(normalize=True)

    # Weight by the same volume measure every other league aggregate uses.
    # This check used to weight by Last_PA, which in the committed artifacts
    # is a PARTIAL season — so it judged the league's batted-ball mix by
    # whoever happened to play early in the year, not by the playing time
    # the projections allocate.
    w = volume_weights(h)
    cols = {"HR": "P_HR", "3B": "P_3B", "2B": "P_2B", "1B": "P_1B",
            "Out": "P_BIPOut"}
    sf = np.average(h["P_SF"], weights=w)
    total = sum(np.average(h[c], weights=w) for c in cols.values()) + sf

    shares = {}
    for ev, col in cols.items():
        proj = np.average(h[col], weights=w) / total
        if ev == "Out":
            proj += sf / total
        shares[ev] = proj
        ratio = proj / actual[ev]
        if ev in ("HR", "2B", "3B"):
            if ratio < XBH_MIN_RATIO:
                status, note = FAIL, (" — suppressed; EVENT_BLEND_WEIGHTS_* "
                                      "regressed, or the imputation sampler?")
            elif ratio > XBH_MAX_RATIO:
                status, note = FAIL, (" — inflated; a sampler that misplaces "
                                      "batted-ball mass can run either way")
            else:
                status, note = PASS, ""
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
                   f"({source}) = {ratio:.2f}x{note}")

    # Total bases per batted ball — the combination offsetting per-event
    # errors do not survive, and the term that drives SLG.
    bases = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}
    tb_proj = sum(n * shares[ev] for ev, n in bases.items())
    tb_real = sum(n * actual[ev] for ev, n in bases.items())
    ratio = tb_proj / tb_real
    lo, hi = TB_PER_BIP_BAND
    checks.add(PASS if lo <= ratio <= hi else FAIL, "TB per BIP",
               f"{tb_proj:.4f} vs {tb_real:.4f} real = {ratio:.3f}x"
               + ("" if lo <= ratio <= hi else
                  " — total bases per batted ball is off; SLG will be too"))


def check_playing_time(checks: Checks, h: pd.DataFrame,
                       p: pd.DataFrame, target_year: int = 2027) -> None:
    """Tiers present, and floor-tier players carry exactly the floor."""
    from pipeline_config import PT_FLOOR_IP, PT_FLOOR_PA
    from team_context import (FREE_AGENT_TEAM_ID, TEAM_OVERRIDE_PATH,
                              load_roster_reserves)

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

        # Every projected-tier player must now carry a volume. Before the
        # playing-time model these were deliberately NaN; a NaN here now means
        # the model did not run, and a NaN volume silently zeroes every
        # rate-times-volume counting stat for that player.
        proj = df[df["pt_tier"].astype(str) == "projected"]
        missing = int(pd.to_numeric(proj[col], errors="coerce").isna().sum()) \
            if len(proj) else 0
        checks.add(PASS if missing == 0 else FAIL, f"{col} coverage",
                   f"{len(proj)} projected rows, {missing} without a volume"
                   + ("" if missing == 0 else
                      " — the playing-time model did not run"))

        # Team closure. A club cannot bat 7,000 times, and the league totals
        # are what let runs scored equal runs allowed.
        #
        # Measured over the PROJECTED players only. The floor tier sits
        # outside the budget by design — a depth arm carried at 1 IP so he
        # is present and joinable is not a claim that he will pitch, and
        # how many of them a club has is a fact about how deep the minor
        # league feed went, not about the club. Summing them in made that
        # coverage artifact take real innings off the major-league staff:
        # clubs lost 5.8% to 9.3% of their pitching budget depending on how
        # many farmhands happened to be translated for them.
        tcol = "Pred_target_team_id"
        if tcol in df.columns:
            budget = 162.0 * 38.0 if col == "Proj_PA" else 162.0 * 9.0
            v = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if "pt_tier" in df.columns:
                v = v.where(df["pt_tier"].astype(str) != "floor", 0.0)
            per = v.groupby(df[tcol]).sum()
            per = per[per.index.notna() & (per.index > 0)]
            # A club that has RESERVED playing time for a signing it has not
            # made is supposed to fall short, by exactly what it reserved.
            # Judging it against the full budget turns a correct projection
            # into a FAIL, so the target is the budget net of the reserve.
            share_key = "pa_share" if col == "Proj_PA" else "ip_share"
            reserves = load_roster_reserves(TEAM_OVERRIDE_PATH(target_year))
            target = pd.Series(
                {int(t): budget * (1.0 - float(
                    (reserves.get(int(t), {}) or {}).get(share_key, 0.0) or 0.0))
                 for t in per.index})
            if len(per):
                off = (per - target.reindex(per.index)).abs()
                worst = float(off.max())
                n_res = sum(1 for t in per.index
                            if (reserves.get(int(t), {}) or {}).get(share_key))
                checks.add(PASS if worst <= budget * 0.02 else FAIL,
                           f"{col} team closure",
                           f"{len(per)} clubs, worst off target by "
                           f"{worst:,.1f} of {budget:,.0f} (projected only"
                           + (f", {n_res} reserving for a signing)" if n_res
                              else ")"))

            # Per-club closure says nothing about FREE AGENTS, who are not on
            # a club and so are not in `per` at all. Their playing time is
            # real and has to come out of what the clubs reserved for the
            # signings they have not made — so the league total is the check
            # that sees them. Without it, 40 unsigned regulars carrying a full
            # season each push the league 15% over and every per-club row
            # still reads PASS.
            league = float(v.sum())
            # Against the clubs actually present, not a hardcoded 30: a
            # missing club is what the team-label check is for, and baking
            # the count in here only makes this check brittle.
            expect = budget * float(len(per))
            n_fa = int((pd.to_numeric(df[tcol], errors="coerce")
                        == FREE_AGENT_TEAM_ID).sum())
            gap = league / expect - 1.0
            status = PASS if abs(gap) <= 0.02 else FAIL
            checks.add(status, f"{col} league total",
                       f"{league:,.0f} vs {expect:,.0f} ({gap:+.2%}), "
                       f"{n_fa} free agent(s)"
                       + ("" if status == PASS else
                          " — free agents need a reserved share to close "
                          "onto (load_roster_reserves), or they are playing "
                          "on top of 30 full clubs"))


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

    Proj_PA is the right weight now that the playing-time model fills it: a
    league aggregate is a statement about the season being projected, so it
    should be weighted by the playing time the projections themselves
    allocate. It used to be unusable — NaN for everyone except the floor
    tier, where it was the 1.0 floor, so weighting by it counted nobody but
    the floor — which is why Career_PA with the floor tier zeroed stood in
    for it. That fallback is still what runs on a frame from before the
    playing-time step, or one where the step did not populate it.

    Career_PA is a career, not a season: it ranks a 36-year-old on his way
    out above the 23-year-old taking his job. Last_PA is worse still in the
    committed artifacts, where it holds a PARTIAL season.
    """
    proj = _num(df, "Proj_PA")
    if proj.isna().all():
        proj = _num(df, "Proj_IP")
    # Usable only if the playing-time model actually populated it. All-floor
    # (every value at the 1.0 floor) or mostly-missing means it did not.
    if proj.notna().mean() > 0.5 and proj.fillna(0).sum() > 2 * len(df):
        w = proj.fillna(0).clip(lower=0)
        # The floor tier is carried so depth players are present, ranked and
        # joinable — not so they vote on what the league looks like. At 1 PA
        # or 1 IP apiece they are 1.2% of the hitter weight but 7.1% of the
        # pitcher weight, which is enough to drag a league aggregate toward
        # organizational filler.
        if "pt_tier" in df.columns:
            w = w.where(df["pt_tier"].astype(str) != "floor", 0.0)
        if w.sum() > 0:
            return w

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
    # Weight by BATTERS FACED, not innings. R_per_PA is runs per batter
    # faced, and innings are not its denominator — a high-strikeout,
    # high-walk pitcher faces more batters per inning than a contact pitcher
    # who works around them. Weighting a per-PA rate by innings understated
    # this gap at +2.3% when it was really +3.8%, by under-counting exactly
    # the pitchers who put men on base.
    from league_reconcile import batters_faced
    wp_all = batters_faced(p)
    if not np.isfinite(wp_all).any() or np.nansum(wp_all) <= 0:
        wp_all = np.asarray(volume_weights(p), dtype=float)
    wp = np.asarray(wp_all, dtype=float)[ok]
    if wp.sum() <= 0:
        checks.add(WARN, "offense/defense",
                   f"no weighted pitcher R_per_PA ({int(ok.sum())} of "
                   f"{len(p)} rows have a rate)")
        return
    rpa_p = float(np.average(rp.to_numpy()[ok], weights=wp))
    gap = abs(rpa_h - rpa_p) / max(rpa_p, 1e-9)
    # Reconciliation closes this identically, so the band is about catching a
    # step that did not run rather than about tolerating a real disagreement.
    checks.add(PASS if gap <= OFFENSE_DEFENSE_MAX_GAP else FAIL,
               "offense/defense",
               f"hitter R/PA {rpa_h:.4f} vs pitcher {rpa_p:.4f} "
               f"({gap * 100:+.1f}%)"
               + ("" if gap <= OFFENSE_DEFENSE_MAX_GAP else
                  " — did league reconciliation run? (step 13c)"))
    checks.add(PASS, "implied R/G", f"{rpa_h * 38:.2f} (MLB ~4.40-4.50)")

    # Runs scored must equal runs allowed, which needs the VOLUMES to agree
    # as well as the rates: every plate appearance is one batter faced.
    pa_total = float(np.nansum(np.asarray(wh, dtype=float)))
    tbf_total = float(np.nansum(np.asarray(wp_all, dtype=float)))
    if pa_total > 0 and tbf_total > 0:
        vol = tbf_total / pa_total - 1.0
        status = PASS if abs(vol) <= VOLUME_MAX_GAP else FAIL
        checks.add(status, "PA vs batters faced",
                   f"{pa_total:,.0f} PA vs {tbf_total:,.0f} faced "
                   f"({vol:+.2%})"
                   + ("" if status == PASS else
                      " — TEAM_PA_BUDGET, TEAM_IP_BUDGET and "
                      "TBF_PER_IP_CALIBRATION are not mutually consistent"))


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
    check_playing_time(checks, h, p, a.target_year)
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

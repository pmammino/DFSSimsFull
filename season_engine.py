"""
season_engine.py
================
The season-long projection layer. Reads the per-PA projections in `out/`,
aligns every player to a target-year team, rebuilds each team's run
environment from its actual projected roster, and rescales R/RBI accordingly.

    python season_engine.py --target-year 2027

This is a LAYER, not a fork. It consumes `out/{hitter,pitcher}_pa_projections_
<year>.csv` and writes `out/season_<year>/`, leaving the per-PA CSVs untouched.
The daily DFS path keeps reading exactly what it read before.

What it does today
------------------
1. **Team alignment.** One shared assignment rule (`team_context.
   assign_target_teams`) for hitters and pitchers, with a roster-override file
   for offseason moves that player history cannot express.
2. **Bottom-up team context.** Each team's offensive run environment is
   derived from the talent projected onto its roster, blended with the
   historical team-RPG prior, and normalized to a league mean of exactly 1.0.
3. **Team-change propagation.** Moving a player recomputes BOTH teams'
   contexts — he leaves one roster aggregate and joins another — and every
   hitter on both teams has R/RBI rescaled off the team-context-free
   `Pred_R_per_PA_neutral`.
4. **Reconciliation diagnostics.** Reports the closure errors that a season
   engine must eventually drive to zero, so progress is measurable.

What it deliberately does NOT do yet
------------------------------------
**Playing time** (PA/G/IP/GS). Without it the roster aggregate must use a
volume proxy (see `team_context.career_pa_weights`), and no counting-stat
total or wins figure can be trusted. Every function takes playing time as an
injectable weight source so the real model drops in without touching this
logic.

Consequently there is **no wins output here**. Wins require league-wide
runs-scored = runs-allowed closure, which requires playing time. The
diagnostics report how far off closure currently is rather than publishing a
wins column that would be wrong. See CURRENT_STATE_ASSESSMENT.md §6, phases
3-4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from team_context import (
    PA_PER_TEAM_GAME,
    TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
    TEAM_OVERRIDE_PATH,
    abbr_for_team_id,
    apply_team_context,
    assign_target_teams,
    blend_team_factors,
    bottom_up_team_factors,
    career_pa_weights,
    depth_weights,
    describe_moves,
    load_team_overrides,
    runs_per_pa,
    team_context_report,
)

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_projections(target_year: int, out_dir: Path
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-PA projection CSVs for `target_year`."""
    hit = out_dir / f"hitter_pa_projections_{target_year}.csv"
    pit = out_dir / f"pitcher_pa_projections_{target_year}.csv"
    for f in (hit, pit):
        if not f.exists():
            raise SystemExit(
                f"missing {f}\nRun: python run_pipeline.py --target-year "
                f"{target_year} --bip-dir bip_inputs --output-dir {out_dir}"
            )
    return pd.read_csv(hit), pd.read_csv(pit)


def resolve_teams(
    df: pd.DataFrame,
    target_year: int,
    *,
    overrides: dict[int, int | None] | None = None,
    history: pd.DataFrame | None = None,
    volume_col: str = "PA",
) -> pd.DataFrame:
    """Attach `team_id` / `team_abbr` / `assign_source` to a projection frame.

    Three sources, in order of authority:

    1. **The override file** — a signing, trade, or manual correction. Always
       wins, and an explicit null means "no team" rather than last year's club.
    2. **`history`** — raw player-season rows, if available, run through the
       shared assignment rule. Only needed when re-deriving teams from
       scratch; the pipeline already did this.
    3. **The projection's own `Pred_target_team_id`** — what the pipeline
       assigned. This is the normal path, since `run_pipeline` now writes it
       for hitters and pitchers alike using the same rule.

    Falling back to the legacy `Team` string is deliberately NOT a source: it
    collapsed 30 franchises into 26 labels and cannot distinguish two clubs
    in one city.
    """
    out = df.copy()
    overrides = dict(overrides or {})

    if history is not None and not history.empty:
        assigned = assign_target_teams(
            history, target_year, id_col="PlayerId", volume_col=volume_col,
            overrides=overrides,
        )
        return out.merge(assigned, on="PlayerId", how="left")

    team_col = None
    for candidate in ("Pred_target_team_id", "Pred_home_team_id",
                      "home_park_team_id"):
        if candidate in out.columns:
            team_col = candidate
            break
    if team_col is None:
        raise KeyError(
            "projection frame carries no numeric team id (looked for "
            "Pred_target_team_id, Pred_home_team_id, home_park_team_id). "
            f"Re-run: python run_pipeline.py --target-year {target_year}"
        )
    if team_col != "Pred_target_team_id":
        # Pre-dates the unified team assignment. `home_park_team_id` is the
        # park the player's projections were adjusted for, which is normally
        # his club — but it is not the same field, and on a legacy pitcher file
        # it was derived by the old unstable `groupby().last()` rule.
        print(f"  NOTE: no Pred_target_team_id; falling back to {team_col!r}. "
              f"Re-run run_pipeline.py for a properly assigned target team.")

    base = pd.to_numeric(out[team_col], errors="coerce")
    out["team_id"] = [
        overrides.get(int(pid), (None if pd.isna(t) else int(t)))
        if pd.notna(pid) else None
        for pid, t in zip(out["PlayerId"], base)
    ]
    out["assign_source"] = [
        "override" if (pd.notna(pid) and int(pid) in overrides)
        else ("history" if pd.notna(t) else "unknown")
        for pid, t in zip(out["PlayerId"], base)
    ]
    out["team_abbr"] = [abbr_for_team_id(t) for t in out["team_id"]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The season layer
# ─────────────────────────────────────────────────────────────────────────────

def build_team_context(
    hitters: pd.DataFrame,
    *,
    bottom_up_weight: float = TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
    weight_fn=depth_weights,
) -> pd.DataFrame:
    """Team run-environment factors from the roster, blended with the prior.

    The prior is each team's existing `Pred_target_team_factor` — the
    historical team-RPG blend — read off any one of its players, since the
    pipeline assigns one factor per team.
    """
    bottom_up = bottom_up_team_factors(hitters, team_col="team_id",
                                       weight_fn=weight_fn)
    prior = None
    if "Pred_target_team_factor" in hitters.columns:
        prior = (hitters.dropna(subset=["team_id"])
                 .groupby("team_id")["Pred_target_team_factor"]
                 .median().to_dict())
        prior = {int(k): float(v) for k, v in prior.items() if pd.notna(v)}
    return blend_team_factors(bottom_up, prior, team_col="team_id",
                              bottom_up_weight=bottom_up_weight)


def reconciliation_report(hitters: pd.DataFrame, pitchers: pd.DataFrame,
                          factors: pd.DataFrame) -> str:
    """Closure diagnostics — the identities a season engine must satisfy.

    None of these are expected to pass yet; they exist so the gap is a number
    that moves rather than a known unknown. The dominant reason they fail is
    the missing playing-time model: without projected PA and IP, a team total
    is a proxy-weighted average rather than a sum.
    """
    lines = []
    # League aggregates weight EVERY player. `depth_weights` is a per-team
    # lineup selector — applied to a whole-league frame it would keep only the
    # nine highest-volume hitters in baseball and report their rate as the
    # league's. Both sides use the same proxy so the gap is apples-to-apples.
    hw = career_pa_weights(hitters)
    pw = career_pa_weights(pitchers)

    rs = runs_per_pa(hitters, hw)
    ra = (float(np.average(pd.to_numeric(pitchers["R_per_PA"], errors="coerce")
                           .fillna(0), weights=pw))
          if "R_per_PA" in pitchers.columns and pw.sum() > 0
          else float("nan"))

    lines.append("league closure")
    lines.append(f"  hitter-implied R/PA        {rs:.4f}")
    lines.append(f"  pitcher-implied R/PA       {ra:.4f}")
    lines.append(f"  gap                        {rs - ra:+.4f}"
                 f"   (must be 0: every run scored is a run allowed)")
    lines.append(f"  implied R/G @ {PA_PER_TEAM_GAME:.0f} PA       "
                 f"{rs * PA_PER_TEAM_GAME:.2f}   (MLB ~4.40-4.50)")

    if "team_factor" in hitters.columns:
        mean_factor = float(np.average(hitters["team_factor"], weights=hw))
        lines.append(f"  volume-wt mean team factor {mean_factor:.4f}"
                     f"   (must be ~1.0: moving players redistributes talent,"
                     f" it does not create it)")

    lines.append("")
    lines.append("R/RBI closure  (per-team, volume-weighted)")
    r_ratios, rbi_ratios = [], []
    for team_id, g in hitters.dropna(subset=["team_id"]).groupby("team_id"):
        w = depth_weights(g)
        if w.sum() <= 0:
            continue
        team_rpa = runs_per_pa(g, w)
        if not np.isfinite(team_rpa) or team_rpa <= 0:
            continue
        if "P_R" in g.columns:
            r_ratios.append(np.average(pd.to_numeric(g["P_R"], errors="coerce")
                                       .fillna(0), weights=w) / team_rpa)
        if "P_RBI" in g.columns:
            rbi_ratios.append(np.average(pd.to_numeric(g["P_RBI"], errors="coerce")
                                         .fillna(0), weights=w) / team_rpa)
    if r_ratios:
        lines.append(f"  mean R/PA   / team R/PA    {np.mean(r_ratios):.3f}"
                     f"   (target ~1.00 — every run is scored by one batter)")
    if rbi_ratios:
        lines.append(f"  mean RBI/PA / team R/PA    {np.mean(rbi_ratios):.3f}"
                     f"   (target ~0.88 — ~12% of runs are not driven in)")
    lines.append("  ^ needs the playing-time model to close; R and RBI are")
    lines.append("    still free-standing player rates, not an allocation of")
    lines.append("    the runs the lineup actually scores.")

    lines.append("")
    lines.append("roster shape")
    n = factors["n_hitters"] if "n_hitters" in factors.columns else pd.Series(dtype=float)
    if len(n):
        lines.append(f"  players per org            min {int(n.min())}"
                     f"  max {int(n.max())}  mean {n.mean():.1f}")
    if "n_projected" in factors.columns:
        p_ = factors["n_projected"]
        lines.append(f"  of which projected         min {int(p_.min())}"
                     f"  max {int(p_.max())}  mean {p_.mean():.1f}")
        lines.append("  ^ the rest are organizational depth at the 1 PA / 1 IP")
        lines.append("    floor: present and joinable, but excluded from team")
        lines.append("    playing time so they cannot move an aggregate.")
    lines.append("  ^ still unconstrained: no position data from statsapi, so no")
    lines.append("    depth chart or batting order can be built yet — though the")
    lines.append("    minors feed DOES carry `position`, which is a way in.")

    if "pt_tier" in hitters.columns:
        lines.append("")
        lines.append("playing-time tiers")
        for label, df in (("hitters", hitters), ("pitchers", pitchers)):
            if "pt_tier" not in df.columns:
                continue
            counts = df["pt_tier"].astype(str).value_counts().to_dict()
            lines.append(f"  {label:<9} {counts}")
        vol = "Proj_PA"
        if vol in hitters.columns:
            unmodeled = int(hitters[vol].isna().sum())
            lines.append(f"  {unmodeled} hitters await a playing-time model"
                         f" (Proj_PA is NaN, not a guess)")
    return "\n".join(lines)


def run(target_year: int, out_dir: Path, *, override_path: Path | None = None,
        bottom_up_weight: float = TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
        write: bool = True) -> dict[str, pd.DataFrame]:
    """Build the season layer. Returns {"hitters","pitchers","teams"}."""
    print("=" * 74)
    print(f"SEASON PROJECTION LAYER — {target_year}")
    print("=" * 74)

    hitters, pitchers = load_projections(target_year, out_dir)
    print(f"  loaded {len(hitters)} hitters, {len(pitchers)} pitchers")

    path = override_path or TEAM_OVERRIDE_PATH(target_year)
    overrides = load_team_overrides(path)
    print(f"  roster overrides: {len(overrides)} from "
          f"{path if path.exists() else f'{path} (absent)'}")

    print("\n" + "-" * 74)
    print("STEP 1: align players to teams")
    print("-" * 74)
    hitters = resolve_teams(hitters, target_year, overrides=overrides,
                            volume_col="PA")
    pitchers = resolve_teams(pitchers, target_year, overrides=overrides,
                             volume_col="TBF")
    for label, df in (("hitters", hitters), ("pitchers", pitchers)):
        teams = df["team_id"].dropna().nunique()
        unknown = int(df["team_id"].isna().sum())
        print(f"  {label:<9} {teams} teams, {unknown} with no team")
    if overrides:
        moved = pd.concat([hitters, pitchers], ignore_index=True)
        names = (dict(zip(moved["PlayerId"], moved["Name"]))
                 if "Name" in moved.columns else None)
        print("  " + describe_moves(moved, names=names)
              .replace("\n", "\n  "))

    print("\n" + "-" * 74)
    print("STEP 2: build team run environment from the roster")
    print("-" * 74)
    factors = build_team_context(hitters, bottom_up_weight=bottom_up_weight)
    print(f"  bottom-up weight {bottom_up_weight:.2f} "
          f"(roster) / {1 - bottom_up_weight:.2f} (historical RPG prior)")
    if not factors.empty:
        print("  " + team_context_report(factors).replace("\n", "\n  "))
        if "prior_factor" in factors.columns:
            gap = (factors["prior_factor"] - factors["bottom_up_factor"]).abs()
            print(f"\n  mean |roster - prior| gap  {gap.mean():.3f}"
                  f"   max {gap.max():.3f}")
            print("  ^ how much the old backward-looking factor disagreed with"
                  " the\n    talent actually projected onto each roster.")

    print("\n" + "-" * 74)
    print("STEP 3: rescale R/RBI to the resolved team context")
    print("-" * 74)
    before = hitters.get("P_R", pd.Series(dtype=float)).copy()
    hitters = apply_team_context(hitters, factors, team_col="team_id")
    if len(before):
        delta = (hitters["P_R"] - before).abs()
        moved = int((delta > 1e-9).sum())
        print(f"  {moved} hitters had R/RBI rescaled "
              f"(mean |change| {delta[delta > 1e-9].mean():.5f} R/PA)"
              if moved else "  no hitters changed team context")

    print("\n" + "-" * 74)
    print("STEP 4: reconciliation")
    print("-" * 74)
    print("  " + reconciliation_report(hitters, pitchers, factors)
          .replace("\n", "\n  "))

    if write:
        dest = out_dir / f"season_{target_year}"
        dest.mkdir(parents=True, exist_ok=True)
        hitters.to_csv(dest / "hitters.csv", index=False)
        pitchers.to_csv(dest / "pitchers.csv", index=False)
        factors.to_csv(dest / "team_context.csv", index=False)
        print(f"\n  wrote {dest}/{{hitters,pitchers,team_context}}.csv")

    return {"hitters": hitters, "pitchers": pitchers, "teams": factors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--out-dir", type=Path, default=HERE / "out",
                    help="directory holding the per-PA projection CSVs")
    ap.add_argument("--overrides", type=Path, default=None,
                    help="roster-override JSON (default: "
                         "rosters/team_assignments_<year>.json)")
    ap.add_argument("--bottom-up-weight", type=float,
                    default=TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
                    help="weight on the roster-derived team factor vs the "
                         "historical team-RPG prior")
    ap.add_argument("--no-write", action="store_true",
                    help="report only; don't write season_<year>/")
    args = ap.parse_args(argv)

    run(args.target_year, args.out_dir, override_path=args.overrides,
        bottom_up_weight=args.bottom_up_weight, write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())

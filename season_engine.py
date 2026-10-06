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
4. **Free agents and roster reserves.** Unsigned players are carried with full
   rate lines under `FREE_AGENT_TEAM_ID`, excluded from team aggregates but
   available for playing time and a role. An unsigned player's playing time is
   whatever his ROLE says, exactly as it is for a player on a club — give him
   a full-time role and he gets a full-time season — and nobody is docked to
   make room: every club is projected at its full budget with the players it
   actually has, and the unsigned sit beside the thirty. The league total then
   reads a season plus an offseason that has not happened yet. Assigning him a
   club settles it at that moment, with nothing to adjust by hand.
5. **Team wins and save / hold opportunity** (`team_wins.py`). Pythagenpat off
   the bottom-up RS and RA factors, normalized so league wins total exactly
   2,430, then converted into per-team save and hold opportunity pools.
6. **Reconciliation diagnostics.** Reports the closure errors that a season
   engine must eventually drive to zero, so progress is measurable.

Wins are available without a playing-time model because **rates do not need
one** — Pythagenpat runs on RS/G and RA/G, and the original audit's 2,618-win
result came from offense and defense being aggregated over differently-selected
pools, not from missing playing time. See the module docstring in
`team_wins.py`.

What it deliberately does NOT do yet
------------------------------------
**Playing time** (PA/G/IP/GS) and **roles**. Without playing time, no
player-level counting stat is trustworthy, and team aggregates fall back to a
volume proxy (`team_context.career_pa_weights`). Playing time is injected as a
`PlayingTimeWeights` callable so the real model drops in without touching this
logic.

Saves and holds are published as TEAM pools only. Dividing them among pitchers
requires a bullpen role (closer / setup / middle) — designed but not built; see
"Designing the role taxonomy" in README_projection_engine.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from team_context import (
    FREE_AGENT_TEAM_ID,
    PA_PER_TEAM_GAME,
    TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
    TEAM_OVERRIDE_PATH,
    abbr_for_team_id,
    apply_team_context,
    assign_target_teams,
    attach_roster_reserves,
    blend_team_factors,
    bottom_up_team_factors,
    career_pa_weights,
    depth_weights,
    describe_moves,
    load_roster_reserves,
    load_team_overrides,
    mlb_clubs_only,
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


# League RBI per run — the reconciliation target for RBI/PA against team R/PA.
#
# This read "~0.88 (~12% of runs are not driven in)" through the first season
# layer ever to run on real data, and it is wrong. Runs that score with NO RBI
# credited are only the ones nobody hit in: errors, wild pitches, passed balls,
# balks, steals of home, and some fielder's-choice plays. Everything else —
# including the batter who homers and drives in himself, a sacrifice fly, and a
# bases-loaded walk — carries an RBI. That is roughly 5% of runs, not 12%.
#
# Measured on the first clean refresh (run 36762891387), across every pool and
# both weightings:
#
#     all hitters      unweighted 0.9540   PA-weighted 0.9613
#     projected tier   unweighted 0.9492   PA-weighted 0.9607
#
# Those projections shrink toward league rates that are literally sum(RBI)/
# sum(PA) over the fetched MLB history, so their ratio IS the real league
# RBI/R. Against 0.88 the report showed a +11% discrepancy and blamed the
# missing playing-time model; the real gap is +2.5%.
#
# A wrong number in a reconciliation report is worse than no number: it is read
# precisely when someone is deciding whether the model is healthy, and this one
# sent the first reader chasing a defect that was not there. Re-derive it the
# same way (ratio of the league R and RBI rates) if the run environment moves.
LEAGUE_RBI_PER_RUN = 0.95

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
        got = float(np.mean(rbi_ratios))
        off = (got - LEAGUE_RBI_PER_RUN) / LEAGUE_RBI_PER_RUN
        lines.append(f"  mean RBI/PA / team R/PA    {got:.3f}"
                     f"   (target ~{LEAGUE_RBI_PER_RUN:.2f} — only errors, "
                     f"wild pitches, passed balls, balks and steals of home "
                     f"score with no RBI)")
        lines.append(f"                             {off:+.1%} vs target")
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
    lines.append("  ^ still unconstrained by position, but the DATA now exists:")
    lines.append("    the fielding fetch returns one row per (player, position)")
    lines.append("    with innings — 13,700 rows over 9 positions on the first")
    lines.append("    clean run — and the minors feed carries `position` too. A")
    lines.append("    depth chart is now a modelling step, not a data gap.")

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
            if unmodeled:
                lines.append(f"  {unmodeled} hitters have no Proj_PA — the "
                             f"playing-time model did not run for them")
            else:
                v = pd.to_numeric(hitters[vol], errors="coerce")
                lines.append(f"  Proj_PA: all {len(hitters)} assigned "
                             f"(max {v.max():,.0f}, median {v.median():,.0f})")
                if "team_id" in hitters.columns:
                    # Projected players only. The floor tier is carried at
                    # 1 PA so a depth player is present and joinable, and
                    # sits outside the budget rather than taking plate
                    # appearances off the major-league roster.
                    vp = v
                    if "pt_tier" in hitters.columns:
                        vp = v.where(hitters["pt_tier"].astype(str) != "floor",
                                     0.0)
                    per = vp.groupby(hitters["team_id"]).sum()
                    per = per[per.index.notna()]
                    # Free agents are not a 31st club. Their pool is a real
                    # total but it is not a budget, so counting it among the
                    # clubs made the closure line read "min 5,941 max 6,464"
                    # when all thirty clubs were in fact at 5,941.
                    fa_pa = float(per.get(FREE_AGENT_TEAM_ID, 0.0))
                    per = per[per.index != FREE_AGENT_TEAM_ID]
                    if len(per):
                        n_floor = int((hitters.get(
                            "pt_tier", pd.Series(dtype=object)
                        ).astype(str) == "floor").sum())
                        lines.append(
                            f"  team PA closure: min {per.min():,.0f} "
                            f"max {per.max():,.0f} "
                            f"(budget {162 * PA_PER_TEAM_GAME:,.0f}, "
                            f"projected only; {n_floor:,} floor players carry "
                            f"1 PA each outside it)")
                        if fa_pa > 0:
                            n_fa = int((hitters["team_id"]
                                        == FREE_AGENT_TEAM_ID).sum())
                            lines.append(
                                f"  free agents: {n_fa} unsigned players "
                                f"holding {fa_pa:,.0f} PA at their roles, "
                                f"beside the {len(per)} clubs rather than "
                                f"inside them")
    return "\n".join(lines)


def run(target_year: int, out_dir: Path, *, override_path: Path | None = None,
        bottom_up_weight: float = TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
        league_rs_per_game: float = 4.45,
        market_odds_path: Path | None = None,
        market_weight: float | None = None,
        write: bool = True) -> dict[str, pd.DataFrame]:
    """Build the season layer. Returns {"hitters","pitchers","teams"}."""
    print("=" * 74)
    print(f"SEASON PROJECTION LAYER — {target_year}")
    print("=" * 74)

    hitters, pitchers = load_projections(target_year, out_dir)
    print(f"  loaded {len(hitters)} hitters, {len(pitchers)} pitchers")

    path = override_path or TEAM_OVERRIDE_PATH(target_year)
    overrides = load_team_overrides(path)
    reserves = load_roster_reserves(path)
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
    factors = attach_roster_reserves(factors, reserves, team_col="team_id")
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
    print("STEP 4: team wins, save & hold opportunity")
    print("-" * 74)
    from team_wins import (
        bottom_up_team_ra, project_save_hold_opportunity, project_team_wins,
        wins_report,
    )

    from market_odds import (
        MARKET_ODDS_PATH, apply_market_to_run_environment, blend_market_wins,
        default_market_weight, load_market_odds, market_expected_wins,
        market_report,
    )

    defense = bottom_up_team_ra(pitchers, team_col="team_id")
    wins = project_team_wins(factors, defense, team_col="team_id",
                             league_rs_per_game=league_rs_per_game)

    # Market prior. Optional — with no odds file the projection is purely
    # bottom-up, exactly as before.
    odds_file = market_odds_path or MARKET_ODDS_PATH(target_year)
    odds = load_market_odds(odds_file)
    if odds.empty:
        print(f"  no market odds at {odds_file} — bottom-up only")
        market = odds
    else:
        market = market_expected_wins(odds)
        # An unset weight resolves from the market type: a win total is a
        # direct estimate of wins and deserves to dominate, a championship
        # future is four playoff rounds removed and does not.
        weight = (default_market_weight(odds.attrs.get("market"))
                  if market_weight is None else market_weight)
        wins = blend_market_wins(wins, market, team_col="team_id",
                                 market_weight=weight)
        wins = apply_market_to_run_environment(wins, team_col="team_id")
        priced = market.attrs.get("priced")
        print(f"  market: {odds.attrs.get('market')} from "
              f"{odds.attrs.get('book') or 'unknown book'} "
              f"as of {odds.attrs.get('as_of') or 'unknown date'}, "
              f"weight {weight:.2f}"
              + ("" if priced is None else
                 ", over/under prices applied" if priced else
                 ", LINES ONLY (no prices — up to ~1.5 wins of information "
                 "left on the table)"))
        print("  " + market_report(market, wins).replace("\n", "\n  "))
        print()

    wins = project_save_hold_opportunity(wins, team_col="team_id")
    print("  " + wins_report(wins).replace("\n", "\n  "))
    print("\n  Saves and holds are TEAM opportunity pools. Dividing them among")
    print("  pitchers needs a bullpen role (closer / setup / middle), which is")
    print("  designed but not built — see README_projection_engine.md.")

    print("\n" + "-" * 74)
    print("STEP 5: reconciliation")
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
    ap.add_argument("--league-rs-per-game", type=float, default=4.45,
                    help="forecast league runs per game; sets the absolute run "
                         "environment for the wins model")
    ap.add_argument("--market-odds", type=Path, default=None,
                    help="market-odds JSON (default: "
                         "rosters/market_odds_<year>.json). Absent means the "
                         "projection stays purely bottom-up.")
    ap.add_argument("--market-weight", type=float, default=None,
                    help="weight on market-implied wins vs the bottom-up "
                         "Pythagenpat. Unset resolves from the market type "
                         "(win_total 0.70, world_series 0.40).")
    ap.add_argument("--no-write", action="store_true",
                    help="report only; don't write season_<year>/")
    args = ap.parse_args(argv)

    run(args.target_year, args.out_dir, override_path=args.overrides,
        bottom_up_weight=args.bottom_up_weight,
        league_rs_per_game=args.league_rs_per_game,
        market_odds_path=args.market_odds, market_weight=args.market_weight,
        write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())

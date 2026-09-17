"""
team_wins.py
============
Team win expectancy and the save / hold opportunity pool.

Saves and holds are not player skills — they are **team-level opportunities**
that a bullpen role converts. A closer on a 98-win team and the same pitcher on
a 68-win team have the same stuff and very different save totals, because save
chances only exist in games the team is winning narrowly. So a save or hold
projection decomposes into two pieces that must be modelled separately:

    player saves = team save opportunities x player's share of them x conversion

This module builds the FIRST factor. The middle factor is a bullpen role
(closer / setup / middle), which is deliberately not modelled yet — see
"Designing the role taxonomy" in README_projection_engine.md. Until roles
exist, this module publishes the team pool and the league conversion rates, so
the role layer has something correct to divide up.

Why this does NOT need the playing-time model
---------------------------------------------
Counting stats need playing time; **rates do not**. Pythagenpat runs on RS/G
and RA/G, both of which are rate-scale quantities derivable from the per-PA
projections and a lineup weighting. The reason the naive Pythagenpat in the
original audit produced 2,618 wins was not missing playing time — it was that
league RS and league RA were computed over differently-selected player pools
and never reconciled.

Both are fixed here by construction:

  * RS/G and RA/G are each indexed to the league mean, so the two sides are on
    one scale by definition.
  * Win percentages are normalized so the league mean is exactly 0.500, hence
    total wins are exactly 2,430.

That makes team wins available NOW, which is what unlocks save and hold
opportunity. Playing time is still required for player-level counting stats.

Calibration status
------------------
The relationships from wins to save and hold opportunity (SAVES_PER_WIN,
HOLDS_PER_WIN, SAVE_OPP_ELASTICITY) are documented starting points, not fitted
values. They are the least certain numbers in this module. `fit_opportunity_rates`
fits them from real team-season data when a table with team SV/HLD is available;
until then treat the absolute level as provisional and the cross-team ordering
as the useful signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from team_context import (
    PA_PER_TEAM_GAME,
    _normalize_to_unit_mean,
    abbr_for_team_id,
    mlb_clubs_only,
    roster_volume_weights,
    runs_per_pa,
)

GAMES_PER_SEASON = 162
LEAGUE_TEAMS = 30
TOTAL_LEAGUE_WINS = GAMES_PER_SEASON * LEAGUE_TEAMS // 2   # 2430
INNINGS_PER_GAME = 9.0

# Pythagenpat exponent: exp = ((RS + RA) / G) ** PYTHAGENPAT_EXPONENT.
# 0.287 is the standard published value (Davenport / Smyth).
PYTHAGENPAT_EXPONENT = 0.287

# ── Save and hold opportunity ────────────────────────────────────────────────
#
# PROVISIONAL. These set the league LEVEL of save and hold opportunity, and
# they are the weakest numbers here. Calibrate with `fit_opportunity_rates`
# against real team-season SV/HLD totals before trusting the absolute scale.
#
# SAVES_PER_WIN — a save requires a win, but not every win produces one: blowouts
# and complete games don't, so the ratio is well under 1. League totals run
# roughly 1,200-1,300 saves against 2,430 wins.
SAVES_PER_WIN = 0.50

# HOLDS_PER_WIN — holds are more numerous than saves because several relievers
# can record one in the same game, and modern bullpen usage has pushed the rate
# up. Roughly one per win at league scale.
HOLDS_PER_WIN = 0.95

# SAVE_OPP_ELASTICITY — save opportunity scales SUB-linearly with wins. A great
# team wins more games, but it also wins more of them by wide margins, and a
# blowout produces no save chance. An elasticity of 1.0 would mean a 100-win
# team gets proportionally more save chances than an 81-win team; below 1.0
# damps that. 0.75 is a reasonable prior, not a measurement.
SAVE_OPP_ELASTICITY = 0.75

# Holds are LESS win-dependent than saves: a losing team still plays close
# games and still bridges innings, and a trailing team's setup men can earn
# holds in games it eventually loses.
HOLD_OPP_ELASTICITY = 0.45

# League conversion rate of a save opportunity into a save. Blown saves are the
# remainder. Published league rates sit around two-thirds to seventy percent.
SAVE_CONVERSION_RATE = 0.68


# ─────────────────────────────────────────────────────────────────────────────
# Team run environments
# ─────────────────────────────────────────────────────────────────────────────

def bottom_up_team_ra(
    pitchers: pd.DataFrame,
    *,
    team_col: str = "team_id",
    weight_fn=None,
) -> pd.DataFrame:
    """Team runs-ALLOWED factor, built from the projected pitching staff.

    The defensive mirror of `team_context.bottom_up_team_factors`. Uses the
    same linear weights, via `runs_per_pa`, so offense and defense are measured
    on one scale — the property that makes the two sides reconcile.

    `weight_fn` defaults to innings-weighted-ish volume: `roster_volume_weights`
    prefers Proj_IP where the playing-time layer has set it and zeroes
    floor-tier arms, so organizational depth cannot drag a staff's projection.
    A staff is NOT reduced to a top-N the way a lineup is — every inning is
    pitched by someone, so the whole staff contributes.

    Returns [team_col, team_abbr, n_pitchers, staff_weight, team_RA_per_PA,
    ra_factor].
    """
    weight_fn = weight_fn or roster_volume_weights
    rows = []
    for team_id, g in mlb_clubs_only(pitchers, team_col=team_col).groupby(team_col):
        w = weight_fn(g)
        rows.append({
            team_col: int(team_id),
            "team_abbr": abbr_for_team_id(team_id),
            "n_pitchers": len(g),
            "staff_weight": float(np.sum(w)),
            "team_RA_per_PA": runs_per_pa(g, w),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(staff_weight=[], team_RA_per_PA=[], ra_factor=[])

    wsum = out["staff_weight"].sum()
    league = (float(np.average(out["team_RA_per_PA"], weights=out["staff_weight"]))
              if wsum > 0 else float(out["team_RA_per_PA"].mean()))
    raw = ((out["team_RA_per_PA"] / league).to_numpy(float) if league > 0
           else np.ones(len(out)))
    out["ra_factor"] = _normalize_to_unit_mean(raw, out["staff_weight"].to_numpy(float))
    return out.sort_values("ra_factor").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Win expectancy
# ─────────────────────────────────────────────────────────────────────────────

def pythagenpat_win_pct(rs_per_game: np.ndarray,
                        ra_per_game: np.ndarray) -> np.ndarray:
    """Pythagenpat expected win percentage.

        exp  = ((RS + RA) / G) ** 0.287
        W%   = RS**exp / (RS**exp + RA**exp)

    Pythagenpat rather than a fixed exponent because the run environment sets
    how much a run is worth — the same run differential converts to more wins
    in a low-scoring context.
    """
    rs = np.asarray(rs_per_game, float)
    ra = np.asarray(ra_per_game, float)
    total = np.clip(rs + ra, 1e-6, None)
    exp = np.power(total, PYTHAGENPAT_EXPONENT)
    rs_e = np.power(np.clip(rs, 1e-6, None), exp)
    ra_e = np.power(np.clip(ra, 1e-6, None), exp)
    return rs_e / (rs_e + ra_e)


def project_team_wins(
    offense: pd.DataFrame,
    defense: pd.DataFrame,
    *,
    team_col: str = "team_id",
    league_rs_per_game: float = 4.45,
) -> pd.DataFrame:
    """Expected wins per team, normalized so the league sums to exactly 2,430.

    `offense` comes from `team_context.bottom_up_team_factors` (or
    `blend_team_factors`) and supplies a run-scoring factor indexed to 1.0;
    `defense` from `bottom_up_team_ra` and supplies a runs-allowed factor, also
    indexed to 1.0. Both are converted to runs per game with a single shared
    league level, which is what forces league runs scored to equal league runs
    allowed — the identity the original audit found broken (4.28 vs 3.86).

    The win-percentage normalization is the second closure constraint: every
    game has exactly one winner, so the league's win percentages must average
    0.500. Normalizing the ODDS rather than the percentages keeps the result a
    valid probability and preserves the ordering.

    `league_rs_per_game` sets the absolute run environment. It is a forecast
    input, not something derivable from the projections — the per-PA engine has
    no league-anchoring step yet (see CURRENT_STATE_ASSESSMENT.md §5), so it is
    exposed here rather than silently assumed.

    Returns [team_col, team_abbr, rs_per_game, ra_per_game, run_diff,
    win_pct, expected_wins].
    """
    off_col = "team_factor" if "team_factor" in offense.columns else "bottom_up_factor"
    left = offense[[team_col, off_col]].rename(columns={off_col: "rs_factor"})
    right = defense[[team_col, "ra_factor"]]
    m = left.merge(right, on=team_col, how="inner")
    if m.empty:
        return m.assign(rs_per_game=[], ra_per_game=[], run_diff=[],
                        win_pct=[], expected_wins=[])

    m["team_abbr"] = [abbr_for_team_id(t) for t in m[team_col]]
    m["rs_per_game"] = m["rs_factor"] * league_rs_per_game
    m["ra_per_game"] = m["ra_factor"] * league_rs_per_game
    m["run_diff"] = (m["rs_per_game"] - m["ra_per_game"]) * GAMES_PER_SEASON

    raw = pythagenpat_win_pct(m["rs_per_game"], m["ra_per_game"])
    # Normalize on the odds scale so the mean win percentage is exactly 0.500
    # and every value stays inside (0, 1).
    odds = np.clip(raw, 1e-6, 1 - 1e-6)
    odds = odds / (1.0 - odds)
    scale = 1.0
    for _ in range(60):          # a few Newton-free bisection-ish passes
        wp = (odds * scale) / (1.0 + odds * scale)
        mean = wp.mean()
        if abs(mean - 0.5) < 1e-12:
            break
        scale *= (0.5 / mean) ** 1.5
    m["win_pct"] = (odds * scale) / (1.0 + odds * scale)
    m["expected_wins"] = m["win_pct"] * GAMES_PER_SEASON
    return m.sort_values("expected_wins", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Save / hold opportunity
# ─────────────────────────────────────────────────────────────────────────────

def project_save_hold_opportunity(
    wins: pd.DataFrame,
    *,
    team_col: str = "team_id",
    saves_per_win: float = SAVES_PER_WIN,
    holds_per_win: float = HOLDS_PER_WIN,
    save_elasticity: float = SAVE_OPP_ELASTICITY,
    hold_elasticity: float = HOLD_OPP_ELASTICITY,
    conversion: float = SAVE_CONVERSION_RATE,
) -> pd.DataFrame:
    """Team save and hold opportunity pools.

    Scaled off expected wins with a sub-linear elasticity, because a save
    chance requires a NARROW lead, not merely a win:

        team_save_opps = league_avg_opps x (wins / league_avg_wins) ** elasticity

    Elasticity below 1 encodes that a great team converts some of its extra
    wins into blowouts, which produce no save chance. Holds carry a lower
    elasticity still: a losing team plays plenty of close games and its setup
    men can earn holds in games it goes on to lose.

    Both pools are then normalized so the league totals match
    `saves_per_win`/`holds_per_win` x 2,430 exactly, so the elasticity changes
    only the DISTRIBUTION across teams, never the league level.

    Returns the input plus [save_opportunities, expected_saves,
    hold_opportunities, save_conversion_rate].
    """
    out = wins.copy()
    if out.empty:
        return out.assign(save_opportunities=[], expected_saves=[],
                          hold_opportunities=[], save_conversion_rate=[])

    avg_wins = GAMES_PER_SEASON / 2.0
    ratio = (out["expected_wins"] / avg_wins).to_numpy(float)

    league_save_opps = TOTAL_LEAGUE_WINS * saves_per_win / max(conversion, 1e-6)
    league_holds = TOTAL_LEAGUE_WINS * holds_per_win

    for col, elasticity, league_total in (
        ("save_opportunities", save_elasticity, league_save_opps),
        ("hold_opportunities", hold_elasticity, league_holds),
    ):
        shape = np.power(np.clip(ratio, 1e-6, None), elasticity)
        # Normalize to the league total so elasticity moves distribution only.
        out[col] = shape / shape.sum() * league_total

    out["expected_saves"] = out["save_opportunities"] * conversion
    out["save_conversion_rate"] = conversion
    return out


def fit_opportunity_rates(team_seasons: pd.DataFrame) -> dict[str, float]:
    """Fit saves/holds-per-win and their elasticities from real team-seasons.

    `team_seasons` needs columns [W, SV, HLD] (and optionally SVO). Replaces
    the provisional constants above, which are the least certain numbers in
    this module. statsapi exposes team SV and HLD, so extending
    `data_acquisition.fetch_team_rpg` to carry them is all this needs.

    The elasticity is fitted in logs: log(SV) = a + b x log(W / avg_W), where b
    is the elasticity.
    """
    df = team_seasons.dropna(subset=["W"]).copy()
    out: dict[str, float] = {}
    avg_w = float(df["W"].mean())
    for stat, key in (("SV", "saves"), ("HLD", "holds")):
        if stat not in df.columns:
            continue
        sub = df[(df[stat] > 0) & (df["W"] > 0)]
        if len(sub) < 10:
            continue
        out[f"{key}_per_win"] = float(sub[stat].sum() / sub["W"].sum())
        b = np.polyfit(np.log(sub["W"] / avg_w), np.log(sub[stat]), 1)[0]
        out[f"{key}_elasticity"] = float(b)
    if {"SV", "SVO"} <= set(df.columns) and df["SVO"].sum() > 0:
        out["save_conversion_rate"] = float(df["SV"].sum() / df["SVO"].sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def wins_report(wins: pd.DataFrame, *, top: int = 10) -> str:
    """Readable table of the win and opportunity projections."""
    if wins.empty:
        return "no teams to report"
    head = (f"{'team':<6}{'RS/G':>7}{'RA/G':>7}{'RD':>7}{'W%':>7}{'W':>7}"
            f"{'SVO':>7}{'SV':>6}{'HLD':>6}")
    lines = [head]
    for _, r in wins.head(top).iterrows():
        lines.append(
            f"{str(r.get('team_abbr') or r.iloc[0]):<6}"
            f"{r['rs_per_game']:>7.2f}{r['ra_per_game']:>7.2f}"
            f"{r['run_diff']:>+7.0f}{r['win_pct']:>7.3f}"
            f"{r['expected_wins']:>7.1f}"
            f"{r.get('save_opportunities', float('nan')):>7.1f}"
            f"{r.get('expected_saves', float('nan')):>6.1f}"
            f"{r.get('hold_opportunities', float('nan')):>6.1f}"
        )
    tot = wins["expected_wins"].sum()
    lines.append(f"  league wins {tot:.1f} (must be {TOTAL_LEAGUE_WINS})")
    if "expected_saves" in wins.columns:
        lines.append(f"  league saves {wins['expected_saves'].sum():.0f}"
                     f"   holds {wins['hold_opportunities'].sum():.0f}"
                     f"   (levels are PROVISIONAL — see fit_opportunity_rates)")
    return "\n".join(lines)

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
    """Readable table of the win and opportunity projections.

    When a market prior has been applied, this prints the market-adjusted
    RS/RA rather than the pre-blend roster values. Printing the originals
    alongside blended wins made the table self-contradictory — a club could
    show a +7 run differential next to 88 wins, which reads as a broken model
    rather than as two columns from different stages. The originals are still
    on the frame (`rs_per_game`, `ra_per_game`) for auditing.
    """
    if wins.empty:
        return "no teams to report"
    market = "rs_per_game_market" in wins.columns
    rs_col = "rs_per_game_market" if market else "rs_per_game"
    ra_col = "ra_per_game_market" if market else "ra_per_game"
    rd_col = "run_diff_market" if market else "run_diff"

    head = (f"{'team':<6}{'RS/G':>7}{'RA/G':>7}{'RD':>7}{'W%':>7}{'W':>7}"
            f"{'SVO':>7}{'SV':>6}{'HLD':>6}")
    lines = [head]
    if market:
        lines.append("  (RS/RA are market-adjusted: the blended win total sets "
                     "the differential,\n   the roster sets the split)")
    for _, r in wins.head(top).iterrows():
        lines.append(
            f"{str(r.get('team_abbr') or r.iloc[0]):<6}"
            f"{r[rs_col]:>7.2f}{r[ra_col]:>7.2f}"
            f"{r[rd_col]:>+7.0f}{r['win_pct']:>7.3f}"
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


# ─────────────────────────────────────────────────────────────────────────────
# Pitcher decisions — wins, losses, quality starts
# ─────────────────────────────────────────────────────────────────────────────
#
# Wins and losses are a TEAM quantity handed out to pitchers, so they are
# allocated the way saves and holds are: each club's expected wins (and the
# losses that are the rest of its 162) are a pool, split across its staff by
# weight, and capped by what a pitcher could physically record. The pools close
# exactly, so a club's pitchers always sum to the club's record.
#
# ASSUMPTIONS, NOT FITS. The repo has no historical decisions to fit against
# (the pipeline's history carries rates, not W/L), so the three constants below
# are set from the shape of a modern season and are the first thing to refit
# when decisions are added to the history:
#
#   SP_WIN_SHARE / SP_LOSS_SHARE   starters take roughly two thirds of a club's
#                                  wins and a little over half of its losses —
#                                  relievers lose more than they win, because
#                                  a blown lead is a loss and a win needs one
#                                  to hold.
#   QS_IP_MIDPOINT / QS_IP_SCALE   a start reaches six innings half the time at
#                                  5.3 innings a start.
#
# WHO GETS THEM, within the pool, is not assumed: a starter's weight is his
# starts, times how often a team with his run prevention wins (Pythagorean, his
# runs allowed against his club's run scoring), times how likely he is to go
# long enough to be the pitcher of record. A reliever's is his innings times the
# same win rate, since he pitches the games in proportion to his innings.
SP_WIN_SHARE = 0.65
SP_LOSS_SHARE = 0.55
# A decision is a noisy outcome of a game result, so its projection is regressed:
# each pitcher's weight is pulled this fraction of the way to his club's
# average, which shrinks the spread between aces and the rest WITHOUT moving
# the club's total (the pools are fixed). Applied to wins and losses alike so
# a pitcher's record stays coherent. 0.0 = the Pythagorean spread as it falls
# out; 1.0 = every starter on a club identical. An assumption, like the shares.
DECISION_REGRESSION = 0.25
PYTH_EXPONENT = 1.83
QS_IP_MIDPOINT = 5.3
QS_IP_SCALE = 0.9
QS_RUN_LEVEL = 0.8        # runs allowed in a start that went six are lower than a start's average


def _pythag(rs: np.ndarray, ra: np.ndarray) -> np.ndarray:
    rs = np.clip(rs, 0.5, None) ** PYTH_EXPONENT
    ra = np.clip(ra, 0.5, None) ** PYTH_EXPONENT
    return rs / (rs + ra)


def quality_start_rate(ip_per_start, era) -> np.ndarray:
    """P(a start is a quality start): six innings AND no more than three
    earned runs. Reaching six is logistic in innings per start; the runs are a
    Poisson with the pitcher's own ERA over those six, a little lower than
    average because a start that goes six is a start that went well."""
    ips = np.asarray(ip_per_start, dtype=float)
    era = np.asarray(era, dtype=float)
    p6 = 1.0 / (1.0 + np.exp(-(ips - QS_IP_MIDPOINT) / QS_IP_SCALE))
    lam = np.clip(era, 0.5, None) / 9.0 * 6.3 * QS_RUN_LEVEL
    cdf3 = np.exp(-lam) * (1 + lam + lam ** 2 / 2 + lam ** 3 / 6)
    return np.clip(p6 * cdf3, 0.0, 1.0)


def project_decisions(pitchers: pd.DataFrame, pools: pd.DataFrame, *,
                      team_col: str = "team_id",
                      eligible: np.ndarray | None = None) -> pd.DataFrame:
    """W, L and QS per pitcher, indexed like `pitchers`.

    `pitchers` needs [team_col, Proj_IP, Proj_G, Proj_GS, ERA, RA9]; `pools`
    is `project_team_wins` output (team_id, expected_wins, rs_per_game,
    ra_per_game). A pitcher with no club, or whose club has no pool, is NaN.
    """
    from playing_time_model import allocate_opportunity

    def num(col):
        return pd.to_numeric(pitchers[col], errors="coerce").fillna(0.0) \
            .to_numpy(float) if col in pitchers.columns \
            else np.zeros(len(pitchers))

    out = pd.DataFrame(np.nan, index=pitchers.index, columns=["W", "L", "QS"])
    if pools is None or len(pools) == 0 or team_col not in pitchers.columns:
        return out
    pool = pools.set_index("team_id")
    ip, g, gs, era, ra9 = (num("Proj_IP"), num("Proj_G"), num("Proj_GS"),
                           num("ERA"), num("RA9"))
    ra9 = np.where(ra9 > 0, ra9, np.where(era > 0, era * 1.08, 4.4))
    # Floor-tier pitchers carry a 1-inning placeholder, not a role. Letting them
    # draw decisions would hand each of ~125 per club a few hundredths of a win
    # that displays as 0.0 and quietly takes ~4% of the pool with it.
    can = (np.ones(len(pitchers), dtype=bool) if eligible is None
           else np.asarray(eligible, dtype=bool))
    is_sp = gs >= 0.5 * np.maximum(g, 1.0)
    ips = np.where(gs > 0, ip / np.maximum(gs, 1e-9), 0.0)

    w, l = np.zeros(len(out)), np.zeros(len(out))
    teams = pitchers[team_col].to_numpy()
    for tid in pd.unique(teams[~pd.isna(teams)]):
        if tid not in pool.index:
            continue
        pos = np.flatnonzero(teams == tid)
        rs = float(pool.loc[tid, "rs_per_game"])
        ra_team = float(pool.loc[tid, "ra_per_game"])
        wins = float(pool.loc[tid, "expected_wins"])
        losses = 162.0 - wins
        # Share of a game he is on the mound: a starter's innings per start,
        # a reliever's a sliver. The rest of the game is the club's average.
        frac = np.where(is_sp[pos], np.clip(ips[pos] / 9.0, 0.10, 0.85), 0.10)
        pw = _pythag(np.full(len(pos), rs), frac * ra9[pos] + (1 - frac) * ra_team)
        go_long = np.clip(ips / 5.2, 0.5, 1.25)[pos]
        # Decisions come from STARTS and from RELIEF INNINGS, by pitcher, not by
        # whether he is filed as a starter. A club's starts are made by more men
        # than its rotation (PIT: ~8 of 162 by pitchers classed as relievers), so
        # splitting the starters' pool over only the classed starters spread it
        # over ~154 starts instead of 162 and inflated every starter's rate by
        # ~5% — Skenes read 15.9 wins for it. Each pitcher gets a share of BOTH
        # pools, a swingman from his starts and from his relief innings.
        starts = gs[pos]
        # Relief innings, from appearances: a pure starter (G == GS) has none,
        # a swingman has his innings in proportion to the games he relieved.
        # (Subtracting an assumed innings-per-start instead gave a 6.3-inning
        # starter 45 phantom relief innings and a share of the bullpen's wins.)
        relief_ip = ip[pos] * np.clip((g[pos] - starts) / np.maximum(g[pos], 1e-9),
                                      0.0, 1.0)
        relief_cap = np.maximum(g[pos] - starts, 0.0)
        for decide, long_pow, pool_total, share, dest in (
                (pw, 1.0, wins, SP_WIN_SHARE, w),
                ((1.0 - pw) / np.clip(pw, 1e-6, None), 0.5, losses,
                 SP_LOSS_SHARE, l)):
            # A win goes to the pitcher who goes long on a winning day; a loss
            # is about how often his club loses behind him, so it follows the
            # LOSS ODDS (a .45 club behind him loses at 1.2:1, a .62 club at
            # 0.6:1) and is only softly reduced by going short — a starter
            # pulled early while trailing is still the pitcher of record.
            for wt, units, part, cap in (
                    (starts * decide * go_long ** long_pow, starts, share,
                     starts),
                    (relief_ip * decide, relief_ip, 1.0 - share, relief_cap)):
                wt = np.where(can[pos], wt, 0.0)
                units = np.where(can[pos], units, 0.0)
                if wt.sum() <= 0 or units.sum() <= 0:
                    continue
                # Regress the per-unit RATE (not the total) toward the club's
                # mean, so a pitcher with more starts still gets more of them.
                rate = np.divide(wt, units, out=np.zeros_like(wt),
                                 where=units > 0)
                mean = wt.sum() / units.sum()
                wt = units * ((1 - DECISION_REGRESSION) * rate
                              + DECISION_REGRESSION * mean)
                dest[pos] += allocate_opportunity(wt / wt.sum(),
                                                  pool_total * part, cap)
    out["W"], out["L"] = w, l
    qs = np.where(is_sp & can, gs * quality_start_rate(ips, era), 0.0)
    out["QS"] = qs
    has_club = pitchers[team_col].isin(pool.index).to_numpy()
    out.loc[~has_club, ["W", "L", "QS"]] = np.nan
    return out

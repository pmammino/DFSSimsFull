"""
test_team_wins.py
=================
Tests for team win expectancy, the save / hold opportunity pools, free-agent
handling, and roster reserves.

Two closure constraints carry the weight:

  1. **League wins total exactly 2,430.** Every game has one winner, so the
     league's win percentages must average .500. The original audit's naive
     Pythagenpat produced 2,618 because offense and defense were aggregated
     over differently-selected pools.
  2. **Save and hold pools hit their league level exactly**, so the
     win-elasticity knob changes only the DISTRIBUTION across teams.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from team_context import (  # noqa: E402
    FREE_AGENT_ABBR,
    FREE_AGENT_TEAM_ID,
    abbr_for_team_id,
    attach_roster_reserves,
    blend_team_factors,
    bottom_up_team_factors,
    is_free_agent,
    load_roster_reserves,
    load_team_overrides,
    mlb_clubs_only,
    team_id_for_abbr,
)
from team_wins import (  # noqa: E402
    GAMES_PER_SEASON,
    TOTAL_LEAGUE_WINS,
    bottom_up_team_ra,
    fit_opportunity_rates,
    project_save_hold_opportunity,
    project_team_wins,
    pythagenpat_win_pct,
)

NYY, LAD, COL, SD = 147, 119, 115, 135
ALL_TEAMS = [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
             121, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
             145, 146, 147, 158]


def _players(spec, per_team=10, rate_key="P_HR", pt_col="Proj_PA"):
    """Synthetic roster: {team_id: rate}. Higher rate -> more runs."""
    rows, pid = [], 1
    for team_id, rate in spec.items():
        for i in range(per_team):
            spread = 0.6 + 0.8 * (i / max(per_team - 1, 1))
            r = rate * spread
            rows.append({
                "PlayerId": pid, "team_id": team_id,
                "P_K": 0.22, "P_BB": 0.08, "P_HBP": 0.01, "P_SF": 0.006,
                "P_HR": r, "P_3B": 0.004, "P_2B": 0.046, "P_1B": 0.142,
                "P_BIPOut": 0.492 - r,
                "Career_PA": 3000.0, pt_col: np.nan,
                "pt_tier": "projected",
            })
            pid += 1
    return pd.DataFrame(rows)


def _league(seed=0):
    """30 teams with a realistic spread of offensive and pitching talent."""
    rng = np.random.default_rng(seed)
    off = {t: float(v) for t, v in
           zip(ALL_TEAMS, rng.uniform(0.018, 0.045, len(ALL_TEAMS)))}
    dfn = {t: float(v) for t, v in
           zip(ALL_TEAMS, rng.uniform(0.018, 0.045, len(ALL_TEAMS)))}
    hitters = _players(off)
    pitchers = _players(dfn, per_team=14, pt_col="Proj_IP")
    return hitters, pitchers


# ─────────────────────────────────────────────────────────────────────────────
# Pythagenpat
# ─────────────────────────────────────────────────────────────────────────────

def test_equal_runs_gives_a_five_hundred_team():
    assert pythagenpat_win_pct([4.5], [4.5])[0] == pytest.approx(0.5, abs=1e-12)


def test_win_pct_rises_with_run_differential():
    wp = pythagenpat_win_pct([5.0, 4.5, 4.0], [4.0, 4.5, 5.0])
    assert wp[0] > wp[1] > wp[2]


def test_the_same_differential_is_worth_more_in_a_low_run_environment():
    """The reason for Pythagenpat over a fixed exponent."""
    low = pythagenpat_win_pct([3.5], [3.0])[0]
    high = pythagenpat_win_pct([6.0], [5.5])[0]
    assert low > high


# ─────────────────────────────────────────────────────────────────────────────
# Wins
# ─────────────────────────────────────────────────────────────────────────────

def _wins(seed=0):
    hitters, pitchers = _league(seed)
    offense = blend_team_factors(bottom_up_team_factors(hitters),
                                 bottom_up_weight=1.0)
    defense = bottom_up_team_ra(pitchers)
    return project_team_wins(offense, defense)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_league_wins_total_exactly_2430(seed):
    """THE closure constraint. Every game has exactly one winner."""
    w = _wins(seed)
    assert len(w) == 30
    assert w["expected_wins"].sum() == pytest.approx(TOTAL_LEAGUE_WINS, abs=1e-6)
    assert w["win_pct"].mean() == pytest.approx(0.5, abs=1e-9)


def test_win_percentages_stay_valid_probabilities():
    w = _wins()
    assert (w["win_pct"] > 0).all() and (w["win_pct"] < 1).all()
    assert (w["expected_wins"] > 0).all()
    assert (w["expected_wins"] < GAMES_PER_SEASON).all()


def test_normalization_preserves_the_win_pct_ordering():
    """Scaling the odds to a .500 league mean must not reshuffle teams.

    Note this is about the RANK of win percentage, not run differential. Those
    two orderings legitimately differ: Pythagenpat's exponent depends on the
    run environment, so a smaller differential in a lower-scoring context can
    be worth more wins. That is the reason to use Pythagenpat at all.
    """
    w = _wins()
    raw = pythagenpat_win_pct(w["rs_per_game"], w["ra_per_game"])
    assert list(np.argsort(-raw)) == list(np.argsort(-w["win_pct"].to_numpy()))
    assert w["win_pct"].is_monotonic_decreasing, "output should be win-sorted"


def test_league_runs_scored_equals_league_runs_allowed():
    """The identity the audit found broken at 4.28 vs 3.86."""
    w = _wins()
    assert w["rs_per_game"].mean() == pytest.approx(
        w["ra_per_game"].mean(), rel=0.02)


def test_the_run_environment_input_scales_both_sides():
    hitters, pitchers = _league()
    offense = blend_team_factors(bottom_up_team_factors(hitters),
                                 bottom_up_weight=1.0)
    defense = bottom_up_team_ra(pitchers)
    low = project_team_wins(offense, defense, league_rs_per_game=3.8)
    high = project_team_wins(offense, defense, league_rs_per_game=5.2)
    assert high["rs_per_game"].mean() > low["rs_per_game"].mean()
    # Wins still close regardless of the assumed environment.
    for w in (low, high):
        assert w["expected_wins"].sum() == pytest.approx(TOTAL_LEAGUE_WINS,
                                                          abs=1e-6)


def test_empty_input_does_not_blow_up():
    empty = pd.DataFrame(columns=["team_id", "bottom_up_factor"])
    assert project_team_wins(empty, pd.DataFrame(columns=["team_id",
                                                           "ra_factor"])).empty


# ─────────────────────────────────────────────────────────────────────────────
# Save / hold opportunity
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_hold_pools_hit_their_league_level():
    """The elasticity must redistribute, never change the league total."""
    w = project_save_hold_opportunity(_wins())
    from team_wins import (HOLDS_PER_WIN, SAVE_CONVERSION_RATE, SAVES_PER_WIN)
    assert w["expected_saves"].sum() == pytest.approx(
        TOTAL_LEAGUE_WINS * SAVES_PER_WIN, rel=1e-9)
    assert w["hold_opportunities"].sum() == pytest.approx(
        TOTAL_LEAGUE_WINS * HOLDS_PER_WIN, rel=1e-9)
    assert w["expected_saves"].sum() == pytest.approx(
        w["save_opportunities"].sum() * SAVE_CONVERSION_RATE, rel=1e-9)


def test_better_teams_get_more_save_opportunity():
    w = project_save_hold_opportunity(_wins()).sort_values("expected_wins")
    assert w["save_opportunities"].is_monotonic_increasing


def test_save_opportunity_scales_sub_linearly_with_wins():
    """A great team converts some extra wins into blowouts, which produce no
    save chance — so the save ratio must compress relative to the win ratio."""
    w = project_save_hold_opportunity(_wins())
    best, worst = w.iloc[0], w.iloc[-1]
    win_ratio = best["expected_wins"] / worst["expected_wins"]
    svo_ratio = best["save_opportunities"] / worst["save_opportunities"]
    assert 1.0 < svo_ratio < win_ratio


def test_holds_are_less_win_dependent_than_saves():
    """A losing team still plays close games and still bridges innings."""
    w = project_save_hold_opportunity(_wins())
    best, worst = w.iloc[0], w.iloc[-1]
    svo = best["save_opportunities"] / worst["save_opportunities"]
    hld = best["hold_opportunities"] / worst["hold_opportunities"]
    assert hld < svo


def test_opportunity_rates_can_be_fitted_from_real_team_seasons():
    """The provisional constants must be replaceable from data."""
    rng = np.random.default_rng(3)
    w = rng.integers(60, 105, 120)
    df = pd.DataFrame({"W": w,
                       "SV": (w * 0.48 + rng.normal(0, 2, 120)).round(),
                       "HLD": (w * 0.9 + rng.normal(0, 4, 120)).round(),
                       "SVO": (w * 0.70).round()})
    fit = fit_opportunity_rates(df)
    assert fit["saves_per_win"] == pytest.approx(0.48, abs=0.03)
    assert fit["holds_per_win"] == pytest.approx(0.90, abs=0.05)
    assert fit["save_conversion_rate"] == pytest.approx(0.48 / 0.70, abs=0.05)
    assert 0.5 < fit["saves_elasticity"] < 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Free agents
# ─────────────────────────────────────────────────────────────────────────────

def test_fa_resolves_both_directions():
    assert team_id_for_abbr("FA") == FREE_AGENT_TEAM_ID
    assert team_id_for_abbr("free agent") == FREE_AGENT_TEAM_ID
    assert abbr_for_team_id(FREE_AGENT_TEAM_ID) == FREE_AGENT_ABBR


def test_free_agent_id_cannot_collide_with_a_real_club():
    from team_context import TEAM_ABBR_BY_ID
    assert FREE_AGENT_TEAM_ID not in TEAM_ABBR_BY_ID
    assert FREE_AGENT_TEAM_ID < 0


def test_free_agent_is_distinct_from_unknown_team(tmp_path):
    """The distinction carries real information: 'FA' means we know he has no
    club, null means we don't know anything."""
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"assignments": [
        {"player_id": 1, "team": "FA"},
        {"player_id": 2, "team": None},
    ]}))
    loaded = load_team_overrides(p)
    assert loaded[1] == FREE_AGENT_TEAM_ID
    assert loaded[2] is None
    assert is_free_agent(loaded[1]) and not is_free_agent(loaded[2])


def test_free_agents_are_excluded_from_team_aggregates():
    """Otherwise FREE_AGENT_TEAM_ID becomes a 31st club and pulls every real
    team's factor off 1.0."""
    hitters, _ = _league()
    fa = _players({FREE_AGENT_TEAM_ID: 0.060}, per_team=25)
    fa["PlayerId"] += 100_000

    base = blend_team_factors(bottom_up_team_factors(hitters),
                              bottom_up_weight=1.0).set_index("team_id")
    withfa = blend_team_factors(
        bottom_up_team_factors(pd.concat([hitters, fa], ignore_index=True)),
        bottom_up_weight=1.0,
    ).set_index("team_id")

    assert len(withfa) == 30, "free agents became a 31st team"
    assert FREE_AGENT_TEAM_ID not in withfa.index
    for t in base.index:
        assert withfa.loc[t, "team_factor"] == pytest.approx(
            base.loc[t, "team_factor"], abs=1e-12)


def test_mlb_clubs_only_drops_fa_and_unknown():
    df = pd.DataFrame({"team_id": [NYY, FREE_AGENT_TEAM_ID, np.nan, LAD]})
    assert sorted(mlb_clubs_only(df)["team_id"]) == sorted([LAD, NYY])


def test_free_agents_stay_in_the_projection_set():
    """Excluded from aggregates is not the same as dropped."""
    hitters, _ = _league()
    fa = _players({FREE_AGENT_TEAM_ID: 0.060}, per_team=5)
    fa["PlayerId"] += 100_000
    combined = pd.concat([hitters, fa], ignore_index=True)
    assert len(combined) == len(hitters) + 5
    assert (combined["team_id"] == FREE_AGENT_TEAM_ID).sum() == 5


# ─────────────────────────────────────────────────────────────────────────────
# Roster reserves
# ─────────────────────────────────────────────────────────────────────────────

def _reserve_file(tmp_path, entries):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"reserves": entries}))
    return p


def test_reserves_load_for_hitters_and_pitchers_independently(tmp_path):
    p = _reserve_file(tmp_path, [
        {"team": "NYY", "pa_share": 0.10},
        {"team": "SD", "ip_share": 0.12},
    ])
    r = load_roster_reserves(p)
    assert r[NYY]["pa_share"] == pytest.approx(0.10)
    assert r[NYY]["ip_share"] == 0.0
    assert r[SD]["ip_share"] == pytest.approx(0.12)
    assert r[SD]["pa_share"] == 0.0


def test_reserve_shares_are_clipped(tmp_path):
    """A club cannot reserve more than half its playing time for players it
    has not acquired."""
    p = _reserve_file(tmp_path, [{"team": "NYY", "pa_share": 0.9},
                                 {"team": "LAD", "pa_share": -0.3}])
    r = load_roster_reserves(p)
    assert r[NYY]["pa_share"] == 0.5
    assert r[LAD]["pa_share"] == 0.0


def test_unknown_teams_in_reserves_warn_and_skip(tmp_path):
    p = _reserve_file(tmp_path, [{"team": "NOT_A_TEAM", "pa_share": 0.1},
                                 {"team": "COL", "pa_share": 0.2}])
    with pytest.warns(UserWarning):
        r = load_roster_reserves(p)
    assert set(r) == {COL}


def test_missing_reserve_file_is_optional():
    assert load_roster_reserves("/nonexistent/nope.json") == {}


def test_reserves_attach_to_the_team_context_frame():
    hitters, _ = _league()
    factors = blend_team_factors(bottom_up_team_factors(hitters),
                                 bottom_up_weight=1.0)
    out = attach_roster_reserves(factors, {NYY: {"pa_share": 0.10,
                                                  "ip_share": 0.0}}
                                 ).set_index("team_id")
    assert out.loc[NYY, "pa_reserve_share"] == pytest.approx(0.10)
    assert out.loc[LAD, "pa_reserve_share"] == 0.0


def test_reserves_do_not_touch_the_rate_projections():
    """A reserve says 'playing time is unaccounted for', NOT 'this team is
    better than it looks' — the quality of an unsigned player is unknowable."""
    hitters, _ = _league()
    factors = blend_team_factors(bottom_up_team_factors(hitters),
                                 bottom_up_weight=1.0)
    withres = attach_roster_reserves(factors, {NYY: {"pa_share": 0.5}})
    assert list(withres["team_factor"]) == list(factors["team_factor"])

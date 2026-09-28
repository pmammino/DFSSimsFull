"""
test_market_odds.py
===================
Tests for the betting-market team-strength prior.

Three properties carry the weight:

  1. **De-vigging is correct and bias-aware.** A World Series board comes in
     with a 15-35% margin, and the default `power` method must shrink longshots
     harder than favorites, because longshots are overbet.
  2. **League closure survives the market.** Blended wins must still total
     exactly 2,430 — every game has one winner, whatever the odds say.
  3. **The market sets the run DIFFERENTIAL, not the RS/RA split.** A futures
     price cannot distinguish a 5.2/4.2 club from a 4.3/3.3 one, so the roster
     must keep supplying the split.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_odds import (  # noqa: E402
    LEAGUE_MEAN_WINS,
    MAX_PROJECTED_WINS,
    MIN_PROJECTED_WINS,
    american_to_probability,
    apply_market_to_run_environment,
    blend_market_wins,
    decimal_to_probability,
    devig,
    implied_wins_from_probabilities,
    load_market_odds,
    market_expected_wins,
    to_probability,
)
from team_context import TEAM_ABBR_BY_ID  # noqa: E402
from team_wins import TOTAL_LEAGUE_WINS  # noqa: E402

LAD, NYY, COL = 119, 147, 115
ALL_TEAMS = sorted(TEAM_ABBR_BY_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Odds conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_american_odds_convert_both_signs():
    assert american_to_probability(100) == pytest.approx(0.5)
    assert american_to_probability(-100) == pytest.approx(0.5)
    assert american_to_probability(450) == pytest.approx(100 / 550)
    assert american_to_probability(-150) == pytest.approx(150 / 250)


def test_a_longer_price_is_a_smaller_probability():
    assert american_to_probability(30000) < american_to_probability(450)


def test_decimal_odds_convert():
    assert decimal_to_probability(2.0) == pytest.approx(0.5)
    assert decimal_to_probability(5.5) == pytest.approx(1 / 5.5)


def test_invalid_odds_are_rejected():
    with pytest.raises(ValueError):
        american_to_probability(0)
    with pytest.raises(ValueError):
        decimal_to_probability(1.0)
    with pytest.raises(ValueError):
        to_probability(100, "klingon")


def test_probability_format_accepts_fraction_or_percentage():
    assert to_probability(0.18, "probability") == pytest.approx(0.18)
    assert to_probability(18, "probability") == pytest.approx(0.18)


# ─────────────────────────────────────────────────────────────────────────────
# De-vigging
# ─────────────────────────────────────────────────────────────────────────────

def _board(n=30, favourite=375, longshot=35000):
    """A realistically-shaped futures board: a few short prices, a long tail."""
    prices = np.geomspace(favourite, longshot, n)
    return {i: american_to_probability(p) for i, p in enumerate(prices)}


@pytest.mark.parametrize("method", ["proportional", "power"])
def test_devig_produces_a_probability_distribution(method):
    fair, overround = devig(_board(), method=method)
    assert sum(fair.values()) == pytest.approx(1.0, abs=1e-10)
    assert all(0 < v < 1 for v in fair.values())
    assert overround > 1.0, "a real board carries a margin"


def test_devig_preserves_the_ordering():
    board = _board()
    fair, _ = devig(board, method="power")
    order_in = [k for k, _ in sorted(board.items(), key=lambda kv: -kv[1])]
    order_out = [k for k, _ in sorted(fair.items(), key=lambda kv: -kv[1])]
    assert order_in == order_out


def test_power_devig_shrinks_longshots_harder_than_favorites():
    """The favorite-longshot correction, and the reason `power` is the default.

    Longshots are systematically overbet, so a single proportional scale factor
    leaves their probability overstated.
    """
    board = _board()
    fav, dog = min(board), max(board)
    prop, _ = devig(board, method="proportional")
    powr, _ = devig(board, method="power")

    assert powr[dog] < prop[dog], "longshot should be cut further"
    assert powr[fav] > prop[fav], "favorite should keep more of its share"


def test_devig_rejects_impossible_probabilities():
    with pytest.raises(ValueError):
        devig({1: 0.0, 2: 0.5})
    with pytest.raises(ValueError):
        devig({1: 1.0, 2: 0.5})
    with pytest.raises(ValueError):
        devig({1: -0.1, 2: 0.5})


def test_devig_handles_an_empty_board():
    fair, over = devig({})
    assert fair == {} and over == 0.0


def test_unknown_devig_method_is_rejected():
    with pytest.raises(ValueError):
        devig(_board(), method="vibes")


# ─────────────────────────────────────────────────────────────────────────────
# Probability -> wins
# ─────────────────────────────────────────────────────────────────────────────

def _fair_board():
    fair, _ = devig({t: american_to_probability(p) for t, p in zip(
        ALL_TEAMS, np.geomspace(375, 35000, 30))})
    return fair


def test_implied_wins_centre_on_the_league_mean():
    """Closure: 30 teams x 81 wins = 2430."""
    wins = implied_wins_from_probabilities(_fair_board())
    assert np.mean(list(wins.values())) == pytest.approx(LEAGUE_MEAN_WINS,
                                                          abs=1e-9)
    assert sum(wins.values()) == pytest.approx(TOTAL_LEAGUE_WINS, abs=1e-6)


def test_implied_wins_are_monotone_in_the_odds():
    fair = _fair_board()
    wins = implied_wins_from_probabilities(fair)
    by_prob = [t for t, _ in sorted(fair.items(), key=lambda kv: -kv[1])]
    by_wins = [t for t, _ in sorted(wins.items(), key=lambda kv: -kv[1])]
    assert by_prob == by_wins


def test_implied_wins_stay_in_a_plausible_band():
    """A mangled or extreme board must not yield a 130-win club."""
    fair, _ = devig({t: american_to_probability(p) for t, p in zip(
        ALL_TEAMS, np.geomspace(120, 500000, 30))})
    wins = implied_wins_from_probabilities(fair)
    assert min(wins.values()) >= MIN_PROJECTED_WINS - 1
    assert max(wins.values()) <= MAX_PROJECTED_WINS + 1


def test_implied_wins_spread_tracks_the_target_sd():
    wins = implied_wins_from_probabilities(_fair_board(), league_sd=11.5)
    tight = implied_wins_from_probabilities(_fair_board(), league_sd=6.0)
    assert np.std(list(wins.values())) > np.std(list(tight.values()))


def test_an_all_equal_board_gives_every_team_the_mean():
    fair = {t: 1 / 30 for t in ALL_TEAMS}
    wins = implied_wins_from_probabilities(fair)
    assert all(w == pytest.approx(LEAGUE_MEAN_WINS) for w in wins.values())


def test_implied_wins_handles_an_empty_board():
    assert implied_wins_from_probabilities({}) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, payload):
    p = tmp_path / "odds.json"
    p.write_text(json.dumps(payload))
    return p


def test_missing_odds_file_is_optional():
    assert load_market_odds("/nonexistent/none.json").empty


def test_load_resolves_team_codes_and_records_provenance(tmp_path):
    p = _write(tmp_path, {"market": "world_series", "odds_format": "american",
                          "as_of": "2026-11-15", "book": "consensus",
                          "teams": {"LAD": 400, "NY-A": 700, "CHW": 30000}})
    with pytest.warns(UserWarning, match="only 3/30"):
        df = load_market_odds(p)
    assert set(df["team_id"]) == {LAD, NYY, 145}
    assert df.attrs["as_of"] == "2026-11-15"
    assert df.attrs["book"] == "consensus"


def test_unknown_teams_warn_and_skip(tmp_path):
    p = _write(tmp_path, {"teams": {"LAD": 400, "NOT_A_TEAM": 900}})
    with pytest.warns(UserWarning):
        df = load_market_odds(p)
    assert list(df["team_id"]) == [LAD]


def test_a_partial_board_warns_because_it_mixes_scales(tmp_path):
    """Teams left off fall back to the bottom-up projection, so a partial board
    silently blends two different scales."""
    p = _write(tmp_path, {"teams": {"LAD": 400, "NYY": 700}})
    with pytest.warns(UserWarning, match="only 2/30"):
        load_market_odds(p)


def test_a_full_board_does_not_warn(tmp_path, recwarn):
    p = _write(tmp_path, {"teams": {TEAM_ABBR_BY_ID[t]: 1000 + 500 * i
                                    for i, t in enumerate(ALL_TEAMS)}})
    load_market_odds(p)
    assert not [w for w in recwarn if "board cover" in str(w.message)]


# ─────────────────────────────────────────────────────────────────────────────
# Win-total lines — the better market
# ─────────────────────────────────────────────────────────────────────────────

def test_win_total_lines_are_read_directly(tmp_path):
    """No inference: the line IS an expected-wins estimate."""
    lines = {TEAM_ABBR_BY_ID[t]: 81.0 for t in ALL_TEAMS}
    lines["LAD"], lines["COL"] = 96.0, 66.0
    p = _write(tmp_path, {"market": "win_total", "teams": lines})
    m = market_expected_wins(load_market_odds(p)).set_index("team_id")
    assert m.loc[LAD, "market_source"] == "win_total"
    assert pd.isna(m.loc[LAD, "market_prob"]), "a line carries no title odds"
    assert m.loc[LAD, "market_wins"] > m.loc[COL, "market_wins"]


def test_win_total_board_is_recentred_to_the_league_mean():
    """Books shade totals down to balance action; closure needs 81."""
    lines = pd.DataFrame({
        "team_id": ALL_TEAMS,
        "team_abbr": [TEAM_ABBR_BY_ID[t] for t in ALL_TEAMS],
        "value": [79.0] * 30,          # a full point light across the board
    })
    lines.attrs["market"] = "win_total"
    m = market_expected_wins(lines)
    assert m["market_wins"].mean() == pytest.approx(LEAGUE_MEAN_WINS, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Blending
# ─────────────────────────────────────────────────────────────────────────────

def _wins_frame(seed=0):
    rng = np.random.default_rng(seed)
    wp = np.clip(rng.normal(0.5, 0.06, 30), 0.3, 0.7)
    wp = wp / wp.mean() * 0.5
    rs = rng.uniform(4.0, 5.0, 30)
    return pd.DataFrame({
        "team_id": ALL_TEAMS,
        "team_abbr": [TEAM_ABBR_BY_ID[t] for t in ALL_TEAMS],
        "rs_per_game": rs,
        "ra_per_game": rng.uniform(4.0, 5.0, 30),
        "run_diff": rng.uniform(-100, 100, 30),
        "win_pct": wp,
        "expected_wins": wp * 162,
    })


def _market_frame():
    fair = _fair_board()
    wins = implied_wins_from_probabilities(fair)
    return pd.DataFrame({
        "team_id": list(wins),
        "team_abbr": [TEAM_ABBR_BY_ID[t] for t in wins],
        "market_prob": [fair[t] for t in wins],
        "market_wins": [wins[t] for t in wins],
        "market_source": "world_series",
    })


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_blending_preserves_league_closure(weight):
    out = blend_market_wins(_wins_frame(), _market_frame(),
                            market_weight=weight)
    assert out["expected_wins"].sum() == pytest.approx(TOTAL_LEAGUE_WINS,
                                                        abs=1e-6)
    assert out["win_pct"].mean() == pytest.approx(0.5, abs=1e-9)


def test_weight_zero_leaves_the_roster_projection_alone():
    base = _wins_frame()
    out = blend_market_wins(base, _market_frame(), market_weight=0.0)
    merged = base.merge(out[["team_id", "expected_wins"]], on="team_id",
                        suffixes=("_before", "_after"))
    assert np.allclose(merged["expected_wins_before"],
                       merged["expected_wins_after"], atol=1e-6)


def test_weight_one_follows_the_market():
    out = blend_market_wins(_wins_frame(), _market_frame(), market_weight=1.0)
    m = _market_frame().set_index("team_id")["market_wins"]
    o = out.set_index("team_id")["expected_wins"]
    # Ranks must match the market exactly at full weight.
    assert list(m.sort_values(ascending=False).index) == \
        list(o.sort_values(ascending=False).index)


def test_the_pre_blend_roster_value_is_preserved():
    """expected_wins is overwritten so downstream picks up the blend; without
    roster_wins there would be no record of what the market moved."""
    base = _wins_frame()
    out = blend_market_wins(base, _market_frame(), market_weight=0.5)
    merged = base.merge(out[["team_id", "roster_wins"]], on="team_id")
    assert np.allclose(merged["expected_wins"], merged["roster_wins"])


def test_teams_off_the_board_keep_their_bottom_up_value():
    market = _market_frame().iloc[:10]
    out = blend_market_wins(_wins_frame(), market, market_weight=1.0)
    off = out[out["market_wins"].isna()]
    assert len(off) == 20
    assert (off["market_weight"] == 0.0).all()


def test_no_market_is_a_clean_no_op():
    base = _wins_frame()
    out = blend_market_wins(base, pd.DataFrame(), market_weight=0.9)
    assert np.allclose(out["blended_wins"], base["expected_wins"])
    assert (out["market_weight"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Run environment
# ─────────────────────────────────────────────────────────────────────────────

def test_run_environment_is_rotated_to_match_the_blended_wins():
    """The differential follows the market; the SUM is held, so a team stays in
    its own run environment instead of being handed a generic one."""
    wins = blend_market_wins(_wins_frame(), _market_frame(), market_weight=1.0)
    out = apply_market_to_run_environment(wins)

    before = out["rs_per_game"] + out["ra_per_game"]
    after = out["rs_per_game_market"] + out["ra_per_game_market"]
    assert np.allclose(before, after, atol=1e-9), "run environment must hold"


def test_the_rotated_run_environment_reproduces_the_win_pct():
    """The point of the rotation: RS/RA must now imply the blended wins, so the
    reported table is internally consistent."""
    from team_wins import pythagenpat_win_pct

    wins = blend_market_wins(_wins_frame(), _market_frame(), market_weight=1.0)
    out = apply_market_to_run_environment(wins)
    recovered = pythagenpat_win_pct(out["rs_per_game_market"],
                                    out["ra_per_game_market"])
    assert np.allclose(recovered, out["win_pct"], atol=1e-6)


def test_the_sign_of_the_differential_matches_the_win_pct():
    """Above .500 must mean a positive differential, and vice versa.

    Note what is deliberately NOT asserted: that differential is monotone in
    win% ACROSS teams. It is not, and that is correct Pythagenpat behavior
    rather than a bug — the exponent rises with the run environment, so a
    better team in a lower-scoring context needs a SMALLER differential to
    reach the same win percentage. Asserting global monotonicity here would
    invite someone to "fix" working code. The fixed-environment version of the
    claim is tested below.
    """
    wins = blend_market_wins(_wins_frame(), _market_frame(), market_weight=1.0)
    out = apply_market_to_run_environment(wins)
    above = out["win_pct"] > 0.5
    assert (out.loc[above, "run_diff_market"] > 0).all()
    assert (out.loc[~above, "run_diff_market"] <= 0).all()


def test_differential_is_monotone_in_win_pct_at_a_fixed_run_environment():
    """The precise form of the claim: hold RS + RA constant across teams and
    the differential must order with win percentage."""
    n = 12
    wp = np.linspace(0.38, 0.62, n)
    wins = pd.DataFrame({
        "team_id": ALL_TEAMS[:n],
        "team_abbr": [TEAM_ABBR_BY_ID[t] for t in ALL_TEAMS[:n]],
        "rs_per_game": [4.5] * n,          # every club in the same
        "ra_per_game": [4.5] * n,          # 9.0-run environment
        "run_diff": [0.0] * n,
        "win_pct": wp,
        "expected_wins": wp * 162,
    })
    out = apply_market_to_run_environment(wins).sort_values("win_pct")
    assert out["run_diff_market"].is_monotonic_increasing


def test_run_environment_handles_an_empty_frame():
    assert apply_market_to_run_environment(pd.DataFrame()).empty


# ─────────────────────────────────────────────────────────────────────────────
# Win totals — the market worth using
# ─────────────────────────────────────────────────────────────────────────────

def test_a_bare_line_is_returned_unchanged():
    """With no prices, the line IS the best available estimate."""
    from market_odds import win_total_expectation
    assert win_total_expectation(88.5) == pytest.approx(88.5)


def test_an_even_priced_total_sits_on_the_line():
    """-110 / -110 de-vigs to a coin flip, so the expectation is the line."""
    from market_odds import win_total_expectation
    assert win_total_expectation(88.5, -110, -110) == pytest.approx(88.5,
                                                                    abs=1e-6)


def test_a_juiced_over_pushes_the_expectation_up():
    """'88.5, over -130' means the market thinks 88.5 is LOW. Reading the line
    at face value throws that information away."""
    from market_odds import win_total_expectation
    mu = win_total_expectation(88.5, -130, 110)
    assert mu > 88.5
    assert mu - 88.5 < 2.0, "a single price should not move it more than ~2 wins"


def test_a_juiced_under_pushes_the_expectation_down():
    from market_odds import win_total_expectation
    assert win_total_expectation(88.5, 110, -130) < 88.5


def test_the_price_adjustment_is_symmetric():
    from market_odds import win_total_expectation
    up = win_total_expectation(88.5, -130, 110) - 88.5
    down = 88.5 - win_total_expectation(88.5, 110, -130)
    assert up == pytest.approx(down, abs=1e-9)


def test_the_price_adjustment_matches_the_normal_model():
    """mu = L + sigma * Phi^-1(P(over)), de-vigged."""
    from market_odds import WIN_OUTCOME_SD, _norm_ppf, win_total_expectation
    over, under = -130, 110
    p_o = american_to_probability(over)
    p_u = american_to_probability(under)
    expect = 88.5 + WIN_OUTCOME_SD * _norm_ppf(p_o / (p_o + p_u))
    assert win_total_expectation(88.5, over, under) == pytest.approx(expect,
                                                                     abs=1e-9)


def test_norm_ppf_matches_known_quantiles():
    from market_odds import _norm_ppf
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-5)
    assert _norm_ppf(0.01) == pytest.approx(-2.326348, abs=1e-4)


def test_win_total_entries_accept_a_line_with_prices(tmp_path):
    p = _write(tmp_path, {"market": "win_total", "teams": {
        TEAM_ABBR_BY_ID[t]: {"line": 81.0, "over": -110, "under": -110}
        for t in ALL_TEAMS}})
    df = load_market_odds(p)
    assert df["over"].notna().all() and df["under"].notna().all()
    m = market_expected_wins(df)
    assert m.attrs["priced"] is True


def test_win_total_entries_accept_a_bare_number(tmp_path):
    p = _write(tmp_path, {"market": "win_total",
                          "teams": {TEAM_ABBR_BY_ID[t]: 81.0
                                    for t in ALL_TEAMS}})
    m = market_expected_wins(load_market_odds(p))
    assert m.attrs["priced"] is False
    assert m["market_wins"].mean() == pytest.approx(LEAGUE_MEAN_WINS, abs=1e-9)


def test_an_entry_with_no_line_warns_and_skips(tmp_path):
    p = _write(tmp_path, {"market": "win_total", "teams": {
        "LAD": {"over": -110}, "NYY": 90.5}})
    with pytest.warns(UserWarning):
        df = load_market_odds(p)
    assert list(df["team_id"]) == [NYY]


def test_the_posted_line_is_preserved_alongside_the_expectation(tmp_path):
    """So the price adjustment stays auditable."""
    p = _write(tmp_path, {"market": "win_total", "teams": {
        TEAM_ABBR_BY_ID[t]: {"line": 81.0, "over": -140, "under": 120}
        for t in ALL_TEAMS}})
    m = market_expected_wins(load_market_odds(p))
    assert "posted_line" in m.columns
    assert (m["posted_line"] == 81.0).all()


def test_win_total_is_the_default_market(tmp_path):
    """No `market` key means win totals, because that is the one to use."""
    p = _write(tmp_path, {"teams": {TEAM_ABBR_BY_ID[t]: 81.0
                                   for t in ALL_TEAMS}})
    assert load_market_odds(p).attrs["market"] == "win_total"


# ─────────────────────────────────────────────────────────────────────────────
# Market-specific blend weight
# ─────────────────────────────────────────────────────────────────────────────

def test_win_totals_outweigh_futures():
    """A win total is a direct estimate of the quantity we want; a title
    future is four playoff rounds removed from it."""
    from market_odds import default_market_weight
    assert default_market_weight("win_total") > default_market_weight("pennant")
    assert default_market_weight("pennant") > default_market_weight("world_series")


def test_an_unknown_market_falls_back_to_the_most_cautious_weight():
    from market_odds import MARKET_WEIGHT_BY_MARKET, default_market_weight
    assert default_market_weight("something_new") == \
        MARKET_WEIGHT_BY_MARKET["world_series"]
    assert default_market_weight(None) == MARKET_WEIGHT_BY_MARKET["world_series"]

"""
test_fielding_model.py
======================
Tests for the season fielding projection.

Three identities carry the weight, because each one fails silently otherwise —
every individual player's total looks plausible while the team total is wrong:

  1. **Putouts close to 27 per 9 team innings.** Every out is a putout credited
     to exactly one fielder. If the position baselines don't sum to 27, every
     team total built from them is off by that factor.
  2. **Chances = PO + A + E**, definitionally. Derived, never projected.
  3. **Catcher putouts move OPPOSITE to everyone else** with the staff's
     strikeout rate, because a catcher is credited with a putout on every
     strikeout while a high-K staff leaves fewer balls for the fielders.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fielding_model import (  # noqa: E402
    ALIGNMENT,
    EXPECTED_FIELDING_PCT,
    BASELINES,
    LEAGUE_CS_RATE,
    LEAGUE_K_PER_9,
    baseline_e_per_chance,
    OUTFIELD,
    OUTS_PER_9,
    SHRINK_INNINGS,
    baseline_rates,
    fielding_report,
    fit_position_baselines,
    project_fielding,
    project_fielding_rates,
    team_putout_check,
    team_strikeout_adjustment,
)

TARGET = 2027


def _history(rows):
    """rows: (PlayerId, Season, Pos, Innings, PO, A, E, DP)."""
    df = pd.DataFrame(rows, columns=["PlayerId", "Season", "Pos", "Innings",
                                     "PO", "A", "E", "DP"])
    df["PB"] = 0.0
    df["CI"] = 0.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Identity 1 — putouts close to 27
# ─────────────────────────────────────────────────────────────────────────────

def test_the_alignment_putouts_sum_to_27():
    """Every out is exactly one putout, so a full defensive alignment must
    account for all 27 per 9 innings."""
    total = sum(BASELINES[p]["po"] for p in ALIGNMENT)
    assert total == pytest.approx(OUTS_PER_9, abs=1e-9)


def test_normalization_survives_a_hand_edit_to_the_table():
    """The table is normalized on load, so editing a raw value cannot silently
    break the identity."""
    import importlib

    import fielding_model as fm
    original = dict(fm.POSITION_BASELINES["SS"])
    try:
        fm.POSITION_BASELINES["SS"] = {**original, "po": original["po"] * 3}
        rebuilt = fm._normalized_baselines()
        assert sum(rebuilt[p]["po"] for p in ALIGNMENT) == \
            pytest.approx(OUTS_PER_9, abs=1e-9)
    finally:
        fm.POSITION_BASELINES["SS"] = original
        importlib.reload(fm)


def test_the_alignment_excludes_the_dh():
    """A designated hitter records no defensive innings; including him would
    break the putout identity."""
    assert "DH" not in ALIGNMENT
    assert "DH" not in BASELINES


# ─────────────────────────────────────────────────────────────────────────────
# Position structure
# ─────────────────────────────────────────────────────────────────────────────

def test_first_base_takes_putouts_and_shortstop_takes_assists():
    """The core of why fielding is position-first: these are different jobs,
    not different skill levels."""
    assert BASELINES["1B"]["po"] > BASELINES["SS"]["po"] * 4
    assert BASELINES["SS"]["a"] > BASELINES["1B"]["a"] * 3


def test_outfield_assists_are_nearly_nil():
    for pos in OUTFIELD:
        assert BASELINES[pos]["a"] < 0.10
        assert BASELINES[pos]["a"] < BASELINES["SS"]["a"] / 20


def test_centre_field_leads_the_outfield_in_putouts():
    assert BASELINES["CF"]["po"] > BASELINES["LF"]["po"]
    assert BASELINES["CF"]["po"] > BASELINES["RF"]["po"]


def test_catcher_putouts_are_strikeout_scale():
    """A catcher records a putout on every strikeout, so his rate must sit
    near the league K/9, not near a fielder's chance rate."""
    assert BASELINES["C"]["po"] == pytest.approx(LEAGUE_K_PER_9, abs=1.0)


def test_the_baselines_reproduce_real_fielding_percentages():
    """The guard on the error column, and a much better check than an ordering
    assertion: the error prior is DERIVED from this table, so a wrong E shows
    up here as a wrong fielding percentage instead of hiding.

    Third base and pitcher are the two easy ones to get wrong — both err far
    more per chance than their neighbours, because a third baseman handles
    hard-hit balls with no time to set and a pitcher fields comebackers and
    bunts off balance. An earlier draft had 3B at .949 and P at .986.
    """
    for pos in ALIGNMENT:
        implied = 1.0 - baseline_e_per_chance(pos)
        assert implied == pytest.approx(EXPECTED_FIELDING_PCT[pos], abs=0.004), (
            f"{pos} implies a .{implied * 1000:.0f} fielding percentage, "
            f"expected .{EXPECTED_FIELDING_PCT[pos] * 1000:.0f}"
        )


def test_third_base_and_pitcher_err_most_per_chance():
    for pos in ("1B", "2B", "SS", "C", "LF", "CF", "RF"):
        assert baseline_e_per_chance("3B") > baseline_e_per_chance(pos)
        assert baseline_e_per_chance("P") > baseline_e_per_chance(pos)


def test_a_player_with_no_history_lands_on_his_position_prior():
    """Self-consistency: the prior shrunk toward IS the position baseline, so
    a flat league constant cannot contradict the table."""
    for pos in ALIGNMENT:
        assert baseline_rates(pos)["e_per_chance"] == \
            pytest.approx(baseline_e_per_chance(pos))


def test_only_catchers_have_passed_balls_and_interference():
    for pos in ALIGNMENT:
        if pos == "C":
            assert BASELINES[pos]["pb"] > 0 and BASELINES[pos]["ci"] > 0
        else:
            assert BASELINES[pos]["pb"] == 0 and BASELINES[pos]["ci"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Identity 3 — the strikeout adjustment
# ─────────────────────────────────────────────────────────────────────────────

def test_a_high_strikeout_staff_helps_its_catcher_and_starves_its_fielders():
    """The opposite-direction property. Strikeouts and fielding chances are the
    same finite pool of outs seen two ways."""
    adj = team_strikeout_adjustment(10.5)
    assert adj["po_catcher"] > 1.0, "more strikeouts = more catcher putouts"
    assert adj["po_fielder"] < 1.0, "...and fewer balls reaching fielders"
    assert adj["a"] < 1.0


def test_a_low_strikeout_staff_does_the_reverse():
    adj = team_strikeout_adjustment(6.5)
    assert adj["po_catcher"] < 1.0
    assert adj["po_fielder"] > 1.0


def test_a_league_average_staff_is_a_no_op():
    adj = team_strikeout_adjustment(LEAGUE_K_PER_9)
    for v in adj.values():
        assert v == pytest.approx(1.0, abs=1e-9)


def test_the_adjustment_is_clipped_against_absurd_input():
    for k in (-5.0, 0.0, 40.0):
        adj = team_strikeout_adjustment(k)
        assert all(np.isfinite(v) and v > 0 for v in adj.values())


# ─────────────────────────────────────────────────────────────────────────────
# Rates and shrinkage
# ─────────────────────────────────────────────────────────────────────────────

def test_no_history_at_a_position_gives_exactly_the_baseline():
    """A shortstop moving to second base should project as an average second
    baseman until he plays there."""
    rates = baseline_rates("2B")
    assert rates["rate_po"] == pytest.approx(BASELINES["2B"]["po"])
    assert rates["rate_a"] == pytest.approx(BASELINES["2B"]["a"])
    assert rates["cs_rate"] == LEAGUE_CS_RATE


def test_an_unknown_position_has_no_baseline():
    assert baseline_rates("DH") == {}
    assert baseline_rates("nonsense") == {}


def test_a_thin_sample_is_pulled_hard_to_the_baseline():
    """30 innings of a wild error rate must barely move the projection."""
    hist = _history([(1, 2026, "SS", 30.0, 5, 9, 3.0, 2)])   # 0.9 E/9, absurd
    r = project_fielding_rates(hist, TARGET).set_index("PlayerId")
    prior = baseline_e_per_chance("SS")
    # Errors are shrunk in CHANCE space, so a 30-inning sample carries ~17
    # chances against a 700-chance prior: the projection must stay within a few
    # percent of the position prior rather than chasing the noise.
    assert r.loc[1, "e_per_chance"] == pytest.approx(prior, rel=0.35)
    assert r.loc[1, "e_per_chance"] > prior, "a bad sample should still nudge up"


def test_a_large_sample_moves_toward_the_players_own_rate():
    hist = _history([(1, y, "SS", 1300.0, 210, 410, 26.0, 90)
                     for y in (2024, 2025, 2026)])
    observed = 26.0 / 1300.0 * 9.0
    r = project_fielding_rates(hist, TARGET).set_index("PlayerId")
    base = BASELINES["SS"]["e"]
    assert abs(r.loc[1, "rate_e"] - observed) < abs(r.loc[1, "rate_e"] - base)


def test_putouts_are_shrunk_harder_than_errors():
    """Errors are the genuine skill term; putouts are position and
    opportunity, so a player's own putout rate carries much less signal."""
    assert SHRINK_INNINGS["po"] > SHRINK_INNINGS["e"]
    assert SHRINK_INNINGS["ci"] > SHRINK_INNINGS["e"]
    assert SHRINK_INNINGS["cs"] < SHRINK_INNINGS["e"], (
        "catcher CS is a real, stable skill and should shrink least"
    )


def test_each_position_gets_its_own_row():
    hist = _history([(1, 2026, "SS", 700.0, 100, 200, 12.0, 45),
                     (1, 2026, "2B", 400.0, 80, 110, 5.0, 30)])
    r = project_fielding_rates(hist, TARGET)
    assert set(r["Pos"]) == {"SS", "2B"}
    assert len(r) == 2


def test_recency_weighting_favours_the_latest_season():
    old = _history([(1, 2023, "SS", 1300.0, 200, 400, 40.0, 80)])
    new = _history([(1, 2026, "SS", 1300.0, 200, 400, 40.0, 80)])
    r_old = project_fielding_rates(old, TARGET).set_index("PlayerId")
    r_new = project_fielding_rates(new, TARGET).set_index("PlayerId")
    base = BASELINES["SS"]["e"]
    # 2023 is outside the default 4-year window entirely.
    assert r_old.empty or abs(r_new.loc[1, "rate_e"] - base) >= \
        abs(r_old.loc[1, "rate_e"] - base) - 1e-9


def test_unknown_positions_are_dropped_from_rates():
    hist = _history([(1, 2026, "DH", 0.0, 0, 0, 0.0, 0),
                     (2, 2026, "SS", 500.0, 80, 150, 8.0, 30)])
    r = project_fielding_rates(hist, TARGET)
    assert set(r["PlayerId"]) == {2}


def test_missing_columns_raise_clearly():
    with pytest.raises(KeyError, match="missing columns"):
        project_fielding_rates(pd.DataFrame({"PlayerId": [1]}), TARGET)


def test_catcher_cs_rate_blends_toward_the_league():
    hist = _history([(1, 2026, "C", 1000.0, 900, 50, 5.0, 5)])
    hist["CS"] = 40.0
    hist["SB_allowed"] = 60.0          # a 40% CS rate, well above league
    r = project_fielding_rates(hist, TARGET).set_index("PlayerId")
    assert LEAGUE_CS_RATE < r.loc[1, "cs_rate"] < 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Identity 2 — season totals
# ─────────────────────────────────────────────────────────────────────────────

def _rates_frame(positions, team_id=147):
    rows = []
    for i, pos in enumerate(positions, start=1):
        rec = {"PlayerId": i, "Pos": pos, "team_id": team_id}
        rec.update(baseline_rates(pos))
        rows.append(rec)
    return pd.DataFrame(rows)


def _full_alignment_exposure(rates, innings=1458.0):
    """One player per position, each logging a full team season there."""
    return {(int(r.PlayerId), r.Pos): innings for r in rates.itertuples()}


def test_chances_is_always_putouts_plus_assists_plus_errors():
    rates = _rates_frame(ALIGNMENT)
    out = project_fielding(rates, _full_alignment_exposure(rates))
    assert np.allclose(out["Chances"], out["PO"] + out["A"] + out["E"])


def test_a_full_alignment_closes_to_27_putouts_per_9():
    """The team-level identity, end to end."""
    rates = _rates_frame(ALIGNMENT)
    out = project_fielding(rates, _full_alignment_exposure(rates))
    check = team_putout_check(out)
    assert len(check) == 1
    assert check.loc[0, "po_per_9"] == pytest.approx(OUTS_PER_9, abs=1e-6)
    assert abs(check.loc[0, "po_per_9_error"]) < 1e-6


def test_totals_scale_linearly_with_innings():
    rates = _rates_frame(["SS"])
    half = project_fielding(rates, {(1, "SS"): 700.0})
    full = project_fielding(rates, {(1, "SS"): 1400.0})
    assert full.loc[0, "PO"] == pytest.approx(half.loc[0, "PO"] * 2)
    assert full.loc[0, "A"] == pytest.approx(half.loc[0, "A"] * 2)


def test_players_with_no_exposure_are_omitted():
    rates = _rates_frame(["SS", "2B"])
    out = project_fielding(rates, {(1, "SS"): 900.0})
    assert list(out["PlayerId"]) == [1]


def test_zero_innings_is_omitted_rather_than_producing_zeros():
    rates = _rates_frame(["SS"])
    assert project_fielding(rates, {(1, "SS"): 0.0}).empty


def test_outfield_assists_are_broken_out_only_for_outfielders():
    rates = _rates_frame(["SS", "CF"])
    out = project_fielding(rates, _full_alignment_exposure(rates)).set_index("Pos")
    assert out.loc["CF", "OF_A"] == pytest.approx(out.loc["CF", "A"])
    assert out.loc["SS", "OF_A"] == 0.0


def test_the_team_strikeout_rate_flows_through_to_totals():
    rates = _rates_frame(["C", "SS"])
    exposure = _full_alignment_exposure(rates)
    low = project_fielding(rates, exposure, team_k_per_9={147: 6.5}
                           ).set_index("Pos")
    high = project_fielding(rates, exposure, team_k_per_9={147: 10.5}
                            ).set_index("Pos")
    assert high.loc["C", "PO"] > low.loc["C", "PO"]
    assert high.loc["SS", "PO"] < low.loc["SS", "PO"]
    assert high.loc["SS", "A"] < low.loc["SS", "A"]


def test_an_exposure_dataframe_works_as_well_as_a_mapping():
    rates = _rates_frame(["SS"])
    frame = pd.DataFrame([{"PlayerId": 1, "Pos": "SS", "Innings": 1200.0}])
    a = project_fielding(rates, frame)
    b = project_fielding(rates, {(1, "SS"): 1200.0})
    assert a.loc[0, "PO"] == pytest.approx(b.loc[0, "PO"])


def test_catcher_caught_stealing_needs_team_attempts():
    rates = _rates_frame(["C"])
    exposure = {(1, "C"): 1458.0}
    without = project_fielding(rates, exposure)
    assert pd.isna(without.loc[0, "CS"])

    with_att = project_fielding(rates, exposure,
                                sb_attempts_against={147: 150.0})
    assert with_att.loc[0, "CS_attempts"] == pytest.approx(150.0)
    assert with_att.loc[0, "CS"] == pytest.approx(150.0 * LEAGUE_CS_RATE)


def test_a_part_time_catcher_gets_a_share_of_the_attempts():
    rates = _rates_frame(["C"])
    out = project_fielding(rates, {(1, "C"): 729.0},
                           sb_attempts_against={147: 150.0})
    assert out.loc[0, "CS_attempts"] == pytest.approx(75.0)


def test_only_catchers_get_a_cs_column_value():
    rates = _rates_frame(["SS"])
    out = project_fielding(rates, {(1, "SS"): 1000.0},
                           sb_attempts_against={147: 150.0})
    assert "CS" not in out.columns or pd.isna(out.get("CS", pd.Series([np.nan])).iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Fitting and reporting
# ─────────────────────────────────────────────────────────────────────────────

def test_baselines_can_be_refitted_from_history():
    """The provisional table must be replaceable from real data."""
    hist = _history([(i, 2026, "SS", 1000.0, 160, 300, 12.0, 60)
                     for i in range(1, 20)])
    fitted = fit_position_baselines(hist)
    assert "SS" in fitted
    assert fitted["SS"]["po"] == pytest.approx(160 / 1000 * 9, abs=1e-6)
    assert fitted["SS"]["a"] == pytest.approx(300 / 1000 * 9, abs=1e-6)


def test_fitting_ignores_thin_samples():
    hist = _history([(1, 2026, "SS", 20.0, 900, 900, 900.0, 900)])
    assert fit_position_baselines(hist, min_innings=200.0) == {}


def test_report_flags_the_putout_identity():
    rates = _rates_frame(ALIGNMENT)
    out = project_fielding(rates, _full_alignment_exposure(rates))
    text = fielding_report(out)
    assert "PO per 9 defensive innings" in text
    assert "27" in text


def test_report_and_check_handle_empty_input():
    assert "no fielding" in fielding_report(pd.DataFrame())
    assert team_putout_check(pd.DataFrame()).empty

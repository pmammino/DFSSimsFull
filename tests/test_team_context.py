"""
test_team_context.py
====================
Tests for the season engine's team layer: canonical team identity, the shared
hitter/pitcher assignment rule, roster overrides for players changing teams,
and the bottom-up team run environment.

The load-bearing property is **league closure**: the volume-weighted mean team
factor must be exactly 1.0. Team factors scale every hitter's R/RBI, so if
they don't average to 1, moving players between teams silently creates or
destroys league runs. Several tests below pin that to machine precision at
various blend weights and after roster changes.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slate_config import canonical_team  # noqa: E402
from team_context import (  # noqa: E402
    TEAM_ABBR_BY_ID,
    TEAM_ID_BY_ABBR,
    abbr_for_team_id,
    apply_team_context,
    assign_target_teams,
    blend_team_factors,
    bottom_up_team_factors,
    career_pa_weights,
    depth_weights,
    load_team_overrides,
    team_id_for_abbr,
)

NYY, LAD, COL, CHC, CWS, NYM = 147, 119, 115, 112, 145, 121


# ─────────────────────────────────────────────────────────────────────────────
# Canonical team identity
# ─────────────────────────────────────────────────────────────────────────────

def test_all_thirty_franchises_are_distinct():
    """The bug this replaces collapsed 30 teams into 26 labels.

    `data_acquisition` used `team["abbreviation"] or team["name"][:3]`, so with
    no abbreviation the name slice merged Chicago's two clubs into "Chi", Los
    Angeles' into "Los", New York's into "New", and San Diego/San Francisco
    into "San".
    """
    assert len(TEAM_ABBR_BY_ID) == 30
    assert len(set(TEAM_ABBR_BY_ID.values())) == 30, "abbreviations collide"


@pytest.mark.parametrize("a,b", [
    (CHC, CWS),   # both "Chi"
    (LAD, 108),   # both "Los"
    (NYY, NYM),   # both "New"
    (135, 137),   # both "San"
])
def test_same_city_clubs_do_not_collide(a, b):
    assert abbr_for_team_id(a) != abbr_for_team_id(b)


def test_id_to_abbr_roundtrips():
    for team_id, abbr in TEAM_ABBR_BY_ID.items():
        assert team_id_for_abbr(abbr) == team_id
        assert TEAM_ID_BY_ABBR[abbr] == team_id


def test_every_abbr_is_canonical():
    """Ids must resolve to codes `canonical_team` already agrees with.

    The daily path keys same-name players by canonical code
    (matchup.sim_name), so a team id that produced a non-canonical string
    would reintroduce the collision drops this fix removes.
    """
    for abbr in TEAM_ABBR_BY_ID.values():
        assert canonical_team(abbr) == abbr, f"{abbr} is not canonical"


def test_unknown_and_missing_ids_degrade_quietly():
    """Minor-league and All-Star team ids appear in statsapi responses; they
    should mean "no team", not crash a pipeline run."""
    for bad in (None, np.nan, 999999, "", "nonsense"):
        assert abbr_for_team_id(bad) is None
    assert team_id_for_abbr(None) is None
    assert team_id_for_abbr("New") is None       # the old truncated label


def test_team_id_accepts_external_vocabularies():
    """Feed codes and full names must resolve, via slate_config's alias tables."""
    assert team_id_for_abbr("NY-A") == NYY       # Rotowire
    assert team_id_for_abbr("CHW") == CWS        # FantasyLabs
    assert team_id_for_abbr("New York Mets") == NYM
    assert team_id_for_abbr("ATH") == team_id_for_abbr("OAK")


# ─────────────────────────────────────────────────────────────────────────────
# Roster overrides
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, payload):
    p = tmp_path / "overrides.json"
    p.write_text(json.dumps(payload))
    return p


def test_missing_override_file_is_optional():
    assert load_team_overrides("/nonexistent/path/nope.json") == {}


def test_override_accepts_abbr_and_explicit_id(tmp_path):
    p = _write(tmp_path, {"assignments": [
        {"player_id": 1, "team": "SF"},
        {"player_id": 2, "team_id": NYY},
        {"player_id": 3, "team": "New York Mets"},
    ]})
    assert load_team_overrides(p) == {1: 137, 2: NYY, 3: NYM}


def test_null_team_means_no_team_not_last_years_club(tmp_path):
    """An unsigned or retired player must not silently keep his old team."""
    p = _write(tmp_path, {"assignments": [{"player_id": 7, "team": None}]})
    loaded = load_team_overrides(p)
    assert 7 in loaded and loaded[7] is None


def test_unresolvable_entries_warn_and_skip(tmp_path):
    p = _write(tmp_path, {"assignments": [
        {"player_id": 1, "team": "NOT_A_TEAM"},
        {"team": "SF"},                      # no player_id
        {"player_id": 3, "team_id": 999999},
        {"player_id": 4, "team": "COL"},     # the one good row
    ]})
    with pytest.warns(UserWarning):
        loaded = load_team_overrides(p)
    assert loaded == {4: COL}


def test_bare_list_payload_is_accepted(tmp_path):
    p = _write(tmp_path, [{"player_id": 5, "team": "COL"}])
    assert load_team_overrides(p) == {5: COL}


# ─────────────────────────────────────────────────────────────────────────────
# Shared assignment rule
# ─────────────────────────────────────────────────────────────────────────────

def _history(rows):
    return pd.DataFrame(rows, columns=["PlayerId", "Season", "TeamId", "PA"])


def test_uses_most_volume_in_the_most_recent_season():
    hist = _history([
        (1, 2025, NYY, 600),    # older season, ignored
        (1, 2026, LAD, 400),    # latest season, more PA here
        (1, 2026, COL, 100),
    ])
    out = assign_target_teams(hist, 2027).set_index("PlayerId")
    assert out.loc[1, "team_id"] == LAD
    assert out.loc[1, "team_abbr"] == "LAD"
    assert out.loc[1, "assign_source"] == "history"


def test_tie_break_is_deterministic_regardless_of_row_order():
    """The pitcher side previously used `groupby().last()` on a frame sorted
    only by (PlayerId, Season) — for a mid-season trade, which club won
    depended on input row order. Ties now break on the lower team id."""
    rows = [(1, 2026, NYY, 300), (1, 2026, LAD, 300)]
    first = assign_target_teams(_history(rows), 2027)
    second = assign_target_teams(_history(rows[::-1]), 2027)
    assert first.loc[0, "team_id"] == second.loc[0, "team_id"] == min(NYY, LAD)


def test_the_same_rule_serves_hitters_and_pitchers():
    """Only the volume column differs: PA for hitters, TBF for pitchers."""
    pit = pd.DataFrame(
        [(1, 2026, NYY, 500), (1, 2026, LAD, 40)],
        columns=["PlayerId", "Season", "TeamId", "TBF"],
    )
    out = assign_target_teams(pit, 2027, volume_col="TBF").set_index("PlayerId")
    assert out.loc[1, "team_id"] == NYY


def test_low_volume_seasons_are_ignored():
    """One relief appearance for a new club must not outrank a real season."""
    hist = _history([(1, 2025, NYY, 600), (1, 2026, LAD, 3)])
    out = assign_target_teams(hist, 2027, min_volume=25).set_index("PlayerId")
    assert out.loc[1, "team_id"] == NYY


def test_target_year_history_is_excluded():
    hist = _history([(1, 2026, NYY, 500), (1, 2027, LAD, 500)])
    out = assign_target_teams(hist, 2027).set_index("PlayerId")
    assert out.loc[1, "team_id"] == NYY


def test_player_with_no_qualifying_season_is_unknown():
    out = assign_target_teams(
        _history([(1, 2026, NYY, 5)]), 2027, min_volume=25,
    ).set_index("PlayerId")
    assert pd.isna(out.loc[1, "team_id"])
    assert out.loc[1, "assign_source"] == "unknown"


def test_override_beats_history_including_a_null():
    hist = _history([(1, 2026, NYY, 600), (2, 2026, LAD, 600)])
    out = assign_target_teams(
        hist, 2027, overrides={1: COL, 2: None},
    ).set_index("PlayerId")
    assert out.loc[1, "team_id"] == COL
    assert out.loc[1, "assign_source"] == "override"
    assert pd.isna(out.loc[2, "team_id"])
    assert out.loc[2, "assign_source"] == "override"


def test_missing_columns_raise_clearly():
    with pytest.raises(KeyError, match="missing columns"):
        assign_target_teams(pd.DataFrame({"PlayerId": [1]}), 2027)


# ─────────────────────────────────────────────────────────────────────────────
# Bottom-up team context
# ─────────────────────────────────────────────────────────────────────────────

def _hitters(spec, per_team=10):
    """Synthetic hitters: {team_id: mean_hr_rate}. Higher HR -> better offense.

    Rates VARY within a team, spanning 0.6x to 1.4x of the team mean, so that
    removing a team's best hitters actually lowers its aggregate. A fixture
    where every teammate is identical would make roster-change tests
    vacuously pass — only renormalization noise would move.
    """
    rows, pid = [], 1
    for team_id, hr_mean in spec.items():
        for i in range(per_team):
            spread = 0.6 + 0.8 * (i / max(per_team - 1, 1))
            hr = hr_mean * spread
            rows.append({
                "PlayerId": pid, "team_id": team_id,
                "P_K": 0.22, "P_BB": 0.08, "P_HBP": 0.01, "P_SF": 0.006,
                "P_HR": hr, "P_3B": 0.004, "P_2B": 0.046,
                "P_1B": 0.142, "P_BIPOut": 0.492 - hr,
                "Career_PA": 1000.0,
                "Pred_R_per_PA_neutral": 0.12,
                "Pred_RBI_per_PA_neutral": 0.11,
                "Pred_target_team_factor": 1.0,
                "P_R": 0.12, "P_RBI": 0.11, "SD_R": 0.01, "SD_RBI": 0.01,
            })
            pid += 1
    return pd.DataFrame(rows)


def _best_n(frame, team_id, n):
    """Index labels of a team's n highest-HR hitters."""
    return frame[frame.team_id == team_id].nlargest(n, "P_HR").index


def test_better_rosters_get_higher_factors():
    f = bottom_up_team_factors(
        _hitters({NYY: 0.050, LAD: 0.032, COL: 0.015}),
    ).set_index("team_id")
    assert f.loc[NYY, "bottom_up_factor"] > f.loc[LAD, "bottom_up_factor"]
    assert f.loc[LAD, "bottom_up_factor"] > f.loc[COL, "bottom_up_factor"]


def test_league_mean_factor_is_exactly_one():
    """The closure constraint. Must hold to machine precision."""
    h = _hitters({NYY: 0.050, LAD: 0.032, COL: 0.015})
    f = bottom_up_team_factors(h)
    mean = np.average(f["bottom_up_factor"], weights=f["team_weight"])
    assert mean == pytest.approx(1.0, abs=1e-12)


def test_closure_holds_with_uneven_roster_sizes():
    """An unweighted mean across clubs drifts when rosters differ in size —
    measured at +3% on the real artifacts. The normalization is volume-weighted
    precisely to prevent that."""
    big = _hitters({NYY: 0.050}, per_team=31)
    small = _hitters({COL: 0.015}, per_team=15)
    small["PlayerId"] += 10_000
    f = bottom_up_team_factors(pd.concat([big, small], ignore_index=True))
    mean = np.average(f["bottom_up_factor"], weights=f["team_weight"])
    assert mean == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("w", [0.0, 0.25, 0.6, 1.0])
def test_blending_never_shifts_league_runs(w):
    """Closure must hold at ANY blend weight, including a pure prior (0.0)."""
    h = _hitters({NYY: 0.050, LAD: 0.032, COL: 0.015})
    bu = bottom_up_team_factors(h)
    prior = {NYY: 1.30, LAD: 0.80, COL: 1.10}   # deliberately not mean-1
    blended = blend_team_factors(bu, prior, bottom_up_weight=w)
    mean = np.average(blended["team_factor"], weights=blended["team_weight"])
    assert mean == pytest.approx(1.0, abs=1e-12)


def test_blend_weight_one_ignores_the_prior():
    h = _hitters({NYY: 0.050, COL: 0.015})
    bu = bottom_up_team_factors(h)
    blended = blend_team_factors(bu, {NYY: 5.0, COL: 0.1}, bottom_up_weight=1.0)
    for _, r in blended.iterrows():
        assert r["team_factor"] == pytest.approx(r["bottom_up_factor"], abs=1e-12)


def test_teams_absent_from_the_prior_fall_back_to_the_roster():
    h = _hitters({NYY: 0.050, COL: 0.015})
    blended = blend_team_factors(
        bottom_up_team_factors(h), {NYY: 1.2}, bottom_up_weight=0.5,
    )
    assert blended["team_factor"].notna().all()


def test_empty_input_does_not_blow_up():
    empty = _hitters({}).assign(team_id=pd.Series(dtype=float))
    f = bottom_up_team_factors(empty)
    assert f.empty
    assert blend_team_factors(f).empty


# ─────────────────────────────────────────────────────────────────────────────
# Applying context / changing teams
# ─────────────────────────────────────────────────────────────────────────────

def test_r_and_rbi_are_rederived_from_the_neutral_column():
    """P_R must be recomputed from the team-context-FREE skill estimate, not
    rescaled from an already-scaled number."""
    h = _hitters({NYY: 0.050, COL: 0.015})
    f = blend_team_factors(bottom_up_team_factors(h), bottom_up_weight=1.0)
    out = apply_team_context(h, f)
    for _, r in out.iterrows():
        assert r["P_R"] == pytest.approx(
            r["Pred_R_per_PA_neutral"] * r["team_factor"], abs=1e-12)
        assert r["P_RBI"] == pytest.approx(
            r["Pred_RBI_per_PA_neutral"] * r["team_factor"], abs=1e-12)


def test_players_with_no_team_get_neutral_context():
    h = _hitters({NYY: 0.050})
    h.loc[h.index[0], "team_id"] = np.nan
    out = apply_team_context(
        h, blend_team_factors(bottom_up_team_factors(h), bottom_up_weight=1.0),
    )
    assert out.iloc[0]["team_factor"] == 1.0
    assert out.iloc[0]["P_R"] == pytest.approx(
        out.iloc[0]["Pred_R_per_PA_neutral"], abs=1e-12)


def test_a_team_change_moves_both_clubs_and_conserves_league_runs():
    """The core requirement: a player switching teams must raise the club he
    joins, lower the club he leaves, and leave league totals untouched."""
    spec = {NYY: 0.050, LAD: 0.032, COL: 0.015}
    before = _hitters(spec)

    def factors_for(frame):
        return blend_team_factors(
            bottom_up_team_factors(frame), bottom_up_weight=1.0,
        ).set_index("team_id")

    f_before = factors_for(before)

    # Move NYY's four best hitters to COL.
    after = before.copy()
    after.loc[_best_n(before, NYY, 4), "team_id"] = COL
    f_after = factors_for(after)

    assert f_after.loc[COL, "team_factor"] > f_before.loc[COL, "team_factor"]
    assert f_after.loc[NYY, "team_factor"] < f_before.loc[NYY, "team_factor"]

    # League closure survives the move — talent is redistributed, not created.
    for f in (f_before, f_after):
        mean = np.average(f["team_factor"], weights=f["team_weight"])
        assert mean == pytest.approx(1.0, abs=1e-12)


def test_unmoved_teammates_feel_a_roster_change():
    """A hitter who did not move still gets a new R/RBI when his lineup
    changes around him — that is the point of a roster-derived context."""
    spec = {NYY: 0.050, COL: 0.015}
    before = _hitters(spec)
    stayer = before.index[before.team_id == COL][0]

    after = before.copy()
    after.loc[_best_n(before, NYY, 5), "team_id"] = COL

    def apply(frame):
        f = blend_team_factors(bottom_up_team_factors(frame),
                               bottom_up_weight=1.0)
        return apply_team_context(frame, f)

    p_before = apply(before).loc[stayer, "P_R"]
    p_after = apply(after).loc[stayer, "P_R"]
    assert p_after > p_before, "adding good hitters must lift teammates' runs"


def test_missing_neutral_column_warns_rather_than_silently_passing_through():
    h = _hitters({NYY: 0.050}).drop(columns=["Pred_R_per_PA_neutral"])
    f = blend_team_factors(bottom_up_team_factors(h), bottom_up_weight=1.0)
    with pytest.warns(UserWarning, match="Pred_R_per_PA_neutral"):
        apply_team_context(h, f)


# ─────────────────────────────────────────────────────────────────────────────
# Playing-time weight sources
# ─────────────────────────────────────────────────────────────────────────────

def test_depth_weights_keep_only_the_top_n():
    df = pd.DataFrame({"Career_PA": [100, 900, 800, 700, 50]})
    w = depth_weights(df, top_n=3)
    assert (w[[1, 2, 3]] > 0).all()
    assert w[0] == 0 and w[4] == 0


def test_depth_weights_pass_through_short_rosters():
    df = pd.DataFrame({"Career_PA": [100, 200]})
    assert (depth_weights(df, top_n=9) > 0).all()


def test_weights_survive_a_missing_or_empty_column():
    assert (career_pa_weights(pd.DataFrame({"x": [1, 2]})) == 1).all()
    df = pd.DataFrame({"Career_PA": [np.nan, 0, -5]})
    assert (career_pa_weights(df) > 0).all(), "weights must stay positive"

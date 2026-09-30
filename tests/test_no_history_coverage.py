"""
test_no_history_coverage.py
===========================
Guards the promise that **every player gets a baseline** — including a debut
player with no MLB record at all, and a minor leaguer who may never reach the
majors.

This is a cross-cutting property, not one module's job, and it has been broken
in a different layer each time it was checked:

  * the rate models dropped anyone under 25 PA or outside a 2-year lookback
  * MLE skipped anyone who HAD MLB history, so thin-history players fell
    through both gates
  * MLE resolved players by NAME, losing every ambiguous or unmatched one —
    and a prospect's name is exactly the kind most likely to be either
  * the fielding model emitted rows only for players in the MLB fielding
    history, so every rookie and minor leaguer got no fielding projection
  * the hitter output silently dropped two team columns the pitcher output kept

Each test below pins one of those. A failure here means some class of player
has quietly vanished from the projection set while it still looks complete,
which is the failure mode that hides best.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fielding_model import (  # noqa: E402
    ALIGNMENT,
    baseline_e_per_chance,
    ensure_rate_coverage,
    project_fielding,
    project_fielding_rates,
)
from pipeline_config import (  # noqa: E402
    MLE_LEVELS,
    PT_FLOOR_IP,
    PT_FLOOR_PA,
    RATE_ACTIVE_LOOKBACK,
    RATE_MIN_PA_ACTIVE,
    SHRINK_K,
)
from playing_time import TIER_FLOOR, apply_playing_time, classify_tier  # noqa: E402
from rate_models import build_inference_panel  # noqa: E402
from team_context import assign_target_teams  # noqa: E402

TARGET = 2027
NYY, LAD = 147, 119


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — rates: does a thin-history player get projected at all?
# ─────────────────────────────────────────────────────────────────────────────

def _rate_history(rows):
    df = pd.DataFrame(rows, columns=["PlayerId", "Season", "TeamId", "PA"])
    for r in ("K%", "BB%", "HBP%", "SF%"):
        df[r] = 0.05
    df["Age"] = 24
    df["Name"] = "Test Player"
    df["Team"] = "NYY"
    return df


@pytest.mark.parametrize("pa,label", [
    (1, "a single MLB plate appearance"),
    (12, "a September callup"),
    (40, "a brief callup"),
])
def test_a_player_with_almost_no_mlb_history_is_still_projected(pa, label):
    panel = build_inference_panel(
        _rate_history([(1, TARGET - 1, NYY, pa)]), TARGET, "PA", SHRINK_K)
    assert 1 in set(panel["PlayerId"].astype(int)), f"dropped: {label}"


def test_a_player_last_seen_several_years_ago_is_still_projected():
    """A 4A player, or someone who lost two seasons to injury."""
    panel = build_inference_panel(
        _rate_history([(1, TARGET - 3, NYY, 40)]), TARGET, "PA", SHRINK_K)
    assert 1 in set(panel["PlayerId"].astype(int))


def test_the_coverage_thresholds_stay_permissive():
    assert RATE_MIN_PA_ACTIVE <= 1
    assert RATE_ACTIVE_LOOKBACK >= 4


def test_a_player_with_literally_no_prior_season_cannot_be_rate_projected():
    """The honest boundary, and the reason MLE exists.

    `build_inference_panel` needs at least one prior season — its
    `len(prior) == 0` gate. A true debut player has none, so the rate models
    CANNOT project him from MLB data, by construction. MLE is what closes this,
    by injecting a synthetic prior season from his minor-league line.
    """
    panel = build_inference_panel(
        _rate_history([(1, TARGET, NYY, 600)]), TARGET, "PA", SHRINK_K)
    # Note the empty frame comes back with no columns at all, so guard before
    # indexing — a caller that assumes PlayerId exists will raise here.
    projected = (set(panel["PlayerId"].astype(int))
                 if "PlayerId" in panel.columns else set())
    assert 1 not in projected, (
        "a player whose only season IS the target year has no prior to "
        "project from; MLE must supply one"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — MLE: the debut player's only route in
# ─────────────────────────────────────────────────────────────────────────────

def test_mle_covers_the_whole_level_ladder():
    """Organizational depth cannot stop at Double-A."""
    assert set(MLE_LEVELS) >= {"AAA", "AA", "A+", "A"}


def test_a_native_mlbam_id_bypasses_the_name_lookup():
    """Name matching is where MLE loses most of its coverage: an unmatched OR
    ambiguous name is dropped, and prospects' names are the likeliest to be
    either. A feed carrying `mlbam_id` must skip that path entirely."""
    from mle_translations import _resolve_mlbam

    # Empty name index: the name path cannot possibly resolve anything.
    assert _resolve_mlbam({"mlbam_id": 660271, "player": "Whoever"}, {}) == 660271
    assert _resolve_mlbam({"player": "Whoever"}, {}) is None


def test_an_ambiguous_name_still_resolves_when_an_id_is_present():
    from mle_translations import _resolve_mlbam

    ambiguous = {"john smith": [1, 2, 3]}
    assert _resolve_mlbam({"player": "John Smith"}, ambiguous) is None
    assert _resolve_mlbam({"player": "John Smith", "mlbam_id": 2},
                          ambiguous) == 2


def test_statsapi_covers_every_configured_level():
    """Every level MLE is configured for must have a sportId, or that level
    silently contributes nothing."""
    from data_acquisition import MINORS_SPORT_IDS

    for level in MLE_LEVELS:
        assert level in MINORS_SPORT_IDS, f"no statsapi sportId for {level}"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — playing time: present, but not claiming playing time
# ─────────────────────────────────────────────────────────────────────────────

def test_a_no_history_player_is_carried_at_the_floor_not_dropped():
    hist = _rate_history([(1, TARGET - 1, NYY, 600),    # regular
                          (2, TARGET - 1, LAD, 80)])    # MLE synthetic row
    tiers = classify_tier(hist, TARGET, volume_col="PA", mle_ids={2})
    out = apply_playing_time(
        pd.DataFrame({"PlayerId": [1, 2], "Career_PA": [4000.0, 80.0]}),
        tiers, role="hitter").set_index("PlayerId")

    assert out.loc[2, "pt_tier"] == TIER_FLOOR
    assert out.loc[2, "Proj_PA"] == PT_FLOOR_PA, (
        "a minor leaguer must be present at the floor, not absent and not zero"
    )


def test_the_floor_is_a_real_number_for_both_roles():
    """Zero would make every rate x volume product zero, so the player would
    vanish from totals while still occupying a row."""
    assert PT_FLOOR_PA == 1.0 and PT_FLOOR_IP == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — team context: an org for the player, no distortion for the team
# ─────────────────────────────────────────────────────────────────────────────

def test_a_thin_season_still_places_the_player_on_that_team():
    """Weak evidence beats none, because "none" costs park and team context.

    The original assertion here was that a 3-PA season yields NO team, "rather
    than a wrong one". The refreshed run showed what that costs: 1,950 of
    2,894 hitters with no Pred_target_team_id, hence no park factor and no
    team context, while their organization sat resolved in `Team` the whole
    time. A single 3-PA line with the Yankees is not a WRONG answer — it is
    the only answer available, and `low_volume` says so.
    """
    hist = pd.DataFrame([(1, TARGET - 1, NYY, 3)],
                        columns=["PlayerId", "Season", "TeamId", "PA"])
    out = assign_target_teams(hist, TARGET, min_volume=25).set_index("PlayerId")
    assert out.loc[1, "team_id"] == NYY
    assert out.loc[1, "assign_source"] == "low_volume"


def test_every_player_in_the_history_frame_gets_a_team_when_one_exists():
    """The property the last refresh violated, stated directly.

    Coverage of team identity must not depend on playing-time volume: an
    MLE-translated prospect carries one row with his parent org and a
    credibility-deflated sample that can land anywhere from 300 PA to 1.
    """
    rows = [(i, TARGET - 1, NYY, pa) for i, pa in
            enumerate([600.0, 310.0, 60.0, 24.0, 8.0, 1.0])]
    hist = pd.DataFrame(rows, columns=["PlayerId", "Season", "TeamId", "PA"])
    out = assign_target_teams(hist, TARGET, min_volume=25)
    assert len(out) == len(rows)
    assert out["team_id"].notna().all(), (
        f"unassigned: {out[out.team_id.isna()].to_dict('records')}")
    assert (out["team_id"] == NYY).all()


def test_an_mle_player_can_be_placed_on_his_parent_org():
    """The minors feed's parent-org field is what makes organizational
    alignment possible; MLE rows used to carry TeamId = NaN."""
    from mle_translations import _parent_org

    assert _parent_org({"currentTeam": "LAD"}) == ("LAD", 119.0)
    # An affiliate name does not resolve to an MLB club, and must not be
    # guessed at.
    abbr, tid = _parent_org({"team": "Oklahoma C"})
    assert np.isnan(tid)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — fielding: the gap this file was written after finding
# ─────────────────────────────────────────────────────────────────────────────

def _fielding_history():
    h = pd.DataFrame([(1, TARGET - 1, "SS", 1200.0, 180, 340, 14.0, 70)],
                     columns=["PlayerId", "Season", "Pos", "Innings",
                              "PO", "A", "E", "DP"])
    h["PB"] = 0.0
    h["CI"] = 0.0
    return h


def test_a_player_with_no_fielding_history_is_dropped_without_coverage():
    """Documents the bug `ensure_rate_coverage` exists to fix, so nobody
    removes that call thinking it is redundant."""
    rates = project_fielding_rates(_fielding_history(), TARGET)
    assert 2 not in set(rates["PlayerId"].astype(int))


def test_coverage_gives_every_rostered_player_a_fielding_rate():
    rates = project_fielding_rates(_fielding_history(), TARGET)
    roster = pd.DataFrame([(1, "SS", NYY), (2, "SS", NYY), (3, "2B", LAD)],
                          columns=["PlayerId", "Pos", "team_id"])
    full = ensure_rate_coverage(rates, roster)

    assert set(full["PlayerId"].astype(int)) == {1, 2, 3}
    sources = full.set_index(["PlayerId", "Pos"])["fielding_source"]
    assert sources[(1, "SS")] == "history"
    assert sources[(2, "SS")] == "baseline"
    assert sources[(3, "2B")] == "baseline"


def test_a_covered_player_produces_real_fielding_totals():
    """Coverage is pointless if the totals still come out empty."""
    rates = project_fielding_rates(_fielding_history(), TARGET)
    roster = pd.DataFrame([(2, "SS", NYY)], columns=["PlayerId", "Pos", "team_id"])
    full = ensure_rate_coverage(rates, roster)
    out = project_fielding(full, {(2, "SS"): 1200.0})
    assert len(out) == 1
    assert out.loc[0, "PO"] > 0 and out.loc[0, "A"] > 0
    assert out.loc[0, "Chances"] == pytest.approx(
        out.loc[0, "PO"] + out.loc[0, "A"] + out.loc[0, "E"])


def test_glove_skill_transfers_across_positions_but_the_job_does_not():
    """A sure-handed shortstop is a sure-handed second baseman, but he takes
    second base's putout and assist rates, not his old ones."""
    rates = project_fielding_rates(_fielding_history(), TARGET)
    roster = pd.DataFrame([(1, "2B", NYY)], columns=["PlayerId", "Pos", "team_id"])
    full = ensure_rate_coverage(rates, roster).set_index(["PlayerId", "Pos"])
    moved = full.loc[(1, "2B")]

    assert moved["fielding_source"] == "transfer"
    # Same skill RATIO to the position prior, not the same absolute rate —
    # 1B (.995) and 3B (.962) err at very different rates per chance.
    own = rates.set_index(["PlayerId", "Pos"]).loc[(1, "SS")]
    ratio_ss = own["e_per_chance"] / baseline_e_per_chance("SS")
    ratio_2b = moved["e_per_chance"] / baseline_e_per_chance("2B")
    assert ratio_2b == pytest.approx(ratio_ss, rel=1e-6)
    # ...but the job is second base's.
    from fielding_model import BASELINES
    assert moved["rate_a"] == pytest.approx(BASELINES["2B"]["a"])


def test_coverage_never_invents_a_defensive_position_for_a_dh():
    rates = project_fielding_rates(_fielding_history(), TARGET)
    roster = pd.DataFrame([(9, "DH", NYY)], columns=["PlayerId", "Pos", "team_id"])
    full = ensure_rate_coverage(rates, roster)
    assert 9 not in set(full["PlayerId"].astype(int))


def test_a_fully_covered_alignment_still_closes_to_27_putouts():
    """Coverage must not break the identity it sits behind."""
    from fielding_model import OUTS_PER_9, team_putout_check

    roster = pd.DataFrame([(i, pos, NYY) for i, pos in enumerate(ALIGNMENT, 1)],
                          columns=["PlayerId", "Pos", "team_id"])
    full = ensure_rate_coverage(pd.DataFrame(), roster)
    out = project_fielding(full, {(i, pos): 1458.0
                                  for i, pos in enumerate(ALIGNMENT, 1)})
    check = team_putout_check(out)
    assert check.loc[0, "po_per_9"] == pytest.approx(OUTS_PER_9, abs=1e-6)


def test_coverage_from_an_empty_history_works():
    """The realistic first-run case: no fielding history cached at all."""
    roster = pd.DataFrame([(1, "CF", NYY)], columns=["PlayerId", "Pos", "team_id"])
    full = ensure_rate_coverage(pd.DataFrame(), roster)
    assert len(full) == 1
    assert full.iloc[0]["fielding_source"] == "baseline"


def test_coverage_requires_the_roster_columns_it_needs():
    with pytest.raises(KeyError, match="roster missing columns"):
        ensure_rate_coverage(pd.DataFrame(), pd.DataFrame({"PlayerId": [1]}))


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 — the output actually carries them
# ─────────────────────────────────────────────────────────────────────────────

def test_both_sides_emit_the_same_team_identity_columns():
    """Hitters silently lost Pred_target_team_abbr and team_assign_source to an
    allowlist that pitchers were not subject to. Asymmetry between the two
    sides is the bug class here, so assert the contract rather than the
    contents."""
    import run_pipeline

    for col in ("Pred_target_team_id", "Pred_target_team_abbr",
                "team_assign_source"):
        assert col in run_pipeline.TEAM_ID_COLS


def test_playing_time_columns_are_declared_for_output():
    import run_pipeline

    for col in ("pt_tier", "pt_source", "Proj_PA", "Proj_IP"):
        assert col in run_pipeline.PLAYING_TIME_COLS

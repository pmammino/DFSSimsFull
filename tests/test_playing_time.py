"""
test_playing_time.py
====================
Tests for playing-time tiers, the 1 PA / 1 IP floor, and the roster-coverage
widening that lets a whole organization into the projection set.

Two invariants matter most:

  1. **Coverage.** Players with thin or stale MLB history — September callups,
     4A players, the long-term injured — must get a projection. They used to
     fall through both gates: `build_inference_panel` dropped them for having
     <25 PA or no recent season, and MLE skipped them for HAVING MLB history.
  2. **Containment.** Being in the output is not a claim of playing time. A
     floor-tier player carries exactly 1 PA / 1 IP and must not move a team or
     league aggregate.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_config import (  # noqa: E402
    MLE_HITTER_FACTORS,
    MLE_LEVELS,
    MLE_PA_CREDIBILITY,
    MLE_PITCHER_FACTORS,
    PT_FLOOR_IP,
    PT_FLOOR_PA,
    RATE_ACTIVE_LOOKBACK,
    RATE_MIN_PA_ACTIVE,
)
from playing_time import (  # noqa: E402
    SOURCE_FLOOR,
    SOURCE_UNMODELED,
    TIER_FLOOR,
    TIER_PROJECTED,
    apply_playing_time,
    classify_tier,
    playing_time_weights,
)
from rate_models import build_inference_panel  # noqa: E402
from pipeline_config import SHRINK_K  # noqa: E402

NYY, LAD = 147, 119


def _history():
    """The five archetypes a full-organization projection set has to cover."""
    rows = [
        (1, "Regular", 2025, NYY, 600), (1, "Regular", 2026, NYY, 620),
        (2, "Callup", 2026, NYY, 12),      # 12 MLB PA — under the old 25 gate
        (3, "Stuck4A", 2024, NYY, 40),     # last MLB 2024 — outside old lookback
        (4, "Injured", 2025, NYY, 500),    # missed all of 2026
        (5, "Prospect", 2026, LAD, 80),    # MLE synthetic row
    ]
    df = pd.DataFrame(rows, columns=["PlayerId", "Name", "Season", "TeamId", "PA"])
    for r in ("K%", "BB%", "HBP%", "SF%"):
        df[r] = 0.05
    df["Age"] = 27
    df["Team"] = "NYY"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Coverage
# ─────────────────────────────────────────────────────────────────────────────

def test_thin_and_stale_history_players_are_projected():
    """The regression this closes: these players were dropped by BOTH gates."""
    panel = build_inference_panel(_history(), 2027, "PA", SHRINK_K)
    got = set(panel["PlayerId"].astype(int))
    for pid, who in [(1, "MLB regular"), (2, "September callup"),
                     (3, "4A player"), (4, "long-term injured"),
                     (5, "MLE prospect")]:
        assert pid in got, f"{who} (id {pid}) has no projection"


def test_coverage_thresholds_admit_everyone_with_any_evidence():
    assert RATE_MIN_PA_ACTIVE <= 1, (
        "the active gate must admit a player with a single PA; anything higher "
        "silently drops September callups that MLE will not rescue"
    )
    assert RATE_ACTIVE_LOOKBACK >= 4, (
        "two lost seasons must not push an injured player out of the set"
    )


def test_mle_covers_the_full_level_ladder():
    """Organizational depth cannot stop at Double-A."""
    assert set(MLE_LEVELS) >= {"AAA", "AA", "A+", "A"}
    for level in MLE_LEVELS:
        assert level in MLE_HITTER_FACTORS, f"no hitter factors for {level}"
        assert level in MLE_PITCHER_FACTORS, f"no pitcher factors for {level}"
        assert level in MLE_PA_CREDIBILITY, f"no credibility for {level}"


def test_credibility_falls_monotonically_down_the_ladder():
    """A Single-A line must carry less evidence than a Triple-A line."""
    ladder = ["AAA", "AA", "A+", "A"]
    creds = [MLE_PA_CREDIBILITY[lvl] for lvl in ladder]
    assert creds == sorted(creds, reverse=True), (
        f"credibility must decrease with level: {dict(zip(ladder, creds))}"
    )
    assert all(0 < c < 1 for c in creds)


def test_strikeouts_rise_and_power_falls_down_the_ladder():
    """Sanity-check the direction of the translation factors."""
    ladder = ["AAA", "AA", "A+", "A"]
    ks = [MLE_HITTER_FACTORS[lvl]["K%"] for lvl in ladder]
    hrs = [MLE_HITTER_FACTORS[lvl]["HR"] for lvl in ladder]
    assert ks == sorted(ks), "K% penalty should grow at lower levels"
    assert hrs == sorted(hrs, reverse=True), "HR should regress more at lower levels"


# ─────────────────────────────────────────────────────────────────────────────
# Tier classification
# ─────────────────────────────────────────────────────────────────────────────

def test_tiers_split_on_real_mlb_evidence():
    tiers = classify_tier(_history(), 2027, volume_col="PA").set_index("PlayerId")
    assert tiers.loc[1, "pt_tier"] == TIER_PROJECTED   # 620 PA in 2026
    assert tiers.loc[4, "pt_tier"] == TIER_PROJECTED   # 500 PA in 2025
    assert tiers.loc[2, "pt_tier"] == TIER_FLOOR       # 12 PA
    assert tiers.loc[3, "pt_tier"] == TIER_FLOOR       # nothing in window


def test_mle_players_are_always_floor_tier():
    """A translated minor-league line is not MLB playing time, however much
    synthetic volume the injected row carries."""
    tiers = classify_tier(_history(), 2027, volume_col="PA",
                          mle_ids={5}).set_index("PlayerId")
    assert tiers.loc[5, "evidence_volume"] == 80.0, "clears the 25-PA bar"
    assert tiers.loc[5, "pt_tier"] == TIER_FLOOR, "but must still be floor"


def test_evidence_uses_the_best_season_in_the_window():
    hist = pd.DataFrame(
        [(1, 2025, NYY, 500), (1, 2026, NYY, 10)],
        columns=["PlayerId", "Season", "TeamId", "PA"],
    )
    tiers = classify_tier(hist, 2027, volume_col="PA").set_index("PlayerId")
    assert tiers.loc[1, "evidence_volume"] == 500.0
    assert tiers.loc[1, "evidence_season"] == 2025
    assert tiers.loc[1, "pt_tier"] == TIER_PROJECTED


def test_target_year_and_out_of_window_seasons_are_excluded():
    hist = pd.DataFrame(
        [(1, 2027, NYY, 600), (1, 2020, NYY, 600)],
        columns=["PlayerId", "Season", "TeamId", "PA"],
    )
    tiers = classify_tier(hist, 2027, volume_col="PA").set_index("PlayerId")
    assert tiers.loc[1, "pt_tier"] == TIER_FLOOR


def test_pitchers_classify_on_batters_faced():
    pit = pd.DataFrame(
        [(1, 2026, NYY, 700), (2, 2026, NYY, 8)],
        columns=["PlayerId", "Season", "TeamId", "TBF"],
    )
    tiers = classify_tier(pit, 2027, volume_col="TBF").set_index("PlayerId")
    assert tiers.loc[1, "pt_tier"] == TIER_PROJECTED
    assert tiers.loc[2, "pt_tier"] == TIER_FLOOR


def test_missing_columns_raise_clearly():
    with pytest.raises(KeyError, match="missing columns"):
        classify_tier(pd.DataFrame({"PlayerId": [1]}), 2027)


# ─────────────────────────────────────────────────────────────────────────────
# The floor
# ─────────────────────────────────────────────────────────────────────────────

def _applied(role="hitter", mle_ids=None):
    hist = _history()
    tiers = classify_tier(hist, 2027,
                          volume_col="PA" if role == "hitter" else "TBF",
                          mle_ids=mle_ids)
    proj = pd.DataFrame({"PlayerId": [1, 2, 3, 4, 5],
                         "Career_PA": [4000., 40., 40., 3000., 80.]})
    return apply_playing_time(proj, tiers, role=role).set_index("PlayerId")


def test_floor_players_get_exactly_one_pa():
    out = _applied(mle_ids={5})
    for pid in (2, 3, 5):
        assert out.loc[pid, "Proj_PA"] == PT_FLOOR_PA == 1.0
        assert out.loc[pid, "pt_source"] == SOURCE_FLOOR


def test_projected_players_get_nan_not_a_guess():
    """A plausible-looking 600 would make every total quietly wrong; a NaN
    fails loudly at the point of use."""
    out = _applied(mle_ids={5})
    for pid in (1, 4):
        assert np.isnan(out.loc[pid, "Proj_PA"])
        assert out.loc[pid, "pt_source"] == SOURCE_UNMODELED


def test_pitchers_get_an_ip_floor():
    hist = _history().rename(columns={"PA": "TBF"})
    tiers = classify_tier(hist, 2027, volume_col="TBF")
    out = apply_playing_time(pd.DataFrame({"PlayerId": [2]}), tiers,
                             role="pitcher").set_index("PlayerId")
    assert out.loc[2, "Proj_IP"] == PT_FLOOR_IP == 1.0
    assert "Proj_PA" not in out.columns


def test_players_absent_from_the_tier_frame_default_to_floor():
    """Unknown provenance is exactly when we must not assert playing time."""
    out = apply_playing_time(
        pd.DataFrame({"PlayerId": [999]}),
        pd.DataFrame(columns=["PlayerId", "pt_tier"]),
        role="hitter",
    ).set_index("PlayerId")
    assert out.loc[999, "pt_tier"] == TIER_FLOOR
    assert out.loc[999, "Proj_PA"] == PT_FLOOR_PA


def test_the_floor_is_one_not_zero():
    """Zero would make every rate-times-volume product zero, so the player
    would vanish from totals while still occupying a row."""
    assert PT_FLOOR_PA == 1.0 and PT_FLOOR_IP == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Containment — floor players must not move aggregates
# ─────────────────────────────────────────────────────────────────────────────

def test_weights_keep_the_floor_instead_of_falling_back_to_career():
    """The containment mechanism. A 4A player with 40 career PA and a 1-PA
    floor must weigh 1, not 40."""
    df = pd.DataFrame({"Proj_PA": [np.nan, 1.0], "Career_PA": [3000.0, 40.0]})
    w = playing_time_weights(df)
    assert w[0] == 3000.0, "unmodeled players fall back to career volume"
    assert w[1] == 1.0, "an explicit floor must survive the fallback"


def test_weights_fall_back_when_playing_time_is_absent():
    df = pd.DataFrame({"Career_PA": [100.0, 900.0]})
    assert list(playing_time_weights(df)) == [100.0, 900.0]
    assert (playing_time_weights(pd.DataFrame({"x": [1, 2]})) == 1).all()


def test_depth_players_cannot_move_a_team_factor():
    """End-to-end containment: adding 200 organizational-depth hitters to one
    club must not shift its run environment."""
    from team_context import blend_team_factors, bottom_up_team_factors

    def roster(team_id, n, hr, tier, start_pid):
        """A roster slice as run_pipeline emits it: pt_tier alongside Proj_PA."""
        floor = tier == TIER_FLOOR
        return pd.DataFrame([{
            "PlayerId": start_pid + i, "team_id": team_id,
            "P_K": 0.22, "P_BB": 0.08, "P_HBP": 0.01, "P_SF": 0.006,
            "P_HR": hr, "P_3B": 0.004, "P_2B": 0.046, "P_1B": 0.142,
            "P_BIPOut": 0.492 - hr,
            "pt_tier": tier,
            "Proj_PA": PT_FLOOR_PA if floor else np.nan,
            "Career_PA": 30.0 if floor else 3000.0,
        } for i in range(n)])

    mlb = pd.concat([roster(NYY, 12, 0.045, TIER_PROJECTED, 1),
                     roster(LAD, 12, 0.025, TIER_PROJECTED, 100)],
                    ignore_index=True)
    base = blend_team_factors(bottom_up_team_factors(mlb),
                              bottom_up_weight=1.0).set_index("team_id")

    # 200 Single-A hitters with wildly different rates, all at the 1-PA floor.
    depth = roster(NYY, 200, 0.001, TIER_FLOOR, 1000)
    with_depth = blend_team_factors(
        bottom_up_team_factors(pd.concat([mlb, depth], ignore_index=True)),
        bottom_up_weight=1.0,
    ).set_index("team_id")

    assert with_depth.loc[NYY, "team_factor"] == pytest.approx(
        base.loc[NYY, "team_factor"], abs=1e-6), (
        "organizational depth changed the MLB team's run environment"
    )

"""
test_season_engine.py
=====================
Integration tests for the season layer: team resolution off a projection
frame, the legacy-column fallback, and an end-to-end run against whatever
per-PA CSVs are in `out/`.

The end-to-end tests skip when `out/` is empty, so a fresh clone without
generated projections still has a green suite.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import season_engine as se  # noqa: E402
from team_context import career_pa_weights, roster_volume_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TARGET = 2027

NYY, LAD, COL = 147, 119, 115

_have_projections = (
    (OUT / f"hitter_pa_projections_{TARGET}.csv").exists()
    and (OUT / f"pitcher_pa_projections_{TARGET}.csv").exists()
)
needs_projections = pytest.mark.skipif(
    not _have_projections,
    reason=f"no per-PA projections in {OUT}; run run_pipeline.py first",
)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_teams
# ─────────────────────────────────────────────────────────────────────────────

def _proj(team_col="Pred_target_team_id"):
    return pd.DataFrame({
        "PlayerId": [1, 2, 3],
        "Name": ["A", "B", "C"],
        team_col: [NYY, LAD, np.nan],
    })


def test_resolves_from_the_pipeline_team_id():
    out = se.resolve_teams(_proj(), TARGET)
    assert out["team_id"].tolist()[:2] == [NYY, LAD]
    assert out["team_abbr"].tolist()[:2] == ["NYY", "LAD"]
    assert out["assign_source"].tolist() == ["history", "history", "unknown"]
    assert pd.isna(out["team_id"].iloc[2])


def test_overrides_win_over_the_pipeline_assignment():
    out = se.resolve_teams(_proj(), TARGET, overrides={1: COL}).set_index("PlayerId")
    assert out.loc[1, "team_id"] == COL
    assert out.loc[1, "team_abbr"] == "COL"
    assert out.loc[1, "assign_source"] == "override"
    assert out.loc[2, "assign_source"] == "history"


def test_an_override_can_place_a_player_with_no_prior_team():
    out = se.resolve_teams(_proj(), TARGET, overrides={3: COL}).set_index("PlayerId")
    assert out.loc[3, "team_id"] == COL
    assert out.loc[3, "assign_source"] == "override"


@pytest.mark.parametrize("legacy", ["Pred_home_team_id", "home_park_team_id"])
def test_legacy_projection_files_still_resolve(legacy):
    """CSVs generated before team ids were unified carry only a park team id.

    They must still work — a user with an old `out/` should get a warning-level
    note, not a crash.
    """
    out = se.resolve_teams(_proj(team_col=legacy), TARGET)
    assert out["team_id"].tolist()[:2] == [NYY, LAD]


def test_a_frame_with_no_team_id_at_all_raises_with_a_fix():
    bare = pd.DataFrame({"PlayerId": [1], "Name": ["A"]})
    with pytest.raises(KeyError, match="run_pipeline"):
        se.resolve_teams(bare, TARGET)


def test_history_path_uses_the_shared_assignment_rule():
    hist = pd.DataFrame(
        [(1, 2026, NYY, 500), (1, 2026, LAD, 100)],
        columns=["PlayerId", "Season", "TeamId", "PA"],
    )
    out = se.resolve_teams(
        pd.DataFrame({"PlayerId": [1], "Name": ["A"]}), TARGET, history=hist,
    ).set_index("PlayerId")
    assert out.loc[1, "team_id"] == NYY


# ─────────────────────────────────────────────────────────────────────────────
# End to end
# ─────────────────────────────────────────────────────────────────────────────

@needs_projections
def test_end_to_end_run_covers_all_thirty_teams():
    res = se.run(TARGET, OUT, write=False)
    hitters, teams = res["hitters"], res["teams"]
    assert len(teams) == 30, "every franchise should carry a team context"
    assert teams["team_abbr"].nunique() == 30, "team labels must be distinct"
    assert hitters["team_id"].notna().all()


@needs_projections
def test_end_to_end_league_factor_closes_to_one():
    """The closure constraint, on real data rather than a fixture.

    Weighted by `roster_volume_weights`, which is what the engine normalizes
    against, so the closure is exact by construction. This used to use
    `career_pa_weights` and still passed at 1e-9 — but only because the
    committed artifacts predated the playing-time step and carried no
    `pt_tier`, which makes the two functions the same function. On a frame
    that has tiers they differ by design, and `team_context` says so: a
    floor-tier player RECEIVES a team factor, so his R/RBI are
    contextualized, but he must not CONSUME a share of the 6,156 plate
    appearances his club actually bats.
    """
    hitters = se.run(TARGET, OUT, write=False)["hitters"]
    mean = np.average(hitters["team_factor"],
                      weights=roster_volume_weights(hitters))
    assert mean == pytest.approx(1.0, abs=1e-9)


@needs_projections
def test_the_floor_tier_cannot_move_league_runs():
    """The property the exact check above is a proxy for.

    Organizational depth is in the frame at the 1-PA floor — 2,179 hitters
    across 30 clubs on the current artifacts, about 73 per club against a
    6,156-PA budget. They are carried so they are present, ranked and
    joinable. If they could shift the league's run total, that would be
    plate appearances conjured out of roster bookkeeping.

    So this is deliberately loose: it asserts the floor tier is NEGLIGIBLE,
    not that it is absent. Counting every one of them at a full share is
    what the tight assertion above would be doing if it used
    `career_pa_weights`.
    """
    hitters = se.run(TARGET, OUT, write=False)["hitters"]
    # Guard: without tiers the two weightings are the SAME function, and
    # both of these tests degenerate into the same trivial check. That is
    # how the old assertion stayed green on artifacts that predated the
    # playing-time step while being wrong about what it measured.
    assert "pt_tier" in hitters.columns, (
        "the committed artifacts predate the playing-time step; this test "
        "and the closure test above are not exercising what they claim")
    assert (hitters["pt_tier"].astype(str) == "floor").any()

    with_floor = np.average(hitters["team_factor"],
                            weights=career_pa_weights(hitters))
    assert with_floor == pytest.approx(1.0, abs=1e-3)


@needs_projections
def test_end_to_end_r_and_rbi_track_the_applied_factor():
    hitters = se.run(TARGET, OUT, write=False)["hitters"]
    expect_r = hitters["Pred_R_per_PA_neutral"] * hitters["team_factor"]
    assert np.allclose(hitters["P_R"], expect_r, atol=1e-9, equal_nan=True)


@needs_projections
def test_a_real_team_change_moves_both_clubs(tmp_path):
    """Move the two highest-volume hitters to Colorado and confirm the club
    they join rises, the clubs they leave fall, and the league still closes."""
    base = se.run(TARGET, OUT, write=False)
    hitters = base["hitters"]
    movers = hitters.nlargest(2, "Career_PA")
    from_teams = {int(t) for t in movers["team_id"]}
    assert COL not in from_teams, "fixture assumes the movers are not Rockies"

    path = tmp_path / "moves.json"
    path.write_text(json.dumps({"target_year": TARGET, "assignments": [
        {"player_id": int(p), "team": "COL"} for p in movers["PlayerId"]
    ]}))
    after = se.run(TARGET, OUT, override_path=path, write=False)

    b = base["teams"].set_index("team_id")["team_factor"]
    a = after["teams"].set_index("team_id")["team_factor"]
    assert a[COL] > b[COL], "Colorado should improve after adding stars"
    for t in from_teams:
        assert a[t] < b[t], f"team {t} should decline after losing a star"

    # Roster counts follow the move.
    nb = base["teams"].set_index("team_id")["n_hitters"]
    na = after["teams"].set_index("team_id")["n_hitters"]
    assert na[COL] == nb[COL] + len(movers)

    # And league runs are conserved — under the weighting the engine closes
    # against, so this is exact rather than approximately true.
    ah = after["hitters"]
    assert np.average(ah["team_factor"],
                      weights=roster_volume_weights(ah)) == pytest.approx(1.0, abs=1e-9)


@needs_projections
def test_moved_players_are_flagged_as_overrides(tmp_path):
    hitters = se.run(TARGET, OUT, write=False)["hitters"]
    pid = int(hitters["PlayerId"].iloc[0])
    path = tmp_path / "moves.json"
    path.write_text(json.dumps(
        {"assignments": [{"player_id": pid, "team": "COL"}]}))

    after = se.run(TARGET, OUT, override_path=path,
                   write=False)["hitters"].set_index("PlayerId")
    assert after.loc[pid, "assign_source"] == "override"
    assert after.loc[pid, "team_abbr"] == "COL"


@needs_projections
def test_writes_the_season_directory(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    for side in ("hitter", "pitcher"):
        src = OUT / f"{side}_pa_projections_{TARGET}.csv"
        (dest / src.name).write_bytes(src.read_bytes())

    se.run(TARGET, dest, write=True)
    season = dest / f"season_{TARGET}"
    for name in ("hitters.csv", "pitchers.csv", "team_context.csv"):
        assert (season / name).exists(), f"{name} was not written"
    teams = pd.read_csv(season / "team_context.csv")
    assert len(teams) == 30


@needs_projections
def test_missing_projections_give_an_actionable_error(tmp_path):
    with pytest.raises(SystemExit, match="run_pipeline"):
        se.run(TARGET, tmp_path, write=False)


# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE_RBI_PER_RUN — a reconciliation target that was wrong
# ─────────────────────────────────────────────────────────────────────────────

def test_league_rbi_per_run_matches_reality():
    """The target read 0.88 through the first real season-layer run.

    Only runs nobody hit in carry no RBI: errors, wild pitches, passed balls,
    balks, steals of home, some fielder's-choice plays. The batter who homers
    drives in himself; a sac fly and a bases-loaded walk both count. That is
    ~5% of runs, not ~12%.

    Measured on run 36762891387 across every pool and both weightings:
    0.9492 / 0.9540 / 0.9607 / 0.9613. A report is read exactly when someone
    is deciding whether the model is healthy, and 0.88 turned a +2.5% gap into
    an apparent +11% one.
    """
    assert 0.93 <= se.LEAGUE_RBI_PER_RUN <= 0.97, (
        f"LEAGUE_RBI_PER_RUN = {se.LEAGUE_RBI_PER_RUN}; real league RBI/R is "
        "~0.95. Re-derive as the ratio of the league RBI and R rates.")


def test_rbi_target_is_not_the_old_wrong_value():
    """Explicitly pin the regression, so a revert is loud."""
    assert se.LEAGUE_RBI_PER_RUN != 0.88


def test_reconciliation_reports_the_rbi_gap_against_the_constant():
    """The printed deviation must be measured against the constant, not a
    literal, so correcting one corrects the other."""
    import inspect
    src = inspect.getsource(se.reconciliation_report)
    assert "LEAGUE_RBI_PER_RUN" in src, (
        "the RBI target must come from the constant, not a hardcoded string")
    assert "0.88" not in src, "the old literal is still in the report"

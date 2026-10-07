"""The playing-time shape, pinned against real innings.

`out/fielding_history_<year>.csv` carries five seasons of real (player,
season, position) rows, which is a measurement of how a club's playing time
is actually distributed. The anchors and the two shape knobs are now fitted
against it by `scripts/fit_role_anchors.py` rather than chosen.

These tests hold the fit in place. They are skipped without the fielding
file, so a fresh clone with no generated artifacts still runs green.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import playing_time_model as M  # noqa: E402
import role_taxonomy as RT  # noqa: E402

FIELDING = ROOT / "out" / "fielding_history_2027.csv"
PITCHERS = ROOT / "out" / "pitcher_pa_projections_2027.csv"
needs_data = pytest.mark.skipif(
    not (FIELDING.exists() and PITCHERS.exists()),
    reason="needs out/fielding_history_*.csv and the per-PA projections")


def _real_ip():
    f = pd.read_csv(FIELDING, low_memory=False)
    p = f[(f["Pos"] == "P") & (f["Season"] >= 2024)]
    g = (p.groupby(["Season", "TeamId", "PlayerId"])["Innings"].sum()
          .reset_index())
    g["rank"] = g.groupby(["Season", "TeamId"])["Innings"].rank(
        "first", ascending=False)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# The constants the fit produced
# ─────────────────────────────────────────────────────────────────────────────

def test_the_two_sides_decay_at_different_rates():
    """One shared constant left the pitching staff far too flat.

    A club was spreading its innings over 32 arms where a real club uses
    29, and the surplus came off the top of the rotation.

    The hitter value moved from 0.78 to 0.75 when the prospect arrivals went
    in, and that is not drift: the decay says how fast a club's playing time
    runs out past its core, so it is fitted against whoever is competing for
    it, and arrivals put one more claimant a club into the ranking. It is
    pinned rather than asserted loosely because it is a FITTED number — if
    it moves again, something changed about the population and the refit
    should be deliberate.
    """
    assert M.ROSTER_DEPTH_DECAY_PITCHER < M.ROSTER_DEPTH_DECAY_HITTER
    # Both move with ROSTER_DEPTH_FLOOR, which decides where the taper ends
    # while they decide how fast it gets there. Fitting either alone finds a
    # compromise that is wrong for both — see the note on the floor.
    assert M.ROSTER_DEPTH_DECAY_PITCHER == pytest.approx(0.70)
    assert M.ROSTER_DEPTH_DECAY_HITTER == pytest.approx(0.75)


def test_the_innings_ceiling_is_not_above_anything_that_has_happened():
    """230 was slack, and the closure spent it.

    The real league leader threw 208.7, 207.0 and 214.0 innings in 2024-26.
    A ceiling 7% above the best of those is not a physical bound.
    """
    real = _real_ip() if FIELDING.exists() else None
    assert M.PT_MAX_IP == pytest.approx(215.0)
    if real is not None:
        assert M.PT_MAX_IP >= real["Innings"].max(), \
            "the ceiling must still admit the best real season in the sample"


def test_the_evidence_band_is_narrower_than_the_spread_it_used_to_allow():
    """Real #1 starters cluster: p25 171, median 179, p75 188."""
    assert M.EVIDENCE_FACTOR_MIN == pytest.approx(0.55)
    assert M.EVIDENCE_FACTOR_MAX == pytest.approx(1.22)
    assert M.EVIDENCE_FACTOR_MAX - M.EVIDENCE_FACTOR_MIN < 0.85, \
        "0.45-1.30 let team talent move a rotation slot more than health does"


def test_the_end_of_rotation_anchor_carries_the_fitted_value():
    row = RT.role_anchor("End-of-Rotation Starter (SP4-5)", "pitcher")
    assert row is not None and row["ip"] == pytest.approx(110)
    ace = RT.role_anchor("Ace (SP1)", "pitcher")
    mid = RT.role_anchor("Mid-Rotation Starter (SP2-3)", "pitcher")
    # The rotation must stay ordered. An unconstrained per-role fit put SP2
    # above SP1, which is why the anchors are not refitted wholesale.
    assert ace["ip"] > mid["ip"] > row["ip"]


# ─────────────────────────────────────────────────────────────────────────────
# What the fit is for
# ─────────────────────────────────────────────────────────────────────────────

@needs_data
def test_the_projected_staff_matches_the_real_one_at_the_top():
    from fit_role_anchors import _projected, real_rank_curve
    from team_context import TEAM_OVERRIDE_PATH, load_roster_reserves

    fielding = pd.read_csv(FIELDING, low_memory=False)
    players = pd.read_csv(PITCHERS, low_memory=False)
    out, vol = _projected(players, "pitcher", fielding,
                          load_roster_reserves(TEAM_OVERRIDE_PATH(2027)))
    curve, _ = real_rank_curve(fielding, "pitcher")

    real = _real_ip()
    ace_real = real[real["rank"] == 1]["Innings"].median()
    ace_proj = out[out["rank"] == 1][vol].median()
    assert ace_proj == pytest.approx(ace_real, rel=0.05), (
        f"ace median {ace_proj:.1f} against a real {ace_real:.1f}")

    # Nobody above the best real season in the sample.
    assert out[vol].max() <= real["Innings"].max() + 1.5

    # And the right NUMBER of workhorses, which is the test the old
    # configuration failed worst: 10 against a real 17.
    n_real = (real["Innings"] > 180).sum() / real["Season"].nunique()
    n_proj = (out[vol] > 180).sum()
    assert abs(n_proj - n_real) <= 5, f"{n_proj} over 180 IP vs a real {n_real:.0f}"

    # The front four of the rotation, where most innings live.
    for r in (1, 2, 3, 4):
        ratio = out[out["rank"] == r][vol].mean() / curve.loc[float(r)]
        assert 0.92 <= ratio <= 1.08, f"rank {r} at {ratio:.2f}x real"


@needs_data
def test_the_hitter_curve_was_already_right_and_stays_right():
    """The sweep moved it half a point with the top getting worse."""
    from fit_role_anchors import _projected, real_rank_curve
    from team_context import TEAM_OVERRIDE_PATH, load_roster_reserves

    fielding = pd.read_csv(FIELDING, low_memory=False)
    players = pd.read_csv(ROOT / "out" / "hitter_pa_projections_2027.csv",
                          low_memory=False)
    out, vol = _projected(players, "hitter", fielding,
                          load_roster_reserves(TEAM_OVERRIDE_PATH(2027)))
    curve, _ = real_rank_curve(fielding, "hitter")
    for r in (1, 2, 3, 4, 5):
        ratio = out[out["rank"] == r][vol].mean() / curve.loc[float(r)]
        assert 0.92 <= ratio <= 1.08, f"rank {r} at {ratio:.2f}x real"


@needs_data
def test_the_games_equivalent_proxy_measures_what_it_claims():
    """The hitter curve rests on it, so it gets its own check.

    Innings at a fielding position over nine, plus games at DH, summed over
    a club, must come to the 1,458 lineup-slot games that 162 games of nine
    slots produce. It comes to 1,459.
    """
    from fit_role_anchors import real_rank_curve
    fielding = pd.read_csv(FIELDING, low_memory=False)
    _, per_club = real_rank_curve(fielding, "hitter")
    assert per_club == pytest.approx(162 * 9, rel=0.01)


@needs_data
def test_the_real_innings_budget_is_below_the_one_in_use():
    """Measured, and deliberately not acted on alone.

    A club's pitchers throw 1,436 innings, not the 162 x 9 = 1,458 the
    budget assumes, because the home team does not pitch the bottom of the
    ninth when it is ahead. Cutting the budget on its own would make the
    PA-versus-batters-faced identity WORSE, because TBF_PER_IP_CALIBRATION
    is low by about the same amount and the two currently cancel. Fixing
    one of three mutually inconsistent constants is not a fix.
    """
    from fit_role_anchors import real_rank_curve
    fielding = pd.read_csv(FIELDING, low_memory=False)
    _, per_club = real_rank_curve(fielding, "pitcher")
    assert per_club < M.TEAM_IP_BUDGET
    assert per_club == pytest.approx(1436, rel=0.01)


@needs_data
def test_nobody_starts_more_games_than_a_rotation_turn_allows():
    """37.4 starts was on the face of the spreadsheet.

    A five-man rotation turns over 32 or 33 times in 162 games. The real
    maxima in 2024-26 were 33, 34 and 34, with nobody reaching 35.
    """
    f = pd.read_csv(FIELDING, low_memory=False)
    p = f[(f["Pos"] == "P") & (f["Season"] >= 2024)]
    real_max = p.groupby(["Season", "PlayerId"])["GS"].sum().max()
    assert M.PT_MAX_GS == pytest.approx(34.0)
    assert M.PT_MAX_GS >= real_max

    from fit_role_anchors import _projected
    from team_context import TEAM_OVERRIDE_PATH, load_roster_reserves
    players = pd.read_csv(PITCHERS, low_memory=False)
    out, _ = _projected(players, "pitcher", f,
                        load_roster_reserves(TEAM_OVERRIDE_PATH(2027)))
    assert out["Proj_GS"].max() <= M.PT_MAX_GS + 1e-6
    assert out["Proj_G"].max() <= M.PT_MAX_G + 1e-6


@needs_data
def test_a_club_starts_exactly_the_games_it_plays():
    """162 games, 162 starting pitchers. The hardest budget of the three,
    and the only one that was not being enforced.

    Unclosed, the anchors simply added up to more than a season — one ace,
    two mid-rotation and two end-of-rotation come to 168 starts before
    anybody is scaled — and the staff started 178.5 games a club, 5,354
    across a league that has 4,859.
    """
    from fit_role_anchors import _projected
    from team_context import TEAM_OVERRIDE_PATH, load_roster_reserves
    f = pd.read_csv(FIELDING, low_memory=False)
    players = pd.read_csv(PITCHERS, low_memory=False)
    out, _ = _projected(players, "pitcher", f,
                        load_roster_reserves(TEAM_OVERRIDE_PATH(2027)))
    per_club = out.groupby("Pred_target_team_id")["Proj_GS"].sum()
    assert per_club.max() <= M.TEAM_GS_BUDGET + 0.5
    assert per_club.min() >= M.TEAM_GS_BUDGET - 1.5
    assert out["Proj_GS"].max() <= M.PT_MAX_GS + 1e-6


def test_the_end_of_rotation_starts_anchor_moved_with_its_innings():
    """They are not independent: innings over starts is how deep a starter
    goes, and cutting one alone sent this role to 4.4 innings a start —
    shallower than a pitcher who is in the rotation at all goes."""
    row = RT.role_anchor("End-of-Rotation Starter (SP4-5)", "pitcher")
    mid = RT.role_anchor("Mid-Rotation Starter (SP2-3)", "pitcher")
    assert row["gs"] == pytest.approx(21)
    depth = row["ip"] / row["gs"]
    assert 4.9 <= depth <= 5.5, f"{depth:.2f} innings per start"
    assert depth < mid["ip"] / mid["gs"], "an SP4 does not outlast an SP2"


def test_the_depth_rank_is_about_the_job_not_the_calendar():
    """A late arrival was charged for arriving twice.

    `pt_raw` already carries the timing share, so a July callup reached the
    ranking at half volume, sorted BELOW the incumbents for it, and then had
    the roster-depth decay take most of what was left. The rank means "he is
    the Nth man at this job on this club" — a statement about the job, not
    about when he takes it up — so the timing is divided back out before
    sorting and left to do its own work once.

    Same family, same role, same anchor; one arrives in July. He must rank
    level with the man who is there all year, not behind him.
    """
    players = pd.DataFrame({
        "PlayerId": [1, 2],
        "Name": ["all year", "arrives in July"],
        "Pred_target_team_id": [120, 120],
        "pt_tier": ["projected", "projected"],
        "pt_role": ["Bench Bat", "Bench Bat"],
        "pt_role_start": ["Opening Day", "Mid Season (~July)"],
        "pt_raw": [200.0, 100.0],
    })
    out = M.apply_roster_depth(players, "hitter")
    assert out["pt_depth_rank"].tolist() == [1.0, 2.0]
    # Level pegging on the standing means neither is pushed past the core
    # into the decay, so the July man keeps his full (halved) volume.
    assert out["pt_depth_factor"].tolist() == [1.0, 1.0]

    # And the one who really is the lesser player still sorts behind him.
    players.loc[1, "pt_raw"] = 40.0
    out = M.apply_roster_depth(players, "hitter")
    assert out["pt_depth_rank"].tolist() == [1.0, 2.0]


def test_the_depth_floor_carries_the_tail_a_real_club_has():
    """At 0.04 the decay ran off a cliff rather than tapering: everyone past
    about the tenth man past the core sat on the same floor, and a club's
    18th-32nd came out at 57% (arms) and 65% (bats) of real. The real curves
    do not end — a real club's 30th pitcher throws three innings and its 24th
    position player bats — so the floor has to carry that."""
    assert M.ROSTER_DEPTH_FLOOR == pytest.approx(0.10)
    # Still a floor rather than a second anchor: the deepest man keeps a
    # tenth of his role, not a third of it.
    assert 0.05 <= M.ROSTER_DEPTH_FLOOR <= 0.15

"""Missing time is not a job description.

A season's plate appearances are the product of two unrelated things — how
many games a player was there for, and how much he plays in a game — and
measuring only the product gets the players it matters most for exactly
backwards.

Byron Buxton is the case these tests are built around. He took 542 plate
appearances in 126 games, 4.30 a game, which is above the median Full Time
hitter's 3.97: when he plays he is an everyday centre fielder. RotoWire's
depth chart has him eighth among Minnesota's centre fielders, because he is
hurt, so the feeds called him an "Injury Replacement / 26th Man" and the
roster-depth discount finished the job. He projected NINE plate appearances.

Aaron Judge is the same story without the collapse: 679 plate appearances in
151 games. He came out "Full Time 0.70 / Strong Side Platoon 0.30", a
sentence that says he might be a platoon bat. He is not one. He is a
full-time player who gets hurt, and those are different claims that belong in
different columns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import durability as D  # noqa: E402
import playing_time_model as M  # noqa: E402
import role_feeds as RF  # noqa: E402

FIELDING = ROOT / "out" / "fielding_history_2027.csv"
has_history = pytest.mark.skipif(
    not FIELDING.exists(), reason="no fielding history in out/")


def _fielding(rows):
    """rows: (PlayerId, Season, Pos, G)"""
    return pd.DataFrame(rows, columns=["PlayerId", "Season", "Pos", "G"])


# ─────────────────────────────────────────────────────────────────────────────
# Counting games
# ─────────────────────────────────────────────────────────────────────────────

def test_a_utility_player_is_not_counted_twice_past_a_season():
    """One row per POSITION, so a man who moved mid-game appears twice. The
    truth is between the max and the sum; the sum capped at 162 is the closer
    of the two, and a cap is needed because without it he out-games the
    schedule."""
    f = _fielding([(1, 2026, "3B", 100), (1, 2026, "LF", 90),
                   (2, 2026, "1B", 140)])
    g = D.games_by_season(f).set_index("PlayerId")["G"]
    assert g[1] == 162.0
    assert g[2] == 140.0


def test_pitchers_and_hitters_are_counted_from_their_own_rows():
    f = _fielding([(1, 2026, "P", 30), (2, 2026, "CF", 150)])
    assert D.games_by_season(f, "hitter")["PlayerId"].tolist() == [2]
    assert D.games_by_season(f, "pitcher")["PlayerId"].tolist() == [1]


def test_no_fielding_history_is_empty_rather_than_fatal():
    assert D.games_by_season(None).empty
    assert D.games_by_season(pd.DataFrame()).empty


# ─────────────────────────────────────────────────────────────────────────────
# Predicting games
# ─────────────────────────────────────────────────────────────────────────────

def _three_seasons(pid, g2024, g2025, g2026, pos="CF"):
    return [(pid, 2024, pos, g2024), (pid, 2025, pos, g2025),
            (pid, 2026, pos, g2026)]


def test_the_iron_man_outranks_the_fragile_star():
    rows = (_three_seasons(1, 162, 162, 162)      # Olson
            + _three_seasons(2, 159, 151, 66)     # Judge
            + _three_seasons(3, 99, 126, 92))     # Buxton
    players = pd.DataFrame({"PlayerId": [1, 2, 3]})
    pred = D.predicted_games(players, D.games_by_season(_fielding(rows)),
                             target_year=2027)
    assert pred[0] > pred[1] > pred[2]


def test_the_recent_season_counts_for_more_than_the_old_one():
    """3/2/1, which the backtest picked over a flat mean (corr 0.428 against
    0.365) — last year says more about next year than three years ago does."""
    rows = _three_seasons(1, 162, 120, 80) + _three_seasons(2, 80, 120, 162)
    pred = D.predicted_games(pd.DataFrame({"PlayerId": [1, 2]}),
                             D.games_by_season(_fielding(rows)),
                             target_year=2027)
    assert pred[1] > pred[0]


def test_games_are_regressed_hard_because_they_barely_predict():
    """Backtested over 557 player-seasons, prior games beat assuming everyone
    is league-average by 8.1%, and among the most durable by 1.8%. A model
    that separated an iron man from a fragile star by the full gap between
    their records would claim far more than the record supports.

    The shrink is exact and fixture-independent: two players share the league
    term, so whatever gap their records hold survives as w/(w+k) of it —
    three fifths, for the 3/2/1 weights and the k the fit chose.
    """
    rows = (_three_seasons(1, 162, 162, 162) + _three_seasons(2, 110, 105, 60)
            + _three_seasons(3, 120, 120, 120) + _three_seasons(4, 110, 110,
                                                                110))
    pred = D.predicted_games(pd.DataFrame({"PlayerId": [1, 2, 3, 4]}),
                             D.games_by_season(_fielding(rows)),
                             target_year=2027)
    record_gap = 162.0 - float(np.average([60, 105, 110],
                                          weights=D.GAMES_WEIGHTS))
    survives = D.GAMES_PRIOR / (D.GAMES_PRIOR + D.GAMES_SHRINK)
    assert survives < 0.7, "the fitted shrink has to be a real one"
    assert (pred[0] - pred[1]) == pytest.approx(record_gap * survives)


def test_a_rookie_is_not_fragile_he_was_in_the_minors():
    """The first run made Yohandy Morales and Rafael Flores Jr. — 47 and 163
    plate appearances as rookies — two of the four most "fragile" players in
    baseball. Docking a player for seasons he spent in Triple-A is the same
    mistake as docking Buxton for being hurt, pointed at someone else. When a
    player arrives is `pt_role_start`'s question, not this one.
    """
    rows = _three_seasons(1, 162, 162, 162) + [(2, 2026, "CF", 30)]
    pred = D.predicted_games(pd.DataFrame({"PlayerId": [1, 2]}),
                             D.games_by_season(_fielding(rows)),
                             target_year=2027)
    assert np.isnan(pred[1]), "a callup has no durability record to read"


# ─────────────────────────────────────────────────────────────────────────────
# Availability
# ─────────────────────────────────────────────────────────────────────────────

def test_availability_is_measured_against_the_same_job():
    """A bench player's 60 games are his job. Judged against the league he
    looks fragile; judged against other bench players he is ordinary — and
    the role anchors were fitted to real accumulated playing time, so they
    already contain league-average missed time."""
    n = D.MIN_COHORT
    players = pd.DataFrame({
        "PlayerId": range(1, 2 * n + 1),
        "pt_role": ["Full Time"] * n + ["Bench Bat"] * n})
    pred = pd.Series([150.0] * (n - 1) + [90.0] + [60.0] * n)
    av = D.availability(players, pred)
    assert av.iloc[n - 1] < 0.95, "90 games is short for a regular"
    assert av.iloc[n:].eq(av.iloc[n]).all()
    assert av.iloc[n] == pytest.approx(1.0), "ordinary for a bench bat"


def test_a_role_too_small_to_measure_is_not_docked():
    """The pool mean is not a substitute for a missing cohort: it is built
    mostly from players in other jobs, so judging a catching tandem against
    it would dock them for catching."""
    players = pd.DataFrame({
        "PlayerId": range(1, 12),
        "pt_role": ["Full Time"] * 9 + ["Catcher - Tandem"] * 2})
    av = D.availability(players, pd.Series([150.0] * 9 + [70.0, 70.0]))
    assert av.iloc[9:].eq(1.0).all()


def test_no_record_is_not_a_dock():
    players = pd.DataFrame({"PlayerId": [1, 2],
                            "pt_role": ["Full Time", "Full Time"]})
    av = D.availability(players, pd.Series([np.nan, np.nan]))
    assert av.eq(1.0).all()


def test_availability_is_bounded():
    players = pd.DataFrame({"PlayerId": range(9),
                            "pt_role": ["Full Time"] * 9})
    pred = pd.Series([162.0] * 8 + [1.0])
    av = D.availability(players, pred)
    assert av.max() <= D.AVAILABILITY_MAX + 1e-9
    assert av.min() >= D.AVAILABILITY_MIN - 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# The rate, which is the job
# ─────────────────────────────────────────────────────────────────────────────

def test_the_rate_comes_from_the_season_the_evidence_came_from():
    players = pd.DataFrame({"PlayerId": [1], "evidence_volume": [542.0],
                            "evidence_season": [2025.0]})
    f = _fielding([(1, 2025, "CF", 126), (1, 2026, "CF", 92)])
    assert D.play_rate(players, D.games_by_season(f))[0] == pytest.approx(
        542 / 126)


def test_a_handful_of_games_is_not_a_rate():
    players = pd.DataFrame({"PlayerId": [1], "evidence_volume": [40.0],
                            "evidence_season": [2026.0]})
    f = _fielding([(1, 2026, "CF", 9)])
    assert np.isnan(D.play_rate(players, D.games_by_season(f))[0])


def test_the_rate_tells_an_everyday_player_from_a_part_time_one():
    """Measured over the projected hitters: Full Time 3.97 a game, Bench Bat
    3.09, Weak Side Platoon 2.96. Buxton's 4.30 is an everyday rate however
    few games he played."""
    assert D.record_read(4.30) == "Full Time"
    assert D.record_read(3.50) == "Strong Side Platoon"
    assert D.record_read(2.90) == "Bench Bat"
    assert D.record_read(np.nan) is None


# ─────────────────────────────────────────────────────────────────────────────
# Into the role
# ─────────────────────────────────────────────────────────────────────────────

def test_a_buried_regulars_own_rate_outranks_the_depth_chart():
    """Buxton. RotoWire has him eighth among Minnesota's centre fielders
    because he is hurt; being hurt is not a job, and the dock belongs in
    availability rather than in his role."""
    mix, src = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "CF", "depth_rank": 8,
         "play_rate": 4.30})
    assert max(mix, key=mix.get) == "Full Time"
    assert "record" in src


def test_the_rate_settles_a_role_one_feed_could_not():
    """Judge. The Yankees have no current batting order, so the depth chart
    alone gave "Full Time 0.70 / Strong Side Platoon 0.30" — which says he
    might be a platoon bat. His 4.50 plate appearances a game say he is not,
    and two sources agreeing is narrow by the module's own rule."""
    thin = RF.hitter_role_from_feeds(
        {"club_has_orders": False, "depth_pos": "RF", "depth_rank": 1})[0]
    with_rate = RF.hitter_role_from_feeds(
        {"club_has_orders": False, "depth_pos": "RF", "depth_rank": 1,
         "play_rate": 4.50})[0]
    assert with_rate["Full Time"] > thin["Full Time"]
    assert with_rate["Full Time"] == pytest.approx(RF.AGREE)


def test_a_part_time_rate_does_not_promote_anyone():
    """The rule only fires for a rate that says EVERYDAY. A bench player's
    own record agreeing he is a bench player must not lift him."""
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "LF", "depth_rank": 5,
         "play_rate": 2.80})
    assert max(mix, key=mix.get) != "Full Time"


def test_the_absence_is_not_charged_twice():
    """`evidence_volume` is a season TOTAL, so it already falls when a player
    is hurt — and availability docks him for the same absence. Compounding
    them would take a fifth of a season off a player twice, which is the
    mistake the batting-order factor made in `role_feeds`."""
    base = pd.DataFrame({
        "PlayerId": [1], "pt_anchor": [630.0], "evidence_volume": [500.0],
        "pt_availability": [1.0]})
    docked = base.assign(pt_availability=[0.80])
    full = M._evidence_factor(base, "hitter")[0]
    part = M._evidence_factor(docked, "hitter")[0]
    assert part > full, "a docked player's RATE evidence must not also fall"
    assert part == pytest.approx(min(500.0 / 0.80 / 630.0,
                                     M.EVIDENCE_FACTOR_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# Against the real history
# ─────────────────────────────────────────────────────────────────────────────

@has_history
def test_the_real_players_come_out_the_way_the_record_reads():
    fielding = pd.read_csv(FIELDING, low_memory=False)
    players = pd.read_csv(ROOT / "out" / "hitter_pa_projections_2027.csv",
                          low_memory=False)
    out, _, _ = M.project_playing_time(players, "hitter", target_year=2027,
                                       fielding=fielding, feed_dir=ROOT /
                                       "feeds")
    byname = out.set_index(out["Name"].astype(str))

    def one(fragment):
        hit = byname[byname.index.str.contains(fragment, na=False)]
        assert len(hit), fragment
        return hit.iloc[0]

    judge, buxton, olson = one("Judge"), one("Buxton"), one("Olson")
    # Both are full-time players. Neither is a platoon bat or a 26th man.
    for r in (judge, buxton, olson):
        assert r["pt_role"] == "Full Time", r["Name"]
    # Buxton used to project NINE plate appearances here.
    assert buxton["Proj_PA"] > 400, buxton["Proj_PA"]
    # And the injury risk shows up where it belongs, not in the role.
    assert judge["pt_availability"] < 0.98
    assert buxton["pt_availability"] < judge["pt_availability"]
    assert olson["pt_availability"] > 1.0

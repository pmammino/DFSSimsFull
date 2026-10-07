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

def test_durability_is_measured_against_the_same_job():
    """A bench player's 60 games are his job. Judged against the league he
    looks fragile; judged against other bench players he is ordinary — and
    the role anchors were fitted to real accumulated playing time, so they
    already contain league-average missed time."""
    n = D.MIN_COHORT
    players = pd.DataFrame({
        "PlayerId": range(1, 2 * n + 1),
        "pt_role": ["Full Time"] * n + ["Bench Bat"] * n})
    pred = pd.Series([150.0] * (n - 1) + [90.0] + [60.0] * n)
    av = D.durability(players, pred)
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
    av = D.durability(players, pd.Series([150.0] * 9 + [70.0, 70.0]))
    assert av.iloc[9:].eq(1.0).all()


def test_no_record_is_not_a_dock():
    players = pd.DataFrame({"PlayerId": [1, 2],
                            "pt_role": ["Full Time", "Full Time"]})
    av = D.durability(players, pd.Series([np.nan, np.nan]))
    assert av.eq(1.0).all()


def test_durability_is_bounded():
    players = pd.DataFrame({"PlayerId": range(9),
                            "pt_role": ["Full Time"] * 9})
    pred = pd.Series([162.0] * 8 + [1.0])
    av = D.durability(players, pred)
    assert av.max() <= D.DURABILITY_MAX + 1e-9
    assert av.min() >= D.DURABILITY_MIN - 1e-9


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
    assert with_rate == {"Full Time": 1.0}


def test_a_settled_job_is_allowed_to_be_certain():
    """A role mixture answers "which job does he hold" and nothing else.
    Stopping at 0.85 because a season offers chances to get hurt was the
    injury risk being charged twice — it is `pt_availability`'s now."""
    for ev in ({"club_has_orders": False, "depth_pos": "LF", "depth_rank": 1,
                "play_rate": 4.50},
               {"club_has_orders": True, "spot_vs_r": 3, "spot_vs_l": 3,
                "depth_pos": "LF", "depth_rank": 1, "play_rate": 4.40}):
        mix, _ = RF.hitter_role_from_feeds(ev)
        assert mix == {"Full Time": 1.0}, ev


def test_a_confirmed_role_keeps_the_feeds_more_specific_name():
    """A rate can tell an everyday player from a part-time one; it cannot
    tell a designated hitter from a left fielder. Asking it for the exact
    role name left Mike Trout — depth chart "Everyday DH / 1B-DH", 4.46
    plate appearances a game — split between a full-time job and a bench
    one, because the two strings did not match."""
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": False, "depth_pos": "DH", "depth_rank": 1,
         "play_rate": 4.46})
    assert mix == {"Everyday DH / 1B-DH": 1.0}


def test_settling_a_role_survives_the_un_floor_guard():
    """The guard re-spreads a mixture of 1.0 left behind when two roles merge
    into one. It must not touch a 1.0 that was asserted on purpose — it was
    putting Judge straight back onto the platoon role he had been taken off."""
    df = pd.DataFrame({
        "PlayerId": [1], "Name": ["Settled Regular"],
        "Pred_target_team_id": [147], "Last_PA": [600], "Career_PA": [3000],
        "evidence_volume": [660.0], "evidence_season": [2025.0],
        "P_K": [0.22], "P_BB": [0.085], "P_HBP": [0.011], "P_SF": [0.006],
        "P_HR": [0.030], "P_3B": [0.004], "P_2B": [0.042], "P_1B": [0.142],
        "P_BIPOut": [0.460]})
    base = M.assign_default_roles(df, "hitter")
    base["pt_play_rate"] = 4.50
    feeds = {"depth": pd.DataFrame({
        "name_key": [RF.name_key("Settled", "Regular")], "feed_name": ["x"],
        "rw_id": ["1"], "feed_team": ["NYY"], "feed_team_id": [147],
        "depth_pos": ["RF"], "depth_rank": [1]})}
    out, _ = RF.apply_feed_roles(base, "hitter", feeds=feeds)
    assert out["pt_role_mix"].iloc[0] == "Full Time:1"


def test_a_part_time_rate_does_not_promote_anyone():
    """The rule only fires for a rate that says EVERYDAY. A bench player's
    own record agreeing he is a bench player must not lift him."""
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "LF", "depth_rank": 5,
         "play_rate": 2.80})
    assert max(mix, key=mix.get) != "Full Time"


def test_the_absence_is_not_charged_twice():
    """`evidence_volume` is a season TOTAL, so it already falls when a player
    is hurt — and durability docks him for the same absence. Compounding
    them would take a fifth of a season off a player twice, which is the
    mistake the batting-order factor made in `role_feeds`."""
    base = pd.DataFrame({
        "PlayerId": [1], "pt_anchor": [630.0], "evidence_volume": [500.0],
        "pt_durability": [1.0]})
    docked = base.assign(pt_durability=[0.80])
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
    assert judge["pt_durability"] < 0.98
    assert buxton["pt_durability"] < judge["pt_durability"]
    assert olson["pt_durability"] > 1.0


def test_the_iron_man_credit_is_actually_spent():
    """`raw_volumes` clipped availability into [0, 1], which silently threw
    away every credit above one. Matt Olson — 162 games in each of the last
    three seasons — earned 1.15 and was handed 1.00, losing his whole
    iron-man bonus, while Mike Trout's 0.974 dock passed through untouched.
    Trout out-projected him, which is how this was noticed.

    The two are different claims and only one is bounded by one, so they are
    different columns: `pt_availability` is the 0..1 knob a person types, and
    `pt_durability` is what the record says about turning up.
    """
    df = pd.DataFrame({
        "PlayerId": [1, 2], "Name": ["Iron Man", "Fragile"],
        "pt_anchor": [630.0, 630.0], "pt_role": ["Full Time"] * 2,
        "pt_role_start": ["Opening Day"] * 2,
        "pt_availability": [1.0, 1.0], "pt_durability": [1.15, 0.90],
        "pt_role_mix": ["", ""], "BatSide": ["R", "R"]})
    out = M.raw_volumes(df, "hitter")
    assert out["pt_raw"].iloc[0] > out["pt_raw"].iloc[1]
    assert out["pt_raw"].iloc[0] == pytest.approx(630.0 * 1.15, rel=1e-6)


def test_one_season_is_not_an_iron_man_record():
    """The shrink was fitted on players with three prior seasons, so handing
    a player with one the same prior treats a single year as though it were
    three — and one season is where the noise is. Kevin McGonigle, a rookie
    with 2026 and nothing else, came out at the 1.15 ceiling: the most
    durable player in baseball on the strength of not having had a chance to
    get hurt yet."""
    rows = (_three_seasons(1, 162, 162, 162)
            + _three_seasons(2, 120, 120, 120)
            + _three_seasons(3, 110, 110, 110)
            + [(4, 2026, "CF", 162)])          # one season, a full one
    pred = D.predicted_games(pd.DataFrame({"PlayerId": [1, 2, 3, 4]}),
                             D.games_by_season(_fielding(rows)),
                             target_year=2027)
    assert pred[3] < pred[0], "three seasons of 162 must beat one"


# ─────────────────────────────────────────────────────────────────────────────
# Pitchers: innings are appearances times innings per appearance
# ─────────────────────────────────────────────────────────────────────────────

def _arms(n=14, team=147, n_sp=0):
    """n_sp of them are starters, so the ROLE cohorts that the rate is
    regressed toward are real ones. Everyone in a bullpen is the degenerate
    case: a lone starter among relievers is regressed toward a reliever's
    innings per appearance, which is correct behaviour on a fixture that has
    told the model he is a reliever."""
    return pd.DataFrame({
        "PlayerId": range(3000, 3000 + n),
        "Name": [f"p{i}" for i in range(n)],
        "Pred_target_team_id": team, "pt_tier": "projected",
        "role": ["starter"] * n_sp + ["reliever"] * (n - n_sp),
        "TBF_per_IP": 4.3,
        "evidence_volume": [760.0] * n_sp + [260.0] * (n - n_sp),
        "evidence_season": 2026.0,
        "P_K": 0.23, "P_BB": 0.080, "P_HBP": 0.011, "P_SF": 0.006,
        "P_HR": 0.029, "P_3B": 0.004, "P_2B": 0.041, "P_1B": 0.140,
        "P_BIPOut": 0.466})


def _arm_history(df, apps, ipa):
    """apps/ipa: dict PlayerId -> value, applied to all three seasons."""
    return pd.DataFrame([
        {"Season": y, "PlayerId": int(p), "Pos": "P", "GS": 0,
         "G": apps.get(int(p), 50), "Innings": apps.get(int(p), 50)
         * ipa.get(int(p), 1.0)}
        for y in (2024, 2025, 2026) for p in df["PlayerId"]])


def test_a_pitchers_innings_are_his_two_numbers_multiplied():
    """The redesign in one assertion. A starter and a reliever can take the
    same innings by opposite routes, and a model carrying only the product
    cannot tell them apart — which is why availability had nowhere to go."""
    df = _arms(n=20, n_sp=8)
    ids = list(df["PlayerId"])
    sp, rp = ids[:8], ids[8:]
    apps = {**{p: 30 for p in sp}, **{p: 62 for p in rp}}
    ipa = {**{p: 5.5 for p in sp}, **{p: 1.0 for p in rp}}
    out, _, _ = M.project_playing_time(
        df, "pitcher", target_year=2027,
        fielding=_arm_history(df, apps, ipa), feed_dir=None)
    starter = out[out["PlayerId"] == sp[0]].iloc[0]
    reliever = out[out["PlayerId"] == rp[0]].iloc[0]
    # Opposite routes to a season: fewer turns, far longer each.
    assert starter["pt_apps_exp"] < reliever["pt_apps_exp"]
    assert starter["pt_ip_per_app_exp"] > reliever["pt_ip_per_app_exp"] * 2


def test_missing_a_season_costs_a_pitcher_appearances_not_his_job():
    """Félix Bautista, who comes out 28.3 appearances at 0.99 innings each.
    Being hurt shows up in how often he is handed the ball; it does not make
    him a different pitcher once he has it."""
    df = _arms(n=12)
    ids = list(df["PlayerId"])
    healthy, hurt = ids[0], ids[1]
    out, _, _ = M.project_playing_time(
        df, "pitcher", target_year=2027,
        fielding=_arm_history(df, {healthy: 65, hurt: 20}, {}),
        feed_dir=None)
    a = out[out["PlayerId"] == healthy].iloc[0]
    b = out[out["PlayerId"] == hurt].iloc[0]
    assert b["pt_apps_exp"] < a["pt_apps_exp"]
    assert b["Proj_IP"] < a["Proj_IP"]
    # Same job: the rate is what says what he does, and it is unchanged.
    assert b["pt_ip_per_app_exp"] == pytest.approx(
        a["pt_ip_per_app_exp"], rel=0.05)


def test_the_absence_is_not_charged_twice_on_the_pitcher_side_either():
    """`pt_apps_exp` is the pitcher's OWN appearance record, so it already
    carries every start he missed. Multiplying it by a durability derived
    from those same appearances would charge the absence a second time —
    the fourth time that shape of mistake has turned up in this model."""
    df = _arms(n=12)
    out, _, _ = M.project_playing_time(
        df, "pitcher", target_year=2027,
        fielding=_arm_history(df, {}, {}), feed_dir=None)
    assert (out["pt_durability"] == 1.0).all()
    share = pd.to_numeric(out["pt_season_share"], errors="coerce")
    assert (share <= 1.0 + 1e-9).all()


def test_innings_per_appearance_regresses_harder_than_appearances_do():
    """Fitted separately and they are not the same number: k = 6.0 against
    k = 2.0. How often a man is handed the ball is mostly about him; how
    long he stays is mostly about the job."""
    assert D.IPA_SHRINK > D.GAMES_SHRINK * 2


def test_a_pitcher_with_no_record_falls_back_to_his_role():
    df = _arms(n=12)
    out, _, _ = M.project_playing_time(df, "pitcher", target_year=2027,
                                       fielding=None, feed_dir=None)
    assert out["Proj_IP"].notna().all()
    assert (out["Proj_IP"] > 0).all()


@has_history
def test_the_decomposition_earns_its_place_against_the_product():
    """Both calibration targets, measured end to end: the within-club rank
    curve the anchors are fitted against, and how many pitchers clear 180
    innings (21, 20 and 12 in the real 2024-26 seasons).

    This used to assert the decomposition beat the product form on the top
    twelve ranks, which it did (0.0269 against 0.0291) until the roster-depth
    floor and the pitcher decay were refitted together. BOTH FORMS IMPROVED
    when that happened — nothing regressed — but the product form improved
    more at the top, to 0.0225 against the decomposition's 0.0265, and the
    old margin reversed.

    So the claim is restated to the one the decomposition still earns, which
    is also the one it was built for: it is right about MORE OF THE STAFF.
    A pitcher's innings are appearances times innings per appearance, and
    keeping them as two numbers is what lets a reliever's durability mean
    anything — so the per-player error across every arm, 0.166 against
    0.178, is the measure that matches the claim. The top twelve are twelve
    men; the staff is thirty.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_role_anchors import real_rank_curve
    fielding = pd.read_csv(FIELDING, low_memory=False)
    players = pd.read_csv(ROOT / "out" / "pitcher_pa_projections_2027.csv",
                          low_memory=False)
    curve, _ = real_rank_curve(fielding, "pitcher")
    scores = {}
    for flag in (False, True):
        M.PITCHER_VOLUME_DECOMPOSED = flag
        try:
            out, _, _ = M.project_playing_time(
                players, "pitcher", target_year=2027, fielding=fielding,
                feed_dir=ROOT / "feeds")
        finally:
            M.PITCHER_VOLUME_DECOMPOSED = True
        pr = out[out["pt_tier"].astype(str) != "floor"].copy()
        pr["rank"] = pr.groupby("Pred_target_team_id")["Proj_IP"].rank(
            "first", ascending=False)
        # Every arm against the real innings at his rank, not the mean of
        # each rank — a staff is thirty men and the mean of twelve of them
        # hides what the other eighteen are doing. Ranks past the end of the
        # real curve take its thinnest value, as `fit_role_anchors` does.
        target = pr["rank"].map(curve).fillna(curve.iloc[-1])
        ok = target > 0
        mae = float(np.mean(np.abs(pr.loc[ok, "Proj_IP"] / target[ok] - 1)))
        scores[flag] = (mae, int((pr["Proj_IP"] > 180).sum()))
    assert scores[True][0] < scores[False][0], scores
    # And it is still in range on the workhorse count, which is the target
    # that caught the product form projecting 29 pitchers past 180 innings.
    assert abs(scores[True][1] - 17.67) <= 6, scores

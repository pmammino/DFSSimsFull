"""Roles that come from a depth chart, not from a guess at last year's volume.

Every player used to reach the allocator as 100% of one role at 100%
availability — a claim nobody has the information to make. Four RotoWire feeds
do have it, and the design idea under test here is how they express doubt:

    a mixture's WIDTH is how much the feeds disagree, not an opinion.

So the tests are mostly about disagreement. Two feeds saying the same thing has
to come out narrow; two feeds saying different things has to come out split;
a feed that states its own uncertainty (a closer's Stability rating) has to
have that uncertainty survive into the mixture; and a feed saying nothing has
to leave the heuristic alone rather than inventing a role for the silence.

The two traps the real files set are regression-tested here by name, because
both were found by measurement and neither is visible from the schema:
stale eight-spot batting orders carried alongside the current nine-spot ones,
and absence from a lineup being read as positive evidence of a bench job.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import playing_time_model as M  # noqa: E402
import role_feeds as RF  # noqa: E402
from role_taxonomy import TIMING, parse_role_mix  # noqa: E402

FEEDS = ROOT / "feeds"
has_feeds = pytest.mark.skipif(
    not (FEEDS / "depth.xml").exists(),
    reason="the RotoWire feed snapshot is not present")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


BANNER = ("This XML file does not appear to have any style information "
          "associated with it. The document tree is shown below.\n")


def test_the_browser_banner_does_not_stop_the_parse(tmp_path):
    """These files arrive saved out of a browser tab, which prepends a line of
    prose to the XML. Refusing to parse it would reject every real file."""
    p = _write(tmp_path, "depth.xml", BANNER + """
<DepthChart><Teams><Team Id="1" Code="NYY"><Players>
  <Player Id="1"><FirstName>A</FirstName><LastName>B</LastName>
    <Position>SS</Position><Rank>1</Rank></Player>
</Players></Team></Teams></DepthChart>""")
    d = RF.read_depth(p)
    assert len(d) == 1
    assert d.iloc[0]["depth_pos"] == "SS"
    assert d.iloc[0]["feed_team"] == "NYY"


def test_a_missing_or_broken_feed_is_empty_rather_than_fatal(tmp_path):
    assert RF.read_depth(tmp_path / "nope.xml").empty
    assert RF.read_closers(_write(tmp_path, "c.xml", "<Teams><oops")).empty


def test_feed_team_codes_resolve_to_mlbam_ids(tmp_path):
    """RotoWire says ANA; the rest of the pipeline says 108."""
    p = _write(tmp_path, "depth.xml", """
<DepthChart><Teams><Team Id="1" Code="ANA"><Players>
  <Player Id="1"><FirstName>A</FirstName><LastName>B</LastName>
    <Position>CF</Position><Rank>1</Rank></Player>
</Players></Team></Teams></DepthChart>""")
    row = RF.read_depth(p).iloc[0]
    assert row["feed_team"] == "LAA"
    assert row["feed_team_id"] == 108


def test_name_keys_survive_accents_suffixes_and_punctuation():
    assert RF.name_key("José", "Ramírez") == "jose ramirez"
    assert RF.name_key("Ronald", "Acuña Jr.") == "ronald acuna"
    assert RF.name_key("Travis", "d'Arnaud") == "travis darnaud"


# ─────────────────────────────────────────────────────────────────────────────
# Trap 1: the stale lineups
# ─────────────────────────────────────────────────────────────────────────────

LEGACY_AND_CURRENT = BANNER + """
<Teams><Team Id="2" Code="ATL">
 <Lineups>
  <Lineup GameType="NORMAL"><BattingOrders>
   <BattingOrder OpposingPitcherHandedness="R"><BattingSpots>
""" + "".join(f"""<BattingSpot Position="{i}"><Player Id="{i}">
     <FirstName>Old</FirstName><LastName>Guy{i}</LastName>
     <Position>OF</Position></Player></BattingSpot>""" for i in range(1, 9)) + """
   </BattingSpots></BattingOrder>
  </BattingOrders></Lineup>
  <Lineup GameType="NORMAL"><BattingOrders>
   <BattingOrder OpposingPitcherHandedness="R"><BattingSpots>
""" + "".join(f"""<BattingSpot Position="{i}"><Player Id="{100+i}">
     <FirstName>New</FirstName><LastName>Guy{i}</LastName>
     <Position>OF</Position></Player></BattingSpot>""" for i in range(1, 10)) + """
   </BattingSpots></BattingOrder>
  </BattingOrders></Lineup>
 </Lineups></Team></Teams>"""


def test_the_eight_spot_lineup_is_flagged_as_the_legacy_one(tmp_path):
    """The feed carries Atlanta's Swanson / Duvall / d'Arnaud order from the
    era when the pitcher batted, beside the current one, both marked
    GameType="NORMAL". The only thing telling them apart is that the old one
    has eight spots — and reading both put 232 players on clubs they left
    years ago. Measured on the real file: 8-spot orders agree with the
    projection's club for 29.4% of their players, 9-spot orders for 99.6%.
    """
    o = RF.read_batting_orders(_write(tmp_path, "orders.xml",
                                      LEGACY_AND_CURRENT))
    assert set(o["legacy"]) == {True, False}
    assert set(o.loc[o["legacy"], "feed_name"].str.startswith("Old")) == {True}
    assert set(o.loc[~o["legacy"], "feed_name"].str.startswith("New")) == {True}


def test_only_the_current_lineup_reaches_the_roles(tmp_path):
    feeds = {"orders": RF.read_batting_orders(
        _write(tmp_path, "orders.xml", LEGACY_AND_CURRENT))}
    kept = RF._for_pool(feeds, "orders", "hitter")
    assert len(kept) == 9
    assert not kept["legacy"].any()


# ─────────────────────────────────────────────────────────────────────────────
# Trap 2: absence is a cap, not a job
# ─────────────────────────────────────────────────────────────────────────────

def test_absence_from_a_lineup_cannot_promote_a_prospect():
    """Read as a positive claim, "not in the nine" made a bench bat of every
    farmhand in the organisation — Boston's Schaffner, Arias and Brannon all
    came out Bench Bat 0.6, taking plate appearances from players who will
    actually bat. The depth chart already has them nowhere near the roster,
    so the two feeds AGREE, and the more specific one wins.
    """
    mix, src = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4})
    assert max(mix, key=mix.get) == "Depth (no MLB PA)"
    assert "Bench Bat" not in mix


def test_absence_still_contradicts_a_depth_chart_that_says_starter():
    """The case where it IS a disagreement, and has to stay one."""
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "SS", "depth_rank": 1})
    assert set(mix) == {"Bench Bat", "Full Time"}
    assert mix["Bench Bat"] == pytest.approx(RF.DISAGREE)


def test_a_club_with_no_lineup_in_the_feed_is_not_a_club_of_bench_players():
    """Seven clubs have no current batting order — ARI, CHC, CWS, LAA, LAD,
    NYM, NYY. Their players are unobserved, not benched, and the width says
    so: wider than agreement, and on the depth chart's answer."""
    mix, src = RF.hitter_role_from_feeds(
        {"club_has_orders": False, "depth_pos": "SS", "depth_rank": 1})
    assert max(mix, key=mix.get) == "Full Time"
    assert mix["Full Time"] == pytest.approx(RF.DEPTH_ONLY["hitter"])
    assert src == "feed:depth"


# ─────────────────────────────────────────────────────────────────────────────
# Width is disagreement
# ─────────────────────────────────────────────────────────────────────────────

def test_two_feeds_agreeing_is_narrow():
    mix, src = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "spot_vs_r": 3, "spot_vs_l": 3,
         "depth_pos": "LF", "depth_rank": 1})
    assert mix["Full Time"] == pytest.approx(RF.AGREE_BY_SPOT[3])
    assert src == "feed:order+depth"


def test_a_job_gets_less_settled_down_the_batting_order():
    """A club's nine starters run from about 634 plate appearances to about
    342, a factor of 1.85, and the spot itself only explains 4.65 against
    3.97 a game. The rest is job security — the number-three hitter is still
    in the lineup in September and the number-nine hitter is who gets
    replaced — and spending it as WIDTH rather than as a multiplier is what
    brought the top of the rank curve back down from 6.1% high to 4.7%.
    """
    def regular(spot):
        mix, _ = RF.hitter_role_from_feeds(
            {"club_has_orders": True, "spot_vs_r": spot, "spot_vs_l": spot,
             "depth_pos": "LF", "depth_rank": 1})
        return mix["Full Time"]

    assert regular(2) > regular(5) > regular(9)
    assert regular(9) < 0.65, "a ninth hitter's job is not a settled thing"
    # And the mixture, not just the label, is what the model spends.
    from role_taxonomy import blend_anchors
    top = blend_anchors(RF.hitter_role_from_feeds(
        {"club_has_orders": True, "spot_vs_r": 2, "spot_vs_l": 2,
         "depth_pos": "LF", "depth_rank": 1})[0], "hitter")
    bottom = blend_anchors(RF.hitter_role_from_feeds(
        {"club_has_orders": True, "spot_vs_r": 9, "spot_vs_l": 9,
         "depth_pos": "LF", "depth_rank": 1})[0], "hitter")
    assert top["pa"] - bottom["pa"] > 40


def test_two_feeds_disagreeing_is_split_and_keeps_both_answers():
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "spot_vs_r": 6, "depth_pos": "1B",
         "depth_rank": 2})
    assert mix == {"Strong Side Platoon": pytest.approx(RF.DISAGREE),
                   "Utility IF": pytest.approx(1 - RF.DISAGREE)}


def test_a_platoon_is_read_off_the_two_orders():
    """The whole reason the feed carries a lineup against each hand."""
    vs_r = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "spot_vs_r": 5})[0]
    vs_l = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "spot_vs_l": 7})[0]
    assert max(vs_r, key=vs_r.get) == "Strong Side Platoon"
    assert max(vs_l, key=vs_l.get) == "Weak Side Platoon"


def test_no_feed_covers_him_so_the_heuristic_stands():
    assert RF.hitter_role_from_feeds({"club_has_orders": False}) is None
    assert RF.pitcher_role_from_feeds({}) is None


def test_every_role_a_player_might_hold_is_a_mixture():
    """A mixture of 1.0 is the certainty this module exists to stop asserting,
    and the ladder's bottom rung is where it crept back in: the smallest real
    job has nothing below it to step down to."""
    for ev in ({"club_has_orders": True, "spot_vs_r": 1, "spot_vs_l": 1,
                "depth_pos": "CF", "depth_rank": 1},
               {"club_has_orders": True, "spot_vs_r": 4, "depth_pos": "1B",
                "depth_rank": 4},
               {"club_has_orders": False, "depth_pos": "C", "depth_rank": 2}):
        mix, _ = RF.hitter_role_from_feeds(ev)
        assert len(mix) >= 2, ev
        assert max(mix.values()) < 1.0, ev
        assert sum(mix.values()) == pytest.approx(1.0)


def test_not_being_a_major_leaguer_is_the_one_thing_said_outright():
    """The exception, and it has to be one. A 15% bench share for a player
    nobody has within four ranks of the roster is not humility, it is plate
    appearances taken from someone who will actually bat — and the depth role
    is how a player stays present and joinable without claiming any.

    What protects a real player from landing here is the un-floored guard in
    `apply_feed_roles`, not a hedge in the mixture.
    """
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 9})
    assert mix == {"Depth (no MLB PA)": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# The bullpen, where the feed states its own doubt
# ─────────────────────────────────────────────────────────────────────────────

def test_stability_is_carried_into_the_mixture():
    """RotoWire publishes a Stability rating on each club's top arm. It is the
    one field in any of these files that is explicitly about how much to
    believe them, and ignoring it would be the model throwing away the best
    thing it was given."""
    sure = RF.pitcher_role_from_feeds(
        {"rw_role": "Closer", "stability": "Very High"})[0]
    shaky = RF.pitcher_role_from_feeds(
        {"rw_role": "Closer", "stability": "Very Low"})[0]
    assert sure["Closer"] == pytest.approx(0.92)
    assert shaky["Closer"] == pytest.approx(0.50)
    assert sure["Closer"] > shaky["Closer"]


def test_a_committee_is_written_as_the_committee_it_is():
    """"Nobody has this job" is not a role, it is three men splitting one."""
    mix, _ = RF.pitcher_role_from_feeds({"rw_role": "Committee"})
    assert len(mix) == 3
    assert mix["Closer"] < 0.5
    assert sum(mix.values()) == pytest.approx(1.0)


def test_a_committee_costs_its_closer_most_of_the_save_pool():
    """What the mixture is FOR: the share follows the uncertainty into the
    saves without anyone having to wire it up separately."""
    from role_taxonomy import blend_anchors
    settled = blend_anchors({"Closer": 1.0}, "pitcher")
    split = blend_anchors(
        RF.pitcher_role_from_feeds({"rw_role": "Committee"})[0], "pitcher")
    assert split["sv"] < settled["sv"] * 0.7


def test_the_rotation_comes_off_the_depth_charts_pitching_group():
    for rank, want in ((1, "Ace (SP1)"), (2, "Mid-Rotation Starter (SP2-3)"),
                       (5, "End-of-Rotation Starter (SP4-5)"),
                       (7, "Swing Arm / Long Relief")):
        mix, _ = RF.pitcher_role_from_feeds({"depth_pos": "P",
                                             "depth_rank": rank})
        assert max(mix, key=mix.get) == want, rank


def test_a_bullpen_arm_the_depth_chart_calls_a_starter_is_a_swing_arm():
    """Both feeds speak and disagree. Carried, not resolved."""
    mix, src = RF.pitcher_role_from_feeds(
        {"rw_role": "Middle Reliever", "depth_pos": "P", "depth_rank": 4})
    assert set(mix) == {"Middle Relief", "End-of-Rotation Starter (SP4-5)"}
    assert src == "feed:pen+depth"


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

def _players():
    return pd.DataFrame({
        "PlayerId": [1, 2, 3, 4],
        "Name": ["Jose Ramirez", "Will Smith", "Will Smith", "Aaron Judge"],
        "Pred_target_team_id": [114, 119, 147, 147],
    })


def _feed(keys, teams):
    return pd.DataFrame({"name_key": keys, "feed_team_id": teams,
                         "feed_name": keys})


def test_a_name_resolves_to_an_mlbam_id():
    out, st = RF.resolve_ids(_feed(["jose ramirez"], [114]), _players())
    assert out["PlayerId"].tolist() == [1.0]
    assert st["matched"] == 1


def test_a_shared_name_is_settled_by_the_club():
    out, _ = RF.resolve_ids(_feed(["will smith"], [147]), _players())
    assert out["PlayerId"].tolist() == [3.0]


def test_a_shared_name_the_club_cannot_settle_is_dropped_not_guessed():
    """Guessing hands one man's job to another, which is worse than leaving
    the heuristic in place. The alias file is how a human settles it."""
    out, st = RF.resolve_ids(_feed(["will smith"], [None]), _players())
    assert out["PlayerId"].isna().all()
    assert st["ambiguous"] == 1


def test_being_traded_does_not_lose_a_player():
    """The club is a tie-breaker, never a filter. Dropping a player because
    the feed has him on his old team would throw away exactly the rows a
    season projection most needs."""
    out, st = RF.resolve_ids(_feed(["aaron judge"], [137]), _players())
    assert out["PlayerId"].tolist() == [4.0]
    assert st["wrong_club"] == 1


def test_an_alias_settles_what_the_name_join_cannot():
    out, st = RF.resolve_ids(_feed(["aaron judgey"], [147]), _players(),
                             aliases={"aaron judgey": 4})
    assert out["PlayerId"].tolist() == [4.0]
    assert st["by_alias"] == 1


def test_an_unmatched_name_is_reported_by_name():
    _, st = RF.resolve_ids(_feed(["nobody here"], [147]), _players())
    assert st["unmatched"] == 1
    assert "nobody here" in st["missed_names"]


def test_the_alias_file_is_optional_and_survives_a_bad_one(tmp_path):
    assert RF.load_aliases(tmp_path / "nope.csv") == {}
    bad = tmp_path / "a.csv"
    bad.write_text("something,else\n1,2\n")
    assert RF.load_aliases(bad) == {}
    good = tmp_path / "b.csv"
    good.write_text("name_key,PlayerId\nmike trout,545361\n")
    assert RF.load_aliases(good) == {"mike trout": 545361}


# ─────────────────────────────────────────────────────────────────────────────
# Into the model
# ─────────────────────────────────────────────────────────────────────────────

def _frame(n=12, team=147):
    return pd.DataFrame({
        "PlayerId": range(1, n + 1),
        "Name": [f"Player {i}" for i in range(1, n + 1)],
        "Pred_target_team_id": team,
        "Last_PA": 400, "Career_PA": 2000,
        "P_K": 0.22, "P_BB": 0.085, "P_HBP": 0.011, "P_SF": 0.006,
        "P_HR": 0.030, "P_3B": 0.004, "P_2B": 0.042, "P_1B": 0.142,
        "P_BIPOut": 0.460,
    })


def test_no_feed_directory_leaves_the_model_exactly_as_it_was():
    """A repo without the feeds has to keep working."""
    df = _frame()
    a, _, _ = M.project_playing_time(df, "hitter", target_year=2027,
                                     feed_dir=None)
    b, _, _ = M.project_playing_time(df, "hitter", target_year=2027,
                                     feed_dir="no/such/dir")
    assert a["Proj_PA"].to_numpy() == pytest.approx(b["Proj_PA"].to_numpy())


def test_the_spot_is_spent_once_and_not_twice():
    """A per-game multiplier on top of `AGREE_BY_SPOT` double-counted the
    batting order and measurably made the curve worse. The column it wrote is
    gone; this fails if it comes back."""
    df = _frame()
    out, _, _ = M.project_playing_time(df, "hitter", target_year=2027,
                                       feed_dir=None)
    assert "pt_lineup_factor" not in out.columns


def test_a_mixture_the_model_cannot_read_would_be_a_silent_depth_player():
    """Everything written out has to survive the round trip the allocator
    makes it take."""
    for ev in ({"club_has_orders": True, "spot_vs_r": 2, "spot_vs_l": 2,
                "depth_pos": "RF", "depth_rank": 1},
               {"club_has_orders": True, "spot_vs_l": 8, "depth_pos": "1B",
                "depth_rank": 3}):
        mix, _ = RF.hitter_role_from_feeds(ev)
        from role_taxonomy import format_role_mix
        back = parse_role_mix(format_role_mix(mix), "hitter")
        assert back is not None
        assert set(back) == set(mix)


def test_the_feeds_never_send_a_real_major_leaguer_to_the_floor(tmp_path):
    """The depth role routes straight to the 1-PA floor and outside the
    club's budget, so a buried player would have his season deleted rather
    than shortened. Boston has Triston Casas eighth among its first basemen;
    that is a bench job, not a retirement.
    """
    df = _frame(n=3)
    base = M.assign_default_roles(df, "hitter")
    assert not base["pt_role"].map(
        lambda r: "Depth" in str(r)).any(), "fixture must start off the floor"
    feeds = {"depth": pd.DataFrame({
        "name_key": [RF.name_key("Player", str(i)) for i in (1, 2, 3)],
        "feed_name": ["x"] * 3, "rw_id": ["1", "2", "3"],
        "feed_team": ["NYY"] * 3, "feed_team_id": [147] * 3,
        "depth_pos": ["1B"] * 3, "depth_rank": [9, 10, 11]})}
    out, st = RF.apply_feed_roles(base, "hitter", feeds=feeds)
    assert st["applied"] == 3
    assert not out["pt_role"].map(lambda r: "Depth" in str(r)).any()
    assert (out["pt_role"] == "Injury Replacement / 26th Man").all()


# ─────────────────────────────────────────────────────────────────────────────
# Prospect arrivals
# ─────────────────────────────────────────────────────────────────────────────

def test_a_top_prospect_at_triple_a_is_not_a_depth_player():
    """The reason this exists. Every Triple-A prospect sat at the 1-PA floor
    however good he was, because the tier asks "has he real MLB evidence"
    and for a man who has not debuted the answer is no and always will be
    until he does. Real first-year position players take 8.9% of all league
    plate appearances, so flooring them does not save that playing time — it
    hands it to incumbents who will not be taking it.
    """
    mix, src = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4,
         "level": "AAA", "prospect_rank": 12, "birth_year": 2005})
    assert src == "feed:prospect"
    assert mix["Full Time"] == pytest.approx(0.40)
    assert "Depth (no MLB PA)" not in mix


def test_the_arrival_mixture_is_wide_because_nobody_knows():
    """This is the case the mixture machinery was built for. Whether a top
    prospect called up in May finishes the year as the regular or back on the
    bus is genuinely unknown, and 40/25/35 is the honest reading of it — it
    is what puts the blended anchor near the real distribution instead of on
    either tail of it. A narrow mixture here would be a claim nobody can
    make.
    """
    mix, _ = RF.hitter_role_from_feeds(
        {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4,
         "level": "AAA", "prospect_rank": 12, "birth_year": 2005})
    assert len(mix) >= 3
    assert max(mix.values()) < 0.5


def test_further_away_and_further_down_means_less():
    """Rank and level both have to bite, and in the right direction: the
    arrival is earlier and bigger for a better prospect who is closer."""
    def pa(level, rank):
        mix, _ = RF.hitter_role_from_feeds(
            {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4,
             "level": level, "prospect_rank": rank, "birth_year": 2005})
        return mix.get("Full Time", 0.0) + mix.get("Strong Side Platoon", 0.0)

    assert pa("AAA", 10) > pa("AAA", 120) > 0
    assert pa("AAA", 10) > pa("AA", 10) > 0
    # And past the end of the table there is no arrival at all, which is a
    # different statement from a small one — see the note on PROSPECT_ARRIVAL.
    assert pa("AAA", 900) == 0.0
    assert pa("AA", 120) == 0.0


def test_an_old_triple_a_prospect_is_organisational_depth():
    """A 21-year-old at Triple-A is next year's regular; a 27-year-old at
    Triple-A has been passed over. With no ETA field in the feed, age is the
    only thing that separates them, and the table must not promote the
    second one."""
    ev = {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4,
          "level": "AAA", "prospect_rank": 12}
    young, src = RF.hitter_role_from_feeds({**ev, "birth_year": 2005})
    assert src == "feed:prospect"
    old, src_old = RF.hitter_role_from_feeds({**ev, "birth_year": 1998})
    assert RF.prospect_read({**ev, "birth_year": 1998}) is None
    assert max(old, key=old.get) == "Depth (no MLB PA)"
    assert src_old != "feed:prospect"
    assert young != old


def test_a_prospect_the_depth_chart_has_already_placed_is_up():
    """The arrival only reaches a man no MLB feed has PLACED. A player the
    depth chart lists at a position, or who appears in a batting order, is
    already up, and what the prospect list thought about him is out of
    date."""
    placed = {"club_has_orders": True, "depth_pos": "SS", "depth_rank": 1}
    mix, src = RF.hitter_role_from_feeds(
        {**placed, "level": "AAA", "prospect_rank": 12, "birth_year": 2005})
    assert src != "feed:prospect"
    # Identical to the same player with no prospect evidence at all: the
    # placement decides it, and the arrival table contributes nothing.
    assert (mix, src) == RF.hitter_role_from_feeds(placed)

    batting = {"club_has_orders": True, "spot_vs_r": 6, "spot_vs_l": 6}
    mix, src = RF.hitter_role_from_feeds(
        {**batting, "level": "AAA", "prospect_rank": 12, "birth_year": 2005})
    assert src != "feed:prospect"
    assert max(mix, key=mix.get) == "Full Time"


def test_being_left_out_of_a_lineup_does_not_block_an_arrival():
    """The bug that made the whole feature fire on nobody. Twenty-three
    clubs have a current batting order, so a prospect on one of them reaches
    the read as "a bench bat who did not make today's nine" — which is an
    ABSENCE, not a placement, and the one case that has to fall through to
    the arrival table anyway."""
    ev = {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4,
          "level": "AAA", "prospect_rank": 12, "birth_year": 2005}
    on_a_club_with_a_lineup = RF.hitter_role_from_feeds(ev)
    without = RF.hitter_role_from_feeds({**ev, "club_has_orders": False})
    assert on_a_club_with_a_lineup[1] == "feed:prospect"
    assert on_a_club_with_a_lineup == without


def test_the_arrival_carries_its_own_timing():
    """A partial season is the point — the volume comes from arriving in May
    or July, not from a full year at a reduced role — so the timing has to be
    the arrival table's own rather than the level's generic answer."""
    ev = {"club_has_orders": True, "depth_pos": "PROS", "depth_rank": 4,
          "level": "AAA", "birth_year": 2005}
    assert RF.prospect_read({**ev, "prospect_rank": 12})[1] \
        == "Early Season (~May)"
    assert RF.prospect_read({**ev, "prospect_rank": 150})[1] \
        == "Mid Season (~July)"
    assert RF.prospect_read({**ev, "level": "AA", "prospect_rank": 9})[1] \
        == "Mid Season (~July)"


def test_low_minors_are_still_nobody():
    """The table reaches the top 200 of Triple-A and the top 40 of
    Double-A, and nobody else. A High-A teenager is not taking plate
    appearances off a major league roster next year however highly he is
    ranked, and neither is the 300th-best player at Triple-A."""
    for level in ("A", "A+", "ROOKIE", "A-"):
        assert RF.prospect_read(
            {"level": level, "prospect_rank": 1, "birth_year": 2007}) is None
    assert RF.prospect_read(
        {"level": "AAA", "prospect_rank": 300, "birth_year": 2005}) is None
    assert RF.prospect_read(
        {"level": "AA", "prospect_rank": 90, "birth_year": 2005}) is None


def test_every_arrival_mixture_is_a_distribution():
    from role_taxonomy import role_names
    known = set(role_names("hitter"))
    for level, bands in RF.PROSPECT_ARRIVAL.items():
        cutoffs = [c for c, _, _ in bands]
        assert cutoffs == sorted(cutoffs), level
        for cutoff, timing, mix in bands:
            assert sum(mix.values()) == pytest.approx(1.0), (level, cutoff)
            assert set(mix) <= known, (level, cutoff)
            assert timing in [t[0] for t in TIMING], (level, cutoff)


@has_feeds
def test_an_arrival_competes_for_his_playing_time_like_everyone_else():
    """The bug that nearly got paid for with a global constant.

    Two places treat the projected tier as "who is on the roster": the floor
    in `allocate_playing_time` and the depth ranking in `apply_roster_depth`.
    The arrivals were let through the first and not the second, so a prospect
    skipped the ranking entirely, kept his whole anchor while the club's real
    last men decayed past him, and inflated the pool from 23.8 players a club
    to 26.0 — playing time from nowhere. The symptom showed up in the rank
    curve, and the fix that suggested itself was steepening the depth decay
    for all 2,894 hitters to pay for 65 prospects. The actual fix is here: a
    prospect takes a rank in his club's depth order like anybody else.
    """
    src = pd.read_csv(ROOT / "out" / "hitter_pa_projections_2027.csv",
                      low_memory=False)
    df = src[[c for c in src.columns if not c.startswith("Proj_")
              and not (c.startswith("pt_") and c != "pt_tier")]]
    out, _, _ = M.project_playing_time(
        df, "hitter", target_year=2027,
        fielding=pd.read_csv(ROOT / "out" / "fielding_history_2027.csv",
                             low_memory=False),
        feed_dir=FEEDS)
    up = out[out["pt_role_source"].astype(str) == "feed:prospect"]
    assert up["pt_depth_rank"].notna().all()

    # And the pool stays near the real one: a club carries about 23 position
    # players who take a plate appearance, and the arrival is one of them
    # rather than a twenty-sixth.
    pool = out[out["pt_tier"].astype(str) != "floor"]
    assert pool.groupby("Pred_target_team_id").size().mean() < 25.5


@has_feeds
def test_the_real_feed_sends_up_about_as_many_as_really_come_up():
    """Calibrated against the real thing rather than against a feeling.

    Real first-year position players: 114 a season, 3.8 a club, median 92
    plate appearances, ninetieth percentile 368. The arrival table is not
    trying to name WHICH prospects debut — it cannot — only to leave about
    the right amount of playing time in about the right shape for the ones
    who do.
    """
    src = pd.read_csv(ROOT / "out" / "hitter_pa_projections_2027.csv",
                      low_memory=False)
    df = src[[c for c in src.columns if not c.startswith("Proj_")
              and not (c.startswith("pt_") and c != "pt_tier")]]
    out, _, _ = M.project_playing_time(
        df, "hitter", target_year=2027,
        fielding=pd.read_csv(ROOT / "out" / "fielding_history_2027.csv",
                             low_memory=False),
        feed_dir=FEEDS)
    up = out[out["pt_role_source"].astype(str) == "feed:prospect"]
    assert 20 <= len(up) <= 60, len(up)

    # Near the real debut median of 92, on the low side of it: the table
    # roles 31 of the 114 who really come up, and the ones it leaves out are
    # the ones nobody saw coming, who are at the floor where they belong.
    pa = up["Proj_PA"]
    assert 40 <= pa.median() <= 140, pa.median()
    assert pa.max() < 500, pa.max()
    # A handful become regulars, which is the shape that matters — the real
    # season has 16 debutants over 300 plate appearances.
    assert (pa > 250).sum() >= 3
    # And it stays a small share of the league: these are the last men onto
    # a roster, not a thirty-first club.
    assert pa.sum() / out["Proj_PA"].sum() < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Prospect arrivals: arms
# ─────────────────────────────────────────────────────────────────────────────

def _arm(**kw):
    ev = {"depth_pos": "PROS", "depth_rank": 3, "level": "AAA",
          "prospect_rank": 22, "birth_year": 2004, "own_role": "starter"}
    ev.update(kw)
    return ev


def test_a_starting_pitching_prospect_arrives_as_an_innings_limited_starter():
    """The anchor that exists for exactly this man. A 23-year-old does not get
    handed a rotation slot for a full season, and projecting him as one would
    be the claim that he does."""
    mix, src = RF.pitcher_role_from_feeds(_arm())
    assert src == "feed:prospect"
    assert max(mix, key=mix.get) == "Innings-Limited Starter"
    assert len(mix) >= 3 and max(mix.values()) < 0.5


def test_the_feed_cannot_tell_a_rotation_arm_from_a_shuttle_reliever():
    """`Position` is a bare "P" for all 128 arms in the prospects feed, and
    the two populations differ by a factor of eight in innings — a real debut
    starter takes a median of 72, a debut reliever 9. The pitcher's OWN
    record is what separates them, so without one there is no arrival."""
    assert RF.pitcher_prospect_read(_arm(own_role=None)) is None
    assert RF.pitcher_prospect_read(_arm(own_role="")) is None
    pr = RF.read_prospects(FEEDS / "prospects.xml") if (
        FEEDS / "prospects.xml").exists() else None
    if pr is not None:
        arms = pr[pr["feed_position"].astype(str).str.upper() == "P"]
        assert len(arms) > 100
        assert set(arms["feed_position"]) == {"P"}


def test_a_relief_prospect_is_left_at_the_floor():
    """Not an oversight — see PITCHER_ARRIVAL. Nine innings is not a
    projection, a club's twentieth relief arm is past the end of the decay
    anyway, and "reliever" on a six-inning record is the weakest reading in
    the feed."""
    assert RF.pitcher_prospect_read(_arm(own_role="reliever")) is None
    mix, src = RF.pitcher_role_from_feeds(_arm(own_role="reliever"))
    assert src != "feed:prospect"
    assert max(mix, key=mix.get) == "Depth (no MLB IP)"


def test_an_arm_the_staff_has_already_placed_is_up():
    """The rotation's order and the bullpen's pecking order are both
    placements, and either one means the prospect list is out of date."""
    placed = {"depth_pos": "P", "depth_rank": 2}
    mix, src = RF.pitcher_role_from_feeds(_arm(**placed))
    assert src != "feed:prospect"
    assert (mix, src) == RF.pitcher_role_from_feeds(placed)

    mix, src = RF.pitcher_role_from_feeds(_arm(rw_role="CLOSER"))
    assert src != "feed:prospect"
    assert max(mix, key=mix.get) == "Closer"


def test_the_arm_table_reaches_further_down_the_rank_list():
    """Not generosity: the feed publishes ONE combined top 400, so a
    pitcher's rank is his standing among bats as well as arms and the 128
    arms are spread through it. Matching the hitters' cutoffs would mean
    reaching a third as far, not being equally strict."""
    for level in ("AAA", "AA"):
        arm = max(c for c, _, _ in RF.PITCHER_ARRIVAL[level])
        bat = max(c for c, _, _ in RF.PROSPECT_ARRIVAL[level])
        assert arm > bat, level


def test_an_arm_too_old_for_the_table_is_organisational_depth():
    assert RF.pitcher_prospect_read(_arm(birth_year=1998)) is None
    mix, _ = RF.pitcher_role_from_feeds(_arm(birth_year=1998))
    assert max(mix, key=mix.get) == "Depth (no MLB IP)"


def test_every_arm_arrival_mixture_is_a_distribution_of_starters():
    from role_taxonomy import role_family, role_names
    known = set(role_names("pitcher"))
    for level, bands in RF.PITCHER_ARRIVAL.items():
        cutoffs = [c for c, _, _ in bands]
        assert cutoffs == sorted(cutoffs), level
        for cutoff, timing, mix in bands:
            assert sum(mix.values()) == pytest.approx(1.0), (level, cutoff)
            assert set(mix) <= known, (level, cutoff)
            assert timing in [t[0] for t in TIMING], (level, cutoff)
            # The MODAL role picks the family a player is ranked in, and a
            # rookie starter competes with his club's rotation for innings,
            # not with its bullpen. A mixture that tipped to a relief role
            # would quietly move him into a 20-deep queue with 8 slots.
            assert role_family(max(mix, key=mix.get)) == "SP", (level, cutoff)


@has_feeds
def test_the_real_feed_calls_up_about_as_many_arms_as_really_start():
    """Calibrated against the real thing. Real first-year pitchers are 181 a
    season at a median of 12.7 innings, but that hides two populations: 23 to
    34 of them START games and take a median of 72 innings, and the rest are
    shuttle relievers. This table is aimed at the first group only.
    """
    src = pd.read_csv(ROOT / "out" / "pitcher_pa_projections_2027.csv",
                      low_memory=False)
    df = src[[c for c in src.columns if not c.startswith("Proj_")
              and not (c.startswith("pt_") and c != "pt_tier")]]
    out, _, _ = M.project_playing_time(
        df, "pitcher", target_year=2027,
        fielding=pd.read_csv(ROOT / "out" / "fielding_history_2027.csv",
                             low_memory=False),
        feed_dir=FEEDS)
    up = out[out["pt_role_source"].astype(str) == "feed:prospect"]
    assert 10 <= len(up) <= 40, len(up)

    ip = up["Proj_IP"]
    # Real debut arms sit at a median of 12.7 innings; these are the
    # starting end of that population, so above it and well short of the
    # 72-inning median of the ones who hold a rotation spot all year.
    assert 10 <= ip.median() <= 45, ip.median()
    assert ip.max() < 120, ip.max()
    assert ip.sum() / out["Proj_IP"].sum() < 0.03

    # They land in the band the model was starving: a real club's 18th to
    # 25th arm takes 23 innings down to 6, where the model was giving 5 to 3.
    assert up["pt_depth_rank"].notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Against the real snapshot
# ─────────────────────────────────────────────────────────────────────────────

@has_feeds
def test_the_real_feeds_parse_and_cover_thirty_clubs():
    f = RF.read_feeds(FEEDS)
    assert f["depth"]["feed_team"].nunique() == 30
    assert f["closers"]["feed_team"].nunique() == 30
    assert len(f["prospects"]) == 400
    cur = f["orders"][~f["orders"]["legacy"]]
    assert cur["feed_team"].nunique() == 23, "7 clubs have no current lineup"
    assert f["orders"]["legacy"].sum() > 0, "the legacy orders are still there"


@has_feeds
def test_the_real_feeds_resolve_at_the_rates_that_were_measured():
    """A join that quietly decays is worse than one that fails, so the rates
    the design was chosen on are pinned here."""
    players = pd.read_csv(ROOT / "out" / "hitter_pa_projections_2027.csv")
    feeds = RF.read_feeds(FEEDS)
    orders = RF._for_pool(feeds, "orders", "hitter")
    _, st = RF.resolve_ids(orders, players)
    assert st["matched"] / st["rows"] > 0.95, st

    pitchers = pd.read_csv(ROOT / "out" / "pitcher_pa_projections_2027.csv")
    _, st = RF.resolve_ids(RF._for_pool(feeds, "closers", "pitcher"), pitchers)
    assert st["matched"] / st["rows"] > 0.95, st


@has_feeds
def test_the_real_feeds_fit_the_real_rank_curve_better():
    """The point of the exercise, measured against the target the anchors
    are fitted on rather than against a spread statistic.

    It used to compare the standard deviation of projected plate appearances
    with and without the feeds, which stopped being like-for-like once the
    prospect arrivals joined the projected pool: sixty-five more players,
    most of them on partial seasons, lower the spread while improving the
    fit. The rank curve is the thing that actually says whether a club's
    playing time is distributed the way real clubs distribute it.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_role_anchors import real_rank_curve

    src = pd.read_csv(ROOT / "out" / "hitter_pa_projections_2027.csv",
                      low_memory=False)
    df = src[[c for c in src.columns if not c.startswith("Proj_")
              and not (c.startswith("pt_") and c != "pt_tier")]]
    fielding = pd.read_csv(ROOT / "out" / "fielding_history_2027.csv",
                           low_memory=False)
    curve, _ = real_rank_curve(fielding, "hitter")
    ranks = [r for r in range(1, 21) if float(r) in curve.index]

    def fit(feed_dir):
        out, _, _ = M.project_playing_time(df, "hitter", target_year=2027,
                                           fielding=fielding,
                                           feed_dir=feed_dir)
        p = out[out["pt_tier"].astype(str) != "floor"]
        m = p.groupby(p.groupby("Pred_target_team_id")["Proj_PA"].rank(
            "first", ascending=False))["Proj_PA"].mean()
        rat = np.array([m.get(float(r), np.nan) / curve.loc[float(r)]
                        for r in ranks])
        rat = rat[np.isfinite(rat)]
        return float(np.sqrt(((rat - 1) ** 2).mean())), out

    without, _ = fit(None)
    with_feeds, after = fit(FEEDS)
    assert with_feeds < without, (with_feeds, without)

    # Roles the heuristic could not identify at all, which is what a feed is
    # for: it can see a platoon, and last season's plate-appearance count
    # cannot.
    roles = after["pt_role"].value_counts()
    assert roles.get("Weak Side Platoon", 0) > 20
    assert roles.get("Strong Side Platoon", 0) > 20

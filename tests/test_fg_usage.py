"""FanGraphs projected usage as role evidence."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import role_feeds as RF                                   # noqa: E402


def test_hitter_roles_follow_the_anchors():
    f = lambda pa, pos="": RF.fg_hitter_role({"fg_pa": pa}, pos)
    assert f(640) == "Full Time"
    assert f(600, "DH") == "Everyday DH / 1B-DH"
    assert f(430) == "Strong Side Platoon"
    assert f(250) == "Weak Side Platoon"
    assert f(150) == "Bench Bat"
    assert f(60) == "Injury Replacement / 26th Man"
    assert f(10) is None                                    # not describing a job
    assert f(520, "C") == "Catcher - Primary"
    assert f(300, "C") == "Catcher - Tandem"
    assert f(150, "C") == "Catcher - Backup"
    assert f(450, float("nan")) == "Strong Side Platoon"   # a missing position is fine


def test_pitcher_roles_follow_innings_and_leverage():
    f = lambda **k: RF.fg_pitcher_role(
        {"fg_ip": 0, "fg_gs": 0, "fg_g": 60, "fg_sv": 0, "fg_hld": 0, **k})
    assert f(fg_ip=190, fg_gs=31, fg_g=31) == "Ace (SP1)"
    assert f(fg_ip=150, fg_gs=28, fg_g=28) == "Mid-Rotation Starter (SP2-3)"
    assert f(fg_ip=110, fg_gs=20, fg_g=22) == "End-of-Rotation Starter (SP4-5)"
    assert f(fg_ip=70, fg_gs=12, fg_g=13) == "Innings-Limited Starter"
    assert f(fg_ip=65, fg_sv=30) == "Closer"
    assert f(fg_ip=65, fg_hld=22) == "Late Inning RP (Setup)"
    assert f(fg_ip=60) == "Middle Relief"
    assert f(fg_ip=25) == "Bullpen Depth Arm"
    assert f(fg_ip=5) is None


def test_blend_only_widens_where_the_feeds_disagree():
    mix = {"Full Time": 0.85, "Strong Side Platoon": 0.15}
    same, how = RF.blend_fg(mix, "Full Time", "hitter")
    assert how == "agree" and same == mix                   # nothing moves
    out, how = RF.blend_fg(mix, "Bench Bat", "hitter", weight=0.4)
    assert how == "blend"
    assert out["Bench Bat"] == pytest.approx(0.4)
    assert out["Full Time"] == pytest.approx(0.51)
    assert sum(out.values()) == pytest.approx(1.0)
    alone, how = RF.blend_fg(None, "Bench Bat", "hitter")
    assert how == "alone" and alone["Bench Bat"] == pytest.approx(RF.FG_ALONE)
    assert RF.blend_fg(mix, None, "hitter") == (mix, "")


def test_read_fg_usage_joins_on_mlbam_id_and_tolerates_a_bom(tmp_path):
    p = tmp_path / "fg_hitters.csv"
    p.write_text("﻿Name,Team,MLBAMID,G,PA\n"
                 '"A","X",111,150,640.5\n"B","Y",,100,400\n"A2","X",111,10,5\n',
                 encoding="utf-8")
    got = RF.read_fg_usage(p, "hitter")
    assert got == {111: {"fg_pa": 640.5, "fg_g": 150.0}}    # duplicate id: the bigger row
    assert RF.read_fg_usage(tmp_path / "nope.csv", "hitter") == {}
    q = tmp_path / "bad.csv"
    q.write_text("x,y\n1,2\n")
    assert RF.read_fg_usage(q, "hitter") == {}


def _players(n=2):
    return pd.DataFrame({
        "PlayerId": [1, 2][:n], "Name": ["a", "b"][:n],
        "Pred_target_team_id": [147] * n, "pt_position": ["SS", "SS"][:n],
        "pt_role": ["Bench Bat"] * n, "pt_role_mix": [""] * n,
        "pt_role_start": ["Opening Day"] * n, "pt_role_source": ["default"] * n})


def test_fg_is_the_only_evidence_for_a_player_no_other_feed_covers():
    out, st = RF.apply_feed_roles(
        _players(1), "hitter", feeds={"depth": pd.DataFrame()},
        fg={1: {"fg_pa": 600.0, "fg_g": 150.0}})
    assert out["pt_role"].iloc[0] == "Full Time"
    assert out["pt_role_source"].iloc[0] == "feed:fg"
    assert st["fg"]["alone"] == 1


def test_no_fg_files_means_no_change():
    a, sa = RF.apply_feed_roles(_players(1), "hitter",
                                feeds={"depth": pd.DataFrame()}, fg={})
    assert sa.get("note")                                   # nothing to apply
    assert a["pt_role"].iloc[0] == "Bench Bat"              # the heuristic stands


def test_explicit_feeds_never_pick_up_files_from_disk():
    # A caller that hands in its own feeds gets exactly those: FG is read only
    # when the feeds themselves came off disk.
    out, st = RF.apply_feed_roles(_players(1), "hitter",
                                  feeds={"depth": pd.DataFrame()})
    assert st.get("note") and out["pt_role"].iloc[0] == "Bench Bat"


def test_a_near_zero_line_is_evidence_for_a_real_role_but_not_for_the_floor():
    real = _players(1)
    real["pt_role"] = "Everyday DH / 1B-DH"                  # the heuristic default
    out, st = RF.apply_feed_roles(real, "hitter",
                                  feeds={"depth": pd.DataFrame()},
                                  fg={1: {"fg_pa": 17.0, "fg_g": 12.0}})
    assert st["fg"]["alone"] == 1
    assert out["pt_role"].iloc[0] == "Injury Replacement / 26th Man"

    floor = _players(1)
    floor["pt_role"] = "Depth (no MLB PA)"
    out, st = RF.apply_feed_roles(floor, "hitter",
                                  feeds={"depth": pd.DataFrame()},
                                  fg={1: {"fg_pa": 1.0, "fg_g": 1.0}})
    assert st["fg"]["alone"] == 0                            # left at the floor
    assert out["pt_role"].iloc[0] == "Depth (no MLB PA)"


def test_the_weighting_is_off_and_files_on_disk_are_ignored(tmp_path):
    assert RF.FG_WEIGHT == {"hitter": 0.0, "pitcher": 0.0}
    (tmp_path / "fg_hitters.csv").write_text("MLBAMID,PA,G\n1,640,150\n")
    for name in ("depth", "orders", "closers", "prospects"):
        (tmp_path / f"{name}.xml").write_text("<x/>")
    out, st = RF.apply_feed_roles(_players(1), "hitter", feed_dir=tmp_path)
    assert out["pt_role"].iloc[0] == "Bench Bat"            # FG did not speak
    assert not st.get("fg", {}).get("rows")


def test_a_saved_role_file_reproduces_the_run_it_was_made_from():
    """The saved files are the product of the FanGraphs run. Applying one must
    put a player in the same role, mixture and timing, and must not change
    whether a prospect is an arrival."""
    import playing_time_model as M
    p = _players(1)
    p["pt_role_source"] = "feed:prospect"
    ov = pd.DataFrame({"PlayerId": [1], "Role": ["Full Time"],
                       "Role Mix": ["Full Time:0.6|Bench Bat:0.4"],
                       "Role Start": ["Mid Season (~July)"]})
    out, st = M.apply_role_overrides(p, ov, kind="hitter")
    assert out["pt_role"].iloc[0] == "Full Time"
    assert out["pt_role_start"].iloc[0] == "Mid Season (~July)"
    assert out["pt_role_mix"].iloc[0] == "Full Time:0.6|Bench Bat:0.4"
    assert bool(M.arriving(out).iloc[0])                     # still an arrival


def test_the_saved_role_files_are_well_formed():
    """The two files in rosters/ are what sets those players' roles now."""
    import playing_time_model as M
    from role_taxonomy import parse_role_mix, role_names
    for kind, plural in (("hitter", "hitters"), ("pitcher", "pitchers")):
        path = ROOT / "rosters" / f"player_roles_{plural}_2027.csv"
        assert path.exists(), path
        raw = pd.read_csv(path)
        ov = M.load_role_overrides(path, kind)
        assert len(raw) > 500
        assert raw["PlayerId"].is_unique
        assert len(ov) == len(raw)                       # nothing dropped as unreadable
        assert ov["Role"].notna().all() and ov["Role Mix"].notna().all()
        assert set(raw["Role"]) <= set(role_names(kind))  # no typos
        for mix in ov["Role Mix"]:
            parsed = parse_role_mix(mix, kind)
            assert parsed and abs(sum(parsed.values()) - 1.0) < 1e-6, mix


def test_prospects_run_down_the_depth_queue_more_slowly():
    import playing_time_model as M
    rows = []
    for i in range(1, 15):                       # one club's bench family
        rows.append(dict(PlayerId=i, Pred_target_team_id=147, pt_tier="projected",
                         pt_role="Bench Bat", pt_role_source="default",
                         pt_role_start="Opening Day", pt_raw=200.0 - i,
                         mle_source="MiLB" if i == 14 else "", Age=24 if i >= 13 else 31,
                         Career_PA=50.0 if i >= 13 else 2000.0))
    df = pd.DataFrame(rows)
    out = M.apply_roster_depth(df, "hitter")
    mask = M._prospect_mask(out)
    assert list(mask[out.PlayerId.isin([13, 14])]) == [True, True]
    assert not mask[out.PlayerId == 1].iloc[0]
    # same rank, same distance past the core: the prospect keeps more
    a = out.loc[out.PlayerId == 13, "pt_depth_factor"].iloc[0]
    M.PROSPECT_DECAY_EXPONENT = 1.0
    try:
        b = M.apply_roster_depth(df, "hitter").loc[lambda d: d.PlayerId == 13,
                                                   "pt_depth_factor"].iloc[0]
    finally:
        M.PROSPECT_DECAY_EXPONENT = 0.6
    assert a > b

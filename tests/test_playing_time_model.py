"""
Tests for the baseline playing-time model.

The properties here are the ones the season engine cannot recover if the model
gets them wrong, plus the three bugs the first real run of this model exposed:

  * ALL 30 "closers" were floor-tier minor leaguers. Staff ranking included the
    MLE-translated tier, whose shrunk RA9 beats every real reliever, so a
    Double-A arm took rank 1 on every club and the actual bullpen was pushed
    past rank 6 into the depth role.
  * 144 PROJECTED-tier hitters were assigned the depth role, which routes a
    player to the 1 PA floor — contradicting the tier they had already cleared.
  * Every player in a role received an IDENTICAL number. Aaron Judge, Ben Rice,
    Ryan McMahon, Trent Grisham and Heliot Ramos all projected 418.1 PA.

Run with:  python -m pytest tests/test_playing_time_model.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import playing_time_model as M  # noqa: E402
from playing_time import TIER_FLOOR, TIER_PROJECTED  # noqa: E402
from role_taxonomy import (  # noqa: E402
    DEPTH_HITTER_ROLE, DEPTH_PITCHER_ROLE, primary_positions, role_anchor,
    role_names, suggest_hitter_role, timing_share,
)

NYY, LAD = 147, 119


def _hitters(n_per_team=20, teams=(NYY, LAD), pa=None):
    rows = []
    pid = 1
    for t in teams:
        for i in range(n_per_team):
            vol = pa[i] if pa else max(30.0, 700.0 - i * 35.0)
            rows.append({"PlayerId": pid, "Name": f"h{pid}",
                         "Pred_target_team_id": t, "pt_tier": TIER_PROJECTED,
                         "evidence_volume": vol, "Last_PA": vol,
                         "BatSide": "R"})
            pid += 1
    return pd.DataFrame(rows)


def _pitchers(n_per_team=18, teams=(NYY, LAD)):
    rows = []
    pid = 1000
    for t in teams:
        for i in range(n_per_team):
            starter = i < 5
            rows.append({
                "PlayerId": pid, "Name": f"p{pid}",
                "Pred_target_team_id": t, "pt_tier": TIER_PROJECTED,
                "role": "starter" if starter else "reliever",
                "weighted_IP_per_G": 6.0 - i * 0.5 if starter else 1.0,
                "RA9": 3.0 + i * 0.15, "TBF_per_IP": 4.3,
                "evidence_volume": (700.0 - i * 60.0) if starter else 250.0,
            })
            pid += 1
    return pd.DataFrame(rows)


def _run(df, kind, **kw):
    return M.project_playing_time(df, kind, target_year=2027, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# closure — the constraints the season engine depends on
# ─────────────────────────────────────────────────────────────────────────────

def test_every_team_closes_on_its_pa_budget():
    out, _, _ = _run(_hitters(), "hitter")
    tot = out.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert np.allclose(tot.to_numpy(), M.TEAM_PA_BUDGET), tot.to_dict()


def test_every_team_closes_on_its_ip_budget():
    out, _, _ = _run(_pitchers(), "pitcher")
    tot = out.groupby("Pred_target_team_id")["Proj_IP"].sum()
    assert np.allclose(tot.to_numpy(), M.TEAM_IP_BUDGET), tot.to_dict()


def test_league_closure_follows_from_team_closure():
    out, _, _ = _run(_hitters(teams=(NYY, LAD, 111)), "hitter")
    assert out["Proj_PA"].sum() == pytest.approx(M.TEAM_PA_BUDGET * 3)


def test_the_floor_tier_does_not_take_playing_time_from_the_roster():
    """Floor players are held at 1.0 and sit OUTSIDE the budget.

    This test used to assert the opposite — that their total came out of the
    budget so the club's rows summed to exactly the budget. That made the
    club's real players share `budget - n_floor`, and how many floor players
    a club carries is a fact about how deep the minor-league feed went for
    that organization, not about the club. On the shipped projections it
    ranged from 85 to 136 pitchers, so clubs were handed 5.8% to 9.3% fewer
    innings than their budget with a 3.5-point spread between them that
    tracked nothing but data coverage.

    `team_context.roster_volume_weights` had it right all along: the 1-PA
    floor exists so a player is present, ranked and joinable, "not so he
    takes playing time away from the major-league roster."
    """
    df = _hitters()
    # Floor SOME of each club, not all of one: a team with no projected
    # players has nobody to allocate to and correctly cannot reach the budget.
    df.loc[df.groupby("Pred_target_team_id").head(6).index, "pt_tier"] = TIER_FLOOR
    out, _, _ = _run(df, "hitter")

    projected = out[out.pt_tier != TIER_FLOOR]
    tot = projected.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert np.allclose(tot.to_numpy(), M.TEAM_PA_BUDGET), \
        "the players who will actually bat get the whole budget"

    floor = out[out.pt_tier == TIER_FLOOR]["Proj_PA"]
    assert (floor == 1.0).all(), "the verifier checks this exactly"

    # And the club's rows therefore sum to budget + one per floor player.
    everyone = out.groupby("Pred_target_team_id")["Proj_PA"].sum()
    n_floor = out[out.pt_tier == TIER_FLOOR].groupby(
        "Pred_target_team_id").size()
    assert np.allclose(everyone.to_numpy(),
                       M.TEAM_PA_BUDGET + n_floor.reindex(everyone.index)
                       .fillna(0).to_numpy())


def test_a_deeper_farm_system_does_not_cost_the_major_league_roster():
    """The cross-club bias the old closure introduced, pinned directly."""
    shallow = _hitters()
    deep = _hitters()
    # Same major-league roster, one club carrying far more depth rows.
    extra = deep.head(40).copy()
    extra["PlayerId"] = range(900_000, 900_040)
    extra["pt_tier"] = TIER_FLOOR
    deep = pd.concat([deep, extra], ignore_index=True)

    s_out, _, _ = _run(shallow, "hitter")
    d_out, _, _ = _run(deep, "hitter")

    def regular_pa(frame):
        f = frame[frame.pt_tier != TIER_FLOOR]
        return f.groupby("Pred_target_team_id")["Proj_PA"].sum().max()

    assert regular_pa(d_out) == pytest.approx(regular_pa(s_out), rel=1e-9)


def test_nobody_exceeds_the_physical_ceiling():
    """A shallow roster must not hand one player the whole budget."""
    out, _, _ = _run(_hitters(n_per_team=3), "hitter")
    assert out["Proj_PA"].max() <= M.PT_MAX_PA + 1e-6
    out, _, _ = _run(_pitchers(n_per_team=3), "pitcher")
    assert out["Proj_IP"].max() <= M.PT_MAX_IP + 1e-6


def test_a_shallow_roster_still_closes_despite_the_ceiling():
    """The clip redistributes rather than discarding the surplus."""
    out, diag, _ = _run(_hitters(n_per_team=10), "hitter")
    tot = out.groupby("Pred_target_team_id")["Proj_PA"].sum()
    # With 10 players and a 760 ceiling the budget is reachable (7,600 > 6,156).
    assert np.allclose(tot.to_numpy(), M.TEAM_PA_BUDGET)


def test_no_player_is_left_without_a_volume():
    for kind, df in (("hitter", _hitters()), ("pitcher", _pitchers())):
        out, _, _ = _run(df, kind)
        col = "Proj_PA" if kind == "hitter" else "Proj_IP"
        assert out[col].notna().all()
        assert (out[col] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# reserves — a team we expect to sign someone must be under-projected
# ─────────────────────────────────────────────────────────────────────────────

def test_a_reserve_leaves_room_for_an_unsigned_player():
    """A reserve is room FOR SOMEBODY, so it is sized by who is unsigned.

    With a free agent in the set, the share says which club gives the
    playing time up and the player's own role says how much — see
    `free_agent_pool`. NYY asked to be the one that pays, so NYY is the one
    that falls short.
    """
    from team_context import FREE_AGENT_TEAM_ID
    df = _hitters()
    df.loc[df.index[:3], "Pred_target_team_id"] = FREE_AGENT_TEAM_ID
    out, diag, _ = _run(df, "hitter", reserves={NYY: {"pa_share": 0.10}})
    tot = out[out.Pred_target_team_id > 0].groupby(
        "Pred_target_team_id")["Proj_PA"].sum()
    fa = out[out.Pred_target_team_id == FREE_AGENT_TEAM_ID]["Proj_PA"].sum()
    assert tot[NYY] == pytest.approx(M.TEAM_PA_BUDGET - fa)
    assert tot[LAD] == pytest.approx(M.TEAM_PA_BUDGET), "untouched team"


def test_a_reserve_with_nobody_to_take_it_is_not_held_back():
    """Playing time reserved for a signing that is not in the data goes
    nowhere: the clubs would fall short and no player would gain, which is
    just a league 10% smaller than the one being projected. Declaring the
    reserve says WHO pays for free agency, and there is no free agency here.

    `FREE_AGENT_PLAYING_TIME = "pool"` is where the literal reading lives,
    for anyone who does want the clubs to hold a spot open.
    """
    out, _, _ = _run(_hitters(), "hitter", reserves={NYY: {"pa_share": 0.10}})
    tot = out.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert tot[NYY] == pytest.approx(M.TEAM_PA_BUDGET)
    assert tot[LAD] == pytest.approx(M.TEAM_PA_BUDGET)


def test_pool_mode_holds_the_declared_reserve_open(monkeypatch):
    monkeypatch.setattr(M, "FREE_AGENT_PLAYING_TIME", "pool")
    out, _, _ = _run(_hitters(), "hitter", reserves={NYY: {"pa_share": 0.10}})
    tot = out.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert tot[NYY] == pytest.approx(M.TEAM_PA_BUDGET * 0.90)
    assert tot[LAD] == pytest.approx(M.TEAM_PA_BUDGET), "untouched team"


def test_pitcher_reserves_are_independent_of_hitter_reserves():
    out, _, _ = _run(_pitchers(), "pitcher",
                     reserves={NYY: {"pa_share": 0.20}})
    tot = out.groupby("Pred_target_team_id")["Proj_IP"].sum()
    assert tot[NYY] == pytest.approx(M.TEAM_IP_BUDGET), (
        "a team shopping for a bat is not shopping for innings")


# ─────────────────────────────────────────────────────────────────────────────
# the three bugs the first run exposed
# ─────────────────────────────────────────────────────────────────────────────

def test_exactly_one_closer_per_team():
    out, _, _ = _run(_pitchers(), "pitcher")
    per_team = out[out.pt_role == "Closer"].groupby("Pred_target_team_id").size()
    assert set(per_team.to_dict().values()) == {1}, per_team.to_dict()


def test_floor_tier_arms_cannot_take_the_closer_job():
    """THE bug: 30 floor-tier 'closers', one per club.

    An MLE-translated arm shrunk toward the league mean can out-rank every
    real reliever on the staff. Ranking must be a standing among players who
    will actually pitch.
    """
    df = _pitchers()
    # A floor-tier arm with the best RA9 in the organisation.
    df = pd.concat([df, pd.DataFrame([{
        "PlayerId": 9999, "Name": "AAA phenom", "Pred_target_team_id": NYY,
        "pt_tier": TIER_FLOOR, "role": "reliever", "RA9": 0.5,
        "weighted_IP_per_G": 1.0, "TBF_per_IP": 4.3, "evidence_volume": 40.0,
    }])], ignore_index=True)
    out, _, _ = _run(df, "pitcher")
    phenom = out[out.PlayerId == 9999].iloc[0]
    assert phenom["pt_role"] == DEPTH_PITCHER_ROLE
    assert phenom["Proj_IP"] == 1.0
    closers = out[out.pt_role == "Closer"]
    assert (closers["pt_tier"] == TIER_PROJECTED).all()


def test_a_projected_hitter_is_never_given_the_depth_role():
    """The depth role routes to the 1 PA floor, contradicting the tier."""
    row = {"pt_tier": TIER_PROJECTED, "evidence_volume": 26.0}
    assert suggest_hitter_role(row, reference=700.0) != DEPTH_HITTER_ROLE
    out, _, _ = _run(_hitters(pa=[40.0] * 20), "hitter")
    proj = out[out.pt_tier == TIER_PROJECTED]
    assert not (proj["pt_role"] == DEPTH_HITTER_ROLE).any()
    assert (proj["Proj_PA"] > 1.0).all()


def test_a_floor_hitter_does_get_the_depth_role():
    assert suggest_hitter_role({"pt_tier": TIER_FLOOR}, 700.0) == \
        DEPTH_HITTER_ROLE


def test_players_in_the_same_role_are_differentiated():
    """THE other bug: five Full Time hitters at identical 418.1 PA."""
    out, _, _ = _run(_hitters(), "hitter")
    ft = out[out.pt_role == "Full Time"]["Proj_PA"]
    assert ft.nunique() > 1, "a role must not flatten its players"
    assert ft.max() / ft.min() > 1.05


def test_differentiation_follows_the_evidence():
    # Enough players that the ceiling does not bind and mask the difference.
    df = _hitters(n_per_team=14, teams=(NYY,))
    df.loc[0, "evidence_volume"] = 700.0
    df.loc[1, "evidence_volume"] = 350.0
    df.loc[[0, 1], "Last_PA"] = df.loc[[0, 1], "evidence_volume"]
    out, _, _ = _run(df, "hitter")
    assert out.loc[0, "Proj_PA"] > out.loc[1, "Proj_PA"]
    assert out.loc[0, "Proj_PA"] < M.PT_MAX_PA, "ceiling would mask the test"


def test_a_team_with_no_projected_players_is_reported():
    """It cannot reach its budget, and that is a data problem worth naming."""
    df = _hitters()
    df.loc[df.Pred_target_team_id == NYY, "pt_tier"] = TIER_FLOOR
    out, diag, _ = _run(df, "hitter")
    row = diag[diag.team_id == NYY].iloc[0]
    assert row["n_projected"] == 0
    report = M.playing_time_report(out, diag, {"overrides": {}}, "hitter")
    assert "no projected players" in report


# ─────────────────────────────────────────────────────────────────────────────
# roster depth
# ─────────────────────────────────────────────────────────────────────────────

def test_players_beyond_their_family_core_slots_are_discounted():
    """Core slots are per FAMILY — nine in the lineup, four on the bench, six
    starters, eight relievers — not a flat 13 per club. A flat count ranked
    starters against relievers and regulars against bench bats, which is the
    wrong competition and is how a closer ended up 13th on his own staff."""
    from role_taxonomy import FAMILY_CORE_SLOTS
    out, _, _ = _run(_hitters(n_per_team=22, teams=(NYY,)), "hitter")
    ranked = out[out.pt_depth_rank.notna()]
    for fam, g in ranked.groupby("pt_family"):
        slots = FAMILY_CORE_SLOTS[fam]
        core = g[g.pt_depth_rank <= slots]
        tail = g[g.pt_depth_rank > slots]
        assert (core["pt_depth_factor"] == 1.0).all(), fam
        assert (tail["pt_depth_factor"] < 1.0).all(), fam


def test_depth_discount_never_reaches_zero():
    """Zero would delete the player from every total while keeping his row."""
    out, _, _ = _run(_hitters(n_per_team=40, teams=(NYY,)), "hitter")
    assert (out["pt_depth_factor"] >= M.ROSTER_DEPTH_FLOOR).all()
    assert (out["Proj_PA"] > 0).all()


def test_depth_ranking_ignores_the_floor_tier():
    df = _hitters(n_per_team=20, teams=(NYY,))
    df.loc[df.index[:15], "pt_tier"] = TIER_FLOOR
    out, _, _ = _run(df, "hitter")
    ranked = out[out["pt_depth_rank"].notna()]
    assert (ranked["pt_tier"] == TIER_PROJECTED).all()


# ─────────────────────────────────────────────────────────────────────────────
# overrides — every default must be replaceable
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, kind, rows):
    p = M.role_override_path(kind, 2027, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return tmp_path


def test_a_role_override_wins_over_the_default(tmp_path):
    base = _hitters()
    root = _write(tmp_path, "hitter", [{"PlayerId": 1, "Role": "Bench Bat"}])
    out, _, stats = _run(base, "hitter", roster_path=root)
    assert out[out.PlayerId == 1].iloc[0]["pt_role"] == "Bench Bat"
    assert out[out.PlayerId == 1].iloc[0]["pt_role_source"] == "override"
    assert stats["overrides"]["role"] == 1


def test_availability_and_timing_overrides_reduce_volume(tmp_path):
    base = _hitters(n_per_team=14, teams=(NYY,))
    plain, _, _ = _run(base, "hitter")
    root = _write(tmp_path, "hitter",
                  [{"PlayerId": 1, "Availability": 0.5},
                   {"PlayerId": 2, "Role Start": "Mid Season (~July)"}])
    out, _, stats = _run(base, "hitter", roster_path=root)
    assert stats["overrides"]["availability"] == 1
    assert stats["overrides"]["timing"] == 1
    # Relative to his own team-mates, each overridden player must fall.
    for pid in (1, 2):
        before = plain[plain.PlayerId == pid].iloc[0]["Proj_PA"]
        after = out[out.PlayerId == pid].iloc[0]["Proj_PA"]
        assert after < before, pid


def test_overrides_still_close_the_team():
    """An override redistributes playing time; it cannot create it."""
    base = _hitters()
    out, _, _ = M.project_playing_time(
        base, "hitter", target_year=2027,
        roster_path="does-not-exist")
    tot = out.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert np.allclose(tot.to_numpy(), M.TEAM_PA_BUDGET)


def test_an_unknown_role_name_is_reported_not_obeyed(tmp_path, capsys):
    base = _hitters()
    root = _write(tmp_path, "hitter",
                  [{"PlayerId": 1, "Role": "Designated Hitter-ish"}])
    out, _, _ = _run(base, "hitter", roster_path=root)
    assert "unknown role name" in capsys.readouterr().out
    assert out[out.PlayerId == 1].iloc[0]["pt_role"] in set(role_names("hitter"))


def test_an_override_file_matching_nobody_is_counted(tmp_path):
    root = _write(tmp_path, "hitter", [{"PlayerId": 777777, "Role": "Full Time"}])
    _, _, stats = _run(_hitters(), "hitter", roster_path=root)
    assert stats["overrides"]["unmatched"] == 1, (
        "a file that silently matches nothing is the usual failure")


def test_a_missing_override_file_is_fine():
    _, _, stats = _run(_hitters(), "hitter", roster_path="no/such/dir")
    assert stats["overrides"]["matched"] == 0


def test_an_override_file_without_playerid_is_ignored(tmp_path, capsys):
    p = M.role_override_path("hitter", 2027, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Name": "somebody", "Role": "Full Time"}]).to_csv(p, index=False)
    _run(_hitters(), "hitter", roster_path=tmp_path)
    assert "no PlayerId column" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# free agents
# ─────────────────────────────────────────────────────────────────────────────

def test_free_agents_are_excluded_from_team_closure():
    """A free agent must not take plate appearances from a club he is not on.

    He does take them from the league, though — there is only one league —
    so every club falls short by an equal slice of what the unsigned class
    holds, and the clubs stay equal to each other.
    """
    from team_context import FREE_AGENT_TEAM_ID
    df = _hitters()
    df.loc[df.index[:3], "Pred_target_team_id"] = FREE_AGENT_TEAM_ID
    out, _, _ = _run(df, "hitter")
    clubs = out[out.Pred_target_team_id > 0]
    tot = clubs.groupby("Pred_target_team_id")["Proj_PA"].sum()
    fa = out[out.Pred_target_team_id == FREE_AGENT_TEAM_ID]
    assert np.allclose(tot.to_numpy(),
                       M.TEAM_PA_BUDGET - fa["Proj_PA"].sum() / len(tot))
    assert (fa["Proj_PA"] > 1.0).all(), "still projected, just not allocated"


def test_a_player_with_no_team_still_gets_a_volume():
    df = _hitters()
    df.loc[df.index[0], "Pred_target_team_id"] = np.nan
    out, _, _ = _run(df, "hitter")
    assert out["Proj_PA"].notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# the catcher ladder — the position data finally exists
# ─────────────────────────────────────────────────────────────────────────────

def _fielding(pid=1, pos="C", innings=900.0, season=2026):
    return pd.DataFrame([{"PlayerId": pid, "Pos": pos, "Innings": innings,
                          "Season": season}])


def test_a_catcher_gets_the_catcher_ladder():
    df = _hitters()
    out, _, _ = _run(df, "hitter", fielding=_fielding(1, "C"))
    assert out[out.PlayerId == 1].iloc[0]["pt_role"].startswith("Catcher")


def test_a_non_catcher_does_not():
    df = _hitters()
    out, _, _ = _run(df, "hitter", fielding=_fielding(1, "SS"))
    assert not out[out.PlayerId == 1].iloc[0]["pt_role"].startswith("Catcher")


def test_the_catcher_anchor_is_below_the_full_time_anchor():
    """Applying Full Time to catchers over-projects all 30 of them by ~120 PA."""
    assert role_anchor("Catcher - Primary", "hitter")["pa"] < \
        role_anchor("Full Time", "hitter")["pa"]


def test_primary_position_prefers_the_most_innings():
    f = pd.DataFrame([
        {"PlayerId": 1, "Pos": "SS", "Innings": 200.0, "Season": 2026},
        {"PlayerId": 1, "Pos": "2B", "Innings": 900.0, "Season": 2026},
    ])
    assert primary_positions(f) == {1: "2B"}


def test_primary_position_uses_the_latest_season():
    f = pd.DataFrame([
        {"PlayerId": 1, "Pos": "SS", "Innings": 1200.0, "Season": 2024},
        {"PlayerId": 1, "Pos": "2B", "Innings": 400.0, "Season": 2026},
    ])
    assert primary_positions(f) == {1: "2B"}, "a move is a move"


def test_dh_is_not_a_position():
    f = pd.DataFrame([
        {"PlayerId": 1, "Pos": "DH", "Innings": 1000.0, "Season": 2026},
        {"PlayerId": 1, "Pos": "C", "Innings": 300.0, "Season": 2026},
    ])
    assert primary_positions(f) == {1: "C"}, (
        "DH says where he does not field, not what job he holds")


def test_primary_position_degrades_on_bad_input():
    for bad in (None, pd.DataFrame(), pd.DataFrame({"x": [1]})):
        assert primary_positions(bad) == {}


# ─────────────────────────────────────────────────────────────────────────────
# taxonomy plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_timing_labels_resolve_and_unknowns_do_not_zero_a_player():
    assert timing_share("Opening Day") == 1.0
    assert timing_share("Mid Season (~July)") == 0.5
    assert timing_share("not a label") == 1.0, (
        "zero would silently delete the player from every total")
    assert timing_share(None) == 1.0
    assert timing_share(0.25) == 0.25


def test_every_role_has_an_anchor_for_its_own_kind():
    for kind, key in (("hitter", "pa"), ("pitcher", "ip")):
        for name in role_names(kind):
            a = role_anchor(name, kind)
            assert a is not None and a[key] > 0, name


def test_an_unknown_role_returns_none_rather_than_raising():
    assert role_anchor("nonsense", "hitter") is None


def test_the_workbook_and_the_model_share_one_taxonomy():
    """Two copies of this table would drift, and a renamed role would fail as
    a silent lookup miss rather than an error."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_role_templates as b
    import role_taxonomy as rt
    assert b.HITTER_ROLES is rt.HITTER_ROLES
    assert b.PITCHER_ROLES is rt.PITCHER_ROLES
    assert b.TIMING is rt.TIMING
    assert b.full_time_reference is rt.full_time_reference, (
        "the workbook's suggestion and the model's default must agree about "
        "what a full-time workload is")


def test_the_workbook_script_still_builds(tmp_path):
    """End-to-end, because moving the taxonomy out of this script silently
    took four of its helper functions with it and 550 passing tests noticed
    nothing — no test opened the workbook path at all."""
    openpyxl = pytest.importorskip("openpyxl")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_role_templates as b

    out = tmp_path / "out"
    out.mkdir()
    for name, df in (("hitter_pa_projections_2027.csv", _hitters()),
                     ("pitcher_pa_projections_2027.csv", _pitchers())):
        d = df.copy()
        d["Name"] = [f"p{i}" for i in range(len(d))]
        d.to_csv(out / name, index=False)

    dest_dir = tmp_path / "rosters"
    dest_dir.mkdir()
    for kind in ("hitter", "pitcher"):
        path = b.build(kind, 2027, out, dest_dir / f"roles_{kind}.xlsx")
        assert path.exists() and path.stat().st_size > 0
        wb = openpyxl.load_workbook(path)
        assert {"Legend", "Assignments", "Roles"} <= set(wb.sheetnames)


def test_full_time_reference_survives_a_partial_season():
    """A half-season file must not collapse everyone into bench roles."""
    from role_taxonomy import full_time_reference
    full = pd.Series(np.linspace(50, 700, 100))
    half = full / 2
    assert full_time_reference(half) == pytest.approx(
        full_time_reference(full) / 2, rel=1e-6)


def test_full_time_reference_handles_nothing_to_measure():
    from role_taxonomy import full_time_reference
    assert full_time_reference(pd.Series(dtype=float)) == 1.0
    assert full_time_reference(pd.Series([0.0, np.nan])) == 1.0
    assert full_time_reference(pd.DataFrame({"nope": [1]})) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# pitcher extras
# ─────────────────────────────────────────────────────────────────────────────

def test_starts_scale_with_innings():
    """A pitcher scaled to 250 innings must not do it in 32 starts."""
    out, _, _ = _run(_pitchers(), "pitcher")
    sp = out[out.pt_role.str.contains("Starter|Ace", regex=True)]
    ratio = sp["Proj_IP"] / sp["Proj_GS"].replace(0, np.nan)
    assert ratio.dropna().between(3.0, 8.0).all(), ratio.describe().to_dict()


def test_save_and_hold_shares_are_attached():
    out, _, _ = _run(_pitchers(), "pitcher")
    closers = out[out.pt_role == "Closer"]
    assert (closers["pt_save_share"] > 0.5).all()
    assert out["pt_save_share"].notna().all()


def test_hitters_get_a_role_based_platoon_share():
    """Fixes a real bug: vL_share currently comes from PAST usage, so a hitter
    moving into a platoon job keeps a stale league-average share."""
    df = _hitters(n_per_team=14, teams=(NYY,))
    df["BatSide"] = "L"
    out, _, _ = _run(df, "hitter")
    assert "Proj_vL_share" in out.columns
    assert out["Proj_vL_share"].between(0.0, 1.0).all()


def test_left_and_right_handers_differ_in_a_platoon_role():
    lhb = role_anchor("Strong Side Platoon", "hitter")["vl_lhb"]
    rhb = role_anchor("Strong Side Platoon", "hitter")["vl_rhb"]
    assert lhb < rhb, "the strong side sits against same-handed pitching"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_save_and_hold_shares_are_normalised_per_team():
    """Role weights are not an allocation.

    Summed over a real staff the raw weights come to 1.16 (saves) and 1.85
    (holds), so multiplying them into a team pool over-allocates — league
    saves read 1,412 against a pool of 1,215 and holds 4,270 against 2,308.
    """
    out, _, _ = _run(_pitchers(), "pitcher")
    for col in ("Proj_SV_share", "Proj_HLD_share"):
        per = out.groupby("Pred_target_team_id")[col].sum()
        assert np.allclose(per.to_numpy(), 1.0), (col, per.to_dict())


def test_the_raw_role_weight_is_kept_alongside():
    out, _, _ = _run(_pitchers(), "pitcher")
    assert (out[out.pt_role == "Closer"]["pt_save_share"] > 0.5).all()


def test_floor_tier_arms_do_not_compete_for_the_pool():
    df = _pitchers()
    df.loc[df.index[:5], "pt_tier"] = TIER_FLOOR
    out, _, _ = _run(df, "pitcher")
    floored = out[out.pt_tier == TIER_FLOOR]
    assert (floored["Proj_SV_share"] == 0.0).all()
    per = out.groupby("Pred_target_team_id")["Proj_SV_share"].sum()
    assert np.allclose(per.to_numpy(), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Closers must project a closer's innings
#
# Found by reading the exported spreadsheet: Justin Martinez projected 1.4
# innings while recording 22.3 saves, Josh Hader 9.3, Mason Miller 22.9. The
# median closer sat at 40 innings against a real 60-65, and 14 of 30 fell
# outside the core roster entirely.
#
# Three separate causes, one per test below.
# ─────────────────────────────────────────────────────────────────────────────

def _staff(n_rp=16, n_sp=11, team=NYY, closer_evidence=400.0):
    """A club with more arms than roster slots — the real shape of the data,
    where 31 projected pitchers compete for 13 jobs."""
    rows, pid = [], 2000
    for i in range(n_sp):
        rows.append({"PlayerId": pid, "Name": f"sp{i}", "Pred_target_team_id": team,
                     "pt_tier": TIER_PROJECTED, "role": "starter",
                     "weighted_IP_per_G": 5.2, "RA9": 4.0 + i * 0.1,
                     "TBF_per_IP": 4.3, "evidence_volume": 700.0 - i * 55})
        pid += 1
    for i in range(n_rp):
        rows.append({"PlayerId": pid, "Name": f"rp{i}", "Pred_target_team_id": team,
                     "pt_tier": TIER_PROJECTED, "role": "reliever",
                     "weighted_IP_per_G": 1.0,
                     # rp0 is the best arm -> the closer
                     "RA9": 2.5 + i * 0.2, "TBF_per_IP": 4.3,
                     "evidence_volume": closer_evidence if i == 0 else 260.0})
        pid += 1
    return pd.DataFrame(rows)


def test_a_closer_is_never_ranked_out_of_the_core_bullpen():
    """He was 13th among his own relievers, behind setup men with better
    evidence factors, because ranking was on raw innings and a closer's
    62-inning anchor loses that race every time."""
    out, _, _ = _run(_staff(), "pitcher")
    closer = out[out.pt_role == "Closer"]
    assert len(closer) == 1
    from role_taxonomy import FAMILY_CORE_SLOTS
    assert closer.iloc[0]["pt_depth_rank"] <= FAMILY_CORE_SLOTS["RP"]
    assert closer.iloc[0]["pt_depth_factor"] == 1.0


def test_a_closer_projects_a_closers_innings():
    out, _, _ = _run(_staff(), "pitcher")
    ip = float(out[out.pt_role == "Closer"].iloc[0]["Proj_IP"])
    assert 35.0 <= ip <= 80.0, f"closer projected {ip:.1f} IP; real is 55-70"


def test_marginal_starters_do_not_outrank_the_bullpen():
    """The 11th starter on a six-slot rotation must be discounted, not the
    closer. Ranking across the whole staff had it backwards."""
    out, _, _ = _run(_staff(), "pitcher")
    rp = out[out.pt_family == "RP"]
    sp = out[out.pt_family == "SP"]
    assert (sp["pt_depth_rank"].max() > 6), "fixture must over-fill the rotation"
    assert sp[sp.pt_depth_rank > 6]["pt_depth_factor"].max() < 1.0
    assert out[out.pt_role == "Closer"].iloc[0]["pt_depth_factor"] == 1.0


def test_a_tiny_sample_does_not_earn_the_ninth_inning():
    """Oakland's closer was Michel Otanez on 28 batters faced — about seven
    innings — where a staff-best RA9 is an accident, not a job."""
    df = _staff()
    # A spectacular RA9 over almost no evidence.
    df.loc[df.Name == "rp5", ["RA9", "evidence_volume"]] = [0.9, 28.0]
    out, _, _ = _run(df, "pitcher")
    assert out[out.Name == "rp5"].iloc[0]["pt_role"] != "Closer"
    assert len(out[out.pt_role == "Closer"]) == 1


def test_every_club_still_gets_a_closer_even_with_thin_arms():
    """The evidence bar must not leave a team without a ninth-inning arm."""
    df = _staff()
    df["evidence_volume"] = 40.0          # nobody clears the bar
    out, _, _ = _run(df, "pitcher")
    assert len(out[out.pt_role == "Closer"]) == 1


def test_saves_cannot_exceed_what_the_innings_allow():
    """A pitcher projected 1.4 innings recorded 22.3 saves. A save is an
    appearance; the share must follow the innings actually thrown."""
    df = _staff()
    out, _, _ = _run(df, "pitcher")
    closer = out[out.pt_role == "Closer"].iloc[0]
    full_share = float(closer["Proj_SV_share"])

    # Same club, but the closer is barely available.
    thin = df.copy()
    thin.loc[thin.Name == "rp0", "evidence_volume"] = 30.0
    out2, _, _ = _run(thin, "pitcher")
    c2 = out2[out2.pt_role == "Closer"]
    if len(c2):
        assert float(c2.iloc[0]["Proj_SV_share"]) < full_share, (
            "a closer who throws fewer innings must take fewer saves")
    for o in (out, out2):
        per = o.groupby("Pred_target_team_id")["Proj_SV_share"].sum()
        assert np.allclose(per.to_numpy(), 1.0)


def test_no_pitcher_takes_a_save_share_without_innings():
    out, _, _ = _run(_staff(), "pitcher")
    no_ip = out[out["Proj_IP"] <= 1.0]
    assert (no_ip["Proj_SV_share"] == 0).all()
    assert (no_ip["Proj_HLD_share"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Save and hold weights must PARTITION the club's pool
#
# Summed over the staff this model produces, the old weights came to 1.16
# (saves) and 1.85 (holds). Normalising that back to 1 diluted the roles that
# should dominate: the closer took 62% of his club's saves against a real
# ~80%, closer saves ran to a 25 median against a real 32-35, and BULLPEN
# DEPTH ARMS took 21% of all league holds against a real ~11% — 0.05 each
# looks modest until it is multiplied by the 15 such arms a club carries.
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_hold_weights_partition_a_real_staff():
    from role_taxonomy import staff_weight_sums
    s = staff_weight_sums()
    assert 0.90 <= s["sv"] <= 1.12, f"save weights sum to {s['sv']:.3f}"
    assert 0.90 <= s["hld"] <= 1.12, f"hold weights sum to {s['hld']:.3f}"


def test_the_closer_takes_the_large_majority_of_the_saves():
    from role_taxonomy import role_anchor, staff_weight_sums
    share = role_anchor("Closer", "pitcher")["sv"] / staff_weight_sums()["sv"]
    assert 0.70 <= share <= 0.90, f"closer takes {share:.0%}; real is ~80%"


def test_setup_men_take_the_large_majority_of_the_holds():
    from role_taxonomy import role_anchor, staff_weight_sums
    two = 2 * role_anchor("Late Inning RP (Setup)", "pitcher")["hld"]
    assert 0.45 <= two / staff_weight_sums()["hld"] <= 0.70


def test_depth_arms_cannot_swamp_the_hold_pool():
    """15 arms x 0.05 came to 41% of all hold weight, for mop-up duty."""
    from role_taxonomy import TYPICAL_STAFF, role_anchor, staff_weight_sums
    n = TYPICAL_STAFF["Bullpen Depth Arm"]
    contrib = n * role_anchor("Bullpen Depth Arm", "pitcher")["hld"]
    assert contrib / staff_weight_sums()["hld"] < 0.20, (
        "a club carries ~15 depth arms, so a per-arm weight that looks small "
        "can still dominate the pool")


def test_no_starter_role_earns_a_save():
    from role_taxonomy import role_anchor
    for role in ("Ace (SP1)", "Mid-Rotation Starter (SP2-3)",
                 "End-of-Rotation Starter (SP4-5)", "Innings-Limited Starter"):
        assert role_anchor(role, "pitcher")["sv"] == 0.0, role


# ─────────────────────────────────────────────────────────────────────────────
# allocate_opportunity — a save is an appearance
# ─────────────────────────────────────────────────────────────────────────────

def test_opportunity_never_exceeds_appearances():
    """Félix Bautista drew 29.6 saves from 28.3 appearances."""
    got = M.allocate_opportunity([0.8, 0.1, 0.1], 40.0, [28.0, 60.0, 60.0])
    assert (got <= np.array([28.0, 60.0, 60.0]) + 1e-9).all()


def test_the_capped_surplus_is_redistributed_not_lost():
    """The pool is the club's opportunity; somebody records it."""
    cap = np.array([28.0, 60.0, 60.0])
    got = M.allocate_opportunity([0.8, 0.1, 0.1], 40.0, cap)
    assert got.sum() == pytest.approx(40.0), got
    assert got[0] == pytest.approx(28.0)


def test_an_uncapped_allocation_is_untouched():
    got = M.allocate_opportunity([0.8, 0.12, 0.08], 40.0, [60.0, 60.0, 60.0])
    assert got == pytest.approx([32.0, 4.8, 3.2])


def test_a_saturated_staff_drops_the_remainder_rather_than_inventing_room():
    """Every arm maxed out: the shortfall is a real statement about a roster
    too thin to finish its own games, not something to paper over."""
    got = M.allocate_opportunity([0.5, 0.5], 40.0, [5.0, 5.0])
    assert got.sum() == pytest.approx(10.0)
    assert (got <= 5.0 + 1e-9).all()


def test_opportunity_handles_degenerate_input():
    assert M.allocate_opportunity([], 40.0, []).size == 0
    assert M.allocate_opportunity([1.0], 0.0, [10.0])[0] == pytest.approx(0.0)


def test_all_zero_shares_allocate_nothing():
    """Not a bug: zero shares across a staff means nobody there holds a save
    role, so nobody records those saves. The headroom fallback inside
    allocate_opportunity exists for redistribution AFTER a cap binds, not to
    hand a pool to pitchers with no claim on it."""
    assert M.allocate_opportunity([0.0, 0.0], 40.0, [10.0, 10.0]).sum() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Hitter roles: four of twelve were never assigned to anybody
#
# Reading the exported spreadsheet: Everyday DH / 1B-DH, Weak Side Platoon,
# Utility IF and Injury Replacement had zero players. The heuristic knew
# nothing about position or platoon usage, so it could only ever emit five of
# the twelve roles it had.
#
# Worse than missing roles, it MISLABELLED: Full Time required a 0.80 share of
# a full-time workload, which gave 5.2 regulars per club against a real nine,
# and the regulars it missed became Strong Side Platoon — a role carrying vL
# shares of 0.12/0.45. Calling an everyday player a platoon bat corrupts his
# handedness exposure, and that feeds the daily sim.
# ─────────────────────────────────────────────────────────────────────────────

def _hitter(pa, pos=None, vl=None, bats="R", pid=1, team=NYY):
    row = {"PlayerId": pid, "Name": f"h{pid}", "Pred_target_team_id": team,
           "pt_tier": TIER_PROJECTED, "evidence_volume": pa, "Last_PA": pa,
           "BatSide": bats}
    if vl is not None:
        row["vL_share"] = vl
    return row


def _roles_for(rows, fielding=None):
    out, _, _ = _run(pd.DataFrame(rows), "hitter", fielding=fielding)
    return dict(zip(out["Name"], out["pt_role"]))


def test_an_everyday_first_baseman_gets_the_dh_role():
    """Never once assigned, because the heuristic had no position data."""
    rows = [_hitter(700, pid=i) for i in range(1, 15)]
    f = pd.DataFrame([{"PlayerId": 1, "Pos": "1B", "Innings": 1200.0,
                       "Season": 2026}])
    assert _roles_for(rows, f)["h1"] == "Everyday DH / 1B-DH"


def test_an_everyday_outfielder_is_full_time_not_dh():
    rows = [_hitter(700, pid=i) for i in range(1, 15)]
    f = pd.DataFrame([{"PlayerId": 1, "Pos": "CF", "Innings": 1200.0,
                       "Season": 2026}])
    assert _roles_for(rows, f)["h1"] == "Full Time"


def test_a_utility_infielder_is_distinguished_from_an_outfielder():
    """Utility IF and Utility OF carry different volume and platoon usage, but
    only the outfield one could ever be suggested."""
    rows = [_hitter(300, pid=1), _hitter(300, pid=2)] + \
           [_hitter(700, pid=i) for i in range(3, 16)]
    f = pd.DataFrame([{"PlayerId": 1, "Pos": "SS", "Innings": 600.0, "Season": 2026},
                      {"PlayerId": 2, "Pos": "LF", "Innings": 600.0, "Season": 2026}])
    got = _roles_for(rows, f)
    assert got["h1"] == "Utility IF"
    assert got["h2"] == "Utility OF / 4th OF"


def test_a_right_hander_used_against_lhp_is_a_weak_side_platoon():
    """Detected from USAGE, not volume. Among real projected hitters, 29 of
    the 30 above a 0.40 vL share are right-handed — that is what a weak-side
    platoon bat is, and the role had never been assigned."""
    rows = [_hitter(250, vl=0.48, bats="R", pid=1)] + \
           [_hitter(600, vl=0.28, pid=i) for i in range(2, 15)]
    assert _roles_for(rows)["h1"] == "Weak Side Platoon"


def test_a_left_hander_who_sits_against_lhp_is_a_strong_side_platoon():
    """44 of the 45 real hitters below a 0.18 vL share are left-handed."""
    rows = [_hitter(450, vl=0.12, bats="L", pid=1)] + \
           [_hitter(600, vl=0.28, pid=i) for i in range(2, 15)]
    assert _roles_for(rows)["h1"] == "Strong Side Platoon"


def test_an_everyday_player_is_not_called_a_platoon_bat():
    """THE mislabelling: Goldschmidt, Chisholm and Garcia were all platoon
    bats, which would have given each of them a platoon's vL exposure."""
    rows = [_hitter(600, vl=0.28, pid=i) for i in range(1, 15)]
    got = _roles_for(rows)
    assert not any(v.endswith("Platoon") for v in got.values()), got


def test_a_full_time_share_yields_about_nine_regulars_a_club():
    """0.80 gave 5.2 per club against a real nine; 0.60 gives ~9."""
    from role_taxonomy import FULL_TIME_SHARE
    assert 0.55 <= FULL_TIME_SHARE <= 0.65


def test_platoon_detection_degrades_without_vl_share():
    """The column is optional; absent it, volume alone still assigns a role."""
    rows = [_hitter(600, pid=i) for i in range(1, 15)]
    got = _roles_for(rows)
    assert all(v in set(role_names("hitter")) for v in got.values())
    assert "Full Time" in got.values()


def test_every_assignable_hitter_role_can_actually_be_reached():
    """Four of twelve were unreachable. Injury Replacement stays override-only
    by design — its volume is conditional on OTHER players getting hurt, which
    no per-player heuristic can see."""
    from role_taxonomy import role_names
    reachable = set()
    f_rows, rows = [], []
    for i, (pa, pos, vl, bats) in enumerate([
        (700, "CF", 0.28, "R"), (700, "1B", 0.28, "R"), (700, "C", 0.28, "R"),
        (300, "C", 0.28, "R"), (80, "C", 0.28, "R"), (300, "SS", 0.28, "R"),
        (300, "LF", 0.28, "R"), (250, "RF", 0.48, "R"), (450, "LF", 0.12, "L"),
        (60, "LF", 0.28, "R"),
    ], start=1):
        rows.append(_hitter(pa, vl=vl, bats=bats, pid=i))
        f_rows.append({"PlayerId": i, "Pos": pos, "Innings": 900.0,
                       "Season": 2026})
    rows += [_hitter(700, vl=0.28, pid=100 + j) for j in range(6)]
    out, _, _ = _run(pd.DataFrame(rows), "hitter", fielding=pd.DataFrame(f_rows))
    reachable = set(out["pt_role"])
    never = set(role_names("hitter")) - reachable - {
        "Injury Replacement / 26th Man", DEPTH_HITTER_ROLE}
    assert not never, f"still unreachable: {sorted(never)}"


def test_the_workbook_shows_the_role_the_model_assigned(tmp_path):
    """The sheet is a view of the assignment, not a second opinion.

    It used to re-derive its own suggestion from `suggest_*_role` called
    WITHOUT the position or the platoon share — the weaker signature from
    before those arguments existed — while the projections beside it carried
    `pt_role` from the playing-time model. The two disagreed badly: the
    workbook showed four hitter roles with no catchers, no designated
    hitters and no platoon bats, and a pitching staff with NO CLOSER on any
    club, where the model had assigned exactly 30. Anyone opening the sheet
    to adjust a role was editing different labels from the ones the numbers
    came from.
    """
    openpyxl = pytest.importorskip("openpyxl")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_role_templates as b

    out = tmp_path / "out"
    out.mkdir()
    assigned = {}
    for kind, frame in (("hitter", _hitters()), ("pitcher", _pitchers())):
        df, _, _ = M.project_playing_time(frame, kind, target_year=2027)
        df["Name"] = [f"p{i}" for i in range(len(df))]
        assigned[kind] = dict(zip(df["PlayerId"].astype(int), df["pt_role"]))
        df.to_csv(out / f"{kind}_pa_projections_2027.csv", index=False)

    dest = tmp_path / "rosters"
    dest.mkdir()
    for kind in ("hitter", "pitcher"):
        path = b.build(kind, 2027, out, dest / f"roles_{kind}.xlsx")
        ws = openpyxl.load_workbook(path)["Assignments"]
        header = [c.value for c in ws[1]]
        i_id = header.index("PlayerId")
        i_role = header.index("Suggested Role")
        seen = 0
        for row in ws.iter_rows(min_row=2, values_only=True):
            pid = row[i_id]
            if not isinstance(pid, (int, float)) or int(pid) not in assigned[kind]:
                continue        # the anchor rows and the example row
            seen += 1
            assert row[i_role] == assigned[kind][int(pid)], (
                f"{kind} {pid}: workbook says {row[i_role]!r}, the model "
                f"assigned {assigned[kind][int(pid)]!r}")
        assert seen > 0, "no player rows were compared"

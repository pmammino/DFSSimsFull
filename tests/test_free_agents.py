"""An unsigned player gets a projection; he gets the SAME one he would get
on a club.

A free agent will play somewhere, so he needs a line, and what he cannot do
is play ON TOP of a league that is already full: the 30 clubs close on all
30 x budget of the playing time there is. Both halves have to hold at once,
and the order they are settled in is what decides whether they do.

Settling the clubs first and handing the free agents the remainder is what
made an everyday regular's season depend on a number in the roster file
rather than on his role — reserving 3.5% and marking eighteen regulars
unsigned gave them 328-389 plate appearances each, which is not a projection
of what any of them will do. Settling the unsigned class FIRST, from their
own role anchors, and making the clubs reserve exactly that much, gets a
full-time free agent a full-time season and still closes the league.

(Before any of this, the same eighteen had a MEDIAN of 39.7 plate
appearances, because every unsigned player in baseball was ranked against
every other as though they were one 40-man roster.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import playing_time_model as M  # noqa: E402
from team_context import FREE_AGENT_TEAM_ID  # noqa: E402

CLUBS = [147, 119, 111]


def _hitters(per_club=16):
    rng = np.random.default_rng(0)
    n = per_club * len(CLUBS)
    return pd.DataFrame({
        "PlayerId": range(1, n + 1),
        "Name": [f"h{i}" for i in range(n)],
        "Pred_target_team_id": [c for c in CLUBS for _ in range(per_club)],
        "Last_PA": rng.integers(250, 620, n),
        "Career_PA": rng.integers(800, 5000, n),
        "P_K": 0.22, "P_BB": 0.085, "P_HBP": 0.011, "P_SF": 0.006,
        "P_HR": 0.030, "P_3B": 0.004, "P_2B": 0.042, "P_1B": 0.142,
        "P_BIPOut": 0.460,
    })


def _run(df, reserves=None, kind="hitter"):
    out, diag, _ = M.project_playing_time(df, kind, target_year=2027,
                                          reserves=reserves)
    return out, diag


def _free(df, n=6):
    """Mark the n highest-volume players unsigned."""
    out = df.copy()
    ids = out.nlargest(n, "Last_PA")["PlayerId"]
    out.loc[out["PlayerId"].isin(ids), "Pred_target_team_id"] = FREE_AGENT_TEAM_ID
    return out, set(ids)


def _projected(df):
    return df[df["pt_tier"].astype(str) != "floor"]


def _per_club(df):
    p = _projected(df)
    on = p[pd.to_numeric(p["Pred_target_team_id"], errors="coerce") > 0]
    return on.groupby("Pred_target_team_id")["Proj_PA"].sum()


# ─────────────────────────────────────────────────────────────────────────────
# Sizing the pool
# ─────────────────────────────────────────────────────────────────────────────

def test_the_pool_is_the_unsigned_share_of_the_leagues_raw_volume():
    """One scale for everybody, which is the whole claim.

    Nine tenths of the role volume is signed, so the clubs keep nine tenths
    of the league and the unsigned tenth takes the rest.
    """
    assert M.free_agent_pool(fa_raw=1_000.0, club_raw=9_000.0,
                             league=20_000.0) == pytest.approx(2_000.0)


def test_a_free_agent_is_scaled_at_the_same_rate_as_everyone_else():
    """What the closed form is for. The league scale and the free agents'
    scale are the same number, so a role means one thing league-wide."""
    league, club_raw, fa_raw = 20_000.0, 17_000.0, 5_000.0
    pool = M.free_agent_pool(fa_raw, club_raw, league)
    assert pool / fa_raw == pytest.approx((league - pool) / club_raw)


def test_nobody_unsigned_reserves_nothing():
    assert M.free_agent_pool(0.0, 9_000.0, 20_000.0) == 0.0


def test_the_pool_cannot_eat_the_league():
    """Self-limiting by construction, so no arbitrary ceiling is needed:
    the clubs keep their share of the raw volume however much is unsigned."""
    pool = M.free_agent_pool(fa_raw=1e9, club_raw=1.0, league=20_000.0)
    assert pool < 20_000.0


def test_nobody_signed_at_all_hands_the_league_to_the_free_agents():
    """Degenerate, but it must not divide by zero or return a league of
    nothing: if no one has a club, the unsigned players ARE the league."""
    assert M.free_agent_pool(5_000.0, 0.0, 20_000.0) == pytest.approx(20_000.0)


def test_pool_mode_takes_the_declared_reserve_instead(monkeypatch):
    monkeypatch.setattr(M, "FREE_AGENT_PLAYING_TIME", "pool")
    assert M.free_agent_pool(1_000.0, 9_000.0, 20_000.0,
                             declared_pool=400.0) == pytest.approx(400.0)


def test_the_declared_share_decides_which_clubs_give_it_up():
    """Its job in "role" mode: a distribution key, not a cap. A club that
    says it expects to sign gives up more of the pool than one that does
    not, and the pool itself is still what the free agents need."""
    held = M._reserve_by_club(300.0, {1: 0.03, 2: 0.01, 3: 0.0}, budget=1_000.0)
    assert sum(held.values()) == pytest.approx(300.0)
    assert held[1] == pytest.approx(225.0)
    assert held[2] == pytest.approx(75.0)
    assert held[3] == pytest.approx(0.0)


def test_with_nothing_declared_every_club_gives_up_the_same():
    held = M._reserve_by_club(300.0, {1: 0.0, 2: 0.0, 3: 0.0}, budget=1_000.0)
    assert set(held) == {1, 2, 3}
    assert list(held.values()) == pytest.approx([100.0] * 3)


# ─────────────────────────────────────────────────────────────────────────────
# A free agent is not a roster spot
# ─────────────────────────────────────────────────────────────────────────────

def test_free_agents_are_not_depth_ranked_against_each_other():
    """The discount means "eighth arm on this staff". He has no staff.

    Grouping by team id made every unsigned player in baseball one
    pseudo-club with nine lineup slots, so they were ranked against one
    another and decayed at 0.78 a rank to the 0.04 floor.
    """
    df, ids = _free(_hitters(), n=20)
    out, _ = _run(df)
    fa = out[out["PlayerId"].isin(ids)]
    assert (fa["pt_depth_factor"] == 1.0).all()
    assert fa["pt_depth_rank"].isna().all()


def test_an_unsigned_regular_projects_like_the_regular_he_is():
    """39.7 plate appearances, and then 328, were the numbers this replaces.

    Being unsigned is not information about how much a player will play once
    he signs, so losing his club should cost him nothing. Measured against
    the same players WITH a club rather than against a threshold, so the
    fixture cannot quietly make it pass.
    """
    base, _ = _run(_hitters())
    df, ids = _free(_hitters(), n=20)
    out, _ = _run(df)

    signed = base[base["PlayerId"].isin(ids)]["Proj_PA"].median()
    unsigned = out[out["PlayerId"].isin(ids)]["Proj_PA"].median()
    assert unsigned == pytest.approx(signed, rel=0.10), (
        f"{unsigned:.1f} PA unsigned against {signed:.1f} on a club")


def test_a_declared_reserve_does_not_change_what_he_plays():
    """The regression this exists for. His season is his ROLE; what the
    clubs wrote in the roster file decides who pays for it, not how much
    he plays. Reserving 3.5% used to cut him to half a season."""
    df, ids = _free(_hitters(), n=10)
    free_run, _ = _run(df)
    declared, _ = _run(df, reserves={c: {"pa_share": 0.035} for c in CLUBS})
    a = free_run[free_run["PlayerId"].isin(ids)].set_index("PlayerId")["Proj_PA"]
    b = declared[declared["PlayerId"].isin(ids)].set_index("PlayerId")["Proj_PA"]
    assert a.to_numpy() == pytest.approx(b.reindex(a.index).to_numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Where the playing time comes from
# ─────────────────────────────────────────────────────────────────────────────

def test_the_league_adds_up_with_nothing_declared_at_all():
    """The case that used to run 15% over. No roster file is needed for the
    league to close: the pool is derived from the free agents themselves."""
    df, _ = _free(_hitters(), n=10)
    out, _ = _run(df)
    assert _projected(out)["Proj_PA"].sum() == pytest.approx(
        M.TEAM_PA_BUDGET * len(CLUBS), rel=1e-9)


def test_the_clubs_fall_short_by_exactly_what_the_free_agents_take():
    df, ids = _free(_hitters(), n=10)
    out, _ = _run(df)
    fa_total = _projected(out).loc[
        _projected(out)["PlayerId"].isin(ids), "Proj_PA"].sum()
    per = _per_club(out)
    assert per.to_numpy() == pytest.approx(
        M.TEAM_PA_BUDGET - fa_total / len(CLUBS))


def test_a_declared_share_moves_the_cost_between_clubs():
    """One club expects to do the signing, so one club pays for it — and
    the league total is untouched either way."""
    df, _ = _free(_hitters(), n=10)
    out, _ = _run(df, reserves={CLUBS[0]: {"pa_share": 0.10}})
    per = _per_club(out)
    assert per.loc[CLUBS[0]] < per.loc[CLUBS[1]] - 1.0
    assert per.loc[CLUBS[1]] == pytest.approx(M.TEAM_PA_BUDGET)
    assert _projected(out)["Proj_PA"].sum() == pytest.approx(
        M.TEAM_PA_BUDGET * len(CLUBS), rel=1e-9)


def test_pool_mode_still_closes_them_onto_the_declared_share(monkeypatch):
    """The other question, kept because it is a real one: "only 10% of
    league plate appearances go to players unsigned today"."""
    monkeypatch.setattr(M, "FREE_AGENT_PLAYING_TIME", "pool")
    df, ids = _free(_hitters(), n=10)
    out, _ = _run(df, reserves={c: {"pa_share": 0.10} for c in CLUBS})
    p = _projected(out)
    assert p[p["PlayerId"].isin(ids)]["Proj_PA"].sum() == pytest.approx(
        M.TEAM_PA_BUDGET * 0.10 * len(CLUBS), rel=1e-6)
    assert _per_club(out).to_numpy() == pytest.approx(M.TEAM_PA_BUDGET * 0.90)


def test_nobody_exceeds_the_ceiling():
    df, _ = _free(_hitters(), n=3)
    out, _ = _run(df)
    assert out["Proj_PA"].max() <= M.PT_MAX_PA + 1e-6


def test_pitchers_work_the_same_way():
    rng = np.random.default_rng(1)
    n = 36
    df = pd.DataFrame({
        "PlayerId": range(1, n + 1),
        "Name": [f"p{i}" for i in range(n)],
        "Pred_target_team_id": [c for c in CLUBS for _ in range(n // 3)],
        "role": "reliever",
        "Last_PA": rng.integers(150, 600, n),
        "Career_PA": rng.integers(400, 3000, n),
        "P_K": 0.23, "P_BB": 0.080, "P_HBP": 0.011, "P_SF": 0.006,
        "P_HR": 0.029, "P_3B": 0.004, "P_2B": 0.041, "P_1B": 0.140,
        "P_BIPOut": 0.466,
    })
    ids = set(df.nlargest(5, "Last_PA")["PlayerId"])
    df.loc[df["PlayerId"].isin(ids), "Pred_target_team_id"] = FREE_AGENT_TEAM_ID
    out, _ = _run(df, kind="pitcher")

    p = out[out["pt_tier"].astype(str) != "floor"]
    assert p["Proj_IP"].sum() == pytest.approx(
        M.TEAM_IP_BUDGET * len(CLUBS), rel=1e-9)
    fa = p[p["PlayerId"].isin(ids)]
    assert (fa["pt_depth_factor"] == 1.0).all()

    # Scaled at the league's own rate, which is the invariant that makes an
    # innings role mean the same thing signed or not.
    raw = pd.to_numeric(p["pt_raw"], errors="coerce")
    rate = p["Proj_IP"].sum() / raw.sum()
    assert (fa["Proj_IP"] / pd.to_numeric(fa["pt_raw"], errors="coerce")
            ).to_numpy() == pytest.approx(rate, rel=1e-6)


def test_leaving_a_club_drops_the_depth_discount_with_it():
    """A consequence worth stating rather than discovering.

    The roster-depth discount is a statement about a DEPTH CHART — "eighth
    arm on this staff, so he throws what an eighth arm throws". A player
    nobody has signed is not eighth on anything, so he carries his role's
    volume undiscounted and can out-project the line he had as somebody's
    long reliever. That is the discount being a fact about the club, which
    is what it was always supposed to be.
    """
    df = _hitters(per_club=18)
    base, _ = _run(df)
    deep = base[base["pt_depth_rank"].fillna(0) >= 10]
    assert len(deep), "fixture has no deep-roster players to test with"
    ids = set(deep["PlayerId"])

    moved = df.copy()
    moved.loc[moved["PlayerId"].isin(ids),
              "Pred_target_team_id"] = FREE_AGENT_TEAM_ID
    out, _ = _run(moved)
    fa = out[out["PlayerId"].isin(ids)]
    assert (fa["pt_depth_factor"] == 1.0).all()
    assert fa["Proj_PA"].median() > deep["Proj_PA"].median()


# ─────────────────────────────────────────────────────────────────────────────
# Saying so
# ─────────────────────────────────────────────────────────────────────────────

def test_the_diagnostics_carry_a_row_for_the_free_agents():
    df, ids = _free(_hitters(), n=6)
    out, diag = _run(df)
    fa = diag[diag["team_id"] == FREE_AGENT_TEAM_ID]
    assert len(fa) == 1
    assert fa.iloc[0]["n"] == 6
    assert fa.iloc[0]["target"] == pytest.approx(
        out[out["PlayerId"].isin(ids)]["Proj_PA"].sum())


def test_the_clubs_report_the_reserve_they_actually_held():
    """Derived, so it has to be reported from the projection and not echoed
    back off the roster file."""
    df, ids = _free(_hitters(), n=6)
    out, diag = _run(df)
    clubs = diag[diag["team_id"] != FREE_AGENT_TEAM_ID]
    held = out[out["PlayerId"].isin(ids)]["Proj_PA"].sum() / len(CLUBS)
    assert clubs["reserved_share"].to_numpy() == pytest.approx(
        held / M.TEAM_PA_BUDGET)


def test_the_report_puts_the_two_scales_side_by_side():
    """The one line that says whether an unsigned player is being treated
    like a signed one."""
    df, _ = _free(_hitters(), n=6)
    out, diag, stats = M.project_playing_time(df, "hitter", target_year=2027)
    text = M.playing_time_report(out, diag, stats, "hitter")
    assert "free agents" in text
    assert "against the clubs'" in text
    assert "so the league still closes" in text


def test_the_club_scale_is_not_polluted_by_the_free_agent_row():
    """It is the number that exposes a mis-sized ANCHOR, so it must not
    move with how many players happen to be unsigned."""
    df, _ = _free(_hitters(), n=10)
    out, diag, stats = M.project_playing_time(df, "hitter", target_year=2027)
    clubs = diag[diag["team_id"] != FREE_AGENT_TEAM_ID]
    assert len(clubs) == len(CLUBS)

    text = M.playing_time_report(out, diag, stats, "hitter")
    assert f"team closure: {len(CLUBS)} teams" in text
    want = pd.to_numeric(clubs["scale"], errors="coerce").dropna().mean()
    assert f"anchor scale: mean {want:.3f}" in text


def test_a_set_with_no_free_agents_reports_none_of_this():
    out, diag, stats = M.project_playing_time(_hitters(), "hitter",
                                              target_year=2027)
    assert "free agents" not in M.playing_time_report(out, diag, stats,
                                                      "hitter")


# ─────────────────────────────────────────────────────────────────────────────
# Downstream
# ─────────────────────────────────────────────────────────────────────────────

def _reserve_file(tmp_path, monkeypatch, share):
    """Write a reserves file where the verifier will look for it."""
    import json
    import team_context as tc
    roster = tmp_path / "rosters"
    roster.mkdir(exist_ok=True)
    (roster / "team_assignments_2027.json").write_text(json.dumps({
        "target_year": 2027,
        "reserves": [{"team_id": c, "pa_share": share, "ip_share": share}
                     for c in CLUBS]}))
    monkeypatch.setattr(tc, "ROSTER_DIR", roster)


def _rows(frame, year=2027):
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_refresh import Checks, check_playing_time
    c = Checks()
    check_playing_time(c, frame, frame.assign(Proj_IP=frame["Proj_PA"]), year)
    return {r[1]: r[0] for r in c.rows}


def test_the_gate_sees_free_agents_in_the_league_total(tmp_path, monkeypatch):
    """Per-club closure cannot: they are not on a club, so they are not in it.

    Ten unsigned regulars carrying a full season each push the league 15%
    over, and every per-club row still reads PASS.
    """
    df, _ = _free(_hitters(), n=10)
    out, _ = _run(df)
    _reserve_file(tmp_path, monkeypatch, 0.0)
    r = _rows(out)
    assert r["Proj_PA team closure"] == "PASS"
    assert r["Proj_PA league total"] == "PASS"


def test_the_gate_reads_the_reserve_off_the_projection(tmp_path, monkeypatch):
    """Not off the roster file, which no longer sets its size.

    A club holding back 6% because that is what the unsigned class needs is
    correct even where the file declared 3.5%, and judging it against the
    declared number turns a correct projection into a FAIL.
    """
    df, _ = _free(_hitters(), n=10)
    out, _ = _run(df, reserves={c: {"pa_share": 0.035} for c in CLUBS})
    for declared in (0.0, 0.035, 0.20):
        _reserve_file(tmp_path, monkeypatch, declared)
        assert _rows(out)["Proj_PA team closure"] == "PASS", declared


def test_the_gate_still_fails_a_club_that_did_not_close(tmp_path, monkeypatch):
    """The check has to keep catching the thing it is for."""
    df, _ = _free(_hitters(), n=10)
    out, _ = _run(df)
    broken = out.copy()
    hit = broken["Pred_target_team_id"] == CLUBS[0]
    broken.loc[hit, "Proj_PA"] = broken.loc[hit, "Proj_PA"] * 0.80
    _reserve_file(tmp_path, monkeypatch, 0.0)
    assert _rows(broken)["Proj_PA team closure"] == "FAIL"

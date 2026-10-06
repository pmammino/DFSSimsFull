"""An unsigned player gets a projection; he does not get a free season.

A free agent will play somewhere, so he needs a line. What he cannot do is
play ON TOP of a league that is already full: the 30 clubs close on all
30 x budget of the playing time there is, so his plate appearances have to
come out of the share a club has reserved for the signing it has not made
yet. `load_roster_reserves` is one half of that mechanism and has existed
since free agency was modelled at all; this is the half that was missing.

Before this, marking 40 everyday regulars unsigned gave them a MEDIAN of
39.7 plate appearances and pushed the league 5% over its own total.
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


def test_an_unsigned_regular_still_projects_like_a_regular():
    """39.7 plate appearances was the number this replaces.

    Being unsigned is not information about how much a player will play
    once he signs, so losing his club should not cost him most of his
    season. Compared against the same players with a club, rather than
    against a threshold, so the fixture cannot quietly make it pass.
    """
    base, _ = _run(_hitters())
    df, ids = _free(_hitters(), n=20)
    out, _ = _run(df)

    signed = base[base["PlayerId"].isin(ids)]["Proj_PA"].median()
    unsigned = out[out["PlayerId"].isin(ids)]["Proj_PA"].median()
    assert unsigned > 0.75 * signed, (
        f"{unsigned:.1f} PA unsigned against {signed:.1f} on a club")


def test_the_clubs_they_left_still_close():
    df, ids = _free(_hitters(), n=10)
    out, _ = _run(df)
    projected = out[out["pt_tier"].astype(str) != "floor"]
    on_club = projected[pd.to_numeric(projected["Pred_target_team_id"],
                                      errors="coerce") > 0]
    totals = on_club.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert totals.to_numpy() == pytest.approx(M.TEAM_PA_BUDGET)


# ─────────────────────────────────────────────────────────────────────────────
# Where the playing time comes from
# ─────────────────────────────────────────────────────────────────────────────

def test_with_a_reserved_share_the_league_adds_up():
    """The two halves of the mechanism, meeting."""
    df, ids = _free(_hitters(), n=10)
    reserves = {c: {"pa_share": 0.10} for c in CLUBS}
    out, _ = _run(df, reserves=reserves)
    projected = out[out["pt_tier"].astype(str) != "floor"]

    league = projected["Proj_PA"].sum()
    assert league == pytest.approx(M.TEAM_PA_BUDGET * len(CLUBS), rel=1e-6)

    # The clubs deliberately fall short by what they reserved...
    on_club = projected[pd.to_numeric(projected["Pred_target_team_id"],
                                      errors="coerce") > 0]
    per = on_club.groupby("Pred_target_team_id")["Proj_PA"].sum()
    assert per.to_numpy() == pytest.approx(M.TEAM_PA_BUDGET * 0.90)

    # ...and the free agents take exactly that much between them.
    fa = projected[projected["PlayerId"].isin(ids)]
    assert fa["Proj_PA"].sum() == pytest.approx(
        M.TEAM_PA_BUDGET * 0.10 * len(CLUBS), rel=1e-6)


def test_without_reserves_they_keep_their_role_volume():
    """The useful line for a player nobody has signed."""
    df, ids = _free(_hitters(), n=6)
    out, _ = _run(df)
    fa = out[out["PlayerId"].isin(ids)]
    raw = pd.to_numeric(fa["pt_raw"], errors="coerce")
    assert fa["Proj_PA"].to_numpy() == pytest.approx(
        np.minimum(raw.to_numpy(), M.PT_MAX_PA))


def test_a_reserve_pool_too_small_for_them_still_closes_onto_it():
    """Reserving 1% for ten everyday regulars is a statement about the
    league, not an error: they cannot all play."""
    df, ids = _free(_hitters(), n=10)
    out, _ = _run(df, reserves={c: {"pa_share": 0.01} for c in CLUBS})
    fa = out[out["PlayerId"].isin(ids)]
    assert fa["Proj_PA"].sum() == pytest.approx(
        M.TEAM_PA_BUDGET * 0.01 * len(CLUBS), rel=1e-6)


def test_nobody_exceeds_the_ceiling_however_much_is_reserved():
    df, ids = _free(_hitters(), n=3)
    out, _ = _run(df, reserves={c: {"pa_share": 0.9} for c in CLUBS})
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
    out, _ = _run(df, reserves={c: {"ip_share": 0.12} for c in CLUBS},
                  kind="pitcher")
    fa = out[out["PlayerId"].isin(ids)]
    assert fa["Proj_IP"].sum() == pytest.approx(
        M.TEAM_IP_BUDGET * 0.12 * len(CLUBS), rel=1e-6)
    assert (fa["pt_depth_factor"] == 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Saying so
# ─────────────────────────────────────────────────────────────────────────────

def test_the_diagnostics_carry_a_row_for_the_free_agents():
    df, ids = _free(_hitters(), n=6)
    _, diag = _run(df, reserves={c: {"pa_share": 0.10} for c in CLUBS})
    fa = diag[diag["team_id"] == FREE_AGENT_TEAM_ID]
    assert len(fa) == 1
    assert fa.iloc[0]["n"] == 6
    assert fa.iloc[0]["target"] == pytest.approx(
        M.TEAM_PA_BUDGET * 0.10 * len(CLUBS))


def test_an_unreserved_league_running_over_is_reported():
    df, ids = _free(_hitters(), n=6)
    out, diag, stats = M.project_playing_time(df, "hitter", target_year=2027)
    text = M.playing_time_report(out, diag, stats, "hitter")
    assert "free agents" in text
    assert "nothing reserved" in text
    assert "league total is over" in text


def test_a_reserved_league_reports_the_pool_instead():
    df, ids = _free(_hitters(), n=6)
    out, diag, stats = M.project_playing_time(
        df, "hitter", target_year=2027,
        reserves={c: {"pa_share": 0.10} for c in CLUBS})
    text = M.playing_time_report(out, diag, stats, "hitter")
    assert "the clubs reserved" in text
    assert "nothing reserved" not in text


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


def test_the_gate_sees_free_agents_in_the_league_total(tmp_path, monkeypatch):
    """Per-club closure cannot: they are not on a club, so they are not in it.

    Ten unsigned regulars carrying a full season each push the league 15%
    over, and every per-club row still reads PASS.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_refresh import Checks, check_playing_time

    df, ids = _free(_hitters(), n=10)
    exact, _ = _run(df, reserves={c: {"pa_share": 0.10} for c in CLUBS})
    _reserve_file(tmp_path, monkeypatch, 0.10)

    def rows(frame):
        c = Checks()
        check_playing_time(c, frame, frame.assign(Proj_IP=frame["Proj_PA"]),
                           2027)
        return {r[1]: r[0] for r in c.rows}

    r = rows(exact)
    assert r["Proj_PA team closure"] == "PASS"
    assert r["Proj_PA league total"] == "PASS"


def test_the_closure_check_knows_what_a_club_reserved(tmp_path, monkeypatch):
    """A club holding 10% back is supposed to fall 10% short.

    Judging it against the full budget turns a correct projection into a
    failure, which is how a reserve-aware league gets reported as broken.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_refresh import Checks, check_playing_time

    df, _ = _free(_hitters(), n=10)
    out, _ = _run(df, reserves={c: {"pa_share": 0.10} for c in CLUBS})

    def verdict():
        c = Checks()
        check_playing_time(c, out, out.assign(Proj_IP=out["Proj_PA"]), 2027)
        return {r[1]: r[0] for r in c.rows}["Proj_PA team closure"]

    _reserve_file(tmp_path, monkeypatch, 0.10)
    assert verdict() == "PASS"

    # Same frame, a file that claims nothing was reserved: now it is wrong.
    _reserve_file(tmp_path, monkeypatch, 0.0)
    assert verdict() == "FAIL"

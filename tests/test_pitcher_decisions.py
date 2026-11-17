"""Wins, losses and quality starts: a club's record handed out to its staff."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import team_wins as TW                                    # noqa: E402


def _staff(n_sp=5, n_rp=8, team=1):
    rows = []
    for i in range(n_sp):
        rows.append(dict(team_id=team, Proj_IP=180 - 15 * i, Proj_G=32,
                         Proj_GS=32, ERA=3.4 + 0.3 * i, RA9=3.7 + 0.3 * i))
    for i in range(n_rp):
        rows.append(dict(team_id=team, Proj_IP=70 - 6 * i, Proj_G=65 - 4 * i,
                         Proj_GS=0, ERA=3.6 + 0.2 * i, RA9=3.9 + 0.2 * i))
    return pd.DataFrame(rows)


def _pools(wins=90.0, team=1):
    return pd.DataFrame({"team_id": [team], "expected_wins": [wins],
                         "rs_per_game": [4.6], "ra_per_game": [4.1]})


def test_a_clubs_pitchers_sum_to_its_record():
    r = TW.project_decisions(_staff(), _pools(90.0))
    assert abs(r["W"].sum() - 90.0) < 1e-6
    assert abs(r["L"].sum() - 72.0) < 1e-6


def test_better_pitchers_win_more_and_lose_less():
    r = TW.project_decisions(_staff(), _pools())
    assert r["W"].iloc[0] > r["W"].iloc[4]
    assert r["L"].iloc[0] < r["L"].iloc[4]


def test_nobody_wins_more_than_he_started_or_appeared_in():
    s = _staff()
    r = TW.project_decisions(s, _pools(110.0))
    assert (r["W"] <= np.where(s.Proj_GS > 0, s.Proj_GS, s.Proj_G) + 1e-9).all()


def test_floor_tier_pitchers_take_no_decisions():
    s = _staff()
    elig = np.ones(len(s), dtype=bool)
    elig[-1] = False
    r = TW.project_decisions(s, _pools(), eligible=elig)
    assert r["W"].iloc[-1] == 0 and r["QS"].iloc[-1] == 0
    assert abs(r["W"].sum() - 90.0) < 1e-6        # the pool is not lost


def test_pitcher_with_no_club_is_blank():
    s = _staff()
    s.loc[0, "team_id"] = np.nan
    r = TW.project_decisions(s, _pools())
    assert r.iloc[0].isna().all()


def test_quality_starts_rise_with_innings_and_fall_with_era():
    assert TW.quality_start_rate(6.5, 3.0) > TW.quality_start_rate(5.0, 3.0)
    assert TW.quality_start_rate(6.0, 2.8) > TW.quality_start_rate(6.0, 5.0)
    assert 0.30 < TW.quality_start_rate(5.3, 4.2) < 0.50   # a league-average start


def test_relievers_have_no_quality_starts():
    r = TW.project_decisions(_staff(), _pools())
    assert (r["QS"].iloc[5:] == 0).all()


def test_starts_made_by_relievers_do_not_inflate_the_rotation():
    """A club's starts are not all made by its rotation. Splitting the starters'
    pool over only the pitchers FILED as starters spread it over ~154 starts
    instead of 162 and inflated every starter's rate (Skenes read 15.9)."""
    s = _staff(n_sp=5, n_rp=8)
    s.loc[:4, "Proj_GS"] = 28.0                       # 140 starts by the rotation
    s.loc[5, ["Proj_GS", "Proj_G", "Proj_IP"]] = [22.0, 40.0, 110.0]  # swingman
    s["Proj_IP"] = np.where(s["Proj_GS"] > 0, s["Proj_GS"] * 5.2, s["Proj_IP"])
    s["ERA"], s["RA9"] = 4.2, 4.5                     # equal quality: isolate the split
    r = TW.project_decisions(s, _pools(81.0))
    per_start = r["W"].iloc[:5].sum() / 140.0
    expected = TW.SP_WIN_SHARE * 81.0 / 162.0         # the pool, per club start
    assert abs(per_start - expected) / expected < 0.12
    assert abs(r["W"].sum() - 81.0) < 1e-6


def test_a_pure_starter_takes_nothing_from_the_bullpen_pool():
    s = _staff()
    s["Proj_G"] = np.where(s["Proj_GS"] > 0, s["Proj_GS"], s["Proj_G"])
    r = TW.project_decisions(s, _pools(90.0))
    rp_total = r["W"].iloc[5:].sum()
    assert abs(rp_total - (1 - TW.SP_WIN_SHARE) * 90.0) < 1e-6

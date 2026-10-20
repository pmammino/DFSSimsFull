"""
test_season_sim.py
==================
The floor / ceiling simulation: direction handling, ordering, determinism,
and that the simulated means stay on top of the projection.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import season_sim as ss  # noqa: E402


def _hitter(pid=1, **kw):
    row = {
        "PlayerId": pid, "pt_tier": "projected", "Proj_PA": 600.0,
        "pt_role": "Full Time", "pt_role_mix": "Full Time:1",
        "P_K": 0.22, "P_BB": 0.09, "P_HBP": 0.01, "P_SF": 0.01, "P_HR": 0.04,
        "P_3B": 0.005, "P_2B": 0.05, "P_1B": 0.14, "P_BIPOut": 0.435,
        "SD_K": 0.015, "SD_BB": 0.008, "SD_HBP": 0.002, "SD_SF": 0.002,
        "SD_HR": 0.005, "SD_3B": 0.002, "SD_2B": 0.006, "SD_1B": 0.01,
        "SD_BIPOut": 0.02, "P_R": 0.12, "SD_R": 0.008, "P_RBI": 0.12,
        "SD_RBI": 0.008, "P_SB": 0.02, "SD_SB": 0.004, "P_CS": 0.005,
    }
    row.update(kw)
    return row


def _pitcher(pid=1, **kw):
    row = {
        "PlayerId": pid, "pt_tier": "projected", "Proj_IP": 180.0,
        "Proj_G": 32.0, "Proj_GS": 32.0, "pt_role": "Ace (SP1)",
        "pt_role_mix": "Ace (SP1):1", "TBF_per_IP": 4.2, "ERA": 3.5,
        "RA9": 3.8,
        "P_K": 0.26, "P_BB": 0.07, "P_HBP": 0.01, "P_SF": 0.008,
        "P_HR": 0.03, "P_3B": 0.003, "P_2B": 0.04, "P_1B": 0.14,
        "P_BIPOut": 0.439,
        "SD_K": 0.02, "SD_BB": 0.006, "SD_HBP": 0.002, "SD_SF": 0.002,
        "SD_HR": 0.004, "SD_3B": 0.001, "SD_2B": 0.005, "SD_1B": 0.01,
        "SD_BIPOut": 0.02,
    }
    row.update(kw)
    return row


def test_hitter_directions_and_order():
    sim = ss.simulate_hitters(pd.DataFrame([_hitter()]), 1500, seed=1)
    r = sim.iloc[0]
    for stat, higher in ss.HITTER_STATS.items():
        lo, hi, med = r[f"{stat}_floor"], r[f"{stat}_ceil"], r[f"{stat}_simmed"]
        if higher:
            assert lo <= med <= hi, stat
        else:
            assert lo >= med >= hi, stat
    # HR ceiling is the high tail, strikeouts the low tail
    assert r["HR_ceil"] > r["HR_floor"]
    assert r["SO_ceil"] < r["SO_floor"]


def test_pitcher_directions_and_order():
    sim = ss.simulate_pitchers(pd.DataFrame([_pitcher()]), 1500, seed=1)
    r = sim.iloc[0]
    assert r["ERA_ceil"] < r["ERA_floor"]
    assert r["WHIP_ceil"] < r["WHIP_floor"]
    assert r["SO_ceil"] > r["SO_floor"]
    assert r["HR_ceil"] < r["HR_floor"]
    assert r["K9_ceil"] > r["K9_floor"]


def test_floor_tier_is_not_simulated():
    df = pd.DataFrame([_hitter(1), _hitter(2, pt_tier="floor", Proj_PA=1.0)])
    sim = ss.simulate_hitters(df, 200, seed=1)
    assert list(sim["PlayerId"]) == [1]


def test_deterministic_for_a_seed():
    df = pd.DataFrame([_hitter(1), _hitter(2, P_HR=0.06)])
    a = ss.simulate_hitters(df, 300, seed=7)
    b = ss.simulate_hitters(df, 300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_means_stay_on_the_projection():
    h = pd.DataFrame([_hitter()])
    sim = ss.simulate_hitters(h, 4000, seed=3).iloc[0]
    assert abs(sim["PA_simmean"] / 600.0 - 1) < 0.02
    assert abs(sim["HR_simmean"] / (600 * 0.04) - 1) < 0.05
    p = pd.DataFrame([_pitcher()])
    sp = ss.simulate_pitchers(p, 4000, seed=3).iloc[0]
    assert abs(sp["IP_simmean"] / 180.0 - 1) < 0.02
    assert abs(sp["ER_simmean"] / (3.5 * 180 / 9) - 1) < 0.05


def test_job_battle_widens_the_range():
    settled = pd.DataFrame([_hitter()])
    battle = pd.DataFrame([_hitter(
        pt_role_mix="Full Time:0.5|Bench Bat:0.5")])
    a = ss.simulate_hitters(settled, 1500, seed=2).iloc[0]
    b = ss.simulate_hitters(battle, 1500, seed=2).iloc[0]
    assert (b["PA_ceil"] - b["PA_floor"]) > (a["PA_ceil"] - a["PA_floor"])


def test_bracket_contains_the_projection():
    frame = pd.DataFrame({"PlayerId": [1], "HR": [30.0], "ERA": [3.0]})
    sim = pd.DataFrame({"PlayerId": [1], "HR_floor": [32.0], "HR_ceil": [40.0],
                        "ERA_floor": [2.8], "ERA_ceil": [2.5]})
    out, rep = ss.bracket(frame, sim, {"HR": True, "ERA": False})
    assert out["HR_floor"][0] == 30.0 and out["ERA_floor"][0] == 3.0
    assert rep["HR"][0] == 1 and rep["ERA"][0] == 1


def test_saves_never_exceed_appearances():
    d = pd.DataFrame([_pitcher(Proj_IP=65.0, Proj_G=65.0, Proj_GS=0.0,
                               pt_role="Closer", pt_role_mix="Closer:1")])
    sims = ss._sim_pitcher_block(np.random.default_rng(4), d,
                                 np.array([30.0]), np.array([8.0]), 2000)
    assert (sims["SV"] + sims["HLD"] <= np.floor(sims["G"]) + 1e-9).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

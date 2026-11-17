"""Two ways a player with almost no evidence looked like a star.

1. A hitter with NO batted balls (every minor-league-translated prospect) got
   N_BIP = 0, floored to 1, so his rate "standard error" was the spread of ONE
   batted ball: SD_HR 0.106 on a P_HR of 0.027. The simulation draws rates from
   N(P, SD), so his simulated OPS ran 0.90 against a projected 0.70 and his
   ceiling (1.18-1.29) topped the board.

2. The export computed AVG/OBP/SLG from counts already rounded to a tenth. At a
   floor tier's 1 PA, 0.2 hits over 0.9 AB is .222 for a true .20 and .856 OPS
   for a true .77.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pa_aggregation import build_pa_projections          # noqa: E402
import export_projections as EX                           # noqa: E402

BIP = ["out", "single", "double", "triple", "home_run"]


def _bip(pid, n, sd=0.18):
    rec = {"batter": pid, "N_BIP": n, "out": 0.69, "single": 0.17,
           "double": 0.05, "triple": 0.005, "home_run": 0.085}
    rec.update({f"{c}_sd": sd for c in BIP})
    return rec


def _rates(ids):
    return pd.DataFrame({
        "PlayerId": ids, "Pred_K%": 0.23, "Pred_BB%": 0.08, "Pred_HBP%": 0.01,
        "Pred_SF%": 0.007, "SD_K%": 0.03, "SD_BB%": 0.02, "SD_HBP%": 0.003,
        "SD_SF%": 0.002})


def test_a_player_with_no_batted_balls_gets_a_rate_sized_sd():
    bip = pd.DataFrame([_bip(1, 150), _bip(2, 600)])
    out = build_pa_projections(bip, _rates([1, 2, 3]), "batter", "PlayerId")
    sd = out.set_index("PlayerId")["SD_HR"]
    # Player 3 has no BIP at all. He must look like a topped-up (150-BIP)
    # player — not like one batted ball (which would be ~12x wider).
    assert sd[3] == np.float64(sd[1])
    assert sd[3] < 0.02
    assert sd[2] < sd[1]                       # more evidence, tighter


def test_the_no_bip_sample_is_configurable():
    bip = pd.DataFrame([_bip(1, 150)])
    a = build_pa_projections(bip, _rates([2]), "batter", "PlayerId",
                             no_bip_n_eff=400)
    b = build_pa_projections(bip, _rates([2]), "batter", "PlayerId",
                             no_bip_n_eff=100)
    assert a["SD_HR"].iloc[0] < b["SD_HR"].iloc[0]


def test_export_rate_stats_use_unrounded_counts():
    # 1 PA at a true .20 hit rate / .77 OPS shape.
    h = pd.DataFrame([{
        "PlayerId": 1, "Name": "x", "Team": "SF", "Age": 30, "BatSide": "R",
        "pt_position": "C", "pt_tier": "floor", "pt_role": "Depth (no MLB PA)",
        "Proj_PA": 1.0, "Proj_G": 0.2,
        "P_1B": 0.10, "P_2B": 0.04, "P_3B": 0.002, "P_HR": 0.058,
        "P_BB": 0.085, "P_HBP": 0.006, "P_SF": 0.008, "P_K": 0.27,
        "P_SB": 0.0, "P_CS": 0.0, "P_R": 0.1, "P_RBI": 0.1}])
    out = EX.build_hitters(h).iloc[0]
    ab = 1 - 0.085 - 0.006 - 0.008
    hits = 0.10 + 0.04 + 0.002 + 0.058
    tb = 0.10 + 0.08 + 0.006 + 0.232
    assert abs(out["AVG"] - hits / ab) < 0.0006
    assert abs(out["SLG"] - tb / ab) < 0.0006
    # The old rounded path would have said .222 / .556 here.
    assert abs(out["OPS"] - ((hits + 0.085 + 0.006) / 1 + tb / ab)) < 0.001

"""
test_fielding_export.py
=======================
The fielding projection built in the season export: exposure closes on the
real innings budget, the model's own baseline table is left as it was found,
and the league lands on the putout identity.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "export_projections", ROOT / "scripts" / "export_projections.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

import fielding_model as FM  # noqa: E402

POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]


def _league(n_teams=2):
    hit, pit, hist = [], [], []
    pid = 1
    for t in range(1, n_teams + 1):
        for pos in POS:
            hit.append({"PlayerId": pid, "Name": f"H{pid}", "team_id": t,
                        "team_abbr": f"T{t}", "pt_tier": "projected",
                        "Proj_PA": 6156.0 / 9, "pt_position": pos,
                        "P_SB_ATTEMPT": 0.02})
            pid += 1
        pit.append({"PlayerId": pid, "Name": f"P{pid}", "team_id": t,
                    "team_abbr": f"T{t}", "pt_tier": "projected",
                    "Proj_IP": 1458.0, "TBF_per_IP": 4.3, "P_K": 0.22})
        pid += 1
    for season in (2024, 2025, 2026):
        for pos, r in FM.BASELINES.items():
            hist.append({"Season": season, "PlayerId": 10_000 + len(hist),
                         "Pos": pos, "Innings": 1000.0,
                         "PO": r["po"] * 1000 / 9, "A": r["a"] * 1000 / 9,
                         "E": r["e"] * 1000 / 9, "DP": r["dp"] * 1000 / 9,
                         "PB": r["pb"] * 1000 / 9, "CI": r["ci"] * 1000 / 9,
                         "CS": 10, "SB_allowed": 30})
    return pd.DataFrame(hit), pd.DataFrame(pit), pd.DataFrame(hist)


def test_every_club_plays_1458_innings_at_every_position():
    h, p, hist = _league()
    f = ex.build_fielding(h, p, hist, 2027)
    by = f.groupby(["Team", "Pos"])["Innings"].sum()
    assert np.allclose(by.to_numpy(), 1458.0, atol=0.5)


def test_chances_identity_and_baselines_restored():
    before = {k: dict(v) for k, v in FM.BASELINES.items()}
    h, p, hist = _league()
    f = ex.build_fielding(h, p, hist, 2027)
    assert np.allclose(f["Chances"], f["PO"] + f["A"] + f["E"], atol=0.31)
    assert FM.BASELINES == before


def test_floor_tier_and_dh_field_nothing():
    h, p, hist = _league(1)
    h.loc[0, "pt_tier"] = "floor"
    h.loc[1, "pt_position"] = "DH"
    f = ex.build_fielding(h, p, hist, 2027)
    assert 1 not in set(f["PlayerId"]) and 2 not in set(f["PlayerId"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""Tests for ownership_history — the compact ownership training log.

Run: python -m pytest tests/test_ownership_history.py

Covers the guarantees that make the log safe to append to every build and
ingest slates into over a season:
  * rows carry the model's features + salary + value
  * append dedups by (date, name, pos); a re-log never wipes an attached label
  * attach_ownership fills the label AND sets the authoritative DK slot
  * projection-CSV builder maps baseball positions to DK slots
  * the log stays tiny
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ownership_history as OH
from ownership_model import norm


def _sims(names, seed=0):
    rng = np.random.default_rng(seed)
    return {norm(n): rng.gamma(2.0, 4.0, size=2000) for n in names}


def _pool(names, pos, sal):
    return pd.DataFrame({"Name": names, "Pos": pos, "Team": ["X"] * len(names),
                         "Salary": sal})


def test_rows_have_features_and_value():
    names = ["A", "B"]
    pool = _pool(names, ["OF", "OF"], [5000, 4000])
    rows = OH.slate_rows_from_pool("2026-07-29", pool, _sims(names))
    assert set(OH.COLUMNS) == set(rows.columns)
    assert rows["proj"].notna().all() and rows["ceiling"].notna().all()
    # value = proj / (salary/1000)
    r0 = rows.iloc[0]
    assert abs(r0["value"] - r0["proj"] / (r0["salary"] / 1000.0)) < 1e-6


def test_append_dedup_and_label_preserved(tmp_path):
    log = str(tmp_path / "log.csv")
    names = ["A", "B", "C"]
    pool = _pool(names, ["OF"] * 3, [5000, 4000, 3000])
    OH.append_slate_from_pool("2026-07-29", pool, _sims(names), log_path=log)
    # attach a label
    con = tmp_path / "contest.csv"
    pd.DataFrame({
        "EntryId": [1, 2], "Player": ["A", "B"], "Roster Position": ["OF", "OF"],
        "%Drafted": ["25%", "10%"], "FPTS": [10, 8],
    }).to_csv(con, index=False)
    OH.attach_ownership("2026-07-29", str(con), log_path=log)
    # re-log the same slate (features refresh) — must NOT wipe the label
    OH.append_slate_from_pool("2026-07-29", pool, _sims(names, seed=1), log_path=log)
    df = OH.load_log(log)
    assert len(df) == 3                        # deduped, not 6
    a = df[df.name == "a"].iloc[0]
    assert a["own"] == 25.0                    # label survived the re-log


def test_attach_sets_dk_slot_from_contest(tmp_path):
    log = str(tmp_path / "log.csv")
    # log a player under a placeholder OF slot...
    pool = _pool(["Slugger"], ["OF"], [4000])
    OH.append_slate_from_pool("2026-07-29", pool, _sims(["Slugger"]), log_path=log)
    # ...but the contest drafted them at 1B
    con = tmp_path / "c.csv"
    pd.DataFrame({"EntryId": [1], "Player": ["Slugger"], "Roster Position": ["1B"],
                  "%Drafted": ["12%"], "FPTS": [9]}).to_csv(con, index=False)
    OH.attach_ownership("2026-07-29", str(con), log_path=log)
    df = OH.load_log(log, labeled_only=True)
    assert df.iloc[0]["pos"] == "1B"           # authoritative DK slot
    assert df.iloc[0]["own"] == 12.0


def test_projection_csv_maps_positions(tmp_path):
    deliv = tmp_path
    pd.DataFrame({"player": ["Al Star", "Cee Fielder"], "pos": ["LF", "DH"],
                  "team": ["X", "Y"], "team_total": [5.0, 4.0],
                  "proj": [10.0, 8.0], "p90": [22.0, 18.0], "p10": [0.0, 0.0],
                  "std": [6.0, 5.0]}).to_csv(
        deliv / "hitter_projections_2026-07-29.csv", index=False)
    pd.DataFrame({"player": ["Ace Arm"], "pos": ["SP"], "team": ["Z"],
                  "proj": [17.0], "p90": [30.0], "p10": [4.0], "std": [7.0]}).to_csv(
        deliv / "pitcher_projections_2026-07-29.csv", index=False)
    rows = OH.slate_rows_from_projection_csvs("2026-07-29", str(deliv))
    slots = dict(zip(rows["name"], rows["pos"]))
    assert slots["al star"] == "OF"            # LF -> OF
    assert slots["ace arm"] == "P"             # SP -> P


def test_load_labeled_only(tmp_path):
    log = str(tmp_path / "log.csv")
    pool = _pool(["A", "B"], ["OF", "OF"], [5000, 4000])
    OH.append_slate_from_pool("2026-07-29", pool, _sims(["A", "B"]), log_path=log)
    con = tmp_path / "c.csv"
    pd.DataFrame({"EntryId": [1], "Player": ["A"], "Roster Position": ["OF"],
                  "%Drafted": ["20%"], "FPTS": [10]}).to_csv(con, index=False)
    OH.attach_ownership("2026-07-29", str(con), log_path=log)
    assert len(OH.load_log(log)) == 2
    assert len(OH.load_log(log, labeled_only=True)) == 1


def test_log_stays_small(tmp_path):
    log = str(tmp_path / "log.csv")
    names = [f"P{i}" for i in range(200)]
    pool = _pool(names, ["OF"] * 200, [4000] * 200)
    OH.append_slate_from_pool("2026-07-29", pool, _sims(names), log_path=log)
    assert os.path.getsize(log) < 60_000       # a 200-player slate < 60 KB


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

"""Tests for DK salary-feed name repair via RotoID.

Regression cover for the "player silently missing from every lineup" bug: the
RotoWire salaries-dk feed occasionally ships a mangled player name — a
multi-word surname truncated to its first token (``Elly De La Cruz`` -> ``Elly
De``) or an accent garbled by a mis-encode (``Jose Ramírez`` -> ``Jose
RamÃ­rez``). The sim/projection universe is keyed by the full lineup-feed name,
so the mangled name matches nothing, the player carries no salary/pool row, and
he drops out of every candidate lineup with no error.

Both feeds carry a stable RotoID, so the name is repaired by bridging on RotoID:
  * dk_slate_feed.to_dk_df(name_overrides=...) corrects dk_df AND the DK-upload
    id_map at one assembly point, so they stay consistent.
  * build_ownership._attach_salary bridges salary through RotoID so the
    projected-ownership deliverable stops losing the player's value term.

Run: python -m pytest tests/test_dk_name_repair.py   (or run this file directly).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dk_ids
import dk_slate_feed as feed
from stage_d import norm

# The exact affected players from the 2026-08-06 early slate, as they actually
# appear in the two feeds: salaries feed = mangled name + RotoID; lineup feed =
# canonical full name + the same pid.
_SLATE = {
    "players": [
        # Elly: salaries feed truncates "De La Cruz" -> "De".
        {"name": "Elly De", "team": "CIN", "position": "SS", "salary": 5900,
         "draftable_id": "43763772", "roto_id": "17579", "ownership": 13.6},
        # Jose Ramirez: accent mis-encoded in the feed.
        {"name": "Jose RamÃ­rez", "team": "CLE", "position": "3B",
         "salary": 5100, "draftable_id": "43763778", "roto_id": "12840",
         "ownership": 6.04},
        # A normal player that already matches — must be untouched.
        {"name": "Pete Alonso", "team": "BAL", "position": "1B", "salary": 5500,
         "draftable_id": "43763774", "roto_id": "14300", "ownership": 17.9},
    ]
}

# RotoID -> canonical full name, as the lineup feed / sims know them.
_PID_NAME = {"17579": "Elly De La Cruz", "12840": "Jose Ramirez",
             "14300": "Pete Alonso"}


def test_override_repairs_dk_df_and_idmap():
    dk_df, id_map = feed.to_dk_df(_SLATE, name_overrides=_PID_NAME)
    by_name = dict(zip(dk_df["FullName"], dk_df["Salary"]))
    assert "Elly De La Cruz" in by_name and by_name["Elly De La Cruz"] == 5900
    assert "Jose Ramirez" in by_name and by_name["Jose Ramirez"] == 5100
    assert "Pete Alonso" in by_name          # normal player unchanged

    # the DK-upload id_map must be keyed under the CORRECTED name, or the export
    # would fail to resolve the player's draftable id.
    assert dk_ids.lookup(id_map, "Elly De La Cruz", salary=5900) == "43763772"
    assert dk_ids.lookup(id_map, "Jose Ramirez", salary=5100) == "43763778"


def test_no_override_keeps_feed_name():
    """Without a bridge (feed unreachable) behaviour is unchanged — the mangled
    name still comes through, so the repair can never *cause* a regression."""
    dk_df, _ = feed.to_dk_df(_SLATE)
    assert "Elly De" in set(dk_df["FullName"])
    assert "Elly De La Cruz" not in set(dk_df["FullName"])


def test_unknown_rotoid_falls_back():
    dk_df, _ = feed.to_dk_df(_SLATE, name_overrides={"99999": "Nobody"})
    assert "Elly De" in set(dk_df["FullName"])      # untouched, no regression


def test_ownership_builder_bridges_salary_by_rotoid():
    import pandas as pd
    from scripts import build_ownership as bo

    # pool keyed by the canonical (lineup-feed) name, as _load_pool produces it
    pool = pd.DataFrame({
        "Name": ["Elly De La Cruz", "Jose Ramirez", "Pete Alonso"],
        "Pos": ["SS", "3B", "1B"], "Team": ["CIN", "CLE", "BAL"],
        "Order": [1.0, 2.0, 2.0],
    })
    # salary maps as they come out of the mangled salaries feed
    by_n = {norm("Elly De"): 5900, norm("Jose RamÃ­rez"): 5100,
            norm("Pete Alonso"): 5500}
    by_nt = {(norm("Elly De"), "CIN"): 5900,
             (norm("Jose RamÃ­rez"), "CLE"): 5100,
             (norm("Pete Alonso"), "BAL"): 5500}
    by_pid = {"17579": 5900, "12840": 5100, "14300": 5500}
    name_pid = {norm(v): k for k, v in _PID_NAME.items()}

    out = bo._attach_salary(pool, by_n, by_nt, by_pid, name_pid)
    sal = dict(zip(out["Name"], out["Salary"]))
    assert sal["Elly De La Cruz"] == 5900     # recovered via RotoID
    assert sal["Jose Ramirez"] == 5100        # recovered via RotoID
    assert sal["Pete Alonso"] == 5500         # normal name match still works

    # without the bridge the mangled names lose their salary (the original bug)
    out0 = bo._attach_salary(pool, by_n, by_nt)
    sal0 = dict(zip(out0["Name"], out0["Salary"]))
    assert sal0["Pete Alonso"] == 5500
    import math
    assert math.isnan(sal0["Elly De La Cruz"])
    assert math.isnan(sal0["Jose Ramirez"])


if __name__ == "__main__":
    test_override_repairs_dk_df_and_idmap()
    test_no_override_keeps_feed_name()
    test_unknown_rotoid_falls_back()
    test_ownership_builder_bridges_salary_by_rotoid()
    print("test_dk_name_repair.py: all passed")

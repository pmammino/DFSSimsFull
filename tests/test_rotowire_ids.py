"""
test_rotowire_ids.py
====================
Joining RotoWire's player master list to our players by name and birth year.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "export_projections", ROOT / "scripts" / "export_projections.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def _master():
    return pd.DataFrame([
        {"rw_id": 1, "name_key": "max muncy", "birth_year": 1990,
         "is_pitcher": False},
        {"rw_id": 2, "name_key": "max muncy", "birth_year": 2002,
         "is_pitcher": False},
        {"rw_id": 3, "name_key": "ann one", "birth_year": 1995,
         "is_pitcher": True},
        {"rw_id": 4, "name_key": "bob two", "birth_year": 1980,
         "is_pitcher": False},
    ])


def _pool(rows):
    return pd.DataFrame(rows, columns=["PlayerId", "name_key", "Age",
                                       "is_pitcher"])


def test_shared_name_is_settled_by_birth_year():
    ids = ex.match_rw_master(_pool([(10, "max muncy", 36, False),
                                    (11, "max muncy", 25, False)]),
                             _master(), 2027)
    assert ids[10] == 1 and ids[11] == 2


def test_lone_name_with_the_wrong_birth_year_is_not_guessed():
    # "bob two" was born 1980; a 22-year-old of the same name is someone else
    ids = ex.match_rw_master(_pool([(20, "bob two", 22, False)]),
                             _master(), 2027)
    assert 20 not in ids.index


def test_unknown_age_still_matches_a_unique_name():
    ids = ex.match_rw_master(_pool([(30, "ann one", float("nan"), True)]),
                             _master(), 2027)
    assert ids[30] == 3


def test_unlisted_player_is_blank_and_ids_are_not_shared():
    ids = ex.match_rw_master(_pool([(40, "nobody here", 30, False),
                                    (41, "ann one", 32, True),
                                    (42, "ann one", 32, True)]),
                             _master(), 2027)
    assert 40 not in ids.index
    # two of our players cannot both own RotoWire id 3
    assert 41 not in ids.index and 42 not in ids.index


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

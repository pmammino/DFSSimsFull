"""Tests for ud_json_export — the Underdog Battle Royale JSON shape."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ud_json_export as U


def test_nkey_matches_accented_and_plain_forms():
    assert U.nkey("José Ramírez") == U.nkey("Jose Ramirez") == "jose ramirez"


def test_nkey_strips_punctuation_and_suffixes():
    assert U.nkey("A.J. Puk") == "a j puk"
    assert U.nkey("Ken Griffey Jr.") == "ken griffey"
    assert U.nkey("Robinson Cano  III") == "robinson cano"


def test_bank_is_sorted_deterministic_and_capped():
    rng = np.random.default_rng(0)
    arr = rng.normal(10, 5, 10000)
    b1 = U._bank(arr, bank_size=500)
    b2 = U._bank(arr, bank_size=500)
    assert b1 == b2                       # deterministic
    assert len(b1) == 500
    assert b1 == sorted(b1)               # preserves sorted shape
    small = U._bank([1, 2, 3], bank_size=500)
    assert len(small) == 3                # never longer than the input


def test_build_ud_json_shape_and_values():
    hitter_ud = {"Yordan Alvarez": np.array([0, 2, 2, 3, 25])}
    pitcher_ud = {"Gerrit Cole": np.array([5, 10, 15])}
    payload = U.build_ud_json(hitter_ud, pitcher_ud, bank_size=5, default_sample=75)
    assert payload["bank_size"] == 5
    assert payload["default_sample"] == 75
    assert payload["meta"] == {"sport": "MLB", "format": "Battle Royale"}
    names = {p["name"] for p in payload["players"]}
    assert names == {"Yordan Alvarez", "Gerrit Cole"}
    yordan = next(p for p in payload["players"] if p["name"] == "Yordan Alvarez")
    assert yordan["pos"] is None
    assert yordan["nkey"] == "yordan alvarez"
    assert yordan["mean"] == round(float(np.mean([0, 2, 2, 3, 25])), 2)
    assert yordan["p90"] == round(float(np.percentile([0, 2, 2, 3, 25], 90)), 2)
    assert len(yordan["bank"]) == 5


def test_write_ud_json_round_trips(tmp_path):
    import json
    hitter_ud = {"Test Player": np.array([1, 2, 3, 4, 5])}
    path = os.path.join(tmp_path, "slate_UD.json")
    U.write_ud_json(hitter_ud, {}, path)
    loaded = json.load(open(path))
    assert loaded["players"][0]["name"] == "Test Player"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

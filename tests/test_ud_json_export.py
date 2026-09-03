"""Tests for ud_json_export — the Underdog Battle Royale JSON shape, with a
focus on the aligned-bank ("team logic") correlation guarantee."""
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


def test_aligned_indices_deterministic_and_capped():
    idx1 = U._aligned_indices(10000, bank_size=500)
    idx2 = U._aligned_indices(10000, bank_size=500)
    assert list(idx1) == list(idx2)       # deterministic (fixed seed)
    assert len(idx1) == 500
    assert list(idx1) == sorted(idx1)     # index order is stable/reproducible
    small = U._aligned_indices(3, bank_size=500)
    assert len(small) == 3                # never longer than the input


def test_banks_preserve_cross_player_correlation():
    """The whole point of 'team logic': two players who are perfectly
    correlated in the underlying sims (same team, same shared latent) must
    still be perfectly correlated bank-slot-by-bank-slot, and a player
    IDENTICAL to another (e.g. a teammate riding the same shared shock) must
    produce banks that line up value-for-value at every index — which an
    independently-sorted-per-player bank would destroy."""
    rng = np.random.default_rng(0)
    n = 10000
    shared = rng.normal(0, 1, n)
    teammate_a = shared * 2 + rng.normal(0, 0.001, n)   # near-perfectly correlated
    teammate_b = shared * 3 + rng.normal(0, 0.001, n)
    unrelated = rng.normal(0, 1, n)                      # independent

    payload = U.build_ud_json(
        {"Teammate A": teammate_a, "Teammate B": teammate_b, "Unrelated": unrelated},
        {}, bank_size=500)
    by_name = {p["name"]: np.array(p["bank"]) for p in payload["players"]}
    # same sim-world indices -> near-perfect linear relationship survives in the bank
    corr_teammates = np.corrcoef(by_name["Teammate A"], by_name["Teammate B"])[0, 1]
    corr_unrelated = np.corrcoef(by_name["Teammate A"], by_name["Unrelated"])[0, 1]
    assert corr_teammates > 0.99
    assert abs(corr_unrelated) < corr_teammates


def test_build_ud_json_shape_and_values():
    hitter_ud = {"Yordan Alvarez": np.array([0, 2, 2, 3, 25])}
    pitcher_ud = {"Gerrit Cole": np.array([5, 10, 15, 4, 8])}
    payload = U.build_ud_json(hitter_ud, pitcher_ud, bank_size=5, default_sample=75)
    assert payload["bank_size"] == 5
    assert payload["default_sample"] == 75
    assert payload["aligned"] is True
    assert payload["meta"] == {"sport": "MLB", "format": "Battle Royale", "n_players": 2}
    names = {p["name"] for p in payload["players"]}
    assert names == {"Yordan Alvarez", "Gerrit Cole"}
    yordan = next(p for p in payload["players"] if p["name"] == "Yordan Alvarez")
    assert yordan["pos"] is None
    assert yordan["nkey"] == "yordan alvarez"
    assert yordan["mean"] == round(float(np.mean([0, 2, 2, 3, 25])), 2)
    assert yordan["p90"] == round(float(np.percentile([0, 2, 2, 3, 25], 90)), 1)
    assert len(yordan["bank"]) == 5


def test_pos_map_overrides_null_default():
    hitter_ud = {"Yordan Alvarez": np.array([1, 2, 3])}
    payload = U.build_ud_json(hitter_ud, {}, bank_size=3,
                              pos_map={"yordan alvarez": "OF"})
    assert payload["players"][0]["pos"] == "OF"


def test_write_ud_json_round_trips(tmp_path):
    import json
    hitter_ud = {"Test Player": np.array([1, 2, 3, 4, 5])}
    path = os.path.join(tmp_path, "slate_UD.json")
    U.write_ud_json(hitter_ud, {}, path, bank_size=5)
    loaded = json.load(open(path))
    assert loaded["players"][0]["name"] == "Test Player"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

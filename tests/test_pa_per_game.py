"""Tests for sim_proj._pa_per_game — the lineup-slot -> plate-appearances curve.

Run: python -m pytest tests/test_pa_per_game.py

Guards the batting-order bias fix: the curve must (a) decrease monotonically
with lineup slot, (b) carry the empirical ~0.8-PA spread / ~0.10 slope (the
prior curve was half that, under-weighting top-of-order bats), and (c) keep the
9-slot average unchanged so the fix reshapes the order without shifting the
overall hitter level.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sim_proj import _pa_per_game, PA_MEAN, PA_SLOPE


def test_monotonic_decreasing():
    pas = [_pa_per_game(s) for s in range(1, 10)]
    assert all(a > b for a, b in zip(pas, pas[1:]))


def test_empirical_spread_and_slope():
    spread = _pa_per_game(1) - _pa_per_game(9)
    assert 0.7 <= spread <= 0.95            # empirical ~0.8, not the old ~0.44
    step = _pa_per_game(1) - _pa_per_game(2)
    assert abs(step - PA_SLOPE) < 1e-9      # slope ~0.10, not the old 0.055


def test_average_preserved():
    # holding the lineup-average PA fixed is what keeps the fix level-neutral
    avg = np.mean([_pa_per_game(s) for s in range(1, 10)])
    assert abs(avg - PA_MEAN) < 1e-9


def test_top_of_order_gets_a_bump_vs_old_curve():
    old = lambda s: 4.2 - (s - 1) * 0.055
    assert _pa_per_game(1) > old(1)         # leadoff up
    assert _pa_per_game(2) > old(2)
    assert _pa_per_game(9) < old(9)         # nine-hole down (redistributed)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

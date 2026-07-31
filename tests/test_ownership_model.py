"""Tests for ownership_model — the projected-ownership scorer.

Run: python -m pytest tests/test_ownership_model.py   (or run this file directly).

These are structural/behavioural guarantees that must hold on any slate,
independent of the fitted coefficients:
  * per-slot invariant  (Σ within a slot = 100% x slot count)
  * per-player cap       (no player above 100%)
  * monotonicity         (more projection => more ownership, all else equal)
  * value effect         (cheaper player of equal projection is more owned)
  * graceful degradation (missing salary / missing sim never crash or break Σ)
  * contest-size chalk    (small field chalkier, large field flatter)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ownership_model as OM
from ownership_model import (
    OwnershipParams, project_ownership, sim_features, size_beta,
    _cap_redistribute, SLOT_COUNT, norm,
)


def _pool_and_sims(seed=0, n_of=12, n_inf=6, n_p=8):
    """A small but realistic slate: OF group + one infield slot + pitchers."""
    rng = np.random.default_rng(seed)
    names, pos = [], []
    for i in range(n_of):
        names.append(f"OF{i}"); pos.append("OF")
    for i in range(n_inf):
        names.append(f"SS{i}"); pos.append("SS")
    for i in range(n_p):
        names.append(f"P{i}"); pos.append("P")
    sims, sal = {}, []
    for n in names:
        mu = rng.uniform(5, 13)
        sims[norm(n)] = rng.gamma(2.0, mu / 2.0, size=4000)
        sal.append(int(rng.integers(3000, 9500)))
    pool = pd.DataFrame({"Name": names, "Pos": pos, "Salary": sal})
    return pool, sims


def _slot_sums(pool, own):
    pool = pool.copy(); pool["own"] = own.values
    return pool.groupby("Pos")["own"].sum().to_dict()


# rounding to 3 dp means a slot of k players can drift up to ~k*5e-4
_TOL = 0.05


def _of_slot(target_names_projsal, n_fillers=10, filler_proj=6.0, seed=1):
    """Build a single OF slot with the named players of interest plus enough
    low-projection fillers that ownership is NOT saturated at the cap (a slot
    with only `slot_count` players trivially pins everyone at 100%)."""
    rng = np.random.default_rng(seed)
    names, sims, sal = [], {}, []
    for nm, proj, s in target_names_projsal:
        names.append(nm); sims[norm(nm)] = np.full(3000, proj); sal.append(s)
    for i in range(n_fillers):
        nm = f"fill{i}"; names.append(nm)
        sims[norm(nm)] = np.full(3000, filler_proj); sal.append(6000)
    pool = pd.DataFrame({"Name": names, "Pos": "OF", "Salary": sal})
    return pool, sims


def test_slot_invariant():
    pool, sims = _pool_and_sims()
    own = project_ownership(pool, sims)
    sums = _slot_sums(pool, own)
    for pos, s in sums.items():
        assert abs(s - 100.0 * SLOT_COUNT[pos]) < _TOL, (pos, s)


def test_invariant_holds_with_contest_size():
    pool, sims = _pool_and_sims()
    for size in (100, 1000, 50000):
        own = project_ownership(pool, sims, contest_size=size)
        for pos, s in _slot_sums(pool, own).items():
            assert abs(s - 100.0 * SLOT_COUNT[pos]) < _TOL, (size, pos, s)


def test_no_player_over_100():
    # force extreme concentration: one monster OF, everyone else tiny
    pool = pd.DataFrame({"Name": [f"OF{i}" for i in range(5)], "Pos": ["OF"] * 5})
    sims = {norm("OF0"): np.full(2000, 40.0)}
    for i in range(1, 5):
        sims[norm(f"OF{i}")] = np.full(2000, 1.0)
    own = project_ownership(pool, sims)
    assert own.max() <= 100.0 + 1e-6
    assert abs(own.sum() - 300.0) < 1e-3      # invariant still holds


def test_cap_redistribute_conserves_total():
    own = np.array([250.0, 30.0, 15.0, 5.0])   # first is impossibly > 100
    out = _cap_redistribute(own, slot_total=300.0, cap=100.0)
    assert out.max() <= 100.0 + 1e-9
    assert abs(out.sum() - 300.0) < 1e-6


def test_projection_monotonic():
    # equal salary, higher projection -> higher ownership
    pool, sims = _of_slot([("A", 12.0, 6000), ("B", 8.0, 6000)])
    own = project_ownership(pool, sims)
    own.index = pool["Name"]
    assert own["A"] > own["B"]


def test_value_effect_of_salary():
    # equal projection, different salary -> cheaper is more owned (value term)
    pool, sims = _of_slot([("cheap", 9.0, 3000), ("pricey", 9.0, 9000)])
    own = project_ownership(pool, sims)
    own.index = pool["Name"]
    assert own["cheap"] > own["pricey"]


def test_missing_salary_graceful():
    pool, sims = _pool_and_sims()
    pool = pool.drop(columns=["Salary"])          # no cost at all
    own = project_ownership(pool, sims)
    for pos, s in _slot_sums(pool, own).items():
        assert abs(s - 100.0 * SLOT_COUNT[pos]) < _TOL


def test_missing_sim_goes_to_floor():
    pool, sims = _of_slot([("has", 10.0, 6000)], n_fillers=3, filler_proj=9.0)
    # add two players with no sim at all
    pool = pd.concat([pool, pd.DataFrame(
        {"Name": ["missing1", "missing2"], "Pos": "OF", "Salary": 6000})],
        ignore_index=True)
    own = project_ownership(pool, sims)
    own.index = pool["Name"]
    assert own["has"] > own["missing1"]
    assert own["missing1"] <= 1.0                 # floored
    assert abs(own.sum() - 300.0) < _TOL


def test_contest_size_chalk_direction():
    # small field should concentrate the top player MORE than a large field
    pool, sims = _pool_and_sims(seed=3)
    small = project_ownership(pool, sims, contest_size=150)
    large = project_ownership(pool, sims, contest_size=50000)
    of = pool["Pos"] == "OF"
    assert small[of].max() >= large[of].max()


def test_size_beta_formula():
    P = OwnershipParams(chalk_k=0.3, n_medium=3000)
    assert size_beta(3000, P.n_medium, P.chalk_k) == 1.0        # medium = neutral
    assert size_beta(300, P.n_medium, P.chalk_k) > 1.0          # small = chalkier
    assert size_beta(30000, P.n_medium, P.chalk_k) < 1.0        # large = flatter


def test_sim_features_shape():
    a = np.array([0, 0, 0, 10, 20, 30], dtype=float)
    f = sim_features(a, ceil_pct=90.0)
    assert f["proj"] == a.mean()
    assert f["ceiling"] >= f["proj"]
    assert 1.0 <= f["ceil_shape"] <= 6.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

"""
test_bip_blend_weights.py
=========================
Guards the extra-base-hit blend fix.

The ported R recipe blended 0.75*mean + 0.25*median into the per-BIP
probability for double/triple/home_run, while single/out used the pure mean.
The per-BIP extra-base probability distribution is extremely right-skewed —
the median per-BIP HR probability is exactly 0.0, because most batted balls
cannot become home runs — so that median weight was a mechanical -25% haircut
on home runs, and the deflated mass was reallocated to singles when the BIP
events were renormalized.

Measured against real 2025 batted balls before the fix: HR 0.81x, 2B 0.83x,
3B 0.84x of their true share of the batted-ball pool, 1B at 1.07x.

These tests fail if anyone reintroduces a median weight on any event.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pa_aggregation import _per_season_event_dist  # noqa: E402
from pipeline_config import (  # noqa: E402
    BIP_OUTCOMES,
    EVENT_BLEND_WEIGHTS_HITTER,
    EVENT_BLEND_WEIGHTS_PITCHER,
)


def _skewed_hr_probs(n=400, seed=0):
    """A realistically right-skewed per-BIP HR probability column.

    ~85% of batted balls carry a near-zero HR probability and a thin tail
    carries most of the mass — median 0.0, mean well above it. This is the
    shape that makes a median blend a pure bias term.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros(n)
    tail = rng.random(n) < 0.15
    p[tail] = rng.uniform(0.15, 0.85, size=tail.sum())
    return p


def _frame(probs, group_col="batter"):
    df = pd.DataFrame({group_col: [1] * len(probs), "Season": [2026] * len(probs)})
    for ev in BIP_OUTCOMES:
        df[f"prob_{ev}"] = 0.0
    df["prob_home_run"] = probs
    return df


@pytest.mark.parametrize(
    "weights,label",
    [(EVENT_BLEND_WEIGHTS_HITTER, "hitter"),
     (EVENT_BLEND_WEIGHTS_PITCHER, "pitcher")],
)
def test_every_event_uses_pure_mean(weights, label):
    """No event may carry a median weight.

    Regression to the mean on batted-ball quality is the adaptive Bayesian
    pull's job (OUT_ADAPTIVE_K_*), which is sample-size aware. A median blend
    here is an uncontrolled, event-asymmetric bias on top of it.
    """
    for event, (w_mean, w_median) in weights.items():
        assert w_median == 0.0, (
            f"{label} blend for {event!r} has median weight {w_median}. "
            "The per-BIP extra-base probability distribution is right-skewed "
            "(median HR probability is 0.0), so a median weight deflates the "
            "event rather than making it robust."
        )
        assert w_mean == 1.0, (
            f"{label} blend for {event!r} has mean weight {w_mean}, expected "
            "1.0 — the weights must not rescale the estimate."
        )


def test_blend_weights_are_symmetric_across_events():
    """All events must be weighted identically.

    The original bug was not the median weight alone but its ASYMMETRY: HR/2B/3B
    were deflated while 1B and out were not, so renormalizing the BIP events
    moved the lost extra-base mass into singles. Uniform weights cannot
    redistribute mass between events.
    """
    assert len(set(EVENT_BLEND_WEIGHTS_HITTER.values())) == 1, (
        f"hitter blend weights differ across events: "
        f"{EVENT_BLEND_WEIGHTS_HITTER} — asymmetric weights redistribute mass "
        "between events on renormalization."
    )


def test_aggregation_recovers_the_mean_on_a_skewed_distribution():
    """End-to-end: the aggregated HR estimate equals the true mean."""
    probs = _skewed_hr_probs()
    assert np.median(probs) == 0.0, "fixture should have a zero median"

    out = _per_season_event_dist(
        _frame(probs), "batter", EVENT_BLEND_WEIGHTS_HITTER,
    )
    assert len(out) == 1
    assert out["home_run"].iloc[0] == pytest.approx(probs.mean(), rel=1e-12)


def test_median_blend_would_deflate_home_runs():
    """Demonstrates the bug this fix removes, so the test documents the why.

    Pinning the magnitude means a future reader can see what reintroducing a
    median weight would cost, without having to rebuild the pipeline.
    """
    probs = _skewed_hr_probs()
    buggy = {**EVENT_BLEND_WEIGHTS_HITTER, "home_run": (0.75, 0.25)}

    fixed_val = _per_season_event_dist(
        _frame(probs), "batter", EVENT_BLEND_WEIGHTS_HITTER,
    )["home_run"].iloc[0]
    buggy_val = _per_season_event_dist(
        _frame(probs), "batter", buggy,
    )["home_run"].iloc[0]

    # Median is 0, so the old recipe returns exactly 75% of the true mean.
    assert buggy_val == pytest.approx(0.75 * fixed_val, rel=1e-12)
    assert buggy_val < fixed_val

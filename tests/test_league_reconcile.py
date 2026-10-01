"""Offense and defense have to describe the same league.

Every plate appearance has one batter and one pitcher, so the hitter frame
and the pitcher frame are two views of one event stream and must agree event
by event once weighted by volume. Nothing made that true: the sides are
projected from separate panels, shrunk toward separately-computed league
means, and given playing time out of separate budgets. On the first clean
refresh the pitchers struck out 3.9% more batters than the hitters struck
out, and the league scored 21,661 runs while allowing 20,761.
"""

import numpy as np
import pandas as pd
import pytest

from league_reconcile import (
    SUFFIXES, batters_faced, league_vector, reconcile_league, reconcile_report,
)
from pipeline_config import PROB_EVENTS

# Roughly the real league, as a starting point for fixtures.
BASE = {"P_K": 0.2230, "P_BB": 0.0840, "P_HBP": 0.0110, "P_SF": 0.0060,
        "P_HR": 0.0300, "P_3B": 0.0037, "P_2B": 0.0420, "P_1B": 0.1420,
        "P_BIPOut": 0.4583}


def _rows(n, tilt=None, seed=0, suffixes=("",)):
    """n players whose events average to BASE scaled by `tilt`, summing to 1."""
    rng = np.random.default_rng(seed)
    tilt = tilt or {}
    mean = np.array([BASE[e] * tilt.get(e, 1.0) for e in PROB_EVENTS])
    mean = mean / mean.sum()
    noise = rng.normal(1.0, 0.08, size=(n, len(PROB_EVENTS))).clip(0.3, 2.0)
    block = mean[None, :] * noise
    block = block / block.sum(axis=1, keepdims=True)
    out = {}
    for s in suffixes:
        for j, e in enumerate(PROB_EVENTS):
            out[f"{e}{s}"] = block[:, j]
    return pd.DataFrame(out)


def _hitters(n=120, tilt=None, pa=600.0, seed=1, suffixes=("",)):
    df = _rows(n, tilt, seed, suffixes)
    df["Proj_PA"] = pa
    return df


def _pitchers(n=150, tilt=None, ip=100.0, seed=2, suffixes=("",)):
    df = _rows(n, tilt, seed, suffixes)
    df["Proj_IP"] = ip
    outs = sum(df[e] for e in ("P_K", "P_BIPOut", "P_SF"))
    df["TBF_per_IP"] = 3.0 / outs * 0.971
    return df


def _gap(h, p, event="P_K"):
    hv = league_vector(h, h["Proj_PA"].to_numpy(float))
    pv = league_vector(p, batters_faced(p))
    return hv[event] / pv[event] - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# The identity
# ─────────────────────────────────────────────────────────────────────────────

def test_the_two_sides_disagree_before_reconciliation():
    """Guard: the fixture must exercise the problem, or this proves nothing."""
    h = _hitters(tilt={"P_K": 0.92, "P_BB": 1.09})
    p = _pitchers()
    assert abs(_gap(h, p, "P_K")) > 0.02


def test_every_event_closes():
    h = _hitters(tilt={"P_K": 0.92, "P_BB": 1.09, "P_3B": 1.15})
    p = _pitchers()
    h2, p2, rep = reconcile_league(h, p)
    assert rep["applied"]
    for e in PROB_EVENTS:
        assert rep["after_hitters"][e] == pytest.approx(rep["after_pitchers"][e],
                                                        rel=1e-4), e


def test_runs_scored_equals_runs_allowed_once_the_volumes_agree():
    """The point of the whole exercise, in runs rather than rates."""
    from pitcher_outputs import LINEAR_WEIGHTS_RUNS as LW
    from pitcher_outputs import RUNS_INTERCEPT_DEFAULT as ICPT
    h = _hitters(tilt={"P_K": 0.92, "P_BB": 1.09})
    p = _pitchers()

    def rpa(vec):
        return sum(lw * vec[e] for e, lw in LW.items() if e in vec) + ICPT

    h2, p2, rep = reconcile_league(h, p)
    assert abs(rpa(rep["before_hitters"]) / rpa(rep["before_pitchers"]) - 1) > 0.02
    assert rpa(rep["after_hitters"]) == pytest.approx(rpa(rep["after_pitchers"]),
                                                      rel=1e-4)


def test_every_player_still_sums_to_one():
    """Nine events partition a plate appearance. That cannot be approximate."""
    h, p = _hitters(), _pitchers(tilt={"P_K": 1.06})
    h2, p2, _ = reconcile_league(h, p)
    for df in (h2, p2):
        assert df[PROB_EVENTS].sum(axis=1).to_numpy() == pytest.approx(1.0,
                                                                       abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Where the agreed value sits
# ─────────────────────────────────────────────────────────────────────────────

def test_a_weight_of_one_keeps_the_hitter_value():
    h, p = _hitters(), _pitchers(tilt={"P_K": 1.08})
    before = league_vector(h, h["Proj_PA"].to_numpy(float))["P_K"]
    _, _, rep = reconcile_league(h, p, hitter_weight=1.0)
    assert rep["after_hitters"]["P_K"] == pytest.approx(before, rel=2e-3)


def test_a_weight_of_zero_keeps_the_pitcher_value():
    h, p = _hitters(), _pitchers(tilt={"P_K": 1.08})
    before = league_vector(p, batters_faced(p))["P_K"]
    _, _, rep = reconcile_league(h, p, hitter_weight=0.0)
    assert rep["after_pitchers"]["P_K"] == pytest.approx(before, rel=2e-3)


def test_a_scalar_weight_applies_to_every_event():
    h, p = _hitters(tilt={"P_K": 0.95}), _pitchers()
    _, _, rep = reconcile_league(h, p, hitter_weight=0.5)
    assert set(rep["hitter_weight"]) == set(PROB_EVENTS)
    assert all(v == 0.5 for v in rep["hitter_weight"].values())


def test_per_event_weights_move_each_event_to_its_own_target():
    """The shipped configuration: batted balls lean hitter, K/BB split."""
    h = _hitters(tilt={"P_K": 0.92, "P_3B": 1.15})
    p = _pitchers()
    hv = league_vector(h, h["Proj_PA"].to_numpy(float))
    pv = league_vector(p, batters_faced(p))
    _, _, rep = reconcile_league(h, p,
                                 hitter_weight={"P_3B": 0.9, "P_K": 0.5})
    # The triple lands near the hitter value, the strikeout near the midpoint.
    assert rep["after_hitters"]["P_3B"] == pytest.approx(
        0.9 * hv["P_3B"] + 0.1 * pv["P_3B"], rel=0.02)
    assert rep["after_hitters"]["P_K"] == pytest.approx(
        0.5 * hv["P_K"] + 0.5 * pv["P_K"], rel=0.02)


def test_an_unlisted_event_falls_back_to_the_midpoint():
    h, p = _hitters(tilt={"P_BB": 1.10}), _pitchers()
    _, _, rep = reconcile_league(h, p, hitter_weight={"P_K": 0.9})
    assert rep["hitter_weight"]["P_BB"] == 0.5


def test_player_differentiation_survives():
    """A league-level correction must not flatten the players inside it."""
    h, p = _hitters(n=300, tilt={"P_K": 0.94}), _pitchers()
    h2, _, _ = reconcile_league(h, p)
    for e in ("P_K", "P_HR", "P_BB"):
        r = pd.Series(h[e]).corr(pd.Series(h2[e]), method="spearman")
        assert r > 0.99, f"{e} ranking scrambled (spearman {r:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# All the probability families move together
# ─────────────────────────────────────────────────────────────────────────────

def test_park_and_platoon_families_are_rescaled_too():
    """Correcting the neutral vector alone leaves the frame self-inconsistent."""
    sfx = ("", "_park", "_vL", "_vR")
    h = _hitters(tilt={"P_K": 0.94}, suffixes=sfx)
    p = _pitchers(suffixes=sfx)
    h2, p2, _ = reconcile_league(h, p)
    for s in sfx:
        cols = [f"{e}{s}" for e in PROB_EVENTS]
        assert h2[cols].sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-12)
        if s:
            # and it actually moved, rather than being silently skipped
            assert not np.allclose(h[cols].to_numpy(), h2[cols].to_numpy())


def test_a_frame_without_the_extra_families_is_fine():
    h, p = _hitters(), _pitchers()
    h2, p2, rep = reconcile_league(h, p)
    assert rep["applied"]
    assert all(f"P_K{s}" not in h2.columns for s in SUFFIXES if s)


# ─────────────────────────────────────────────────────────────────────────────
# Batters faced is the denominator, not innings
# ─────────────────────────────────────────────────────────────────────────────

def test_batters_faced_uses_the_pipelines_own_conversion():
    p = _pitchers(n=10, ip=100.0)
    assert batters_faced(p) == pytest.approx(
        (p["Proj_IP"] * p["TBF_per_IP"]).to_numpy())


def test_batters_faced_falls_back_before_the_pitcher_summary_step():
    p = _pitchers(n=10).drop(columns=["TBF_per_IP"])
    outs = p[["P_K", "P_BIPOut", "P_SF"]].sum(axis=1)
    assert batters_faced(p) == pytest.approx((p["Proj_IP"] * 3.0 / outs).to_numpy())


def test_innings_are_not_batters_faced():
    """Why the verifier's old weighting understated the gap.

    A high-strikeout, high-walk pitcher faces more batters per inning than a
    contact pitcher who works around them, so weighting a per-PA rate by
    innings under-counts exactly the pitchers who put men on base.
    """
    wild = _pitchers(n=1, tilt={"P_K": 1.3, "P_BB": 1.8}, seed=5)
    calm = _pitchers(n=1, tilt={"P_K": 0.7, "P_BB": 0.5}, seed=5)
    assert batters_faced(wild)[0] > batters_faced(calm)[0]
    assert wild["Proj_IP"].iloc[0] == calm["Proj_IP"].iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate input must not corrupt a frame
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("drop", ["Proj_PA", "Proj_IP"])
def test_missing_volume_skips_rather_than_guessing(drop):
    h, p = _hitters(), _pitchers()
    frame = h if drop == "Proj_PA" else p
    frame.drop(columns=[drop], inplace=True)
    h2, p2, rep = reconcile_league(h, p)
    assert not rep["applied"]
    assert h2 is h and p2 is p


def test_missing_probability_columns_skip_rather_than_guessing():
    h, p = _hitters(), _pitchers()
    h = h.drop(columns=["P_3B"])
    _, _, rep = reconcile_league(h, p)
    assert not rep["applied"]
    assert "missing" in rep["reason"]


def test_a_row_that_carries_no_distribution_is_left_alone():
    h, p = _hitters(), _pitchers(tilt={"P_K": 1.05})
    h.loc[h.index[0], PROB_EVENTS] = 0.0
    h.loc[h.index[1], PROB_EVENTS] = np.nan
    h2, _, rep = reconcile_league(h, p)
    assert rep["applied"]
    assert (h2.loc[h2.index[0], PROB_EVENTS] == 0).all()
    assert h2.loc[h2.index[1], PROB_EVENTS].isna().all()


def test_zero_volume_players_are_still_corrected():
    """They do not vote on the league, but their rates must still be right."""
    h, p = _hitters(), _pitchers(tilt={"P_K": 1.08})
    h.loc[h.index[0], "Proj_PA"] = 0.0
    before = h.loc[h.index[0], "P_K"]
    h2, _, _ = reconcile_league(h, p)
    assert h2.loc[h2.index[0], "P_K"] != pytest.approx(before, rel=1e-6)


def test_identical_sides_are_left_where_they_are():
    h = _hitters(seed=9)
    p = _pitchers(seed=9)
    p[PROB_EVENTS] = h[PROB_EVENTS].to_numpy()[:len(p)] if len(p) <= len(h) \
        else p[PROB_EVENTS]
    h = _hitters(n=100, seed=11)
    p = _pitchers(n=100, seed=11)
    p[PROB_EVENTS] = h[PROB_EVENTS].to_numpy()
    before = h[PROB_EVENTS].copy()
    h2, _, rep = reconcile_league(h, p)
    assert max(abs(v - 1) for v in rep["hitter_factors"].values()) < 0.02
    assert np.allclose(before.to_numpy(), h2[PROB_EVENTS].to_numpy(), atol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# The report
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# The R/RBI model has to score runs in the same league
# ─────────────────────────────────────────────────────────────────────────────

def _with_runs(h, r_per_pa=0.1250, rbi_ratio=0.95):
    h = h.copy()
    h["Pred_R_per_PA_neutral"] = r_per_pa
    h["Pred_RBI_per_PA_neutral"] = r_per_pa * rbi_ratio
    h["P_R"] = h["Pred_R_per_PA_neutral"]
    h["P_RBI"] = h["Pred_RBI_per_PA_neutral"]
    h["SD_R"] = r_per_pa * 0.2
    return h


def test_runs_are_scaled_to_the_runs_the_events_imply():
    """P_R comes from its own model and was tied to nothing.

    Every run is scored by exactly one batter, so the league's run total is
    one number — it cannot be one thing by the batting lines and another by
    the R column.
    """
    from league_reconcile import runs_per_pa
    h = _with_runs(_hitters(tilt={"P_K": 0.92, "P_BB": 1.09}))
    p = _pitchers()
    _, _, rep0 = reconcile_league(h, p)
    assert abs(0.1250 / runs_per_pa(rep0["after_hitters"]) - 1) > 0.02, \
        "fixture no longer shows the disagreement"

    h2, _, rep = reconcile_league(h, p)
    pa = h2["Proj_PA"].to_numpy(float)
    assert rep["runs"]["applied"]
    assert np.average(h2["P_R"], weights=pa) == pytest.approx(
        runs_per_pa(rep["after_hitters"]), rel=1e-6)


def test_the_neutral_rates_scale_too_or_the_season_layer_undoes_it():
    """season_engine rebuilds P_R as neutral * team_factor."""
    h = _with_runs(_hitters(tilt={"P_K": 0.92}))
    h2, _, rep = reconcile_league(h, _pitchers())
    f = rep["runs"]["factor"]
    assert h2["Pred_R_per_PA_neutral"].iloc[0] == pytest.approx(0.1250 * f)
    assert h2["P_R"].iloc[0] == pytest.approx(h2["Pred_R_per_PA_neutral"].iloc[0])


def test_the_rbi_per_run_ratio_is_untouched():
    h = _with_runs(_hitters(tilt={"P_K": 0.92}), rbi_ratio=0.95)
    h2, _, _ = reconcile_league(h, _pitchers())
    assert (h2["P_RBI"] / h2["P_R"]).to_numpy() == pytest.approx(0.95)


def test_the_sds_scale_with_the_rates():
    h = _with_runs(_hitters(tilt={"P_K": 0.92}))
    h2, _, rep = reconcile_league(h, _pitchers())
    assert h2["SD_R"].iloc[0] == pytest.approx(0.1250 * 0.2 * rep["runs"]["factor"])


def test_a_frame_without_runs_is_left_alone():
    h2, _, rep = reconcile_league(_hitters(), _pitchers())
    assert not rep["runs"]["applied"]
    assert "P_R" not in h2.columns


def test_the_report_names_the_size_of_the_correction():
    h, p = _hitters(tilt={"P_K": 0.94}), _pitchers()
    _, _, rep = reconcile_league(h, p)
    text = reconcile_report(rep)
    assert "largest correction" in text
    assert "batters faced" in text
    for e in PROB_EVENTS:
        assert e in text


def test_a_skipped_reconciliation_says_so():
    h, p = _hitters(), _pitchers()
    assert "SKIPPED" in reconcile_report(
        reconcile_league(h.drop(columns=["Proj_PA"]), p)[2])

"""The imputation has to put synthetic batted balls where real ones are.

Outcome is a sharply non-linear function of (launch_speed, launch_angle):
the difference between a double and a routine fly out is a few degrees at
the same exit velocity. So a sampler that gets the mean and covariance of
the joint right but its SHAPE wrong does not merely add noise — it moves
mass across outcome boundaries, and the boundaries it crosses are the
extra-base ones, because those occupy narrow bands.

That is what the legacy multivariate-normal sampler did. Measured against
the real 2024-26 batted balls in `bip_inputs/`, a Gaussian matched to the
league's own mean and covariance holds 51% of the real mass in the gap band
(EV >= 95, LA 8-20) and 64% in the barrel band (EV >= 98, LA 24-33), a
total variation distance of 0.255 from the real grid. Scored through the
classifier the synthetic rows came out at 1.083x real on doubles, 0.855x on
triples and 1.064x on home runs — on a pool that is 75% synthetic.

These tests pin the sampler that replaced it: a three-pool bootstrap, which
cannot misplace mass because every draw is a real observation.
"""

import numpy as np
import pandas as pd
import pytest

import bip_imputation as bi
from bip_imputation import (
    _draw_rows, _impute_player, _row_pool, _sample_bootstrap,
    _sample_gaussian, impute_bip,
)
from pipeline_config import BIP_CONT_VARS

VARS = BIP_CONT_VARS
TEAMS = ["NYY", "BOS", "LAD"]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: a curved joint, which is what the real (EV, LA) cloud is
# ─────────────────────────────────────────────────────────────────────────────

def _curved_pool(n: int, seed: int, ev_lo: float = 60.0,
                 ev_hi: float = 115.0) -> pd.DataFrame:
    """Rows lying along an arc in (launch_speed, launch_angle).

    Launch angle is a deterministic function of exit velocity plus a little
    noise, so the cloud is a thin curve. A Gaussian fitted to it has to fill
    in the whole enclosing ellipse, including the corners the curve never
    visits — the same failure mode as on real batted balls, in miniature.
    """
    rng = np.random.default_rng(seed)
    ev = rng.uniform(ev_lo, ev_hi, n)
    la = 45.0 - 0.012 * (ev - 70.0) ** 2 + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "launch_speed":   ev,
        "launch_angle":   la,
        "adjusted_angle": rng.normal(0, 20, n),
        "stand":          rng.choice(["L", "R"], n),
        "home_team":      rng.choice(TEAMS, n),
    })


def _on_curve(arr: np.ndarray, tol: float = 3.0) -> float:
    """Fraction of rows within `tol` degrees of the generating arc."""
    ev, la = arr[:, 0], arr[:, 1]
    return float(np.mean(np.abs(la - (45.0 - 0.012 * (ev - 70.0) ** 2)) <= tol))


def _pool_dict(frame: pd.DataFrame) -> dict:
    """A one-season population dict of the shape _impute_player expects."""
    df = frame.copy()
    df["Season"] = 2026
    pool = _row_pool(df, [2026], "Season", VARS, np.array([1.0]))
    vals = df[VARS].to_numpy(dtype=float)
    return {
        "mean":      vals.mean(axis=0),
        "cov":       np.cov(vals.T),
        "stand":     df["stand"].value_counts(normalize=True),
        "home_team": df["home_team"].value_counts(normalize=True),
        "vars":      VARS,
        **pool,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The shape claim
# ─────────────────────────────────────────────────────────────────────────────

def test_gaussian_sampler_leaves_the_curve_and_bootstrap_does_not():
    """The regression this sampler exists to prevent."""
    pop_frame = _curved_pool(4000, seed=1)
    pop = _pool_dict(pop_frame)
    real_share = _on_curve(pop_frame[VARS].to_numpy(dtype=float))
    assert real_share > 0.95, "the fixture itself should sit on the curve"

    rng = np.random.default_rng(7)
    boot = _sample_bootstrap(pop_frame.iloc[:0].copy(), None, pop, VARS,
                             w_curr=0.0, w_player=0.0, n=4000, rng=rng)
    gaus = _sample_gaussian(pop["mean"], pop["cov"], VARS, 4000,
                            np.random.default_rng(7))

    assert _on_curve(boot) == pytest.approx(real_share, abs=0.01)
    # The Gaussian scatters most of its draws into the empty interior of the
    # ellipse. The exact figure depends on the fixture's curvature; the point
    # is that it is nowhere near the real share.
    assert _on_curve(gaus) < 0.60


def test_bootstrap_draws_are_real_observations():
    pop_frame = _curved_pool(500, seed=2)
    pop = _pool_dict(pop_frame)
    drawn = _sample_bootstrap(pop_frame.iloc[:0].copy(), None, pop, VARS,
                              w_curr=0.0, w_player=0.0, n=300,
                              rng=np.random.default_rng(3))
    source = {tuple(np.round(r, 9)) for r in pop["rows"]}
    assert all(tuple(np.round(r, 9)) in source for r in drawn)


def test_bootstrap_cannot_invent_contact_quality_below_the_bound():
    """1.15% of real batted balls leave the bat under 40 mph.

    BIP_BOUNDS clips the Gaussian's unbounded tails at 40, which is right
    for a Gaussian and wrong for a real observation: clipping a 25 mph
    squibber up to 40 invents contact quality that did not happen. The
    bootstrap path is deliberately left unclipped, so a real weak-contact
    observation survives into the synthetic pool.
    """
    weak = _curved_pool(200, seed=4)
    weak.loc[:, "launch_speed"] = 25.0
    pop = _pool_dict(weak)
    player = weak.iloc[:1].copy()
    out = _impute_player(player, None, pop, min_obs=25, n_target=50,
                         rng=np.random.default_rng(5))
    assert out is not None
    assert out["launch_speed"].max() == pytest.approx(25.0, abs=0.05)

    bi_sampler = bi.IMP_SAMPLER
    try:
        bi.IMP_SAMPLER = "gaussian"
        clipped = _impute_player(player, None, pop, min_obs=25, n_target=50,
                                 rng=np.random.default_rng(5))
    finally:
        bi.IMP_SAMPLER = bi_sampler
    assert clipped["launch_speed"].min() >= 40.0


# ─────────────────────────────────────────────────────────────────────────────
# The mixture has to keep the shrinkage semantics it replaced
# ─────────────────────────────────────────────────────────────────────────────

def test_mixture_mean_matches_the_gaussian_shrinkage_target():
    """Same first moment, different shape — that is the whole trade."""
    player_frame = _curved_pool(400, seed=6, ev_lo=100.0, ev_hi=115.0)
    pop_frame    = _curved_pool(8000, seed=7, ev_lo=60.0, ev_hi=95.0)
    pop = _pool_dict(pop_frame)

    w_player = 0.65
    drawn = _sample_bootstrap(player_frame, None, pop, VARS, w_curr=1.0,
                              w_player=w_player, n=120_000,
                              rng=np.random.default_rng(8))
    target = (w_player * player_frame[VARS].to_numpy(dtype=float).mean(axis=0)
              + (1 - w_player) * pop["mean"])
    assert drawn.mean(axis=0) == pytest.approx(target, abs=0.35)


def test_player_weight_controls_how_often_his_own_profile_is_used():
    """w_player is the shrinkage dial, and it still is."""
    # Disjoint in exit velocity, so the source of each draw is identifiable.
    player_frame = _curved_pool(300, seed=9, ev_lo=108.0, ev_hi=115.0)
    pop_frame    = _curved_pool(3000, seed=10, ev_lo=60.0, ev_hi=80.0)
    pop = _pool_dict(pop_frame)

    def own_share(w_player):
        drawn = _sample_bootstrap(player_frame, None, pop, VARS, w_curr=1.0,
                                  w_player=w_player, n=20_000,
                                  rng=np.random.default_rng(11))
        return float(np.mean(drawn[:, 0] >= 100.0))

    assert own_share(0.90) == pytest.approx(0.90, abs=0.02)
    assert own_share(0.10) == pytest.approx(0.10, abs=0.02)
    assert own_share(0.00) == pytest.approx(0.00, abs=0.005)


def test_history_pool_is_used_in_proportion_to_w_curr():
    curr_frame = _curved_pool(200, seed=12, ev_lo=110.0, ev_hi=115.0)
    hist_frame = _curved_pool(200, seed=13, ev_lo=90.0, ev_hi=95.0)
    pop_frame  = _curved_pool(2000, seed=14, ev_lo=60.0, ev_hi=70.0)
    hist = hist_frame.copy()
    hist["Season"] = 2025
    hist_pool = _row_pool(hist, [2025], "Season", VARS, np.array([1.0]))

    drawn = _sample_bootstrap(curr_frame, hist_pool, _pool_dict(pop_frame),
                              VARS, w_curr=0.25, w_player=1.0, n=20_000,
                              rng=np.random.default_rng(15))
    ev = drawn[:, 0]
    assert float(np.mean(ev >= 105)) == pytest.approx(0.25, abs=0.02)
    assert float(np.mean((ev >= 85) & (ev < 105))) == pytest.approx(0.75, abs=0.02)
    assert float(np.mean(ev < 85)) == pytest.approx(0.0, abs=0.005)


def test_population_pool_decays_older_seasons():
    """A row three seasons back should be drawn less often than a recent one."""
    old   = _curved_pool(1000, seed=16, ev_lo=110.0, ev_hi=115.0)
    recent = _curved_pool(1000, seed=17, ev_lo=60.0, ev_hi=65.0)
    old["Season"], recent["Season"] = 2024, 2026
    df = pd.concat([old, recent], ignore_index=True)

    decay = 0.6
    pool = _row_pool(df, [2024, 2026], "Season", VARS,
                     np.array([decay ** 2, decay ** 0]))
    drawn = _draw_rows(pool, 40_000, np.random.default_rng(18))
    expected_old = decay ** 2 / (decay ** 2 + 1.0)
    assert float(np.mean(drawn[:, 0] >= 105)) == pytest.approx(expected_old,
                                                               abs=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate pools must fall back rather than raise or silently mis-sample
# ─────────────────────────────────────────────────────────────────────────────

def test_no_pool_at_all_returns_none_so_the_caller_can_fall_back():
    empty = {"rows": None, "row_w": None, "mean": np.zeros(3),
             "cov": np.eye(3), "vars": VARS}
    got = _sample_bootstrap(pd.DataFrame(columns=VARS), None, empty, VARS,
                            w_curr=1.0, w_player=1.0, n=10,
                            rng=np.random.default_rng(19))
    assert got is None


def test_player_with_one_observation_still_fills_its_quota():
    pop_frame = _curved_pool(2000, seed=20)
    player = _curved_pool(1, seed=21)
    out = _impute_player(player, None, _pool_dict(pop_frame), min_obs=25,
                         n_target=150, rng=np.random.default_rng(22))
    assert len(out) == 149
    assert out[VARS].notna().all().all()
    # n=1 against min_obs=25 is w_player = 1/(1+625): essentially all
    # population, which is the intended behaviour for a player we know
    # nothing about. His one batted ball was 103 mph, and that must NOT
    # turn into a synthetic pool of 103 mph contact.
    pop_share = float(np.mean(pop_frame["launch_speed"] >= 100))
    assert float(np.mean(out["launch_speed"] >= 100)) == pytest.approx(
        pop_share, abs=0.12)


def test_a_full_time_player_is_not_imputed_at_all():
    pop_frame = _curved_pool(2000, seed=23)
    player = _curved_pool(200, seed=24)
    assert _impute_player(player, None, _pool_dict(pop_frame), min_obs=25,
                          n_target=150, rng=np.random.default_rng(25)) is None


def test_missing_population_rows_fall_back_to_the_player_pool():
    """If only the player has rows, every draw must come from him."""
    player_frame = _curved_pool(40, seed=26, ev_lo=108.0, ev_hi=115.0)
    pop = _pool_dict(_curved_pool(100, seed=27))
    pop["rows"], pop["row_w"] = None, None
    drawn = _sample_bootstrap(player_frame, None, pop, VARS, w_curr=1.0,
                              w_player=0.4, n=500,
                              rng=np.random.default_rng(28))
    assert drawn is not None and len(drawn) == 500
    assert float(np.mean(drawn[:, 0] >= 105)) == pytest.approx(1.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# End to end
# ─────────────────────────────────────────────────────────────────────────────

def _bip_frame(seed: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for season in (2025, 2026):
        for i, n in enumerate([400, 200, 40, 5]):
            f = _curved_pool(n, seed=seed + season + i)
            f["Season"]  = season
            f["batter"]  = 100 + i
            f["pitcher"] = 500 + rng.integers(0, 3, n)
            f["events"]  = rng.choice(["field_out", "single", "double"], n)
            parts.append(f)
    return pd.concat(parts, ignore_index=True)


def test_impute_bip_tops_every_batter_up_and_flags_the_synthetic_rows():
    df = _bip_frame()
    out = impute_bip(df, target_n_batter=150, target_n_pitcher=200,
                     verbose=False)
    assert out["_synthetic"].dtype == bool or out["_synthetic"].notna().all()
    real = out[~out["_synthetic"].astype(bool)]
    assert len(real) == len(df.dropna(subset=VARS))

    per_batter = (out[out["batter"].notna()]
                  .groupby(["Season", "batter"]).size())
    assert per_batter.min() >= 150
    # Synthetic rows carry no outcome — they are inputs to the classifier,
    # never training labels.
    synth = out[out["_synthetic"].astype(bool)]
    assert synth["events"].isna().all()


def test_end_to_end_synthetic_rows_land_on_the_real_support():
    df = _bip_frame(seed=40)
    out = impute_bip(df, target_n_batter=150, target_n_pitcher=200,
                     verbose=False)
    synth = out[out["_synthetic"].astype(bool)]
    assert len(synth) > 0
    # Rounding is applied after sampling, so compare against the curve with
    # a tolerance that covers it rather than against exact source rows.
    assert _on_curve(synth[VARS].to_numpy(dtype=float), tol=4.0) > 0.95

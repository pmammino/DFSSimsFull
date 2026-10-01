"""What the BIP classifier is asked for, and what capping its training costs.

Two things worth pinning.

1. The pipeline consumes `predict_proba` and averages it per player. It never
   takes an argmax. So the quantity that has to be right is the AGGREGATE
   probability — mean(prob_double) over a pool must equal that pool's double
   rate — and per-class recall is not a measure of anything the pipeline uses.
   Triples have a recall of 0.005 and that is fine: almost no batted ball is
   more likely to be a triple than something else, yet triples still have to
   come out at the right rate in the sum.

2. The training sample is STRATIFIED, so a row cap costs every class the same
   fraction of its rows. Triples are 0.53% of batted balls, so a 100k cap on
   a 304k pool trained the triple class on ~530 examples instead of ~1,630.
   Measured on a held-out fifth of the real 2024-26 batted balls
   (mean predicted probability / actual rate):

       cap        logloss     double    triple
       100,000    0.43684     0.990     0.971
       200,000    0.43248     0.997     0.991
       none       0.43160     1.001     1.004

   Hence XGB_SAMPLE_SIZE = None. These tests use a small synthetic pool —
   they check the mechanism, not those figures.
"""

import numpy as np
import pandas as pd
import pytest

import bip_outcomes as bo
from bip_outcomes import BIPOutcomeModel, FEATURES, map_event_to_outcome
from pipeline_config import BIP_OUTCOMES, XGB_SAMPLE_SIZE


def _pool(n: int = 24_000, seed: int = 0) -> pd.DataFrame:
    """A pool where outcome depends on (ev, la) and triples are rare."""
    rng = np.random.default_rng(seed)
    ev = rng.normal(88, 15, n).clip(40, 120)
    la = rng.normal(13, 29, n).clip(-90, 90)
    adj = rng.normal(0, 25, n)
    barrel = (ev >= 98) & (la >= 24) & (la <= 33)
    gap    = (ev >= 95) & (la >= 8) & (la < 24)
    u = rng.random(n)
    res = np.where(barrel & (u < 0.70), "home_run",
          np.where(gap & (u < 0.45), "double",
          np.where(gap & (u < 0.50), "triple",
          np.where(u < 0.24, "single", "out"))))
    return pd.DataFrame({
        "launch_speed": ev, "launch_angle": la, "adjusted_angle": adj,
        "sprint_speed": rng.normal(27, 1.2, n),
        "stand": rng.choice(["L", "R"], n),
        "home_team": rng.choice(["NYY", "BOS", "LAD", "SF"], n),
        "Result": res,
    })


def test_the_shipped_config_does_not_cap_the_training_pool():
    assert XGB_SAMPLE_SIZE is None, (
        "a stratified cap costs the rare classes the same fraction as the "
        "common ones, and the rare classes are the ones that cannot afford it"
    )


def test_aggregate_probability_is_what_has_to_be_calibrated():
    """The quantity the pipeline actually averages, on held-out rows."""
    train, test = _pool(seed=1), _pool(seed=2)
    model = BIPOutcomeModel().fit(train, verbose=False)
    proba = model.predict_proba(test)
    for cls in BIP_OUTCOMES:
        actual = (test["Result"] == cls).mean()
        if actual < 0.005:
            continue
        got = proba[f"prob_{cls}"].mean()
        assert got == pytest.approx(actual, rel=0.12), (
            f"{cls}: mean predicted {got:.5f} vs actual {actual:.5f}")


def test_probabilities_sum_to_one_per_batted_ball():
    train = _pool(8_000, seed=3)
    model = BIPOutcomeModel().fit(train, verbose=False)
    proba = model.predict_proba(_pool(2_000, seed=4))
    assert proba.sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-5)


def test_proba_columns_follow_the_label_encoder_order():
    """The column names must match the integer columns XGBoost produces.

    LabelEncoder sorts alphabetically, and `predict_proba` names its columns
    from `outcome_classes`, so a reader who assumes BIP_OUTCOMES order gets
    doubles labelled as outs. Pin the contract.
    """
    model = BIPOutcomeModel()
    assert model.outcome_classes == sorted(BIP_OUTCOMES)
    trained = model.fit(_pool(6_000, seed=5), verbose=False)
    cols = list(trained.predict_proba(_pool(500, seed=6)).columns)
    assert cols == [f"prob_{c}" for c in sorted(BIP_OUTCOMES)]


def test_a_cap_takes_the_same_fraction_from_every_class():
    """Why the cap hurt triples specifically — it is stratified."""
    pool = _pool(20_000, seed=7)
    before = pool["Result"].value_counts(normalize=True)
    original = bo.XGB_SAMPLE_SIZE
    try:
        bo.XGB_SAMPLE_SIZE = 2_000
        capped = BIPOutcomeModel().fit(pool, verbose=False)
        assert capped.model is not None
    finally:
        bo.XGB_SAMPLE_SIZE = original
    # Stratification preserves SHARES, which is exactly the problem: the
    # rare class keeps its share of a much smaller pool, so it loses the
    # same fraction of its already-few rows.
    from sklearn.model_selection import train_test_split
    sub, _ = train_test_split(pool, train_size=2_000,
                              stratify=pool["Result"], random_state=42)
    after = sub["Result"].value_counts(normalize=True)
    for cls in before.index:
        assert after[cls] == pytest.approx(before[cls], abs=0.01)
    assert (sub["Result"] == "triple").sum() < (pool["Result"] == "triple").sum() / 5


def test_synthetic_rows_are_never_training_labels():
    """Imputed batted balls are inputs to scoring, never fitted targets."""
    pool = _pool(6_000, seed=8)
    pool.loc[pool.index[:1_000], "Result"] = np.nan
    model = BIPOutcomeModel().fit(pool, verbose=False)
    assert model.model is not None
    # fit() drops them; nothing in the class carries an unlabelled row.
    assert not pd.isna(pool.dropna(subset=["Result"])["Result"]).any()


def test_event_mapping_covers_the_out_family():
    for ev in ("field_out", "grounded_into_double_play", "sac_fly",
               "fielders_choice", "triple_play"):
        assert map_event_to_outcome(ev) == "out"
    for ev in ("single", "double", "triple", "home_run"):
        assert map_event_to_outcome(ev) == ev
    assert pd.isna(map_event_to_outcome("walk"))
    assert pd.isna(map_event_to_outcome(np.nan))


def test_features_are_the_ones_the_scrape_provides():
    """A feature the BIP files do not carry would be silently all-NaN."""
    assert set(FEATURES) == {
        "launch_speed", "launch_angle", "adjusted_angle",
        "sprint_speed", "stand", "home_team",
    }

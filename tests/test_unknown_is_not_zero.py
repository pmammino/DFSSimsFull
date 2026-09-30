"""
Unknown is not zero.

One pathology, found in four places in a single review, so it gets a
cross-cutting test rather than four scattered ones:

    a count of 0.0 standing in for "we have no data", then read back as an
    observation of zero.

The shrinkage projectors all share a shape — filter a prior pool, build
`league_rate = sum(count) / sum(PA)`, then blend each player's own rate
against it. Feeding them a fabricated zero does damage twice:

  - the player's own rate reads as a real 0.000 season, and because n_eff comes
    from the same row, the MORE fake data he has the harder his zero outweighs
    the prior;
  - his PA lands in the league-rate denominator with nothing in the numerator,
    which biases EVERY player low, including real MLB ones.

Measured on realistic pools before the fix:

    R/PA    real MLB hitter   0.1092 vs 0.1150 truth   (-5.0%)
            MLE hitter        0.0399                   (35% of league)
    WP/PA   real MLB pitcher  0.00656 vs 0.00729       (-10.0%)
            MLE pitcher       0.00376                  (52% of league)

The MLE rows now carry NaN for everything the minors feed does not supply
(R, RBI, IBB, SH, WP, BK) and the projectors separate evidence from mere
presence. Coverage is preserved: a player with no history gets the league
rate, which is the honest estimate, not a dropped row and not a zero.

Run with:  python -m pytest tests/test_unknown_is_not_zero.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitcher_outputs import project_wp_per_pa
from runs_rbi_model import _project_neutral_rate

REAL_R_PER_PA = 0.115
REAL_WP_PER_PA = 5.1 / 700.0


# ─────────────────────────────────────────────────────────────────────────────
# the MLE rows themselves
# ─────────────────────────────────────────────────────────────────────────────

def _feed():
    return {"AAA": {"hitters": [{
        "mlbam_id": 691234, "player": "A Prospect", "games": "120",
        "ab": "450", "walks": "40", "strikes": "95", "hits": "130",
        "doubles": "28", "triples": "3", "hr": "18", "steals": "9",
        "caught": "3", "runs": "70", "rbi": "75", "parent_org_id": 140,
    }], "pitchers": [{
        "mlbam_id": 691235, "player": "An Arm", "games": "25", "gs": "25",
        "ip": "140.2", "h": "120", "er": "55", "hr": "14", "k": "150",
        "bb": "45", "bf": "580", "parent_org_id": 140,
    }]}}


@pytest.mark.parametrize("role,cols", [
    ("hitter",  ["R", "RBI", "IBB", "SH"]),
    ("pitcher", ["IBB", "SH", "WP", "BK"]),
])
def test_untranslated_columns_are_nan_not_zero(role, cols):
    """The feed does not carry these, so the row must not claim zero."""
    from mle_translations import build_synthetic_rows
    rows, _, stats = build_synthetic_rows(
        _feed(), role, 2027, {}, None, existing_ids=set(), levels=("AAA",))
    assert stats["used"] == 1, stats
    r = rows.iloc[0]
    for c in cols:
        assert pd.isna(r[c]), f"{role} row has {c}={r[c]!r}, expected NaN"


def test_translated_columns_are_still_populated():
    """The fix must not blank out what IS translated."""
    from mle_translations import build_synthetic_rows
    rows, _, _ = build_synthetic_rows(
        _feed(), "hitter", 2027, {}, None, existing_ids=set(), levels=("AAA",))
    r = rows.iloc[0]
    for c in ("PA", "AB", "K", "BB", "H", "HR", "2B", "3B", "SB", "CS"):
        assert pd.notna(r[c]) and float(r[c]) > 0, f"{c} = {r[c]!r}"


def test_pitcher_ip_stays_raw_for_role_classification():
    """IP is deliberately NOT deflated — see the note in _pitcher_row.

    role_from_ip_per_g reads IP/G, and a AAA starter starts games. Deflating
    IP would turn 5.6 IP/G into 3.1 and misclassify him as a reliever.
    """
    from mle_translations import build_synthetic_rows
    rows, _, _ = build_synthetic_rows(
        _feed(), "pitcher", 2027, {}, None, existing_ids=set(), levels=("AAA",))
    r = rows.iloc[0]
    ip_per_g = float(r["IP"]) / float(r["G"])
    assert ip_per_g >= 3.5, f"IP/G {ip_per_g:.2f} would misclassify a starter"


# ─────────────────────────────────────────────────────────────────────────────
# R / RBI projector
# ─────────────────────────────────────────────────────────────────────────────

def _hitter_pool(mle_value, n_real=400, n_mle=600):
    rows = []
    for i in range(n_real):
        for season in (2025, 2026):
            rows.append({"PlayerId": i, "Season": season, "PA": 500.0,
                         "R": 500.0 * REAL_R_PER_PA, "TeamId": 147})
    for j in range(n_mle):
        rows.append({"PlayerId": 10_000 + j, "Season": 2026, "PA": 275.0,
                     "R": mle_value, "TeamId": 147})
    df = pd.DataFrame(rows)
    df["R_per_PA"] = df["R"] / df["PA"].replace(0, np.nan)
    return df


def _project_r(df):
    """Frame only. `_project_neutral_rate` also returns the league rate, which
    project_runs_and_rbi needs so a player with no projection can be filled
    with league average instead of a confident 0.000."""
    frame, _league = _project_neutral_rate(
        df, 2027, "R_per_PA", "R", {},
        k_pa=200.0, decay=0.85, max_history_years=5)
    return frame


def _project_r_with_league(df):
    return _project_neutral_rate(df, 2027, "R_per_PA", "R", {},
                                 k_pa=200.0, decay=0.85, max_history_years=5)


def test_projector_reports_the_league_rate_it_used():
    """The caller needs it to fill players who got no projection."""
    _frame, league = _project_r_with_league(_hitter_pool(np.nan))
    assert league == pytest.approx(REAL_R_PER_PA, abs=1e-4)


def test_unknown_runs_do_not_bias_real_hitters():
    """The league rate must not absorb PA with no runs attached."""
    out = _project_r(_hitter_pool(np.nan))
    real = out[out.PlayerId < 10_000]["Pred_R_per_PA_neutral"]
    assert real.mean() == pytest.approx(REAL_R_PER_PA, abs=1e-4)


def test_a_fabricated_zero_would_bias_them():
    """Guard: the pool must actually exercise the bug, or the test is vacuous."""
    out = _project_r(_hitter_pool(0.0))
    real = out[out.PlayerId < 10_000]["Pred_R_per_PA_neutral"]
    assert real.mean() < REAL_R_PER_PA * 0.98, (
        "fixture no longer reproduces the league-rate deflation")


def test_unknown_runs_project_to_league_average():
    out = _project_r(_hitter_pool(np.nan))
    mle = out[out.PlayerId >= 10_000]["Pred_R_per_PA_neutral"]
    assert mle.mean() == pytest.approx(REAL_R_PER_PA, abs=1e-4)
    assert (mle > 0).all(), "no projection may be zero"


def test_unknown_runs_keep_their_rows():
    """Coverage is the point: nobody may be dropped for lacking run data."""
    df = _hitter_pool(np.nan)
    out = _project_r(df)
    assert len(out) == df["PlayerId"].nunique()
    assert out["Pred_R_per_PA_neutral"].notna().all()


def test_players_with_no_run_data_carry_zero_n_eff():
    """n_eff must report the evidence, and there is none."""
    out = _project_r(_hitter_pool(np.nan))
    mle = out[out.PlayerId >= 10_000]
    assert (mle["n_eff_R_per_PA"] == 0.0).all()
    real = out[out.PlayerId < 10_000]
    assert (real["n_eff_R_per_PA"] > 0).all()


def test_real_hitters_still_differentiate():
    """Shrinkage toward a correct league rate, not flattening."""
    df = _hitter_pool(np.nan, n_real=2)
    df.loc[df.PlayerId == 0, "R"] = 500.0 * 0.16      # strong
    df.loc[df.PlayerId == 1, "R"] = 500.0 * 0.07      # weak
    df["R_per_PA"] = df["R"] / df["PA"].replace(0, np.nan)
    out = _project_r(df).set_index("PlayerId")
    assert (out.loc[0, "Pred_R_per_PA_neutral"]
            > out.loc[1, "Pred_R_per_PA_neutral"])


def test_no_usable_run_history_returns_empty_not_zeros():
    """Degenerate pool: better no projection than a confident 0.000."""
    df = _hitter_pool(np.nan, n_real=0, n_mle=50)
    out = _project_r(df)
    assert out.empty or (out["Pred_R_per_PA_neutral"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# WP projector
# ─────────────────────────────────────────────────────────────────────────────

def _pitcher_pool(mle_value, n_real=400, n_mle=800):
    rows = []
    for i in range(n_real):
        for season in (2025, 2026):
            rows.append({"PlayerId": i, "Season": season, "TBF": 700.0,
                         "WP": 5.1})
    for j in range(n_mle):
        rows.append({"PlayerId": 10_000 + j, "Season": 2026, "TBF": 275.0,
                     "WP": mle_value})
    return pd.DataFrame(rows)


def test_unknown_wild_pitches_do_not_bias_real_pitchers():
    out = project_wp_per_pa(_pitcher_pool(np.nan), 2027)
    real = out[out.PlayerId < 10_000]["Pred_WP_per_PA"]
    assert real.mean() == pytest.approx(REAL_WP_PER_PA, abs=1e-5)


def test_a_fabricated_zero_wp_would_bias_them():
    out = project_wp_per_pa(_pitcher_pool(0.0), 2027)
    real = out[out.PlayerId < 10_000]["Pred_WP_per_PA"]
    assert real.mean() < REAL_WP_PER_PA * 0.98, (
        "fixture no longer reproduces the WP league-rate deflation")


def test_unknown_wild_pitches_project_to_league_average():
    out = project_wp_per_pa(_pitcher_pool(np.nan), 2027)
    mle = out[out.PlayerId >= 10_000]["Pred_WP_per_PA"]
    assert mle.mean() == pytest.approx(REAL_WP_PER_PA, abs=1e-5)


def test_unknown_wild_pitches_keep_their_rows():
    df = _pitcher_pool(np.nan)
    out = project_wp_per_pa(df, 2027)
    assert len(out) == df["PlayerId"].nunique()
    assert out["Pred_WP_per_PA"].notna().all()


def test_wp_stays_within_its_clip():
    out = project_wp_per_pa(_pitcher_pool(np.nan), 2027)
    v = out["Pred_WP_per_PA"]
    assert (v >= 0).all() and (v <= 1).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

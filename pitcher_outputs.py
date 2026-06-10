"""
pitcher_outputs.py
==================
Derive pitcher-level summary stats from per-PA event projections.

Adds four pitcher outputs:
    1. RA9 — runs allowed per 9 IP, computed from per-PA rates via linear
       weights. Uses the same machinery as wOBA/Run Creation but calibrated
       to absolute MLB R/PA. Both neutral and park-adjusted versions are
       produced.
    2. ERA — earned run average. Derived from RA9 via the league-stable
       ER/RA ratio. Historical analysis on 2018-2025 data shows this ratio
       is essentially a league constant ≈ 0.918 (range 0.912-0.927).
    3. TBF_per_IP — batters faced per inning, derived from the projected
       per-PA out rate (K + BIPOut + SF). Slightly inflated relative to
       actuals because not every "BIPOut" is exactly one out (GIDP saves an
       inning), so an empirical calibration factor is applied.
    4. WP_per_PA — wild pitches per PA, projected via PA-weighted recency-
       decay shrinkage on the pitcher's history. YoY r ≈ 0.40 so we shrink
       moderately heavily to the league mean (~0.0075 per PA).

The HBP rate is already in the existing per-PA output as `P_HBP`. To make
that more obvious for pitcher consumers, we expose it as both the original
`P_HBP` and a renamed `HBP_pct` column scaled to a percentage. Same value,
just two views.

Linear weights for RA9
----------------------
Standard per-PA run weights (variant of Tom Tango's "The Book" / FanGraphs
wOBA scale, calibrated empirically to absolute runs):

    BB:    +0.30
    HBP:   +0.34
    1B:    +0.45
    2B:    +0.77
    3B:    +1.06
    HR:    +1.38
    K:     -0.03 (slightly worse than non-K out — no productive out potential)
    BIPOut: 0 (reference event)
    SF:    0 (treated as BIPOut; SF rate is ~0.005 per PA so negligible)

After summing weight × P_event, we add an empirical intercept calibrated
against ~1700 pitcher-seasons (2022-2025, 40+ IP). The intercept (-0.047
on the absolute scale) brings the league mean predicted R/PA into line with
actual ~0.115 R/PA.

Validation against held-out 2024-2025 pitcher seasons:
    - Predicted vs actual RA9 correlation: r = 0.877
    - Mean absolute error: 0.47 runs/9
    - Mean predicted = mean actual = 4.32 (well-calibrated by construction)

The 0.47 MAE is essentially the FIP-vs-ERA gap — sequencing and bullpen
inheritance noise that no PA-based model can predict.

ERA from RA9
------------
Earned runs are runs scored without the aid of an error (subject to scorer
judgment, but consistent enough at the league level). Historical analysis
of ER/RA on 2018-2025 statsapi data shows the LEAGUE ratio is very stable
(0.912-0.927), but the cleanest predictive split is by ROLE:

    Starters (IP/G >= 3.5):   ER/RA = 0.934  → 6.6% unearned
    Relievers (IP/G < 3.5):   ER/RA = 0.889  → 11.1% unearned
    League pooled:            ER/RA = 0.918

Per-pitcher YoY stability of individual ER/RA is essentially noise
(r = 0.15-0.28), so we DON'T project per-pitcher ratios. Instead we infer
each pitcher's role from their historical IP/G and apply the role constant:

    starter ratio:  0.934
    reliever ratio: 0.889

Why the gap? Relievers face systematically different base/out states.
Closers often inherit jammed innings where a single error has outsized
scoring impact, while starters work from a clean first inning and benefit
from larger samples that smooth out scoring quirks. By skill quintile
ER/RA only varies 0.896 (elite) to 0.924 (worst) — much smaller than the
role split, so we skip a skill-based adjustment.

Validation against 2024-2025 ERA actuals:
    Predicted vs actual ERA correlation: r ≈ 0.875 (matches RA9, since ERA
    is RA9 scaled by a near-constant). Mean predicted ERA ≈ mean actual ERA
    by construction.

Both neutral and park-adjusted ERAs are produced.

TBF_per_IP calibration
----------------------
Naive formula: TBF/IP = 3 / (P_K + P_BIPOut + P_SF), since each out finishes
one of three needed for an inning. This over-predicts by ~3% because GIDPs
and CS provide "extra" outs not counted as PAs. Multiplying by 4.219/4.345
= 0.971 brings the mean into alignment with historical reality.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# Linear weights for runs allowed per PA. Calibrated against 2022-2025
# pitcher-seasons with 40+ IP (see experiment_pitcher_outputs.py).
LINEAR_WEIGHTS_RUNS = {
    "P_BB":     0.30,
    "P_HBP":    0.34,
    "P_1B":     0.45,
    "P_2B":     0.77,
    "P_3B":     1.06,
    "P_HR":     1.38,
    "P_K":     -0.03,
    "P_BIPOut": 0.00,   # reference
    "P_SF":     0.00,   # treated as BIPOut
}

# Intercept that brings the sum-of-weights mean prediction in line with
# absolute league R/PA. Empirically -0.047 on 2022-2025 data — see the
# calibration block in `experiment_pitcher_outputs.py`.
RUNS_INTERCEPT_DEFAULT = -0.047

# Empirical calibration factor for TBF/IP. Naive formula 3/(K+BIPOut+SF)
# over-predicts by ~3% because not every BIPOut consumes exactly one out
# (GIDPs, CS in non-PA situations). 0.971 brings the mean into alignment.
TBF_PER_IP_CALIBRATION = 0.971

# ER/RA ratios by role. Pooled from 2022-2025 pitcher-seasons with 40+ IP.
#   Starters (IP/G >= 3.5):  0.934 — 6.6% of runs are unearned
#   Relievers (IP/G < 3.5):  0.889 — 11.1% of runs are unearned
#   League pooled fallback:  0.918
# Per-pitcher YoY stability of individual ER/RA is r=0.15-0.28, so role
# constants — not pitcher-specific projections — are the right fidelity.
ER_RA_RATIO_STARTER  = 0.934
ER_RA_RATIO_RELIEVER = 0.889
ER_RA_RATIO_LEAGUE   = 0.918   # fallback when role can't be determined
STARTER_IP_PER_G_THRESHOLD = 3.5


def determine_pitcher_role(pit_df: pd.DataFrame, target_year: int,
                            max_history_years: int = 3,
                            decay: float = 0.85,
                            ip_threshold: float = STARTER_IP_PER_G_THRESHOLD,
                            ) -> pd.DataFrame:
    """Classify each pitcher as 'starter' or 'reliever' for the target year.

    Uses an IP-weighted recency-decayed average of IP-per-game from the
    pitcher's prior seasons. Players with IP/G >= 3.5 are starters.

    A pitcher who's switched roles recently (e.g., a starter shifted to the
    bullpen) is weighted by recent seasons, so the role classification
    follows the latest usage pattern.

    Returns
    -------
    DataFrame with PlayerId, weighted_IP_per_G, role ('starter' or 'reliever').
    """
    prior = pit_df[(pit_df["Season"] < target_year) &
                   (pit_df["Season"] >= target_year - max_history_years) &
                   (pit_df["IP"] > 0) & (pit_df["G"] > 0)].copy()
    if prior.empty:
        return pd.DataFrame(columns=["PlayerId", "weighted_IP_per_G", "role"])

    prior["IP_per_G"] = prior["IP"] / prior["G"]
    rows = []
    for pid, g in prior.groupby("PlayerId"):
        g = g.sort_values("Season").copy()
        g["yb"] = target_year - g["Season"].astype(int)
        # Weight by IP × recency decay so recent heavy usage dominates
        w = g["IP"].astype(float).values * (decay ** g["yb"].values)
        if w.sum() == 0:
            continue
        weighted = float(np.sum(g["IP_per_G"].astype(float).values * w)
                          / w.sum())
        role = "starter" if weighted >= ip_threshold else "reliever"
        rows.append({
            "PlayerId":          int(pid),
            "weighted_IP_per_G": weighted,
            "role":              role,
        })
    return pd.DataFrame(rows)


def role_to_er_ra_ratio(role: str | pd.Series) -> float | pd.Series:
    """Map role to ER/RA ratio. Accepts a string or Series."""
    if isinstance(role, pd.Series):
        return role.map({
            "starter":  ER_RA_RATIO_STARTER,
            "reliever": ER_RA_RATIO_RELIEVER,
        }).fillna(ER_RA_RATIO_LEAGUE)
    return {
        "starter":  ER_RA_RATIO_STARTER,
        "reliever": ER_RA_RATIO_RELIEVER,
    }.get(role, ER_RA_RATIO_LEAGUE)


def compute_runs_allowed_per_pa(probs_df: pd.DataFrame,
                                 weights: dict[str, float] = None,
                                 intercept: float = RUNS_INTERCEPT_DEFAULT,
                                 suffix: str = "",
                                 ) -> pd.Series:
    """Compute expected runs allowed per PA from per-PA event probabilities.

    Sum of (weight × probability) plus a calibrated intercept that aligns
    the league mean of predictions to absolute MLB R/PA (~0.115).

    Parameters
    ----------
    probs_df : pd.DataFrame
        Must contain per-PA probability columns P_BB, P_HBP, P_1B, P_2B,
        P_3B, P_HR, P_K, P_BIPOut. Optionally P_SF (defaults to 0).
    weights : dict, optional
        Override the default linear weights.
    intercept : float
        Additive constant to match absolute league R/PA.
    suffix : str
        Append to each event column name (e.g., "_park") to compute the
        park-adjusted runs allowed.

    Returns
    -------
    Series indexed identically to probs_df: expected runs allowed per PA.
    """
    if weights is None:
        weights = LINEAR_WEIGHTS_RUNS
    out = pd.Series(0.0, index=probs_df.index)
    for ev_col, w in weights.items():
        col = ev_col + suffix
        if col in probs_df.columns:
            out += w * probs_df[col].fillna(0)
    return out + intercept


def compute_tbf_per_ip(probs_df: pd.DataFrame, suffix: str = "",
                        calibration: float = TBF_PER_IP_CALIBRATION,
                        ) -> pd.Series:
    """Compute expected TBF per IP from per-PA out probabilities.

    Three outs are needed per inning. Each PA produces an expected
    (P_K + P_BIPOut + P_SF) outs, so TBF/IP = 3 / out_rate, scaled by an
    empirical calibration factor to match historical reality (the formula
    slightly over-predicts due to GIDP and CS that aren't full PAs).
    """
    out_rate = (probs_df.get(f"P_K{suffix}", 0).fillna(0)
                + probs_df.get(f"P_BIPOut{suffix}", 0).fillna(0)
                + probs_df.get(f"P_SF{suffix}", 0).fillna(0))
    # Guard against degenerate zero out rate
    out_rate = out_rate.clip(lower=0.10)
    return calibration * 3.0 / out_rate


def compute_ra9(probs_df: pd.DataFrame, suffix: str = "",
                intercept: float = RUNS_INTERCEPT_DEFAULT,
                er_ra_ratio: "float | pd.Series" = ER_RA_RATIO_LEAGUE,
                ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute RA9, ERA, R/PA, and TBF/IP from per-PA event probabilities.

    RA9 = R/PA × TBF/IP × 9
    ERA = RA9 × er_ra_ratio

    `er_ra_ratio` may be a single float (league constant) or a Series
    indexed identically to probs_df (e.g., role-specific ratios where
    starters use 0.934 and relievers 0.889).

    Per-pitcher YoY r for ER/RA is only 0.15-0.28, so individual
    projections aren't worthwhile — role-based constants are the right
    fidelity. Pass a Series produced by `role_to_er_ra_ratio` to apply
    role-specific ratios.

    Returns
    -------
    Tuple of (RA9_series, ERA_series, R_per_PA_series, TBF_per_IP_series).
    """
    r_per_pa = compute_runs_allowed_per_pa(probs_df, intercept=intercept,
                                            suffix=suffix)
    tbf_per_ip = compute_tbf_per_ip(probs_df, suffix=suffix)
    ra9 = r_per_pa * tbf_per_ip * 9.0
    era = ra9 * er_ra_ratio
    return ra9, era, r_per_pa, tbf_per_ip


def project_wp_per_pa(pit_df: pd.DataFrame, target_year: int,
                      k_pa: float = 600.0,
                      decay: float = 0.85,
                      max_history_years: int = 5,
                      ) -> pd.DataFrame:
    """Per-pitcher projection of wild pitches per PA.

    YoY stability is moderate (r ≈ 0.40), and per-pitcher WP totals are small
    (~5-15 per season for everyday starters; many relievers have 0-1). High
    shrinkage warranted: k_pa=600 means a pitcher needs ~600 weighted TBF
    to halfway pull from league average (~0.0075 WP/PA).

    Returns
    -------
    DataFrame with PlayerId, Pred_WP_per_PA, n_eff_WP.
    """
    df = pit_df.copy()
    df["WP_per_PA"] = df["WP"] / df["TBF"].replace(0, np.nan)

    prior = df[(df["Season"] < target_year) &
               (df["Season"] >= target_year - max_history_years) &
               (df["TBF"] >= 25)].copy()
    if prior.empty:
        return pd.DataFrame(columns=["PlayerId", "Pred_WP_per_PA", "n_eff_WP"])

    league_rate = (float(prior["WP"].sum())
                   / max(float(prior["TBF"].sum()), 1.0))

    rows = []
    for pid, g in prior.groupby("PlayerId"):
        g = g.sort_values("Season").copy()
        g["yb"] = target_year - g["Season"].astype(int)
        w = g["TBF"].astype(float).values * (decay ** g["yb"].values)
        if w.sum() == 0:
            continue
        weighted = (float(np.sum(g["WP_per_PA"].astype(float).fillna(0).values * w))
                    / float(w.sum()))
        n_eff = float(w.sum())
        pred = (n_eff * weighted + k_pa * league_rate) / (n_eff + k_pa)
        rows.append({
            "PlayerId":       int(pid),
            "Pred_WP_per_PA": float(np.clip(pred, 0.0, 1.0)),
            "n_eff_WP":       n_eff,
        })
    return pd.DataFrame(rows)

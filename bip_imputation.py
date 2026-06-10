"""
bip_imputation.py
=================
Python port of impute_observations.R.

For each season, ensures every batter and pitcher has at least TARGET_N
records by generating synthetic observations using a three-layer
hierarchical blend with quadratic shrinkage.

Three layers:
    Layer 1 (innermost):  Current-season player data
    Layer 2 (middle):     Player's own historical data (decay-weighted)
    Layer 3 (outermost):  Season population prior (decay-weighted)

Two blend weights:
    w_curr   = n_current / (n_current + n_hist_eff)
        Controls how much current data overrides player history.
        Uses DECAY-WEIGHTED history count — recency anchors the profile.
    w_player = n_total_raw² / (n_total_raw² + MIN_OBS²)
        Controls how much the combined player profile resists the
        population. Uses RAW history count — actual data volume earns
        credibility regardless of decay applied to the mean.

Continuous variables (launch_speed, launch_angle, adjusted_angle) are
modeled as a multivariate normal per player-season. Categorical variables
(stand, home_team) are sampled from blended categorical distributions.

This is a memory-conscious port — it processes one season at a time and
discards the synthetic data of completed seasons rather than holding all
in memory at once.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_config import (
    IMP_TARGET_N_BATTER, IMP_TARGET_N_PITCHER,
    IMP_MIN_OBS_BATTER, IMP_MIN_OBS_PITCHER,
    IMP_HIST_DECAY, IMP_HIST_MAX_LOOKBACK,
    IMP_PLAYER_HIST_DECAY, IMP_PLAYER_MAX_LOOKBACK,
    BIP_CONT_VARS, BIP_BOUNDS,
)

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Quadratic shrinkage weight (Layer 2 → Layer 3 / player → population)
# ─────────────────────────────────────────────────────────────────────────────

def _player_weight(n_obs: float, min_obs: float) -> float:
    """w = n² / (n² + min_obs²). Players with virtually no data collapse to pop."""
    return (n_obs ** 2) / (n_obs ** 2 + min_obs ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Population-level statistics for one target season
# ─────────────────────────────────────────────────────────────────────────────

def _build_population_stats(df: pd.DataFrame, szn: int, season_col: str,
                            vars_cont: list[str],
                            decay: float = IMP_HIST_DECAY,
                            max_lookback: int = IMP_HIST_MAX_LOOKBACK
                            ) -> dict:
    """Decay-weighted blend of population stats from szn and prior seasons.

    Returns dict with keys: mean, cov, stand (probs), home_team (probs).
    """
    all_seasons   = sorted(df[season_col].unique())
    prior_seasons = [s for s in all_seasons if s <= szn]
    use_seasons   = prior_seasons[-(max_lookback + 1):]
    steps_back    = np.array([szn - s for s in use_seasons])
    decay_w       = decay ** steps_back

    means, covs, eff_ns = [], [], []
    for i, s in enumerate(use_seasons):
        sdf = df[df[season_col] == s][vars_cont].dropna()
        n_s = len(sdf)
        eff_ns.append(decay_w[i] * n_s)
        if n_s >= 1:
            means.append(sdf.mean().values)
        else:
            means.append(np.zeros(len(vars_cont)))
        if n_s >= 3:
            cmat = sdf.cov().values
            if not np.any(np.isnan(cmat)) and np.linalg.det(cmat) > 0:
                covs.append((i, cmat))
    eff_ns = np.array(eff_ns)
    total  = eff_ns.sum()

    pop_mean = np.zeros(len(vars_cont))
    for i, m in enumerate(means):
        pop_mean += m * eff_ns[i]
    pop_mean /= total

    if covs:
        cov_total = sum(eff_ns[i] for i, _ in covs)
        pop_cov = sum(cmat * eff_ns[i] for i, cmat in covs) / cov_total
    else:
        pop_cov = np.eye(len(vars_cont))

    # Categorical: decay-weighted proportions over all stand / home_team levels
    stand_levels = sorted(df["stand"].dropna().unique().tolist())
    ht_levels    = sorted(df["home_team"].dropna().unique().tolist())
    stand_w      = {lv: 0.0 for lv in stand_levels}
    ht_w         = {lv: 0.0 for lv in ht_levels}
    for i, s in enumerate(use_seasons):
        sdf = df[df[season_col] == s]
        n_s = len(sdf)
        if n_s == 0:
            continue
        w = decay_w[i] * n_s
        st_counts = sdf["stand"].value_counts(normalize=True)
        ht_counts = sdf["home_team"].value_counts(normalize=True)
        for lv, p in st_counts.items():
            stand_w[lv] = stand_w.get(lv, 0.0) + w * p
        for lv, p in ht_counts.items():
            ht_w[lv] = ht_w.get(lv, 0.0) + w * p

    stand_probs = pd.Series(stand_w) / sum(stand_w.values())
    ht_probs    = pd.Series(ht_w)    / sum(ht_w.values())

    return {
        "mean":      pop_mean,
        "cov":       pop_cov,
        "stand":     stand_probs,
        "home_team": ht_probs,
        "vars":      vars_cont,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Player-level historical statistics
# ─────────────────────────────────────────────────────────────────────────────

def _get_player_hist_stats(player_hist_df: pd.DataFrame, szn: int,
                           season_col: str, vars_cont: list[str],
                           decay: float = IMP_PLAYER_HIST_DECAY,
                           max_lookback: int = IMP_PLAYER_MAX_LOOKBACK
                           ) -> Optional[dict]:
    """Decay-weighted player history. Returns None if no usable data.

    Returns dict with mean, cov (or None), stand probs, home_team probs,
    n_eff (decay-weighted count → drives w_curr),
    n_raw (undecayed count → drives w_player).
    """
    if player_hist_df is None or len(player_hist_df) == 0:
        return None
    all_seasons = sorted(player_hist_df[season_col].unique().tolist())
    prior = [s for s in all_seasons if s < szn]
    use_seasons = prior[-max_lookback:]
    if not use_seasons:
        return None
    steps_back = np.array([szn - s for s in use_seasons])
    decay_w    = decay ** steps_back

    means, covs, eff_ns, raw_ns = [], [], [], []
    for i, s in enumerate(use_seasons):
        sdf = player_hist_df[player_hist_df[season_col] == s][vars_cont].dropna()
        n_s = len(sdf)
        eff_ns.append(decay_w[i] * n_s)
        raw_ns.append(n_s)
        if n_s >= 1:
            means.append((i, sdf.mean().values))
        if n_s >= 3:
            cmat = sdf.cov().values
            if not np.any(np.isnan(cmat)) and np.linalg.det(cmat) > 0:
                covs.append((i, cmat))

    n_eff = float(sum(eff_ns))
    n_raw = int(sum(raw_ns))
    if n_eff == 0:
        return None

    eff_arr = np.array(eff_ns)
    if means:
        mean_total = sum(eff_arr[i] for i, _ in means)
        hist_mean  = sum(m * eff_arr[i] for i, m in means) / mean_total
    else:
        return None
    if covs:
        cov_total = sum(eff_arr[i] for i, _ in covs)
        hist_cov  = sum(c * eff_arr[i] for i, c in covs) / cov_total
    else:
        hist_cov = None

    # Categorical proportions
    stand_w, ht_w = defaultdict(float), defaultdict(float)
    for i, s in enumerate(use_seasons):
        sdf = player_hist_df[player_hist_df[season_col] == s]
        if len(sdf) == 0:
            continue
        w = decay_w[i] * len(sdf)
        for lv, p in sdf["stand"].value_counts(normalize=True).items():
            stand_w[lv] += w * p
        for lv, p in sdf["home_team"].value_counts(normalize=True).items():
            ht_w[lv] += w * p

    stand_probs = (pd.Series(stand_w) / sum(stand_w.values())
                   if stand_w else None)
    ht_probs    = (pd.Series(ht_w)    / sum(ht_w.values())
                   if ht_w    else None)

    return {
        "mean":      hist_mean,
        "cov":       hist_cov,
        "stand":     stand_probs,
        "home_team": ht_probs,
        "n_eff":     n_eff,
        "n_raw":     n_raw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for categorical blending and sampling
# ─────────────────────────────────────────────────────────────────────────────

def _blend_cat_probs(curr_vals: pd.Series, hist_probs: Optional[pd.Series],
                     w_curr: float) -> Optional[pd.Series]:
    """Blend player's current observed proportions with historical, by w_curr."""
    curr_clean = curr_vals.dropna()
    has_curr = len(curr_clean) > 0
    has_hist = hist_probs is not None and hist_probs.sum() > 0
    if not has_curr and not has_hist:
        return None
    levels = set()
    if has_curr:
        levels.update(curr_clean.unique())
    if has_hist:
        levels.update(hist_probs.index)
    levels = sorted(levels)

    curr_p = pd.Series(0.0, index=levels)
    if has_curr:
        c = curr_clean.value_counts(normalize=True)
        for lv, p in c.items():
            if lv in curr_p.index:
                curr_p[lv] = p
    hist_p = pd.Series(0.0, index=levels)
    if has_hist:
        for lv, p in hist_probs.items():
            if lv in hist_p.index:
                hist_p[lv] = p
        s = hist_p.sum()
        if s > 0:
            hist_p = hist_p / s
    blended = w_curr * curr_p + (1 - w_curr) * hist_p
    s = blended.sum()
    return blended / s if s > 0 else None


def _sample_categorical(player_probs: Optional[pd.Series], pop_probs: pd.Series,
                        n: int, w_player: float, rng) -> np.ndarray:
    """Final categorical sample using player→population blend."""
    levels = set(pop_probs.index)
    if player_probs is not None:
        levels.update(player_probs.index)
    levels = sorted(levels)

    pp = pd.Series(0.0, index=levels)
    if player_probs is not None and player_probs.sum() > 0:
        for lv, p in player_probs.items():
            if lv in pp.index:
                pp[lv] = p
        pp = pp / pp.sum()
    else:
        w_player = 0.0  # no player info — fall fully to population

    pop_p = pd.Series(0.0, index=levels)
    for lv, p in pop_probs.items():
        if lv in pop_p.index:
            pop_p[lv] = p
    s = pop_p.sum()
    if s > 0:
        pop_p = pop_p / s

    final_p = w_player * pp + (1 - w_player) * pop_p
    final_p = final_p / final_p.sum()
    return rng.choice(levels, size=n, p=final_p.values)


# ─────────────────────────────────────────────────────────────────────────────
# Per-player imputation
# ─────────────────────────────────────────────────────────────────────────────

def _impute_player(player_data: pd.DataFrame, player_hist_stats: Optional[dict],
                   pop: dict, min_obs: int, n_target: int, rng) -> Optional[pd.DataFrame]:
    """Generate synthetic rows for a single player-season."""
    n_existing = len(player_data)
    n_needed   = n_target - n_existing
    if n_needed <= 0:
        return None
    vars_cont = pop["vars"]

    n_hist_eff = player_hist_stats["n_eff"] if player_hist_stats else 0.0
    n_hist_raw = player_hist_stats["n_raw"] if player_hist_stats else 0

    w_curr   = (n_existing / (n_existing + n_hist_eff)) if (n_existing + n_hist_eff) > 0 else 1.0
    w_player = _player_weight(n_existing + n_hist_raw, min_obs)

    # Player mean: blend current and history
    if n_existing >= 1:
        curr_mean = player_data[vars_cont].mean(skipna=True).values
        if player_hist_stats and player_hist_stats["mean"] is not None:
            player_mean = w_curr * curr_mean + (1 - w_curr) * player_hist_stats["mean"]
        else:
            player_mean = curr_mean
    elif player_hist_stats and player_hist_stats["mean"] is not None:
        player_mean = player_hist_stats["mean"]
    else:
        player_mean = pop["mean"]

    # Player cov: same blend
    curr_cov = None
    if n_existing >= 3:
        cmat = player_data[vars_cont].cov().values
        if not np.any(np.isnan(cmat)) and np.linalg.det(cmat) > 0:
            curr_cov = cmat
    has_curr_cov = curr_cov is not None
    has_hist_cov = player_hist_stats is not None and player_hist_stats["cov"] is not None
    if has_curr_cov and has_hist_cov:
        player_cov = w_curr * curr_cov + (1 - w_curr) * player_hist_stats["cov"]
    elif has_curr_cov:
        player_cov = curr_cov
    elif has_hist_cov:
        player_cov = player_hist_stats["cov"]
    else:
        player_cov = pop["cov"]

    # Shrink toward population
    shrunk_mean = w_player * player_mean + (1 - w_player) * pop["mean"]
    shrunk_cov  = w_player * player_cov  + (1 - w_player) * pop["cov"]

    # Sample continuous variables
    try:
        synth = rng.multivariate_normal(shrunk_mean, shrunk_cov, size=n_needed,
                                        check_valid="ignore")
    except Exception:
        # Cov can be near-singular for low-data players; regularize and retry
        reg = shrunk_cov + np.eye(len(vars_cont)) * 1e-3
        synth = rng.multivariate_normal(shrunk_mean, reg, size=n_needed,
                                        check_valid="ignore")

    out = pd.DataFrame(synth, columns=vars_cont)
    for v, (lo, hi) in BIP_BOUNDS.items():
        if v in out.columns:
            out[v] = out[v].clip(lower=lo, upper=hi)
    out["launch_speed"]   = out["launch_speed"].round(1)
    out["launch_angle"]   = out["launch_angle"].round(0)
    out["adjusted_angle"] = out["adjusted_angle"].round(0)

    # Categorical
    stand_pp = _blend_cat_probs(player_data["stand"],
                                player_hist_stats["stand"] if player_hist_stats else None,
                                w_curr)
    ht_pp    = _blend_cat_probs(player_data["home_team"],
                                player_hist_stats["home_team"] if player_hist_stats else None,
                                w_curr)
    out["stand"]     = _sample_categorical(stand_pp, pop["stand"], n_needed, w_player, rng)
    out["home_team"] = _sample_categorical(ht_pp,    pop["home_team"], n_needed, w_player, rng)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def impute_bip(df: pd.DataFrame, season_col: str = "Season",
               target_n_batter: int = IMP_TARGET_N_BATTER,
               target_n_pitcher: int = IMP_TARGET_N_PITCHER,
               min_obs_batter: int = IMP_MIN_OBS_BATTER,
               min_obs_pitcher: int = IMP_MIN_OBS_PITCHER,
               verbose: bool = True) -> pd.DataFrame:
    """Run the full multi-season imputation. Returns real + synthetic rows
    combined, with a `_synthetic` flag column for traceability.

    Different TARGET_N values are used for batters vs pitchers:
        * Batters get a smaller target to preserve their real BIP profile
          (preventing over-regression of elite hitters' HR / 2B rates).
          Empirically tuned via experiment_target_n.py.
        * Pitchers get a higher target to stabilize their batted-ball
          predictions, since pitcher year-to-year BIP outcome stability is
          lower and they benefit more from the population-prior pull.
    """
    rng = np.random.default_rng(42)
    df = df.dropna(subset=BIP_CONT_VARS).copy()
    if verbose:
        print(f"  Loaded {len(df):,} real BIP rows")
        print(f"  Seasons: {sorted(df[season_col].unique())}")
        print(f"  Unique batters: {df['batter'].nunique():,}")
        print(f"  Unique pitchers: {df['pitcher'].nunique():,}")
        print(f"  Batter target_n: {target_n_batter}, Pitcher target_n: {target_n_pitcher}")

    df["_synthetic"] = False
    all_synth_batter, all_synth_pitcher = [], []
    seasons = sorted(df[season_col].unique())

    for szn in seasons:
        if verbose:
            print(f"\n  ── Season {szn} ──")
        pop = _build_population_stats(df, szn, season_col, BIP_CONT_VARS)
        season_df = df[df[season_col] == szn]

        # ── Batters ──
        if verbose:
            print("    Imputing batters...", end=" ", flush=True)
        bcount = 0
        for b in season_df["batter"].dropna().unique():
            bdata = season_df[season_df["batter"] == b]
            if len(bdata) >= target_n_batter:
                continue
            hist = df[(df["batter"] == b) & (df[season_col] < szn)]
            phs  = _get_player_hist_stats(hist, szn, season_col, BIP_CONT_VARS)
            synth = _impute_player(bdata, phs, pop, min_obs_batter, target_n_batter, rng)
            if synth is None or len(synth) == 0:
                continue
            synth["batter"]  = b
            synth["pitcher"] = pd.NA
            synth["events"]  = pd.NA
            synth[season_col] = szn
            synth["_synthetic"] = True
            all_synth_batter.append(synth)
            bcount += 1
        if verbose:
            print(f"{bcount} batter-seasons", flush=True)

        # ── Pitchers ──
        if verbose:
            print("    Imputing pitchers...", end=" ", flush=True)
        pcount = 0
        for p in season_df["pitcher"].dropna().unique():
            pdata = season_df[season_df["pitcher"] == p]
            if len(pdata) >= target_n_pitcher:
                continue
            hist = df[(df["pitcher"] == p) & (df[season_col] < szn)]
            phs  = _get_player_hist_stats(hist, szn, season_col, BIP_CONT_VARS)
            synth = _impute_player(pdata, phs, pop, min_obs_pitcher, target_n_pitcher, rng)
            if synth is None or len(synth) == 0:
                continue
            synth["batter"]  = pd.NA
            synth["pitcher"] = p
            synth["events"]  = pd.NA
            synth[season_col] = szn
            synth["_synthetic"] = True
            all_synth_pitcher.append(synth)
            pcount += 1
        if verbose:
            print(f"{pcount} pitcher-seasons", flush=True)

    parts = [df]
    if all_synth_batter:
        parts.append(pd.concat(all_synth_batter, ignore_index=True))
    if all_synth_pitcher:
        parts.append(pd.concat(all_synth_pitcher, ignore_index=True))
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"\n  Combined: {len(df):,} real + "
              f"{sum(len(x) for x in all_synth_batter):,} synthetic-batter + "
              f"{sum(len(x) for x in all_synth_pitcher):,} synthetic-pitcher "
              f"= {len(out):,}")
    return out

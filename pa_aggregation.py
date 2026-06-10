"""
pa_aggregation.py
=================
The final step: combine the per-BIP outcome probabilities with the per-PA
K%/BB%/HBP%/SF% rates into a single normalized event distribution per
player.

Two-stage construction:
    1. For each (player, season) collect every BIP prediction the XGBoost
       made (real + imputed), then aggregate to a per-BIP outcome
       distribution via the R script's mean/median blend. Pull the OUT
       probability toward the league mean (the R script's `out_pop_pull`).
       Weight seasons (last 3) and combine.
    2. Take the K%/BB%/HBP%/SF% projections and treat them as "carved out"
       of the PA budget. The remaining mass (1 - K - BB - HBP - SF) is
       allocated across {1B, 2B, 3B, HR, BIPOut} using the season-weighted
       BIP probabilities.

Final per-player frame has Pr() and SD() for all 9 events. SDs:
    K, BB, HBP, SF      — from rate_models
    HR, 3B, 2B, 1B, BIPOut — std-dev of the per-BIP probability across
                          the player's pool of BIPs (real + imputed),
                          scaled by the player's BIP share. This captures
                          "how variable was this player's predicted hit
                          quality?" — a player with mostly weak contact
                          gets a tighter HR SD, a guy who hits some lasers
                          and some grounders gets a wider one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_config import (
    SEASON_WEIGHTS_FROM_TARGET, OUT_POP_PULL_HITTER, OUT_POP_PULL_PITCHER,
    EVENT_BLEND_WEIGHTS_HITTER, EVENT_BLEND_WEIGHTS_PITCHER, BIP_OUTCOMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-season aggregation of XGBoost predictions
# ─────────────────────────────────────────────────────────────────────────────

def _per_season_event_dist(bip_with_proba: pd.DataFrame, group_col: str,
                           blend_weights: dict) -> pd.DataFrame:
    """Aggregate per-BIP predictions into one row per (player, Season).

    Applies the R script's mean/median blend per event (default: 1.0/0.0 for
    out + single, 0.75/0.25 for 2B/3B/HR). Also captures per-event std-dev
    of the per-BIP probability — feeds SD output later.
    """
    rows = []
    for (player, season), g in bip_with_proba.groupby([group_col, "Season"]):
        rec = {group_col: player, "Season": season, "N_BIP": len(g)}
        for ev in BIP_OUTCOMES:
            col = f"prob_{ev}"
            if col not in g.columns:
                rec[ev] = 0.0
                rec[f"{ev}_sd"] = 0.0
                continue
            wm, wmd = blend_weights.get(ev, (1.0, 0.0))
            rec[ev] = wm * g[col].mean() + wmd * g[col].median()
            rec[f"{ev}_sd"] = g[col].std(ddof=0) if len(g) > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def _per_season_profile_meta(bip_real_only: pd.DataFrame, group_col: str
                             ) -> pd.DataFrame:
    """Per (player, Season) computation of launch-profile means and
    covariances on REAL BIP data only. This drives the Hotelling T² divergence
    test for adaptive season weighting.

    The synthetic imputation rows are excluded — divergence should reflect
    what we actually observed, not noise from the imputation prior.
    """
    profile_vars = ["launch_speed", "launch_angle", "adjusted_angle"]
    real = bip_real_only.dropna(subset=profile_vars)
    rows = []
    for (player, season), g in real.groupby([group_col, "Season"]):
        n = len(g)
        if n == 0:
            continue
        means = g[profile_vars].mean().values
        rec = {
            group_col: player, "Season": int(season),
            "N_REAL_BIP": int(n),
            "mean_speed": float(means[0]),
            "mean_angle": float(means[1]),
            "mean_spray": float(means[2]),
        }
        if n >= 3:
            c = g[profile_vars].cov().values
            rec["cov_ss"] = float(c[0, 0])
            rec["cov_aa"] = float(c[1, 1])
            rec["cov_pp"] = float(c[2, 2])
            rec["cov_sa"] = float(c[0, 1])
            rec["cov_sp"] = float(c[0, 2])
            rec["cov_ap"] = float(c[1, 2])
        else:
            # Defaults; small samples won't be used for divergence anyway
            rec.update({"cov_ss": 1.0, "cov_aa": 1.0, "cov_pp": 1.0,
                        "cov_sa": 0.0, "cov_sp": 0.0, "cov_ap": 0.0})
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted multi-season aggregation
# ─────────────────────────────────────────────────────────────────────────────

# Hotelling T² statistic for two-sample mean comparison on multivariate launch
# profile. Used to detect when a player's recent season differs meaningfully
# from their earlier seasons — like Jordan Walker's jump in 2026.

def _hotelling_t2(g1_means, g1_cov, n1, g2_means, g2_cov, n2):
    """Two-sample Hotelling T² statistic on two multivariate samples.
    Larger value = more different. Returns nan if sample too small.
    """
    if n1 < 5 or n2 < 5:
        return float("nan")
    sp = ((n1 - 1) * g1_cov + (n2 - 1) * g2_cov) / (n1 + n2 - 2)
    # Small ridge for numerical stability
    sp = sp + np.eye(len(g1_means)) * 1e-4
    try:
        sp_inv = np.linalg.inv(sp)
    except np.linalg.LinAlgError:
        return float("nan")
    diff = (g1_means - g2_means).reshape(-1, 1)
    return (n1 * n2 / (n1 + n2)) * float((diff.T @ sp_inv @ diff).item())


def _compute_adaptive_weights(per_season_meta: pd.DataFrame, target_year: int,
                              k_reliability: float | None = None,
                              recency_decay: dict | None = None,
                              divergence_threshold: float | None = None,
                              divergence_max_boost: float | None = None,
                              ) -> pd.DataFrame:
    """For each player in the per_season_meta dataframe, compute adaptive
    season weights based on (a) sample-size-driven reliability, (b) a
    recency-decay preference, and (c) a Hotelling-T² boost on the most
    recent season when its launch profile diverges from older seasons.

    Empirically validated against held-out data (see experiment_adaptive_weights.py):
        * For partial-season 2026 (~100 BIPs), fixed weight 0.65 has MAE 0.0346
          vs adaptive 0.0310 (a 10% improvement)
        * For very small samples (50 BIPs), the improvement is 29%
        * For divergent players (T²>5), the boost provides an additional small lift

    Input frame must have columns:
        group_col, Season, N_REAL_BIP, mean_speed, mean_angle, mean_spray,
        cov_xx (or full launch cov)

    Returns a frame with: group_col, Season, w (adaptive weight, sum=1 per player)
    """
    # Pull config defaults
    from pipeline_config import (
        ADAPTIVE_K_RELIABILITY, ADAPTIVE_RECENCY_DECAY,
        ADAPTIVE_DIV_THRESHOLD, ADAPTIVE_DIV_MAX_BOOST,
    )
    if k_reliability is None:
        k_reliability = ADAPTIVE_K_RELIABILITY
    if recency_decay is None:
        recency_decay = ADAPTIVE_RECENCY_DECAY
    if divergence_threshold is None:
        divergence_threshold = ADAPTIVE_DIV_THRESHOLD
    if divergence_max_boost is None:
        divergence_max_boost = ADAPTIVE_DIV_MAX_BOOST

    out_rows = []
    group_col = per_season_meta.columns[0]  # batter or pitcher
    profile_vars = ["mean_speed", "mean_angle", "mean_spray"]
    cov_keys = ["cov_ss", "cov_aa", "cov_pp", "cov_sa", "cov_sp", "cov_ap"]

    for player, g in per_season_meta.groupby(group_col):
        g = g.sort_values("Season").reset_index(drop=True)
        if len(g) == 0:
            continue
        # years_back: target_year - Season
        g["years_back"] = target_year - g["Season"].astype(int)
        g = g[g["years_back"].between(1, 3)].copy()  # only last 3 prior seasons
        if len(g) == 0:
            continue

        # Reliability per season
        g["reliability"] = g["N_REAL_BIP"] / (g["N_REAL_BIP"] + k_reliability)

        # Recency importance
        g["recency"] = g["years_back"].map(recency_decay).fillna(0.0)

        # Divergence boost: only applied to the MOST RECENT season,
        # comparing its profile vs pooled older seasons.
        boost = 0.0
        if len(g) >= 2:
            recent_idx = g["years_back"].idxmin()
            recent = g.loc[recent_idx]
            older  = g.drop(recent_idx)
            if (recent["N_REAL_BIP"] >= 30 and
                len(profile_vars) > 0 and
                older["N_REAL_BIP"].sum() >= 50):
                # Pool older seasons' means and covariances, weighted by N
                w_older = older["N_REAL_BIP"].values
                older_means = np.average(
                    older[profile_vars].values, axis=0, weights=w_older
                )
                # Average pooled covariance
                older_cov = np.zeros((3, 3))
                for _, row in older.iterrows():
                    c = np.array([
                        [row.get("cov_ss", 1), row.get("cov_sa", 0), row.get("cov_sp", 0)],
                        [row.get("cov_sa", 0), row.get("cov_aa", 1), row.get("cov_ap", 0)],
                        [row.get("cov_sp", 0), row.get("cov_ap", 0), row.get("cov_pp", 1)],
                    ])
                    older_cov += c * row["N_REAL_BIP"]
                older_cov /= w_older.sum()

                recent_means = recent[profile_vars].values
                recent_cov = np.array([
                    [recent.get("cov_ss", 1), recent.get("cov_sa", 0), recent.get("cov_sp", 0)],
                    [recent.get("cov_sa", 0), recent.get("cov_aa", 1), recent.get("cov_ap", 0)],
                    [recent.get("cov_sp", 0), recent.get("cov_ap", 0), recent.get("cov_pp", 1)],
                ])
                t2 = _hotelling_t2(older_means, older_cov, int(w_older.sum()),
                                   recent_means, recent_cov, int(recent["N_REAL_BIP"]))
                if not np.isnan(t2):
                    boost = max(0.0, min((t2 - divergence_threshold) / 10.0,
                                         divergence_max_boost))

        # Compute raw weights: reliability * recency, with boost on most recent
        g["w_raw"] = g["reliability"] * g["recency"]
        if boost > 0 and len(g) >= 2:
            most_recent_idx = g["years_back"].idxmin()
            g.loc[most_recent_idx, "w_raw"] *= (1.0 + boost)

        total = g["w_raw"].sum()
        if total <= 0:
            continue
        g["w"] = g["w_raw"] / total

        # Save details for diagnostics
        for _, row in g.iterrows():
            out_rows.append({
                group_col: player,
                "Season": int(row["Season"]),
                "N_REAL_BIP": int(row["N_REAL_BIP"]),
                "reliability": float(row["reliability"]),
                "recency": float(row["recency"]),
                "w": float(row["w"]),
                "boost_recent": float(boost) if int(row["years_back"]) == 1 else 0.0,
            })
    return pd.DataFrame(out_rows)


def _weighted_multi_season(per_season: pd.DataFrame, target_year: int,
                           group_col: str,
                           per_season_meta: pd.DataFrame | None = None
                           ) -> pd.DataFrame:
    """Combine the last 3 seasons of per-season event distributions.

    If per_season_meta is provided (with launch-profile means/cov per player-season
    and REAL BIP counts), use adaptive weights based on sample size and
    divergence. Otherwise fall back to fixed SEASON_WEIGHTS_FROM_TARGET.
    """
    df = per_season.copy()

    if per_season_meta is not None and len(per_season_meta):
        # ADAPTIVE: compute per-(player,season) weights from meta
        weight_df = _compute_adaptive_weights(per_season_meta, target_year)
        # Merge weights into df (group_col + Season key)
        df = df.merge(
            weight_df[[group_col, "Season", "w"]],
            on=[group_col, "Season"], how="left",
        )
        # Drop rows with no weight (e.g., 4+ years back)
        df = df.dropna(subset=["w"])
    else:
        # FIXED: legacy path
        df["w_raw"] = df["Season"].map(
            lambda s: SEASON_WEIGHTS_FROM_TARGET.get(int(s) - target_year, 0.0))
        df = df[df["w_raw"] > 0]
        df["w"] = df.groupby(group_col)["w_raw"].transform(lambda x: x / x.sum())

    agg_cols = list(BIP_OUTCOMES)
    sd_cols  = [f"{c}_sd" for c in BIP_OUTCOMES]
    rows = []
    for player, g in df.groupby(group_col):
        rec = {group_col: player, "N_BIP": int(g["N_BIP"].sum())}
        # Weighted mean of point estimates
        for c in agg_cols:
            rec[c] = float((g[c] * g["w"]).sum())
        # SD aggregates: weighted RMS of per-season SDs.
        for c in sd_cols:
            rec[c] = float(np.sqrt((g[c] ** 2 * g["w"]).sum()))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Apply OUT pull toward league average.
#
# Two pull modes are supported:
#
# 1. FIXED PULL — historical mode that mirrors the R script: every player
#    gets the same pull strength regardless of sample size. Set via
#    OUT_POP_PULL_HITTER / OUT_POP_PULL_PITCHER in pipeline_config.
#
# 2. SAMPLE-SIZE-ADAPTIVE PULL — Bayesian-shrinkage formulation:
#       pull(N) = K / (N + K)
#    where N is the player's total real BIP count across all included
#    seasons and K is a role-specific shrinkage prior strength. Players
#    with rich samples (Judge, Witt, Soto) keep most of their observed
#    profile; players with thin samples regress hard toward league.
#
#    K is empirically calibrated to preserve the true-skill spread of
#    observed BABIP, which equals sqrt(r) × observed_spread where r is
#    the year-to-year correlation of BIP-out rates.
#
#    Calibrated values (from experiment_adaptive_out_pull.py):
#       K_HITTER  = 200  (r=0.39, avg N≈500 BIPs → avg pull ≈ 0.29)
#       K_PITCHER = 400  (r=0.18, avg N≈400 BIPs → avg pull ≈ 0.50)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_out_pull_fixed(df: pd.DataFrame, league_out: float,
                          pull: float) -> pd.DataFrame:
    """Apply a constant pull to all players (legacy / fallback mode)."""
    out = df.copy()
    out["out"] = (1 - pull) * out["out"] + pull * league_out
    return out


def _apply_out_pull_adaptive(df: pd.DataFrame, league_out: float,
                              K: float, n_col: str = "N_REAL_BIP"
                              ) -> pd.DataFrame:
    """Apply sample-size-adaptive pull. Each player gets pull = K/(N+K).

    df must have an N_REAL_BIP column (real BIPs across all seasons used).
    Falls back to fixed pull of 0.5 if N_REAL_BIP is missing.
    """
    out = df.copy()
    if n_col not in out.columns:
        out["out"] = 0.5 * out["out"] + 0.5 * league_out
        return out
    n = out[n_col].fillna(0).clip(lower=1)
    pulls = K / (n + K)
    out["out"] = (1 - pulls) * out["out"] + pulls * league_out
    return out


def _rescale_hits_to_complement_out(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror R's lines 336-338:
        cols <- c("single", "double", "triple", "home_run")
        row_sums <- rowSums(hitters[, cols])
        hitters[, cols] <- (hitters[, cols] / row_sums) * (1-hitters$out)

    Final step that makes the per-BIP probabilities sum to 1 again, with
    the regressed OUT% fixed and hits filling the remaining mass according
    to the player's own hit-type distribution.
    """
    out = df.copy()
    hit_cols = ["single", "double", "triple", "home_run"]
    cur_hits = out[hit_cols].sum(axis=1).replace(0, np.nan)
    target_hits = 1 - out["out"]
    scale = (target_hits / cur_hits).fillna(0)
    for c in hit_cols:
        out[c] = out[c] * scale
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Final PA-level distribution by combining with rate projections
# ─────────────────────────────────────────────────────────────────────────────

def build_pa_projections(bip_aggregated: pd.DataFrame,
                         rate_projections: pd.DataFrame,
                         group_col: str,
                         id_col: str = "PlayerId") -> pd.DataFrame:
    """Combine BIP probabilities (per-BIP, summing to 1) with PA-level rates
    (K%, BB%, HBP%, SF%) into a 9-event PA-level distribution that sums to 1.

    Parameters
    ----------
    bip_aggregated : per-player frame with columns
        [group_col, N_BIP, out, single, double, triple, home_run,
         out_sd, single_sd, double_sd, triple_sd, home_run_sd]
    rate_projections : per-player frame with columns
        [id_col, Pred_K%, SD_K%, Pred_BB%, SD_BB%, Pred_HBP%, SD_HBP%,
         Pred_SF%, SD_SF%, ...]
    group_col : 'batter' or 'pitcher' — joins to id_col via int match
    """
    # Join — bip_aggregated has the Statcast id (batter/pitcher), rate
    # projections have statsapi PlayerId. They are the same MLBAM id.
    bip = bip_aggregated.copy()
    bip[group_col] = pd.to_numeric(bip[group_col], errors="coerce").astype("Int64")
    rates = rate_projections.copy()
    rates[id_col] = pd.to_numeric(rates[id_col], errors="coerce").astype("Int64")

    df = rates.merge(bip, left_on=id_col, right_on=group_col, how="left")

    # For players in rates but not in bip (no observed BIPs), we'll need a
    # fallback BIP distribution. Use the population BIP probabilities — when
    # we don't have player-specific info, league-average is the right prior.
    has_bip = df["N_BIP"].notna()
    n_missing = (~has_bip).sum()
    if n_missing > 0:
        print(f"    Note: {n_missing} players in rates have no BIP data — "
              f"using league-average BIP distribution for them")
        # Compute population means from those with BIP
        pop_bip = bip_aggregated[BIP_OUTCOMES].mean().to_dict()
        pop_sd  = bip_aggregated[[f"{c}_sd" for c in BIP_OUTCOMES]].mean().to_dict()
        for c in BIP_OUTCOMES:
            df[c] = df[c].fillna(pop_bip[c])
            df[f"{c}_sd"] = df[f"{c}_sd"].fillna(pop_sd[f"{c}_sd"])
        df["N_BIP"] = df["N_BIP"].fillna(0)

    # ── Apportion the PA budget ──────────────────────────────────────────────
    # K% + BB% + HBP% + SF% come straight from rate models.
    # Remaining mass goes to BIP → (1 - K - BB - HBP - SF) × {bip probs}
    rate_cols   = ["Pred_K%", "Pred_BB%", "Pred_HBP%", "Pred_SF%"]
    for c in rate_cols:
        if c not in df.columns:
            df[c] = 0.0
    # Clamp each individually to keep sane bounds
    for c in rate_cols:
        df[c] = df[c].clip(0, 1)
    # If the sum of non-BIP rates exceeds 1, scale them down proportionally.
    # (Shouldn't happen in practice, but guard for it.)
    non_bip_total = df[rate_cols].sum(axis=1)
    over = non_bip_total > 0.95  # leave at least 5% for BIP
    if over.any():
        scl = 0.95 / non_bip_total[over]
        for c in rate_cols:
            df.loc[over, c] = df.loc[over, c] * scl
        non_bip_total = df[rate_cols].sum(axis=1)
    bip_share = (1 - non_bip_total).clip(lower=0)

    # Final PA-level probabilities — 9 events
    pa_K   = df["Pred_K%"]
    pa_BB  = df["Pred_BB%"]
    pa_HBP = df["Pred_HBP%"]
    pa_SF  = df["Pred_SF%"]
    pa_HR  = df["home_run"] * bip_share
    pa_3B  = df["triple"]   * bip_share
    pa_2B  = df["double"]   * bip_share
    pa_1B  = df["single"]   * bip_share
    pa_OUT = df["out"]      * bip_share

    out = pd.DataFrame({
        id_col:      df[id_col],
        "Name":      df.get("Name"),
        "Team":      df.get("Team"),
        "Age":       df.get("Age"),
        "Last_PA":   df.get("p1_PA"),
        "Career_PA": df.get("career_PA"),
        "N_BIP":     df["N_BIP"].astype(int),
        # Point probabilities
        "P_K":      pa_K,
        "P_BB":     pa_BB,
        "P_HBP":    pa_HBP,
        "P_SF":     pa_SF,
        "P_HR":     pa_HR,
        "P_3B":     pa_3B,
        "P_2B":     pa_2B,
        "P_1B":     pa_1B,
        "P_BIPOut": pa_OUT,
    })

    # ── Final consistency check ──────────────────────────────────────────────
    # By construction:
    #   P_K + P_BB + P_HBP + P_SF = non_bip_total = 1 - bip_share
    #   P_HR + P_3B + P_2B + P_1B + P_BIPOut = bip_share * (sum of per-BIP probs)
    # If the per-BIP probabilities sum to exactly 1 (which they should after
    # _rescale_hits_to_complement_out), the BIP events sum to bip_share and
    # the grand total is 1.0 exactly.
    #
    # In practice the per-BIP probs can drift by ~1e-5 from 1 due to floating
    # point. We renormalize ONLY the BIP events to make them sum exactly to
    # bip_share, leaving the rate-model outputs (K, BB, HBP, SF) untouched —
    # those came directly from the rate model and we want to preserve them
    # as predicted. The PA budget is "the remaining 1 - K - BB - HBP - SF"
    # and that's what the BIP events get split across.
    bip_cols = ["P_HR", "P_3B", "P_2B", "P_1B", "P_BIPOut"]
    bip_sum  = out[bip_cols].sum(axis=1).replace(0, np.nan)
    for c in bip_cols:
        out[c] = out[c] / bip_sum * bip_share

    # ── SDs ──────────────────────────────────────────────────────────────────
    # For rate-model events (K, BB, HBP, SF) we have explicit SDs.
    out["SD_K"]   = df["SD_K%"].fillna(0)
    out["SD_BB"]  = df["SD_BB%"].fillna(0)
    out["SD_HBP"] = df["SD_HBP%"].fillna(0)
    out["SD_SF"]  = df["SD_SF%"].fillna(0)
    # For BIP-derived events, SD = (sd of per-BIP prob in this player's pool)
    #   / sqrt(N_BIP)  × bip_share.
    # The first term — std of per-BIP HR probability across the player's BIPs —
    # captures within-player BIP-quality variability. Dividing by sqrt(N_BIP)
    # converts it to the standard error of the mean estimate. Multiplying by
    # bip_share scales it back to the PA-level event. This gives intervals
    # in roughly the right ballpark to compare directly with the rate-model
    # SDs (K, BB) without dwarfing them.
    # We use the player's BIP count for the SE term (real + imputed; imputed
    # rows add weight but don't carry true outcome signal, so this is a
    # conservative — slightly tight — estimator).
    n_eff = np.sqrt(np.maximum(df["N_BIP"].fillna(1), 1))
    out["SD_HR"]     = (df["home_run_sd"].fillna(0) / n_eff) * bip_share
    out["SD_3B"]     = (df["triple_sd"].fillna(0)   / n_eff) * bip_share
    out["SD_2B"]     = (df["double_sd"].fillna(0)   / n_eff) * bip_share
    out["SD_1B"]     = (df["single_sd"].fillna(0)   / n_eff) * bip_share
    out["SD_BIPOut"] = (df["out_sd"].fillna(0)      / n_eff) * bip_share

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_bip_to_player(bip_with_proba: pd.DataFrame, target_year: int,
                            group_col: str, is_pitcher: bool,
                            league_out_pop: float) -> pd.DataFrame:
    """End-to-end: BIPs-with-proba → per-player BIP distribution.

    Pipeline:
      1. Per (player, season): mean/median blend across BIPs (no OUT pull yet).
      2. Adaptive weighted multi-season combine using Hotelling-T² divergence
         to determine season weights (e.g., Walker's 2026 gets a boost).
      3. SAMPLE-SIZE-ADAPTIVE OUT pull, applied ONCE on the combined
         per-player BIP-out:
            pull(N) = K / (N + K)
         where N is the player's total real BIP count and K is the role-
         specific shrinkage prior. Star hitters with 800+ BIPs (Judge, Soto)
         keep most of their observed BIP profile; thin samples regress hard.
      4. Rescale hit columns so per-BIP probabilities sum to 1.

    Key change from prior versions: the OUT pull is no longer applied
    per-season with a fixed strength. Instead it's applied once at the
    player level with strength dependent on sample size. This addresses
    the over-regression of star hitters with rich samples while still
    properly shrinking small-sample projections.
    """
    from pipeline_config import (
        OUT_ADAPTIVE_K_HITTER, OUT_ADAPTIVE_K_PITCHER,
    )

    blend = EVENT_BLEND_WEIGHTS_PITCHER if is_pitcher else EVENT_BLEND_WEIGHTS_HITTER
    K     = OUT_ADAPTIVE_K_PITCHER if is_pitcher else OUT_ADAPTIVE_K_HITTER

    # Step 1: per-season mean/median blend (across all BIPs incl. synthetic)
    per_season = _per_season_event_dist(bip_with_proba, group_col, blend)

    # Build per-season profile meta from REAL BIPs only (drives adaptive
    # season weights AND provides N_REAL_BIP for adaptive OUT pull).
    if "_synthetic" in bip_with_proba.columns:
        real_bip = bip_with_proba[~bip_with_proba["_synthetic"].astype(bool)]
    else:
        real_bip = bip_with_proba
    per_season_meta = _per_season_profile_meta(real_bip, group_col)

    # Step 2: weighted multi-season combine with adaptive weights
    combined = _weighted_multi_season(
        per_season, target_year, group_col,
        per_season_meta=per_season_meta if len(per_season_meta) else None,
    )

    # Compute each player's TOTAL real BIP count across the included seasons
    # (only the last 3 prior seasons count toward the adaptive pull strength).
    if len(per_season_meta):
        meta = per_season_meta.copy()
        meta["years_back"] = target_year - meta["Season"].astype(int)
        meta_recent = meta[meta["years_back"].between(1, 3)]
        n_real = meta_recent.groupby(group_col)["N_REAL_BIP"].sum().reset_index()
        n_real.rename(columns={"N_REAL_BIP": "N_REAL_BIP_TOTAL"}, inplace=True)
        combined = combined.merge(n_real, on=group_col, how="left")
        combined["N_REAL_BIP"] = combined["N_REAL_BIP_TOTAL"].fillna(0)
        combined.drop(columns=["N_REAL_BIP_TOTAL"], inplace=True)
    else:
        combined["N_REAL_BIP"] = 0

    # Step 3: SAMPLE-SIZE-ADAPTIVE OUT pull
    combined = _apply_out_pull_adaptive(combined, league_out_pop, K=K,
                                        n_col="N_REAL_BIP")

    # Step 4: final rescale of hits to fill (1 - out)
    final = _rescale_hits_to_complement_out(combined)
    return final

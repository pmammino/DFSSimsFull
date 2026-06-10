"""
splits_model.py
===============
Per-platoon-side projections (vs LHP / vs RHP for hitters; vs LHB / vs RHB for
pitchers). Produces a parallel set of `_vL` / `_vR` event-probability columns
alongside the season/ROS projection.

Why splits exist
----------------
The season-level model produces blended per-PA event rates that aggregate
across all platoon matchups a player saw. For daily projections — picking
lineups, modeling DFS contests, picking betting lines for a specific game —
we need to know how a hitter performs against a specific starting pitcher's
handedness and how that pitcher fares against the lineup's lefties and
righties. The platoon gap is large:

    Garrett Crochet (LHP):  vL HR%=0.010, vR HR%=0.034  (3.5x gap)
    Tarik Skubal (LHP):     vL HR%=0.008, vR HR%=0.022  (2.7x gap)
    Yordan Alvarez (LHH):   vL K%=0.187,  vR K%=0.131   (large platoon advantage vs RHP)

Architecture: direct per-side projection with overall-anchored constraint
-------------------------------------------------------------------------
For each rate metric (K%, BB%, HBP%, HR%, 1B%, 2B%, 3B%, SF%) and each side:

    1. Project the side-specific rate via PA-weighted recency-decay shrinkage
       on that side's history. Same machinery as the rate_models — just
       applied to a slice of the data.

    2. The league prior for that side is the population mean of that
       metric on that side (e.g., HR% on hitters vs LHP across all hitters).

    3. After projecting both sides, RESCALE the pair so that the player's
       expected PA-share-weighted average equals their projected overall
       rate from the main pipeline. This ensures internal consistency: a
       hitter's projected season K% is the natural blend of their vL and vR
       K% projections, weighted by the share of PAs each side will get.

The rescaling is implemented as a multiplicative scale factor:

    implied_overall = vL_share × proj_vL + vR_share × proj_vR
    scale           = main_overall / implied_overall
    final_vL        = proj_vL × scale
    final_vR        = proj_vR × scale

Where vL_share is derived from the player's historical PA distribution
(typical hitter sees ~25% LHP, 75% RHP). The constraint anchors splits to
the main projection rather than letting them drift due to sparser per-side
samples.

Why direct projection over a multiplier approach?
- Multiplier YoY r is only 0.32-0.42 (per-player platoon gap not very
  stable as a ratio)
- Direct per-side YoY r is 0.63 (vL) and 0.78 (vR) — much higher
- Direct projection more naturally handles the case where one side has
  far less history than the other (BIP-like player vs reliever splits)

Sample sizes
------------
Typical PA per side per season:
    Hitters: 145 vL / 410 vR for everyday hitters
    Pitchers: 100-250 BF per side for full-time pitchers

Both are sparser than full-season totals, so we apply HEAVIER shrinkage on
the per-side rate models (k_pa=150-200) than the overall models (k=100).

How splits are used
-------------------
For a daily lineup projection: identify the starting pitcher's hand → use
the hitter's vL or vR projection for that PA. For the pitcher's projection
vs the entire lineup: weight the pitcher's vL and vR projections by the
lineup composition.

The pipeline emits both vL and vR sets of columns; downstream consumers
pick by handedness.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# Metrics we project per-side. Each is a per-PA rate (count / PA).
# Note: P_BIPOut is computed as the residual to enforce probability sum=1.
SPLIT_RATE_METRICS = {
    "P_K":   ("K",   100),  # (count_col, default k_pa shrinkage)
    "P_BB":  ("BB",  150),
    "P_HBP": ("HBP", 300),
    "P_SF":  ("SF",  400),
    "P_HR":  ("HR",  200),
    "P_1B":  ("_1B", 200),
    "P_2B":  ("2B",  300),
    "P_3B":  ("3B",  500),
}

# Default historical PA-share vs LHP if a player has no history to infer from.
# League-average is roughly 26% vL for hitters; pitchers see roughly 41% LHB.
DEFAULT_HITTER_VL_SHARE   = 0.26
DEFAULT_PITCHER_VL_SHARE  = 0.41


def _compute_1B(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 1B = H - 2B - 3B - HR. Adds `_1B` column."""
    df = df.copy()
    df["_1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["_1B"] = df["_1B"].clip(lower=0)
    return df


def _project_one_metric_one_side(splits_df: pd.DataFrame, target_year: int,
                                  side: str, count_col: str, k_pa: float,
                                  decay: float = 0.85,
                                  max_history_years: int = 5) -> pd.DataFrame:
    """Project one rate metric (e.g., K%) for one side (vL or vR).

    Returns DataFrame with PlayerId, projected rate, and n_eff.
    """
    prior = splits_df[(splits_df["Season"] < target_year) &
                      (splits_df["Season"] >= target_year - max_history_years) &
                      (splits_df["Side"] == side) &
                      (splits_df["PA"] >= 20)].copy()
    if prior.empty:
        return pd.DataFrame()

    # League rate ON THIS SIDE
    league_rate = (float(prior[count_col].sum())
                   / max(float(prior["PA"].sum()), 1.0))

    prior["_rate"] = prior[count_col] / prior["PA"].replace(0, np.nan)
    rows = []
    for pid, g in prior.groupby("PlayerId"):
        g = g.sort_values("Season").copy()
        g["yb"] = target_year - g["Season"].astype(int)
        w = g["PA"].astype(float).values * (decay ** g["yb"].values)
        if w.sum() == 0:
            continue
        weighted = float(np.sum(g["_rate"].fillna(0).astype(float).values * w)
                         / w.sum())
        n_eff = float(w.sum())
        pred = (n_eff * weighted + k_pa * league_rate) / (n_eff + k_pa)
        rows.append({
            "PlayerId":              int(pid),
            f"rate_{side}":          float(pred),
            f"n_eff_{side}":         n_eff,
        })
    return pd.DataFrame(rows)


def _project_pa_share(splits_df: pd.DataFrame, target_year: int,
                      default_share: float,
                      max_history_years: int = 3,
                      shrink_pa: float = 200.0) -> pd.DataFrame:
    """Per-player expected vL_share = expected PAs vs LHP / total PAs.

    Uses recent-season PA counts on each side. Shrunk toward the population
    default (e.g., 0.26 for hitters) — particularly important for low-PA
    players where the split observed is noisy.

    Returns DataFrame with PlayerId, vL_share, vR_share.
    """
    prior = splits_df[(splits_df["Season"] < target_year) &
                      (splits_df["Season"] >= target_year - max_history_years)]
    if prior.empty:
        return pd.DataFrame(columns=["PlayerId", "vL_share", "vR_share"])

    grp = prior.groupby(["PlayerId", "Side"])["PA"].sum().unstack(fill_value=0)
    if "vL" not in grp:
        grp["vL"] = 0
    if "vR" not in grp:
        grp["vR"] = 0
    grp["total"] = grp["vL"] + grp["vR"]
    # Shrink toward default
    grp["vL_share"] = (grp["vL"] + shrink_pa * default_share) / (
        grp["total"] + shrink_pa
    )
    grp["vR_share"] = 1.0 - grp["vL_share"]
    return grp[["vL_share", "vR_share"]].reset_index()


def project_splits(splits_df: pd.DataFrame, target_year: int,
                   overall_proj: pd.DataFrame,
                   group: str = "hitting",
                   decay: float = 0.85,
                   max_history_years: int = 5) -> pd.DataFrame:
    """End-to-end per-side rate projection with overall-anchored constraint.

    Parameters
    ----------
    splits_df : pd.DataFrame
        Output of `fetch_split_rates` for hitters or pitchers, with columns
        Season, PlayerId, Side ('vL'/'vR'), PA, K, BB, HBP, SF, H, HR, 2B, 3B.
    target_year : int
        Year to project for.
    overall_proj : pd.DataFrame
        The main pipeline's per-PA event probabilities. Must have PlayerId,
        P_K, P_BB, P_HBP, P_SF, P_HR, P_1B, P_2B, P_3B, P_BIPOut.
        Used to anchor the per-side projections so the implied weighted
        average matches the overall projection.
    group : str
        'hitting' or 'pitching'. Determines the default vL_share fallback.

    Returns
    -------
    DataFrame with one row per player. New columns:
        P_K_vL, P_BB_vL, P_HBP_vL, P_SF_vL, P_HR_vL, P_1B_vL, P_2B_vL,
        P_3B_vL, P_BIPOut_vL  (and same for _vR)
        vL_share, vR_share (used for the anchoring weight)
        n_eff_K_vL, n_eff_K_vR (sample sizes for the K projection)
    """
    df = _compute_1B(splits_df)
    default_share = (DEFAULT_HITTER_VL_SHARE if group == "hitting"
                     else DEFAULT_PITCHER_VL_SHARE)

    # PA share
    share_df = _project_pa_share(df, target_year, default_share)
    out = overall_proj[["PlayerId"]].drop_duplicates().merge(
        share_df, on="PlayerId", how="left",
    )
    out["vL_share"] = out["vL_share"].fillna(default_share)
    out["vR_share"] = 1.0 - out["vL_share"]

    # Project each metric per side
    for ev_col, (count_col, k_pa) in SPLIT_RATE_METRICS.items():
        for side in ("vL", "vR"):
            sub = _project_one_metric_one_side(
                df, target_year, side, count_col, k_pa,
                decay=decay, max_history_years=max_history_years,
            )
            if sub.empty:
                out[f"{ev_col}_{side}"] = np.nan
                continue
            sub = sub.rename(columns={
                f"rate_{side}": f"{ev_col}_{side}",
                f"n_eff_{side}": f"n_eff_{ev_col}_{side}",
            })
            out = out.merge(sub, on="PlayerId", how="left")

    # Anchor to overall: for each metric, compute the implied overall and
    # rescale so it matches the projection from the main pipeline. This is a
    # mild correction in most cases but prevents drift.
    for ev_col in SPLIT_RATE_METRICS:
        overall = overall_proj.set_index("PlayerId")[ev_col]
        out = out.merge(
            overall.rename(f"{ev_col}_overall"),
            left_on="PlayerId", right_index=True, how="left",
        )
        # Players might not have one side projected; backfill with overall
        for side in ("vL", "vR"):
            out[f"{ev_col}_{side}"] = out[f"{ev_col}_{side}"].fillna(
                out[f"{ev_col}_overall"]
            )

        implied = (out["vL_share"] * out[f"{ev_col}_vL"]
                   + out["vR_share"] * out[f"{ev_col}_vR"])
        # Avoid division by zero; if implied is 0, scale=1
        scale = np.where(implied > 1e-9,
                         out[f"{ev_col}_overall"] / implied.replace(0, 1),
                         1.0)
        out[f"{ev_col}_vL"] = out[f"{ev_col}_vL"] * scale
        out[f"{ev_col}_vR"] = out[f"{ev_col}_vR"] * scale

    # Compute BIPOut as residual on each side, enforcing sum = 1
    for side in ("vL", "vR"):
        non_bipout_sum = sum(
            out[f"{ev}_{side}"].fillna(0) for ev in SPLIT_RATE_METRICS
        )
        # Clamp to [0.05, 0.95] for safety, scale others if necessary
        excess = (non_bipout_sum > 0.95).fillna(False)
        if excess.any():
            # Scale down non_bipout events so sum stays at 0.95
            factor = pd.Series(1.0, index=out.index)
            factor[excess] = 0.95 / non_bipout_sum[excess]
            for ev in SPLIT_RATE_METRICS:
                out.loc[excess, f"{ev}_{side}"] = (
                    out.loc[excess, f"{ev}_{side}"] * factor[excess]
                )
            non_bipout_sum = sum(
                out[f"{ev}_{side}"].fillna(0) for ev in SPLIT_RATE_METRICS
            )
        out[f"P_BIPOut_{side}"] = (1.0 - non_bipout_sum).clip(lower=0.05,
                                                              upper=0.95)

    return out


def attach_splits_to_main(main_df: pd.DataFrame,
                          splits_proj: pd.DataFrame) -> pd.DataFrame:
    """Merge per-side projection columns onto the main projection frame."""
    keep = ["PlayerId", "vL_share", "vR_share"]
    for ev in list(SPLIT_RATE_METRICS) + ["P_BIPOut"]:
        for side in ("vL", "vR"):
            col = f"{ev}_{side}"
            if col in splits_proj.columns:
                keep.append(col)
    return main_df.merge(splits_proj[keep], on="PlayerId", how="left")

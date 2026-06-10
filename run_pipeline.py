#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end PA-projection pipeline orchestrator.

USAGE:
    python run_pipeline.py                          # use config defaults
    python run_pipeline.py --target-year 2027
    python run_pipeline.py --bip-dir /path/to/bip   # already-converted CSVs
    python run_pipeline.py --skip-2026-scrape       # don't re-fetch live data

Steps:
    1. Acquire data (statsapi rates, sprint speeds, target-year-1 Statcast).
    2. Build & validate K%/BB% models (Ridge+HGB ensemble, quantile SDs).
    3. Project HBP% & SF% via beta-binomial shrinkage.
    4. Run multi-season BIP imputation (port of impute_observations.R).
    5. Train BIP-outcome XGBoost on real BIP data (port of mlb_projections_statcast.R).
    6. Score all (real + imputed) BIPs, aggregate per player, per season,
       apply OUT pull and season weighting.
    7. Combine K%/BB%/HBP%/SF% with BIP outcomes into 9-event PA-level
       distributions for every active hitter and pitcher.

Outputs:
    OUTPUT_DIR/hitter_pa_projections_<year>.csv
    OUTPUT_DIR/pitcher_pa_projections_<year>.csv

Both contain one row per player with point probabilities (P_*) and standard
deviations (SD_*) for K, BB, HBP, SF, HR, 3B, 2B, 1B, BIPOut. Probabilities
sum to 1.0 per player.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import (
    TARGET_YEAR, RATE_HIST_START, CACHE_DIR, OUTPUT_DIR,
    INPUT_BIP_FILES, INPUT_HISTORICAL,
    SHRINK_K, SHRINK_K_PITCHER,
    RATE_MIN_PA_TRAIN, RATE_MIN_PA_ACTIVE, RATE_ACTIVE_LOOKBACK,
    BIP_OUTCOMES,
)
from data_acquisition import (
    fetch_rate_data, fetch_sprint_speeds, fetch_statcast_season,
    fetch_chadwick_lookup, fetch_team_rpg, fetch_park_factors,
)
from rate_models import (
    build_rate_panel, build_inference_panel, fit_and_predict_rate,
    predict_rates, project_simple_rate,
)
from bip_imputation import impute_bip
from bip_outcomes import BIPOutcomeModel, map_event_to_outcome
from pa_aggregation import aggregate_bip_to_player, build_pa_projections

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders for BIP data — accepts pre-converted CSVs or live Statcast scrape
# ─────────────────────────────────────────────────────────────────────────────

def _load_bip_csvs(bip_dir: Path) -> dict[int, pd.DataFrame]:
    """Load any bip_<year>.csv from bip_dir."""
    out = {}
    if not bip_dir.exists():
        return out
    for p in sorted(bip_dir.glob("bip_*.csv")):
        try:
            yr_str = p.stem.split("_")[1]
            yr = int(yr_str)
            df = pd.read_csv(p)
            if "Season" not in df.columns:
                df["Season"] = yr
            out[yr] = df
            print(f"  Loaded {p.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")
    return out


def _load_historical_csv(p: Path) -> pd.DataFrame:
    """Load the historical 2015-2019 BIP CSV (already split by Season)."""
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"  Loaded historical: {len(df):,} rows, seasons={sorted(df['Season'].unique())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step orchestration
# ─────────────────────────────────────────────────────────────────────────────

def step1_acquire_rates(target_year: int, force: bool):
    print("\n" + "═" * 70)
    print(f"STEP 1: Acquire rate data (statsapi) for {RATE_HIST_START}–{target_year - 1}")
    print("═" * 70)
    return fetch_rate_data(target_year, RATE_HIST_START, force=force)


def step2_build_rate_models(hit_df: pd.DataFrame, pit_df: pd.DataFrame,
                            target_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "═" * 70)
    print(f"STEP 2: Rate models (K%, BB%, HBP%, SF%) → {target_year}")
    print("═" * 70)
    from pipeline_config import (
        RATE_DECAY, RATE_SHRINK_K_HITTER, RATE_SHRINK_K_PITCHER,
        RATE_MAX_HISTORY_YEARS,
    )
    from rate_models import fit_and_predict_decay_rate

    def _do(role_df, shrink_dict, pa_col, role_name, rate_shrink_k):
        print(f"\n  Building {role_name} panel...")
        panel = build_rate_panel(role_df, pa_col, shrink_dict)
        print(f"  Panel: {len(panel):,} player-season rows")

        infer = build_inference_panel(role_df, target_year, pa_col, shrink_dict)
        print(f"  Inference (active players): {len(infer):,}")

        # K% and BB% — PA-weighted recency-decay shrinkage with bounded window.
        # Bounded window addresses the "skill evolution" problem (e.g., Judge's
        # walk rate climbed dramatically from 2022 onwards, and his 2018-21
        # data drags him down if we include all history).
        for rate in ["K%", "BB%"]:
            print(f"\n  -- {role_name} {rate} (decay={RATE_DECAY}, k={rate_shrink_k}, "
                  f"max_history={RATE_MAX_HISTORY_YEARS}) --")
            infer, metrics, scale = fit_and_predict_decay_rate(
                role_df, target_year, pa_col, rate,
                prior_k=rate_shrink_k, decay=RATE_DECAY, infer=infer,
                max_history_years=RATE_MAX_HISTORY_YEARS,
            )
            if len(metrics):
                print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"  SD calibration scale: {scale:.3f}")
            print(f"  Mean Pred_{rate}: {infer[f'Pred_{rate}'].mean():.4f}  "
                  f"Mean SD: {infer[f'SD_{rate}'].mean():.4f}")

        # HBP% and SF% via beta-binomial shrinkage (unchanged)
        for rate in ["HBP%", "SF%"]:
            print(f"\n  -- {role_name} {rate} (beta-binomial shrinkage) --")
            infer = project_simple_rate(
                role_df, target_year, pa_col, rate,
                prior_k=shrink_dict[rate], infer=infer
            )
            print(f"  Mean Pred_{rate}: {infer[f'Pred_{rate}'].mean():.4f}  "
                  f"Mean SD: {infer[f'SD_{rate}'].mean():.4f}")
        return infer

    hit_rates = _do(hit_df, SHRINK_K, "PA", "Hitter", RATE_SHRINK_K_HITTER)
    pit_rates = _do(pit_df, SHRINK_K_PITCHER, "TBF", "Pitcher", RATE_SHRINK_K_PITCHER)
    return hit_rates, pit_rates


def step3_load_bip_data(target_year: int, bip_dir: Path | None,
                       skip_2026_scrape: bool, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "═" * 70)
    print(f"STEP 3: Load BIP data")
    print("═" * 70)

    historical = pd.DataFrame()
    recent = {}

    # User-provided BIP CSVs
    if bip_dir:
        bip_dir = Path(bip_dir)
        recent.update(_load_bip_csvs(bip_dir))

    # Historical RDS/CSV — if user pre-converted to CSV we use it, else
    # attempt to convert via Rscript on-the-fly (requires R installed).
    if INPUT_HISTORICAL.exists():
        if INPUT_HISTORICAL.suffix == ".csv":
            historical = _load_historical_csv(INPUT_HISTORICAL)
        else:
            csv_path = CACHE_DIR / "bip_historical.csv"
            if csv_path.exists():
                historical = pd.read_csv(csv_path)
                print(f"  Loaded cached historical: {len(historical):,} rows")
            else:
                # Try Rscript conversion as fallback
                import subprocess, shutil
                if shutil.which("Rscript"):
                    print("  Converting historical RDS → CSV via Rscript...")
                    # Falls back to user running their own conversion
                    print("  (would call Rscript here — install r-base-core)")

    # Live 2026 (or target_year - 1) scrape from Statcast — only if not provided
    most_recent = target_year - 1
    if most_recent not in recent and not skip_2026_scrape:
        print(f"\n  Scraping {most_recent} BIP data from Statcast...")
        recent[most_recent] = fetch_statcast_season(most_recent, force=force)

    # Concatenate recent + historical
    parts = []
    if not historical.empty:
        parts.append(historical)
    for yr in sorted(recent.keys()):
        parts.append(recent[yr])
    bip_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Drop the historical (2015-2019) from the imputation pool — only the
    # last 3 seasons feed the season-weighted aggregation. But XGBoost
    # benefits from all of it.
    bip_imp_pool = bip_all[bip_all["Season"].isin(
        list(range(target_year - 3, target_year))
    )].copy()

    print(f"\n  Training pool (all years): {len(bip_all):,}")
    print(f"  Imputation pool (last 3 seasons): {len(bip_imp_pool):,}")
    print(f"  Imputation seasons: {sorted(bip_imp_pool['Season'].unique())}")
    return bip_all, bip_imp_pool


def step4_impute(bip_imp_pool: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═" * 70)
    print("STEP 4: Multi-season BIP imputation")
    print("═" * 70)
    imputed = impute_bip(bip_imp_pool)
    return imputed


def step5_train_xgb(bip_all: pd.DataFrame, sprint: pd.DataFrame) -> BIPOutcomeModel:
    print("\n" + "═" * 70)
    print("STEP 5: BIP outcome XGBoost training")
    print("═" * 70)
    # Map raw events → 5-class outcome
    df = bip_all.dropna(subset=["events"]).copy()
    df["Result"] = df["events"].apply(map_event_to_outcome)
    df = df.dropna(subset=["Result"])
    # Merge sprint speed by (Season, batter)
    df = _attach_sprint(df, sprint)
    # Fill missing sprint speed with 27 ft/s (R script default)
    df["sprint_speed"] = df["sprint_speed"].fillna(27)
    print(f"  Training rows after event mapping: {len(df):,}")
    model = BIPOutcomeModel()
    model.fit(df)
    return model


def step6_score_and_aggregate(imputed: pd.DataFrame, model: BIPOutcomeModel,
                              sprint: pd.DataFrame,
                              target_year: int,
                              league_out_pop: float,
                              debug_dir: Path | None = None
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "═" * 70)
    print("STEP 6: Score imputed BIPs and aggregate per player")
    print("═" * 70)
    # Attach sprint speed (batter-side only; for pitchers' BIPs we still
    # use the batter's sprint speed — this is correct because sprint speed
    # is about how the runner converts contact to outcomes).
    scored = _attach_sprint(imputed, sprint)
    scored["sprint_speed"] = scored["sprint_speed"].fillna(27)

    print(f"  Scoring {len(scored):,} BIPs (real + imputed)...")
    proba = model.predict_proba(scored)
    scored = pd.concat([scored.reset_index(drop=True),
                       proba.reset_index(drop=True)], axis=1)

    # Per the R script (line 296: `x <- mean(results$prob_out)`), the OUT
    # reference for the 10/90 regression is the mean of the MODEL's
    # predicted out probability across ALL scored BIPs — real + imputed.
    # This is subtly different from the empirical OUT% on real BIPs;
    # using the model's own mean keeps the regression internally consistent.
    league_out_pop_model = float(scored["prob_out"].mean())
    print(f"  Model-mean OUT% across all scored BIPs: {league_out_pop_model:.4f}")

    # Per-player aggregation (separate for batter / pitcher views)
    print("\n  Aggregating BIPs per batter...")
    bip_batter = scored[scored["batter"].notna()].copy()
    bip_batter["batter"] = bip_batter["batter"].astype(int)
    hitters_bip = aggregate_bip_to_player(
        bip_batter, target_year, "batter",
        is_pitcher=False, league_out_pop=league_out_pop_model
    )

    print("  Aggregating BIPs per pitcher...")
    bip_pitcher = scored[scored["pitcher"].notna()].copy()
    bip_pitcher["pitcher"] = bip_pitcher["pitcher"].astype(int)
    pitchers_bip = aggregate_bip_to_player(
        bip_pitcher, target_year, "pitcher",
        is_pitcher=True, league_out_pop=league_out_pop_model
    )

    print(f"  Hitter BIP aggregations: {len(hitters_bip)}")
    print(f"  Pitcher BIP aggregations: {len(pitchers_bip)}")

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        hitters_bip.to_parquet(debug_dir / "hitters_bip_agg.parquet")
        pitchers_bip.to_parquet(debug_dir / "pitchers_bip_agg.parquet")
    return hitters_bip, pitchers_bip


def step7_combine(hit_rates: pd.DataFrame, pit_rates: pd.DataFrame,
                  hitters_bip: pd.DataFrame, pitchers_bip: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "═" * 70)
    print("STEP 7: Combine rate + BIP projections into PA-level distributions")
    print("═" * 70)
    print("\n  Hitters:")
    h_final = build_pa_projections(hitters_bip, hit_rates, "batter", "PlayerId")
    print(f"    {len(h_final)} hitter projections")

    print("\n  Pitchers:")
    p_final = build_pa_projections(pitchers_bip, pit_rates, "pitcher", "PlayerId")
    print(f"    {len(p_final)} pitcher projections")
    return h_final, p_final


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attach_sprint(df: pd.DataFrame, sprint: pd.DataFrame) -> pd.DataFrame:
    """Attach sprint_speed to BIP rows by (Season, batter)."""
    df = df.copy()
    df["batter"]    = pd.to_numeric(df["batter"],   errors="coerce").astype("Int64")
    df["Season"]    = pd.to_numeric(df["Season"],   errors="coerce").astype(int)
    s = sprint.rename(columns={"year": "Season", "player_id": "batter"})
    s["batter"] = s["batter"].astype("Int64")
    s["Season"] = s["Season"].astype(int)
    s = s[["Season", "batter", "sprint_speed"]].drop_duplicates(
        subset=["Season", "batter"])
    return df.merge(s, on=["Season", "batter"], how="left")


def _compute_league_out(bip_with_events: pd.DataFrame) -> float:
    """Empirical league-average OUT% over recent seasons."""
    df = bip_with_events.dropna(subset=["events"]).copy()
    df["Result"] = df["events"].apply(map_event_to_outcome)
    df = df.dropna(subset=["Result"])
    return (df["Result"] == "out").mean()


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

# Probabilities for the 9-event PA distribution
PROB_COLS = ["P_K", "P_BB", "P_HBP", "P_SF", "P_HR", "P_3B", "P_2B", "P_1B", "P_BIPOut"]
SD_COLS   = ["SD_K", "SD_BB", "SD_HBP", "SD_SF", "SD_HR", "SD_3B", "SD_2B", "SD_1B", "SD_BIPOut"]

# SB column group (hitters only). P_SB is per-PA successful steals;
# P_SB_ATTEMPT is per-PA attempts (SB+CS); P_CS is per-PA caught-stealing.
# The decomposition fields expose the underlying components for auditability.
SB_COLS    = ["P_SB_ATTEMPT", "P_SB", "P_CS",
              "SD_SB_ATTEMPT", "SD_SB",
              "Pred_attempts_per_opp", "Pred_success_rate",
              "Pred_steal_opp_per_PA",
              "n_eff_opp", "n_eff_attempts",
              "sprint_speed_used"]

# R/RBI column group (hitters only). P_R and P_RBI are per-PA rates at the
# player's projected team's run environment. The "neutral" versions are the
# projection at league-average run environment, and Pred_target_team_factor
# is what we multiplied by. Pred_lineup_slot is the estimated 1-9 slot.
RUNS_RBI_COLS = ["P_R", "P_RBI", "SD_R", "SD_RBI",
                 "Pred_R_per_PA_neutral", "Pred_RBI_per_PA_neutral",
                 "Pred_target_team_factor", "Pred_target_team_id",
                 "Pred_lineup_slot",
                 "n_eff_R_per_PA", "n_eff_RBI_per_PA"]

# Park-adjusted column groups (both hitters and pitchers). All neutral
# probabilities have a `_park` counterpart. Effective park factors (after
# the 50/50 home/away blend) are exposed via `eff_HR`, `eff_1B`, etc., and
# raw home-park factors via `pf_HR`, `pf_1B`, etc.
PARK_PROB_COLS = ["P_K_park", "P_BB_park", "P_HBP_park", "P_HR_park",
                  "P_1B_park", "P_2B_park", "P_3B_park",
                  "P_SF_park", "P_BIPOut_park"]
PARK_SUMMARY_COLS = ["AVG_park", "OBP_park", "BABIP_park",
                     "AVG_against_park", "BABIP_against_park"]
PARK_SB_COLS  = ["P_SB_ATTEMPT_park", "P_SB_park", "P_CS_park",
                 "Pred_steal_opp_per_PA_park"]
PARK_FACTOR_COLS = ["home_park_team_id",
                    "pf_HR", "pf_1B", "pf_2B", "pf_3B", "pf_BB", "pf_SO",
                    "eff_HR", "eff_1B", "eff_2B", "eff_3B", "eff_BB", "eff_SO"]


# Pitcher summary columns. RA9 and TBF_per_IP are derived from per-PA event
# probabilities via linear weights. ERA is RA9 × role-specific ER/RA ratio
# (starter 0.934, reliever 0.889 — see pitcher_outputs.py for justification).
# Both neutral and park-adjusted versions are produced when park factors are
# applied. HBP_pct is P_HBP × 100 for readability. WP/PA is separately
# projected via shrinkage on pitcher history.
PITCHER_SUMMARY_COLS = ["role", "weighted_IP_per_G", "er_ra_ratio",
                        "RA9", "RA9_park",
                        "ERA", "ERA_park",
                        "R_per_PA", "R_per_PA_park",
                        "TBF_per_IP", "TBF_per_IP_park",
                        "HBP_pct", "HBP_pct_park",
                        "Pred_WP_per_PA", "n_eff_WP"]

# Platoon-split column group. Each per-PA event probability has a `_vL` and
# `_vR` counterpart. For hitters: vL = facing LHP, vR = facing RHP. For
# pitchers: vL = facing LHB, vR = facing RHB. `vL_share` / `vR_share` are
# the player's expected PA distribution; downstream daily-projection
# consumers can use these to weight matchup outcomes. `BatSide` (hitters)
# or `PitchHand` (pitchers) flags the player's own handedness.
SPLITS_COLS = ["vL_share", "vR_share", "BatSide", "PitchHand",
               "P_K_vL", "P_K_vR",
               "P_BB_vL", "P_BB_vR",
               "P_HBP_vL", "P_HBP_vR",
               "P_HR_vL", "P_HR_vR",
               "P_1B_vL", "P_1B_vR",
               "P_2B_vL", "P_2B_vR",
               "P_3B_vL", "P_3B_vR",
               "P_SF_vL", "P_SF_vR",
               "P_BIPOut_vL", "P_BIPOut_vR"]


def _attach_names(df: pd.DataFrame, chadwick: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Use Chadwick lookup to ensure name is filled even when rate-only path."""
    df = df.copy()
    if chadwick.empty:
        return df  # statsapi names already there
    cw = chadwick.rename(columns={"key_mlbam": id_col,
                                  "name_first": "_first",
                                  "name_last":  "_last"})
    cw[id_col] = cw[id_col].astype("Int64")
    df[id_col] = df[id_col].astype("Int64")
    df = df.merge(cw, on=id_col, how="left")
    df["Name"] = df["Name"].where(df["Name"].notna(),
                                  df["_first"].fillna("") + " " + df["_last"].fillna(""))
    df.drop(columns=["_first", "_last"], inplace=True)
    return df


def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Round probabilities and SDs to 5 dp; reorder columns. SB, R/RBI, and
    park-adjusted columns are included only when present (hitters get them;
    pitchers don't except for park-adjusted on both)."""
    df = df.copy()
    round_cols = (PROB_COLS + SD_COLS
                  + ["P_SB_ATTEMPT", "P_SB", "P_CS",
                     "SD_SB_ATTEMPT", "SD_SB",
                     "Pred_attempts_per_opp", "Pred_success_rate",
                     "Pred_steal_opp_per_PA",
                     "P_R", "P_RBI", "SD_R", "SD_RBI",
                     "Pred_R_per_PA_neutral", "Pred_RBI_per_PA_neutral",
                     "Pred_target_team_factor"]
                  + PARK_PROB_COLS + PARK_SUMMARY_COLS + PARK_SB_COLS
                  + ["pf_HR", "pf_1B", "pf_2B", "pf_3B", "pf_BB", "pf_SO",
                     "eff_HR", "eff_1B", "eff_2B", "eff_3B", "eff_BB", "eff_SO"]
                  + ["RA9", "RA9_park", "ERA", "ERA_park",
                     "R_per_PA", "R_per_PA_park",
                     "TBF_per_IP", "TBF_per_IP_park",
                     "HBP_pct", "HBP_pct_park",
                     "Pred_WP_per_PA"]
                  + [c for c in SPLITS_COLS if c.startswith("P_")
                     or c in ("vL_share", "vR_share")])
    for c in round_cols:
        if c in df.columns:
            df[c] = df[c].round(5)
    keep_meta = ["PlayerId", "Name", "Team", "Age", "Last_PA", "Career_PA", "N_BIP"]
    extra_sb = [c for c in SB_COLS if c in df.columns]
    extra_rr = [c for c in RUNS_RBI_COLS if c in df.columns]
    extra_park_prob = [c for c in PARK_PROB_COLS if c in df.columns]
    extra_park_sum  = [c for c in PARK_SUMMARY_COLS if c in df.columns]
    extra_park_sb   = [c for c in PARK_SB_COLS if c in df.columns]
    extra_park_fac  = [c for c in PARK_FACTOR_COLS if c in df.columns]
    extra_pit_summ  = [c for c in PITCHER_SUMMARY_COLS if c in df.columns]
    extra_splits    = [c for c in SPLITS_COLS if c in df.columns]
    cols = ([c for c in keep_meta if c in df.columns]
            + PROB_COLS + SD_COLS
            + extra_sb + extra_rr
            + extra_park_prob + extra_park_sum + extra_park_sb + extra_park_fac
            + extra_pit_summ
            + extra_splits)
    return df[cols]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def step8_project_sb(h_final: pd.DataFrame, hit_df: pd.DataFrame,
                     sprint: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Project SB/PA and SB-attempts/PA for each hitter using a chained
    decomposition:
        P_SB_ATTEMPT = (P_1B + P_BB + P_HBP) × Pred_attempts_per_opp
        P_SB         = P_SB_ATTEMPT × Pred_success_rate
        P_CS         = P_SB_ATTEMPT × (1 - Pred_success_rate)

    Attempts/opp is the most stable signal in the SB chain (YoY r=0.81),
    so we project it directly. Success rate has very low YoY stability
    (r ~ 0.12-0.20), so it's heavily shrunk to league average.

    Only hitters get SB projections (pitchers don't bat in the AL/NL post-2022).
    """
    print("\n" + "═" * 70)
    print("STEP 8: Project SB attempts & SB/PA (hitters)")
    print("═" * 70)
    from sb_model import (
        project_attempts_per_opp, project_success_rate,
        combine_sb_components, derive_sb_per_pa,
    )
    from pipeline_config import (
        SB_K_ATTEMPTS, SB_K_SUCCESS, SB_SPRINT_COEF, SB_SPRINT_BASELINE,
        SB_MAX_HISTORY_YEARS, SB_DECAY,
    )

    # Stage 2a: attempts per opportunity
    att_proj = project_attempts_per_opp(
        hit_df, target_year, sprint,
        k_attempts=SB_K_ATTEMPTS, sprint_coef=SB_SPRINT_COEF,
        sprint_baseline=SB_SPRINT_BASELINE,
        max_history_years=SB_MAX_HISTORY_YEARS, decay=SB_DECAY,
    )
    print(f"  Attempts/opp projections: {len(att_proj)} players")
    print(f"  Mean Pred_attempts_per_opp: {att_proj['Pred_attempts_per_opp'].mean():.4f}")
    print(f"  Top 5 attempt-prone:")
    top_att = (att_proj.merge(hit_df[["PlayerId", "Name"]].drop_duplicates("PlayerId"),
                               on="PlayerId", how="left")
               .nlargest(5, "Pred_attempts_per_opp"))
    print(top_att[["Name", "Pred_attempts_per_opp",
                   "sprint_speed_used", "n_eff_opp"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Stage 2b: success rate
    succ_proj = project_success_rate(
        hit_df, target_year,
        k_success=SB_K_SUCCESS,
        max_history_years=SB_MAX_HISTORY_YEARS, decay=SB_DECAY,
    )
    print(f"\n  Success rate projections: {len(succ_proj)} players")
    print(f"  Mean Pred_success_rate: {succ_proj['Pred_success_rate'].mean():.4f}")

    # Combine into rate frame
    h_final = combine_sb_components(h_final, att_proj, succ_proj)
    h_final = derive_sb_per_pa(h_final, k_attempts=SB_K_ATTEMPTS,
                                k_success=SB_K_SUCCESS)

    print(f"\n  Mean P_SB across all hitters:         {h_final['P_SB'].mean():.5f}")
    print(f"  Mean P_SB_ATTEMPT across all hitters: {h_final['P_SB_ATTEMPT'].mean():.5f}")
    print(f"  → SB/600 PA:       {h_final['P_SB'].mean() * 600:.2f}")
    print(f"  → Attempts/600 PA: {h_final['P_SB_ATTEMPT'].mean() * 600:.2f}")
    return h_final


def step9_project_runs_rbi(h_final: pd.DataFrame, hit_df: pd.DataFrame,
                            team_rpg: pd.DataFrame, target_year: int
                            ) -> pd.DataFrame:
    """Project R/PA and RBI/PA for each hitter.

    Detrends each season's historical rate by team_RPG/league_RPG, projects
    via PA-weighted recency-decay shrinkage, then re-applies the target
    team's run-environment factor. Estimates lineup slot 1-9 from the
    neutralized rates against the published slot-base-rate table.
    """
    print("\n" + "═" * 70)
    print("STEP 9: Project R/PA and RBI/PA (hitters)")
    print("═" * 70)
    from runs_rbi_model import project_runs_and_rbi
    from pipeline_config import (
        RUNS_RBI_K_PA, RUNS_RBI_DECAY, RUNS_RBI_MAX_HISTORY_YEARS,
    )

    if team_rpg is None or team_rpg.empty:
        print("  No team_rpg data available — skipping R/RBI projection")
        return h_final

    proj = project_runs_and_rbi(
        hit_df, team_rpg, target_year,
        k_pa=RUNS_RBI_K_PA, decay=RUNS_RBI_DECAY,
        max_history_years=RUNS_RBI_MAX_HISTORY_YEARS,
    )
    print(f"  Projections for {len(proj)} players")
    print(f"  League avg projected R/PA:   {proj['Pred_R_per_PA'].mean():.4f}")
    print(f"  League avg projected RBI/PA: {proj['Pred_RBI_per_PA'].mean():.4f}")
    print(f"  → R/600 PA:   {proj['Pred_R_per_PA'].mean() * 600:.1f}")
    print(f"  → RBI/600 PA: {proj['Pred_RBI_per_PA'].mean() * 600:.1f}")

    # Lineup slot distribution
    print(f"\n  Lineup slot distribution among projections:")
    for slot, n in proj["Pred_lineup_slot"].value_counts().sort_index().items():
        print(f"    slot {slot}: {n:>3} players")

    print(f"\n  Top 5 R/PA (post team-adjustment):")
    top = (proj.merge(hit_df[["PlayerId", "Name"]].drop_duplicates("PlayerId"),
                      on="PlayerId", how="left")
                .nlargest(5, "Pred_R_per_PA"))
    print(top[["Name", "Pred_R_per_PA_neutral", "Pred_target_team_factor",
               "Pred_R_per_PA", "Pred_lineup_slot"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Merge into final frame
    merge_cols = ["PlayerId", "Pred_R_per_PA_neutral", "Pred_RBI_per_PA_neutral",
                  "Pred_R_per_PA", "Pred_RBI_per_PA",
                  "Pred_target_team_factor", "Pred_target_team_id",
                  "Pred_lineup_slot",
                  "n_eff_R_per_PA", "n_eff_RBI_per_PA",
                  "SD_R_per_PA", "SD_RBI_per_PA"]
    h_final = h_final.merge(
        proj[merge_cols].rename(columns={
            "Pred_R_per_PA":   "P_R",
            "Pred_RBI_per_PA": "P_RBI",
            "SD_R_per_PA":     "SD_R",
            "SD_RBI_per_PA":   "SD_RBI",
        }),
        on="PlayerId", how="left",
    )
    return h_final


def step10_apply_park_factors(h_final: pd.DataFrame, p_final: pd.DataFrame,
                               hit_df: pd.DataFrame, pit_df: pd.DataFrame,
                               target_year: int, force: bool = False
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Statcast park factors to per-PA event probabilities.

    Produces a parallel set of `_park` columns alongside the existing neutral
    projections. Park factors apply BEFORE the SB calculation (since they
    change the opportunity rate); they are NOT applied to R/RBI (since the
    team_RPG factor already encodes park effects).

    For hitters: home park comes from `Pred_target_team_id` (already set by
    the R/RBI step). For pitchers: we derive it from each pitcher's most
    recent team in the history data.
    """
    print("\n" + "═" * 70)
    print("STEP 10: Apply park factors (neutral + park-adjusted output)")
    print("═" * 70)
    from park_factors import (
        apply_park_factors_to_projections,
        derive_park_adjusted_sb,
        derive_park_adjusted_summary_stats,
    )
    from pipeline_config import (
        PARK_HOME_SHARE, PARK_ROLLING_YEARS,
    )

    park_df = fetch_park_factors(target_year - 1, rolling=PARK_ROLLING_YEARS,
                                  force=force)
    if park_df is None or park_df.empty:
        print("  Park factors unavailable — skipping park adjustment")
        return h_final, p_final
    print(f"  Loaded {len(park_df)} venues; year_range={park_df['year_range'].iloc[0] if 'year_range' in park_df else '?'}")
    print(f"  Home/away share: {PARK_HOME_SHARE}/{1-PARK_HOME_SHARE} (81-81 schedule)")

    # ── Hitters ────────────────────────────────────────────────────────────
    if "Pred_target_team_id" in h_final.columns:
        h_final = apply_park_factors_to_projections(
            h_final, park_df,
            team_id_col="Pred_target_team_id",
            home_share=PARK_HOME_SHARE,
        )
        # Recompute SB on park-adjusted opportunity rate
        if "Pred_attempts_per_opp" in h_final.columns:
            h_final = derive_park_adjusted_sb(h_final)
        h_final = derive_park_adjusted_summary_stats(h_final)
        # Adjust P_R, P_RBI to use park-adjusted base if applicable —
        # we intentionally don't, per design decision (see pipeline_config).
        print(f"  Hitters: {(h_final['home_park_team_id'].notna()).sum()} with park assigned, "
              f"{(h_final['home_park_team_id'].isna()).sum()} fall back to neutral")
    else:
        print("  Hitter R/RBI step didn't run — no team_id column; skipping hitters")

    # ── Pitchers — need to derive home team from history ──────────────────
    if p_final is not None and len(p_final) > 0:
        # Map each pitcher's most recent team
        pit_history = (pit_df[pit_df["TeamId"].notna() & (pit_df["TBF"] >= 25)]
                       .sort_values(["PlayerId", "Season"]))
        most_recent = (pit_history.groupby("PlayerId")
                       .agg(TeamId=("TeamId", "last")).reset_index())
        p_final = p_final.merge(
            most_recent.rename(columns={"TeamId": "Pred_home_team_id"}),
            on="PlayerId", how="left",
        )
        p_final = apply_park_factors_to_projections(
            p_final, park_df,
            team_id_col="Pred_home_team_id",
            home_share=PARK_HOME_SHARE,
        )
        p_final = derive_park_adjusted_summary_stats(p_final)
        print(f"  Pitchers: {(p_final['home_park_team_id'].notna()).sum()} with park assigned, "
              f"{(p_final['home_park_team_id'].isna()).sum()} fall back to neutral")

    # Summary
    print()
    if "P_HR_park" in h_final.columns:
        print(f"  Hitter mean P_HR neutral={h_final['P_HR'].mean():.4f}, "
              f"park-adj mean={h_final['P_HR_park'].mean():.4f}")
        # Coors example
        coors = h_final[h_final["home_park_team_id"] == 115]
        if len(coors):
            print(f"  Coors hitters ({len(coors)}): mean P_HR_park / P_HR ratio = "
                  f"{(coors['P_HR_park'] / coors['P_HR'].replace(0,1)).mean():.3f}, "
                  f"mean P_1B_park / P_1B = "
                  f"{(coors['P_1B_park'] / coors['P_1B'].replace(0,1)).mean():.3f}")
    return h_final, p_final


def step11_pitcher_summary_outputs(p_final: pd.DataFrame, pit_df: pd.DataFrame,
                                    target_year: int) -> pd.DataFrame:
    """Derive RA9, ERA, TBF_per_IP, and WP/PA for pitchers.

    RA9 and TBF/IP come from the projected per-PA event probabilities via
    linear weights (see pitcher_outputs.py for calibration). Both neutral
    and park-adjusted versions are produced when park columns are present.

    ERA is derived from RA9 via a role-specific ER/RA ratio:
        Starters (IP/G >= 3.5):  0.934
        Relievers (IP/G < 3.5):  0.889
    Role is determined from each pitcher's recent IP/G history. Per-pitcher
    YoY stability of individual ER/RA is r=0.15-0.28 (mostly noise), so role
    constants are the right fidelity rather than per-pitcher modeling.

    WP/PA is projected via PA-weighted shrinkage on each pitcher's history.
    """
    print("\n" + "═" * 70)
    print("STEP 11: Pitcher RA9, ERA, TBF/IP, WP/PA outputs")
    print("═" * 70)
    from pitcher_outputs import (
        compute_ra9, project_wp_per_pa, RUNS_INTERCEPT_DEFAULT,
        determine_pitcher_role, role_to_er_ra_ratio,
        ER_RA_RATIO_STARTER, ER_RA_RATIO_RELIEVER,
    )
    from pipeline_config import (
        PITCHER_WP_K_PA, PITCHER_WP_DECAY, PITCHER_WP_MAX_HISTORY_YEARS,
    )

    if p_final is None or p_final.empty:
        print("  No pitchers to process — skipping")
        return p_final

    # Classify each pitcher's role (starter vs reliever) from recent IP/G
    role_df = determine_pitcher_role(pit_df, target_year)
    p_final = p_final.merge(role_df, on="PlayerId", how="left")
    p_final["role"] = p_final["role"].fillna("reliever")
    p_final["er_ra_ratio"] = role_to_er_ra_ratio(p_final["role"])
    n_starters = (p_final["role"] == "starter").sum()
    n_relievers = (p_final["role"] == "reliever").sum()
    print(f"  Role split: {n_starters} starters (ER/RA={ER_RA_RATIO_STARTER}) "
          f"/ {n_relievers} relievers (ER/RA={ER_RA_RATIO_RELIEVER})")

    # Neutral RA9 + ERA (with role-specific ER/RA)
    ra9, era, r_per_pa, tbf_per_ip = compute_ra9(
        p_final, suffix="",
        intercept=RUNS_INTERCEPT_DEFAULT,
        er_ra_ratio=p_final["er_ra_ratio"],
    )
    p_final["R_per_PA"]    = r_per_pa
    p_final["TBF_per_IP"]  = tbf_per_ip
    p_final["RA9"]         = ra9
    p_final["ERA"]         = era
    p_final["HBP_pct"]     = p_final["P_HBP"] * 100.0

    # Park-adjusted RA9 + ERA
    if "P_K_park" in p_final.columns:
        ra9_p, era_p, r_pa_p, tbf_ip_p = compute_ra9(
            p_final, suffix="_park",
            intercept=RUNS_INTERCEPT_DEFAULT,
            er_ra_ratio=p_final["er_ra_ratio"],
        )
        p_final["R_per_PA_park"]   = r_pa_p
        p_final["TBF_per_IP_park"] = tbf_ip_p
        p_final["RA9_park"]        = ra9_p
        p_final["ERA_park"]        = era_p
        p_final["HBP_pct_park"]    = p_final["P_HBP_park"] * 100.0

    # WP/PA projection
    wp_proj = project_wp_per_pa(
        pit_df, target_year,
        k_pa=PITCHER_WP_K_PA,
        decay=PITCHER_WP_DECAY,
        max_history_years=PITCHER_WP_MAX_HISTORY_YEARS,
    )
    p_final = p_final.merge(wp_proj, on="PlayerId", how="left")
    league_wp_per_pa = wp_proj["Pred_WP_per_PA"].median() if len(wp_proj) else 0.0075
    p_final["Pred_WP_per_PA"] = p_final["Pred_WP_per_PA"].fillna(league_wp_per_pa)

    print(f"  Mean RA9: {p_final['RA9'].mean():.2f}, Mean ERA: {p_final['ERA'].mean():.2f}")
    star = p_final[p_final["role"] == "starter"]
    rel  = p_final[p_final["role"] == "reliever"]
    if len(star) and len(rel):
        print(f"  Starters: mean RA9={star['RA9'].mean():.2f}, mean ERA={star['ERA'].mean():.2f}")
        print(f"  Relievers: mean RA9={rel['RA9'].mean():.2f}, mean ERA={rel['ERA'].mean():.2f}")
    print(f"  Mean TBF/IP: {p_final['TBF_per_IP'].mean():.3f}")
    print(f"  Mean WP/PA: {p_final['Pred_WP_per_PA'].mean():.5f}")

    print(f"\n  Top 5 best by projected ERA (park-adjusted):")
    era_col = "ERA_park" if "ERA_park" in p_final.columns else "ERA"
    top_cols = ["Name", "Team", "role", "TBF_per_IP", "RA9", "ERA"]
    if "ERA_park" in p_final.columns:
        top_cols.append("ERA_park")
    top_cols.extend(["Pred_WP_per_PA", "HBP_pct"])
    top = p_final.nsmallest(5, era_col)[top_cols]
    print(top.to_string(index=False,
        float_format=lambda x: f"{x:.3f}" if isinstance(x,(int,float)) else str(x)))
    return p_final


def step12_project_splits(h_final: pd.DataFrame, p_final: pd.DataFrame,
                           target_year: int, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project per-platoon-side rates (vL/vR) for hitters and pitchers.

    For hitters: vL = facing LHP, vR = facing RHP.
    For pitchers: vL = facing LHB, vR = facing RHB.

    Uses the same PA-weighted recency-decay shrinkage as the main rate models,
    applied to side-specific historical data. The resulting (vL, vR) pair is
    rescaled to anchor to the main pipeline's overall projection — ensuring
    internal consistency between the per-side projections and the season-
    level projection.

    Adds the per-side P_*_vL and P_*_vR columns plus vL_share/vR_share.
    """
    print("\n" + "═" * 70)
    print("STEP 12: Platoon-split (vL/vR) projections")
    print("═" * 70)
    from data_acquisition import fetch_split_rates, fetch_player_handedness
    from splits_model import project_splits
    from pipeline_config import SPLITS_DECAY, SPLITS_MAX_HISTORY_YEARS

    hit_splits, pit_splits = fetch_split_rates(target_year, force=force)
    if hit_splits.empty and pit_splits.empty:
        print("  No split data available — skipping")
        return h_final, p_final

    # Hitters
    if not hit_splits.empty and h_final is not None:
        h_split_proj = project_splits(
            hit_splits, target_year, h_final, group="hitting",
            decay=SPLITS_DECAY, max_history_years=SPLITS_MAX_HISTORY_YEARS,
        )
        # Drop existing columns we're about to overwrite, then merge
        cols_to_add = [c for c in h_split_proj.columns
                       if c not in h_final.columns or c == "PlayerId"]
        h_final = h_final.merge(h_split_proj[cols_to_add], on="PlayerId", how="left")
        print(f"  Hitters: {len(h_split_proj)} player projections")
        print(f"  Mean vL_share among hitters: "
              f"{h_split_proj['vL_share'].mean():.3f}")
        # Sanity: implied avg should match overall
        sample = h_final.dropna(subset=["P_K_vL", "P_K_vR", "P_K"]).head(5)
        if len(sample):
            implied = (sample["vL_share"] * sample["P_K_vL"]
                       + sample["vR_share"] * sample["P_K_vR"])
            print(f"  Sanity check (implied K% vs P_K, first 5 hitters):")
            for _, r in sample.iterrows():
                imp = r["vL_share"] * r["P_K_vL"] + r["vR_share"] * r["P_K_vR"]
                print(f"    {r['Name']:25s} P_K={r['P_K']:.4f}, implied={imp:.4f}, "
                      f"vL={r['P_K_vL']:.4f}, vR={r['P_K_vR']:.4f}")

    # Pitchers
    if not pit_splits.empty and p_final is not None:
        p_split_proj = project_splits(
            pit_splits, target_year, p_final, group="pitching",
            decay=SPLITS_DECAY, max_history_years=SPLITS_MAX_HISTORY_YEARS,
        )
        cols_to_add = [c for c in p_split_proj.columns
                       if c not in p_final.columns or c == "PlayerId"]
        p_final = p_final.merge(p_split_proj[cols_to_add], on="PlayerId", how="left")
        print(f"  Pitchers: {len(p_split_proj)} player projections")
        print(f"  Mean vL_share among pitchers: "
              f"{p_split_proj['vL_share'].mean():.3f}")

    # Attach handedness for context
    all_pids = []
    if h_final is not None:
        all_pids.extend(h_final["PlayerId"].dropna().astype(int).tolist())
    if p_final is not None:
        all_pids.extend(p_final["PlayerId"].dropna().astype(int).tolist())
    if all_pids:
        hand_df = fetch_player_handedness(all_pids, force=force)
        if h_final is not None and not hand_df.empty:
            h_final = h_final.merge(
                hand_df[["PlayerId", "BatSide"]], on="PlayerId", how="left",
            )
        if p_final is not None and not hand_df.empty:
            p_final = p_final.merge(
                hand_df[["PlayerId", "PitchHand"]], on="PlayerId", how="left",
            )

    return h_final, p_final


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-year", type=int, default=TARGET_YEAR)
    parser.add_argument("--bip-dir",     type=str, default=None,
                        help="Directory containing bip_<year>.csv files "
                             "(if pre-converted from RDS).")
    parser.add_argument("--skip-2026-scrape", action="store_true",
                        help="Skip live Statcast scrape (use only what's in --bip-dir).")
    parser.add_argument("--force",       action="store_true",
                        help="Force re-fetch of all cached data sources.")
    parser.add_argument("--output-dir",  type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = args.target_year

    # Step 1
    hit_df, pit_df = step1_acquire_rates(target, args.force)

    # Step 2
    hit_rates, pit_rates = step2_build_rate_models(hit_df, pit_df, target)

    # Step 3 — load BIP
    bip_all, bip_imp_pool = step3_load_bip_data(
        target, Path(args.bip_dir) if args.bip_dir else None,
        args.skip_2026_scrape, args.force
    )
    if bip_imp_pool.empty:
        print("\n[!] No BIP data available — only rate projections will be saved.")
        # Bail with rate-only output
        chadwick = fetch_chadwick_lookup(force=args.force)
        for df, name in [(hit_rates, "hitter"), (pit_rates, "pitcher")]:
            df2 = _attach_names(df, chadwick, "PlayerId")
            path = out_dir / f"{name}_rates_only_{target}.csv"
            df2.to_csv(path, index=False)
            print(f"Wrote {path}")
        return

    # Step 4 — impute
    imputed = step4_impute(bip_imp_pool)

    # Sprint speeds covering all imputation seasons (+1 for safety)
    sprint_years = sorted(set(bip_all["Season"].astype(int)))
    sprint = fetch_sprint_speeds(sprint_years, force=args.force)

    # Step 5 — train XGB on ALL real BIP data (not synthetic)
    real_bip = bip_all[~bip_all.get("_synthetic", False).astype(bool)
                       if "_synthetic" in bip_all.columns else bip_all.index ==
                       bip_all.index]
    model = step5_train_xgb(real_bip, sprint)

    # League OUT% on real recent BIPs
    league_out = _compute_league_out(
        bip_all[bip_all["Season"].isin(range(target - 3, target))])
    print(f"\n  League OUT% over recent seasons: {league_out:.4f}")

    # Step 6 — score & aggregate
    hitters_bip, pitchers_bip = step6_score_and_aggregate(
        imputed, model, sprint, target, league_out,
        debug_dir=Path(args.output_dir) / "debug"
    )

    # Step 7 — combine
    h_final, p_final = step7_combine(hit_rates, pit_rates, hitters_bip, pitchers_bip)

    # Step 8 — SB projection (hitters only)
    h_final = step8_project_sb(h_final, hit_df, sprint, target)

    # Step 9 — R/RBI projection (hitters only). Fetches team RPG for
    # historical seasons; uses most-recent team RPG as the target-year forecast.
    team_rpg = fetch_team_rpg(
        sorted(set(hit_df["Season"].astype(int))),
        force=args.force,
    )
    h_final = step9_project_runs_rbi(h_final, hit_df, team_rpg, target)

    # Step 10 — Park factor adjustment (both hitters and pitchers).
    # Produces _park columns alongside neutral projections.
    h_final, p_final = step10_apply_park_factors(
        h_final, p_final, hit_df, pit_df, target, force=args.force,
    )

    # Step 11 — Pitcher summary outputs (RA9, TBF/IP, WP/PA, HBP%).
    # Uses linear weights × per-PA probabilities; produces both neutral and
    # park-adjusted RA9. WP/PA projected via shrinkage on pitcher history.
    p_final = step11_pitcher_summary_outputs(p_final, pit_df, target)

    # Step 12 — Per-side platoon splits (vL/vR) for daily projections.
    # Each rate metric is projected per side and anchored back to the main
    # projection. Also fetches each player's handedness for downstream use.
    h_final, p_final = step12_project_splits(h_final, p_final, target, args.force)

    # Names
    chadwick = fetch_chadwick_lookup(force=args.force)
    h_final = _attach_names(h_final, chadwick, "PlayerId")
    p_final = _attach_names(p_final, chadwick, "PlayerId")

    # Format & save
    h_out = _format_output(h_final).sort_values("P_HR", ascending=False)
    p_out = _format_output(p_final).sort_values("P_K", ascending=False)

    hpath = out_dir / f"hitter_pa_projections_{target}.csv"
    ppath = out_dir / f"pitcher_pa_projections_{target}.csv"
    h_out.to_csv(hpath, index=False)
    p_out.to_csv(ppath, index=False)
    print(f"\nWrote {hpath} ({len(h_out)} hitters)")
    print(f"Wrote {ppath} ({len(p_out)} pitchers)")

    # Quick summary
    print("\n── Summary: top 10 hitters by P_HR ──")
    print(h_out.head(10)[["Name", "Team", "Last_PA",
                          "P_K", "P_BB", "P_HR", "P_BIPOut"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n── Summary: top 10 pitchers by P_K ──")
    print(p_out.head(10)[["Name", "Team", "Last_PA",
                          "P_K", "P_BB", "P_HR", "P_BIPOut"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

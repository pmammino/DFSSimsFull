# MLB Per-PA Projection Pipeline

End-to-end pipeline that produces handedness-specific per-plate-appearance
projections for every MLB hitter and pitcher. Outputs neutral, park-
adjusted, and platoon-split (vL/vR) versions of every per-PA event
probability, plus derived stats (R/PA, RBI/PA, SB, RA9, ERA, TBF/IP, WP/PA).

Intended use:
  - **Season / ROS projections** straight from the output CSVs
  - **Daily game-level projections** by feeding the per-PA splits into the
    game simulator (see `game_sim/`)

## Required inputs

These should sit in a `bip_inputs/` subdirectory alongside the pipeline code:

| File | Source | Purpose |
|---|---|---|
| `bip_2024.csv` | converted from `all_2024.rds` | Per-pitch BIP data for 2024 (launch_speed, launch_angle, adjusted_angle, stand, etc.) |
| `bip_2025.csv` | converted from `all_2025.rds` | Same for 2025 |
| `bip_2026.csv` | converted from `all_2026.rds` | Same for current season |
| `bip_historical.csv` | converted from `all_pitches.rds` | Multi-year BIP backfill (2018-2023) for the XGBoost outcome model |

The `.rds` files come from a statsapi/Baseball Savant scrape (whatever
process you already use). To convert them:

```bash
Rscript convert_rds_to_csv.R bip_inputs/all_2024.rds bip_inputs/bip_2024.csv 2024
Rscript convert_rds_to_csv.R bip_inputs/all_2025.rds bip_inputs/bip_2025.csv 2025
Rscript convert_rds_to_csv.R bip_inputs/all_2026.rds bip_inputs/bip_2026.csv 2026
Rscript convert_historical_lean.R bip_inputs/all_pitches.rds bip_inputs/bip_historical.csv
```

`convert_historical_lean.R` is memory-friendly for the huge multi-year
file (3.5M+ rows) — uses chunked reading and writes only the columns the
pipeline needs.

## Required Python packages

```
pandas pyarrow numpy scikit-learn scipy xgboost requests pybaseball pyyaml
```

(`pybaseball` is only used for the Chadwick name lookup; if it's missing
the pipeline still runs but some output rows will lack the Name column.)

## How to run

```bash
python run_pipeline.py \
    --target-year 2027 \
    --bip-dir bip_inputs \
    --output-dir out
```

Key flags:
  - `--target-year`: year you're projecting (e.g., 2027 for full-season
    or current-year ROS projections)
  - `--force`: bypass cache and re-fetch all data from statsapi
  - `--skip-2026-scrape`: skip the partial-current-season fetch (useful
    early in dev cycles)

Total runtime: ~6 minutes on a fresh fetch (most of that is the BIP
imputation + XGBoost training); ~2 minutes when caches are warm.

## Outputs

Two CSVs in `--output-dir`:

  - **`hitter_pa_projections_<year>.csv`** — ~595 hitters × ~99 columns
  - **`pitcher_pa_projections_<year>.csv`** — ~754 pitchers × ~88 columns

Each row is one player. Columns are grouped:

### Hitter columns
  - **Per-PA event probabilities (neutral):** `P_K`, `P_BB`, `P_HBP`,
    `P_HR`, `P_1B`, `P_2B`, `P_3B`, `P_SF`, `P_BIPOut` (sum to 1.0)
  - **SD per event:** `SD_K`, `SD_BB`, ...
  - **SB stack:** `Pred_attempts_per_opp`, `Pred_success_rate`,
    `P_SB_ATTEMPT`, `P_SB`, `P_CS`, `Pred_steal_opp_per_PA`
  - **R/RBI stack:** `P_R`, `P_RBI`, `SD_R`, `SD_RBI`,
    `Pred_R_per_PA_neutral`, `Pred_RBI_per_PA_neutral`,
    `Pred_target_team_factor`, `Pred_lineup_slot`
  - **Park-adjusted parallel set:** `P_K_park`, `P_HR_park`, ...,
    `AVG_park`, `OBP_park`, `BABIP_park`, plus the raw park-factor columns
    `pf_HR`, `pf_1B`, ... and effective factors `eff_HR`, `eff_1B`, ...
  - **Platoon splits (vL/vR):** `P_K_vL`, `P_K_vR`, `P_HR_vL`, `P_HR_vR`,
    etc. for all 9 events; plus `vL_share` and `vR_share`
  - **Identity:** `BatSide` ('L', 'R', 'S')

### Pitcher columns
  - Same 9 per-PA probabilities and SDs as hitters
  - Same park-adjusted parallel set
  - **Pitcher summary:** `role` ('starter'/'reliever'), `weighted_IP_per_G`,
    `er_ra_ratio`, `RA9`, `ERA`, `RA9_park`, `ERA_park`,
    `R_per_PA`, `R_per_PA_park`, `TBF_per_IP`, `TBF_per_IP_park`,
    `HBP_pct`, `HBP_pct_park`, `Pred_WP_per_PA`, `n_eff_WP`
  - **Platoon splits (vL/vR):** same as hitters (vL=facing LHB,
    vR=facing RHB)
  - **Identity:** `PitchHand` ('L', 'R')

## Pipeline modules — what each does

| File | Role |
|---|---|
| `pipeline_config.py` | All tunable constants (decay, shrinkage strength, history window, etc.) — every magic number documented inline |
| `data_acquisition.py` | All external data fetches: statsapi (rates + splits), team RPG, Statcast park factors, sprint speeds, Chadwick names, player handedness; everything cached as parquet |
| `rate_models.py` | K%/BB%/HBP%/SF% projections via PA-weighted recency-decay shrinkage with an adaptive divergence boost |
| `bip_imputation.py` | 3-layer BIP imputation (own player → handedness pool → league) targeting 150 BIPs for hitters / 400 for pitchers |
| `bip_outcomes.py` | XGBoost classifier on 738K real BIPs predicting 1B/2B/3B/HR/Out from launch speed, launch angle, spray angle (~82% test acc) |
| `pa_aggregation.py` | Combines rate-event projections + BIP-event projections into a 9-event PA distribution summing to 1.0 |
| `sb_model.py` | Chained SB projection: attempts/opp (sprint-speed adjusted, k=50) × success rate (heavy league shrinkage, k=200) |
| `runs_rbi_model.py` | Team-context detrended R/PA and RBI/PA via shrinkage, then re-applies the target team's forecast factor; estimates lineup slot 1-9 |
| `park_factors.py` | Statcast 3-yr rolling park factors with handedness-specific factors, applied to each event probability then renormalized via the BIPOut residual |
| `pitcher_outputs.py` | Linear-weights derivation of RA9 from per-PA events; ERA via role-specific ER/RA ratio (starter 0.934, reliever 0.889); TBF/IP and WP/PA |
| `splits_model.py` | Per-side (vL/vR) projection with overall-anchored constraint — uses same shrinkage machinery on side-specific history, then rescales so PA-weighted average matches the main projection |
| `run_pipeline.py` | Orchestrator — runs all 12 steps in order, prints progress + validation, and writes the final CSVs |

## Important config knobs (in `pipeline_config.py`)

```python
TARGET_YEAR              = 2027     # year to project
RATE_HIST_START          = 2018     # earliest historical year
RATE_DECAY               = 0.85     # PA weight decay per year-back
RATE_MAX_HISTORY_YEARS   = 5
RATE_SHRINK_K_HITTER     = 100      # PA equivalent of prior weight
RATE_SHRINK_K_PITCHER    = 150

BIP_BATTER_TARGET_N      = 150      # BIPs needed before imputation fully trusts player
BIP_PITCHER_TARGET_N     = 400

SB_K_ATTEMPTS            = 50
SB_SPRINT_COEF           = 0.005
SB_K_SUCCESS             = 200

PARK_HOME_SHARE          = 0.5      # 50/50 home/road blend for effective factor
PARK_ROLLING_YEARS       = 3

ER_RA_RATIO_STARTER      = 0.934    # 6.6% unearned for starters
ER_RA_RATIO_RELIEVER     = 0.889    # 11.1% unearned for relievers
STARTER_IP_PER_G_THRESHOLD = 3.5

PITCHER_WP_K_PA          = 600      # heavy WP shrinkage (sparse data)

SPLITS_DECAY             = 0.85
SPLITS_MAX_HISTORY_YEARS = 5
```

All other constants are inline-documented at point of use.

## What the projection actually represents

For the target year, each per-PA event probability represents the player's
**expected event rate against a league-average opposing player at a
neutral park**. Layered context columns add:

  - `*_park` columns: same expectations applied at the player's home park
    (50% home games + 50% neutral away games)
  - `*_vL` / `*_vR` columns: expectations facing a specifically-handed
    opponent (matchup-conditional, used downstream for daily projections)
  - For pitchers, RA9 / ERA / TBF_per_IP are derived via linear weights
    on the same per-PA probabilities, giving an internally-consistent
    full pitcher stat line

## Validation summary (from walk-forward 2024→2025 tests)

  - **K% projection:** Bounded window=5 + k=100, Q5 K% bias -0.011
  - **BB% projection:** Judge BB% projected 17.5% vs actual 18.4%
  - **SB attempts/opp:** YoY r=0.81 (most stable signal)
  - **SB success rate:** YoY r=0.12-0.20 (noisy; heavy shrinkage applied)
  - **R/PA (team-detrended):** R²=0.30
  - **Lineup slot:** 26-28% exact match, 62-64% within ±1
  - **RA9 from linear weights:** r=0.877, MAE 0.47 runs/9 (~ FIP-ERA gap)
  - **ER/RA ratio by role:** Starters 0.934 (6.6% unearned), Relievers
    0.889 (11.1% unearned) — pooled 2022-2025
  - **Splits (vL/vR):** K%_vL wMAE 0.0414 vs baseline 0.0427 (better
    on the smaller-sample side where it matters most)

## Caches written to `cache/`

The first run populates these parquet files; subsequent runs read from
cache unless `--force` is passed:

```
statsapi_hitting_<start>_to_<target-1>.parquet
statsapi_pitching_<start>_to_<target-1>.parquet
statsapi_hitting_splits_<start>_to_<target-1>.parquet
statsapi_pitching_splits_<start>_to_<target-1>.parquet
team_rpg_<start>_to_<target-1>.parquet
park_factors_<target-1>_rolling_<n>.parquet
sprint_speed_<start>_to_<target-1>.parquet
player_handedness.parquet
chadwick_lookup.parquet
```

Refresh policy: rate + split files should be re-fetched daily during the
season (`--force` invalidates them). Park factors update weekly. Sprint
speeds update weekly. Team RPG can update daily.

"""
pipeline_config.py
==================
Central configuration for the PA projection pipeline.

Edit TARGET_YEAR here (or pass via CLI) to project a different season.
Everything else is tuned but reasonable to override if you want to experiment.
"""

from pathlib import Path

# ── Target year and historical scope ─────────────────────────────────────────
TARGET_YEAR = 2027
# Years of FanGraphs-style discipline data to pull from statsapi for K%/BB%.
# Going back further increases training signal but adds older noise.
RATE_HIST_START = 2022

# ── Filesystem layout ────────────────────────────────────────────────────────
# All intermediate artifacts go in CACHE_DIR. Re-runs reuse cached fetches
# unless you delete them. Final outputs land in OUTPUT_DIR.
CACHE_DIR  = Path("./cache")
OUTPUT_DIR = Path("./out")

# Pre-existing user inputs. Both 2024 and 2025 are full-season RDS files;
# CSV equivalents (post-BIP-filter) live here if you ran the conversion step.
# If the .csv exists it is preferred over the .rds; the run_pipeline script
# handles the conversion when only .rds is present.
INPUT_BIP_FILES = {
    2024: Path("./inputs/all_2024.rds"),
    2025: Path("./inputs/all_2025.rds"),
}
# Historical pre-2020 data for training the BIP-outcome XGBoost on a large
# stable sample. Multi-season RDS already containing a Season column.
INPUT_HISTORICAL = Path("./bip_historical.csv")

# ── Rate-model thresholds ────────────────────────────────────────────────────
RATE_MIN_PA_TRAIN  = 50   # PA threshold to enter K%/BB% training panel
RATE_MIN_PA_ACTIVE = 25   # PA threshold to be projected for the target year
RATE_ACTIVE_LOOKBACK = 2  # Active = appeared within this many years before TARGET

# ── Minor-league translations (MLE) for no-MLB-history players ────────────────
# The rate/BIP models only project players with prior MLB data — a debut rookie
# has none, so they are dropped entirely (build_inference_panel's `len(prior)==0`
# gate). To give those players a credible baseline we translate their most-recent
# minor-league line into a Major-League-Equivalent (MLE) "prior season" row and
# feed it through the SAME shrinkage/decay machinery as everyone else.
#
# MLE_ENABLE toggles the whole feature. When on, run_pipeline loads a minors
# feed (RotoWire minors tables), translates each hitter/pitcher line per the
# per-level, per-stat factors below, links the RotoWire id to an MLBAM id via
# the Chadwick lookup, and injects a synthetic prior row for any player NOT
# already present in the statsapi history. Every injected player is flagged
# with mle_source in the output so downstream consumers can treat them as
# low-confidence.
MLE_ENABLE          = True
MLE_LEVELS          = ("AAA", "AA")   # levels to translate (highest signal first)
MLE_SEASON_OFFSET   = 1               # inject as (target_year - offset) season row

# Per-level, per-component multipliers applied to a player's observed minor-
# league RATE (per PA for hitters, per TBF for pitchers) to estimate the MLB
# equivalent. These are published-consensus starting points (Davenport / Szymborski
# style component MLEs, adjusted for the modern AAA offensive environment) and are
# meant to be TUNED once you can validate against players who graduated. Values
# > 1.0 mean the event gets MORE frequent in MLB, < 1.0 LESS frequent.
#
# Hitters (from the batter's perspective):
#   K%   rises going up a level (better pitching) → factor > 1
#   BB%  falls slightly (fewer free passes) → factor < 1
#   HR / 2B / 3B / BABIP all regress down (better defense + pitching) → < 1
#   SB attempt rate roughly holds
MLE_HITTER_FACTORS = {
    "AAA": {"K%": 1.20, "BB%": 0.92, "HR": 0.80, "2B": 0.90, "3B": 0.85,
            "BABIP": 0.95, "SB": 0.90},
    "AA":  {"K%": 1.28, "BB%": 0.88, "HR": 0.70, "2B": 0.85, "3B": 0.80,
            "BABIP": 0.93, "SB": 0.85},
}
# Pitchers (events the pitcher ALLOWS, per TBF):
#   K%   allowed falls in MLB (hitters harder to miss) → factor < 1
#   BB%  allowed rises (better plate discipline against them) → > 1
#   HR   allowed rises → > 1
MLE_PITCHER_FACTORS = {
    "AAA": {"K%": 0.85, "BB%": 1.08, "HR": 1.15, "BABIP": 1.02},
    "AA":  {"K%": 0.78, "BB%": 1.12, "HR": 1.25, "BABIP": 1.03},
}

# Credibility discount: a full minor-league season is NOT worth a full MLB
# season of evidence. We deflate the plate-appearance (hitter) / batters-faced
# (pitcher) count carried by the synthetic row so the shrinkage machinery pulls
# these players harder toward the league mean. AAA carries more weight than AA.
MLE_PA_CREDIBILITY = {"AAA": 0.55, "AA": 0.35}

# Default age for a translated player when the Chadwick lookup has no birth year
# (prospects skew young; this only affects display + the unused ML rate path).
MLE_DEFAULT_AGE     = 24

# Local fallback feed used when the live RotoWire fetch is blocked (as it is in
# the hosted web sandbox). Point MLE at a JSON export with the same shape as the
# live endpoint: {level: {"hitters": [...], "pitchers": [...]}}.
MLE_LOCAL_FEED      = Path("./minors_inputs/minors_<season>.json")

# Bayesian shrinkage prior weights (PA-equivalent) for each rate.
# Note: K% and BB% no longer use these — they use the simpler PA-weighted
# recency-decay projection (RATE_DECAY_*, RATE_SHRINK_K_*) below. SHRINK_K
# is retained because the per-(player, season) panel features still use
# shrunk historical rates for the BIP imputation step.
SHRINK_K = {
    "K%":   60,
    "BB%":  60,
    "HBP%": 80,  # rarer events get a stronger prior
    "SF%":  80,
}

# Pitcher equivalents (TBF-equivalent units)
SHRINK_K_PITCHER = {
    "K%":   60,
    "BB%":  60,
    "HBP%": 80,
    "SF%":  120,
}

# ── K% / BB% rate model parameters ───────────────────────────────────────────
# The pipeline projects K% and BB% via PA-weighted recency-decay shrinkage:
#
#     weighted = Σ (decay^years_back × PA_t × rate_t) / Σ (decay^years_back × PA_t)
#     pred     = (n_eff × weighted + prior_k × league_mean) / (n_eff + prior_k)
#
# Two refinements from the original spec address a residual problem where
# elite walkers (Judge in particular) were still projected too low:
#
#   1. RATE_MAX_HISTORY_YEARS limits the per-player window to N seasons.
#      The original config used all available history (~9 years for older
#      players). A player like Aaron Judge developed into an elite walker
#      around 2022-23 (BB% jumped from ~13% to 18%+), and his 2018-21
#      developmental years were dragging the projection down. Capping at
#      5 years keeps his recent skill-level visible.
#
#   2. RATE_SHRINK_K is lowered substantially (200 → 100 for hitters,
#      200 → 150 for pitchers). With a smaller window, players have less
#      effective sample size, but we also have more confidence that the
#      remaining data reflects current skill — so we shrink less to mean.
#
# Walk-forward validation 2024-2026 (combined):
#   Old (window=∞, k=200): hitter avg wMAE ≈ 0.0207, Q5 BB% bias = -0.028
#   New (window=5,  k=100): hitter avg wMAE ≈ 0.0208, Q5 BB% bias = -0.027
#
# MAE is essentially unchanged; the change is meaningful for individual
# elite-walker projections:
#   Judge BB%:      0.161 → 0.175  (real 0.184)  improvement: +1.4 pp
#   Schwarber BB%: 0.143 → 0.149  (real 0.151)  improvement: +0.6 pp
#   Soto BB%:      0.176 → 0.176  (real 0.175)  unchanged
RATE_DECAY               = 0.85   # Per-year decay on player history (0.85 → 15%/year)
RATE_MAX_HISTORY_YEARS   = 5      # Per-player history window (None = all)
RATE_SHRINK_K_HITTER     = 100    # PA-equivalent shrinkage prior strength (hitters)
RATE_SHRINK_K_PITCHER    = 150    # TBF-equivalent shrinkage prior strength

# ── BIP imputation parameters ───────────────────────────────────────────────
# TARGET_N — the per-player-season row count after imputation. Real BIPs are
# kept as-is and synthetic rows are added to reach this target. Higher values
# stabilize small-sample players but compress real player-to-player skill
# differences (the "regression to mean" effect).
#
# Empirical tuning on 2025 BIP data (see experiment_target_n.py) compared
# TARGET_N ∈ {50, 100, 150, 200, 250, 300, 400, 500} for hitters via 50/50
# split-half ground-truth tests. Results:
#
#   TARGET_N | HR spread retained | HR MAE  | HR bias
#   ---------|--------------------|---------|---------
#       100  |       94.0%        | 0.00696 | +0.0004
#   *   150  |       93.3%        | 0.00708 | +0.0005
#       200  |       84.3%        | 0.00707 | +0.0002
#       300  |       69.4%        | 0.00783 | -0.0004
#       400  |       64.8%        | 0.00829 | -0.0006   ← was over-regressing
#       500  |       59.4%        | 0.00811 | -0.0009
#
# TARGET_N=150 retains 93%+ of real cross-hitter spread on every event with
# near-zero bias and competitive MAE. Going below 150 yields marginally
# better spread retention but adds variance noise for the smallest-sample
# players. 150 is the sweet spot.
#
# Pitchers stay at 400 per user preference — their imputation behavior is
# different (more pitchers face fewer batters, so they need more padding;
# they also have less true year-to-year stability in batted-ball results).
IMP_TARGET_N_BATTER     = 150    # Empirically tuned — preserves elite skill
IMP_TARGET_N_PITCHER    = 400    # Match R script default
IMP_MIN_OBS_BATTER      = 25
IMP_MIN_OBS_PITCHER     = 200
IMP_HIST_DECAY          = 0.6    # Population prior decay/season
IMP_HIST_MAX_LOOKBACK   = 2
IMP_PLAYER_HIST_DECAY   = 0.1    # Player history decay/season — fast fade
IMP_PLAYER_MAX_LOOKBACK = 2

# Backwards-compatibility alias (legacy single-value entry point — should
# not be referenced in new code; use the role-specific values above).
IMP_TARGET_N            = IMP_TARGET_N_BATTER

# Continuous variables modeled as multivariate normal per player-season.
BIP_CONT_VARS = ["launch_speed", "launch_angle", "adjusted_angle"]

# Sane physical bounds for the multivariate normal samples.
BIP_BOUNDS = {
    "launch_speed":   (40,  120),
    "launch_angle":   (-90, 90),
    "adjusted_angle": (-90, 90),
}

# ── XGBoost (BIP-outcome) parameters ─────────────────────────────────────────
# These mirror the R script (xgb_grid with caret) but use direct xgboost
# to skip caret's CV overhead. The grid is small and fixed for reproducibility.
XGB_SAMPLE_SIZE = 100_000
XGB_PARAMS = dict(
    objective="multi:softprob",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

# Outcome classes the XGBoost predicts (single / double / triple / home_run / out)
BIP_OUTCOMES = ["out", "single", "double", "triple", "home_run"]

# ── Season-weighting for the per-player aggregation step ─────────────────────
# These are the LEGACY fixed weights — kept for backwards compatibility but
# overridden by adaptive weights when launch-profile meta is available.
# Adjust if you trust 2026 more or less. Weights are normalized per player to
# handle missing seasons.
SEASON_WEIGHTS_FROM_TARGET = {
    -1: 0.65,   # Most recent (target - 1)
    -2: 0.30,   # 2 years before target
    -3: 0.05,   # 3 years before target
}

# ── Adaptive season-weighting parameters ─────────────────────────────────────
# These drive the smart per-player weight assignment that replaces the fixed
# SEASON_WEIGHTS_FROM_TARGET when launch-profile data is available.
#
# K_RELIABILITY: Bayesian shrinkage constant — when n_bip = K, the season
#   gets half its theoretical max weight. Empirically tuned to ~280 via
#   experiment_adaptive_weights.py (predicts 2025-second-half from
#   2025-first-half + 2024 data).
# RECENCY_DECAY: per-years-back multiplier on top of reliability. Encodes
#   the prior preference for recent data; combines multiplicatively with the
#   reliability-based weight.
# DIVERGENCE_THRESHOLD: Hotelling T² above which we start boosting the most
#   recent season's weight. T²≈5 corresponds to a "moderately different"
#   launch profile (loose 95% CI threshold for ~100-BIP samples).
# DIVERGENCE_MAX_BOOST: cap on the proportional boost — even with extreme
#   divergence, we don't let any season dominate.
ADAPTIVE_K_RELIABILITY     = 280.0
ADAPTIVE_RECENCY_DECAY     = {1: 1.00, 2: 0.70, 3: 0.30}
ADAPTIVE_DIV_THRESHOLD     = 5.0
ADAPTIVE_DIV_MAX_BOOST     = 0.20

# Regression to the mean on per-BIP OUT rate (BABIP).
#
# We use a SAMPLE-SIZE-ADAPTIVE pull, applied once per player after combining
# their multi-season BIP-out estimate. The pull strength is:
#
#     pull(N) = K / (N + K)
#
# where N is the player's total real BIP count across the seasons used (up to
# 3 prior years) and K is the role-specific Bayesian shrinkage prior strength.
#
# Star hitters with rich samples (Judge 800+, Soto 900+) get pull ≈ 0.20,
# letting them keep most of their observed BABIP profile. Players with thin
# samples get pull → 1.0, regressing them hard toward league average.
#
# K is empirically calibrated to PRESERVE THE TRUE-SKILL SPREAD of observed
# BABIP (i.e., make projection σ ≈ sqrt(r) × observed σ, where r is the
# year-to-year correlation):
#
#   Hitter:  K = 200   → avg pull ≈ 0.29  for typical 500-BIP cohort
#   Pitcher: K = 400   → avg pull ≈ 0.50  for typical 400-BIP cohort
#
# Pitchers get higher K (more regression) because their year-to-year BABIP
# stability is much lower (r=0.18 vs hitter r=0.39) — they genuinely control
# BABIP less, so their observed samples carry less signal regardless of N.
#
# These values are validated against real-data BABIP distributions and
# preserve player differentiation more faithfully than a fixed pull.
OUT_ADAPTIVE_K_HITTER  = 200   # Bayesian prior strength for hitter BIP-out
OUT_ADAPTIVE_K_PITCHER = 400   # Bayesian prior strength for pitcher BIP-out

# Legacy fixed-pull values — no longer used in the main pipeline path but
# retained for backwards compatibility. Set via _apply_out_pull_fixed.
OUT_POP_PULL_HITTER  = 0.38
OUT_POP_PULL_PITCHER = 0.58

# The mean/median blend weights for each event (R script's exact recipe)
EVENT_BLEND_WEIGHTS_HITTER = {
    "out":      (1.0, 0.0),   # only mean (population pull does the work)
    "single":   (1.0, 0.0),
    "double":   (0.75, 0.25),
    "triple":   (0.75, 0.25),
    "home_run": (0.75, 0.25),
}
EVENT_BLEND_WEIGHTS_PITCHER = EVENT_BLEND_WEIGHTS_HITTER

# ── Data acquisition rate limiting ───────────────────────────────────────────
STATSAPI_TIMEOUT = 60
SAVANT_TIMEOUT   = 120
SAVANT_DAYS_PER_CHUNK = 5  # mirrors the R script's chunk size

# League means used as fallbacks when a metric has no prior-season pool.
DEFAULT_LEAGUE_RATES = {
    "K%":   0.225,
    "BB%":  0.085,
    "HBP%": 0.012,
    "SF%":  0.010,
}

# ── Stolen base projection parameters ────────────────────────────────────────
# SB and SB-attempts per PA are projected as a chained decomposition:
#
#     P_SB_ATTEMPT = (P_1B + P_BB + P_HBP) × Pred_attempts_per_opp
#     P_SB         = P_SB_ATTEMPT × Pred_success_rate
#     P_CS         = P_SB_ATTEMPT × (1 - Pred_success_rate)
#
# Stage 1 — steal opportunity per PA — flows from the pipeline's existing rate
# and BIP models. Nothing new is needed for stage 1.
#
# Stage 2a — attempts/opp — projected via PA-weighted recency-decay shrinkage,
# plus an additive sprint-speed adjustment. We project attempts (not SB)
# because attempts is the most-stable signal — it represents the player's
# decision to try to steal, which is what's player-controlled.
#
# Stage 2b — success rate — shrunk heavily to league average. Per-player
# success rate has low YoY stability (r ≈ 0.12-0.20); most variation is
# situational, not player skill.
#
# Empirical findings driving the parameter choices:
#   - Attempts/opp YoY r = 0.81  ← most stable, strongest projection signal
#   - SB/opp YoY r       = 0.76
#   - Success rate YoY r = 0.12-0.20  ← mostly noise
#   - Sprint speed YoY r = 0.92, correlates +0.56 with SB/PA across players
#   - OBP / 1B / BB% correlate only ≈ +0.05 with SB/PA — attempt rate, not
#     opportunity rate, drives the difference between high/low SB players.
#     Stage 1 handles opportunity; we don't need OBP as a stage-2 feature.
#   - Adding age as a feature: no marginal value once sprint is included.
#
# Tuning (sweep on 2024 and 2025 walk-forward — see experiment_sb_model.py):
#   SB_K_ATTEMPTS=50, SB_SPRINT_COEF=0.005 minimize weighted MAE on both
#   SB/PA and attempts/PA, with appropriate elite-runner calibration.
#   SB_K_SUCCESS=200 keeps almost everyone within a few points of league.
SB_K_ATTEMPTS      = 50.0    # Bayesian prior on attempts/opp (opportunity units)
SB_SPRINT_COEF     = 0.005   # +0.005 per (ft/s - SB_SPRINT_BASELINE) on attempts/opp
SB_SPRINT_BASELINE = 27.0    # Average sprint speed in ft/s (population median)
SB_K_SUCCESS       = 200.0   # Bayesian prior on success rate (attempts units)
SB_MAX_HISTORY_YEARS = 5     # Window of prior seasons for SB history
SB_DECAY           = 0.85    # Per-year decay multiplier (matches rate models)

# Legacy: older code may import SB_K_PRIOR; keep as alias.
SB_K_PRIOR = SB_K_ATTEMPTS

# ── Runs and RBI projection parameters ───────────────────────────────────────
# R/PA and RBI/PA are projected via PA-weighted recency-decay shrinkage on the
# team-neutralized rate (i.e., each season's rate is divided by the team's
# RPG factor before aggregation). After projection, the player's projected
# target-team factor is re-applied to get the final raw rate.
#
# This explicit team detrending lifts walk-forward R² on 2025 from 0.20 to
# 0.30 for R/PA, and 0.20 to 0.27 for RBI/PA. Without it, the projection
# conflates team context (which is fluid year-to-year as players change teams)
# with player skill.
#
# k_pa is higher than the rate models' shrinkage because R/PA YoY r is only
# ~0.45 — much less stable than HR/PA (0.65) or BB/PA (0.74) — so we lean
# more heavily on the league prior.
#
# Lineup slot is estimated from the player's neutralized projection vs
# published slot-base-rate table (Tango's "The Book", Baseball Prospectus
# research). It's output as context only — does NOT feed back into the
# projection itself. Independent prior research showed lineup slot explains
# 51% of R/PA and 44% of RBI/PA variance, but a player's own R/PA history
# already implicitly encodes their typical slot.
RUNS_RBI_K_PA              = 200.0   # Shrinkage prior strength (PA units)
RUNS_RBI_DECAY             = 0.85    # Per-year decay (matches rate models)
RUNS_RBI_MAX_HISTORY_YEARS = 5       # History window (matches rate models)

# ── Park factor parameters ───────────────────────────────────────────────────
# Statcast park factors are applied to per-PA event probabilities to produce
# a parallel "park-adjusted" set of projections alongside the neutral set.
# Each player's home park factors are blended 50/50 with road-park-neutral to
# account for the 81-81 schedule:
#
#     effective_factor = home_share × park_factor + (1 − home_share) × 1.0
#
# Default home_share = 0.5 (matches the MLB schedule). The Statcast factors
# are indexed so 1.0 = league average — a 1.27 HR factor means 27% more HR
# per PA at that park than at other parks.
#
# Park factors use 3-year rolling averages by default for stability. For
# parks too new to have 3 years (e.g., Sutter Health Park for the A's in
# 2025), the fetcher falls back to 1-year rolling.
#
# Renormalization: park-adjusted projections always sum to 1.0 across the
# 9 event categories. P_BIPOut is the residual that absorbs any leftover
# probability after the factored events are multiplied. A min_bipout floor
# prevents extreme parks from producing degenerate projections.
PARK_HOME_SHARE        = 0.5      # 81-81 home/away split
PARK_ROLLING_YEARS     = 3        # 3-year rolling park factors (stable)
PARK_MIN_BIPOUT        = 0.05     # Safety floor on P_BIPOut after adjustment

# Should park factors apply BEFORE the SB and R/RBI models?
#   - SB: YES. Park affects opportunity rate (P_1B + P_BB + P_HBP), which is
#     the foundation of SB calculation. The player's attempt rate per
#     opportunity and success rate stay constant — they're behavioral, not
#     park-driven. We produce both neutral SB and park-adjusted SB columns.
#   - R/RBI: NO. The team_RPG factor used by the R/RBI model already encodes
#     park effects implicitly (a hitter-friendly park naturally lifts team
#     RPG). Applying park factors on top would double-count, since the same
#     batters and pitchers contributing to the RPG factor also feed the
#     Statcast park factor. R/RBI use only the team-RPG approach.
PARK_APPLY_BEFORE_SB     = True
PARK_APPLY_BEFORE_RUNS   = False  # documentation only — R/RBI uses team_RPG

# ── Pitcher-level summary outputs ────────────────────────────────────────────
# Three additional pitcher columns derived from per-PA projections:
#
#   RA9          — runs allowed per 9 IP, via linear weights × per-PA probs
#   TBF_per_IP   — batters faced per inning, derived from out rates
#   Pred_WP_per_PA — wild pitches per PA, projected via shrinkage
#
# The RA9 calculation uses standard linear-weights for runs (BB +0.30, HBP
# +0.34, 1B +0.45, 2B +0.77, 3B +1.06, HR +1.38, K -0.03) plus an empirical
# intercept (-0.047) calibrated against 2022-2025 actual data so the mean
# predicted RA9 matches actual league RA9 (4.32 in our sample). Walk-forward
# validation shows r=0.877 between predicted and actual RA9, MAE 0.47.
#
# The TBF_per_IP formula 3/(P_K + P_BIPOut + P_SF) over-predicts by ~3%
# (GIDP, CS not in PA), so a 0.971 calibration factor is applied.
#
# WP_per_PA YoY r ≈ 0.40 — moderate stability — so we shrink heavily to the
# league rate (~0.0075). k_pa=600 means most pitchers stay near league
# unless they have a long, distinctive history.
PITCHER_WP_K_PA       = 600.0   # Heavy shrinkage on WP/PA (sparse signal)
PITCHER_WP_DECAY      = 0.85
PITCHER_WP_MAX_HISTORY_YEARS = 5

# ── Platoon splits projection ────────────────────────────────────────────────
# Per-side projections (vL/vR) allow daily/matchup-specific use of the model.
# For hitters: vL = facing LHP, vR = facing RHP.
# For pitchers: vL = facing LHB, vR = facing RHB.
#
# We project each rate metric independently for each side using the same
# PA-weighted recency-decay shrinkage machinery as the main rate models, then
# rescale the (vL, vR) pair so their PA-share-weighted average matches the
# main pipeline's overall projection. This anchoring keeps the per-side
# projections internally consistent with the season-level projection.
#
# Sample sizes per side are sparser than full-season (typical hitter has ~145
# PA vL / ~410 PA vR), so we use heavier shrinkage priors on per-side metrics
# than on overall metrics. The k_pa values are encoded in splits_model's
# SPLIT_RATE_METRICS dict (K=100, BB=150, HBP=300, HR=200, ...).
#
# YoY stability of per-side rates (validated 2024→2025):
#   K%_vL: r=0.63    K%_vR: r=0.78
#   Stable enough to be useful, though noisier than full-season.
#
# The pipeline emits both vL and vR columns; downstream daily-projection
# consumers pick by handedness of the matchup.
SPLITS_DECAY            = 0.85
SPLITS_MAX_HISTORY_YEARS = 5

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
| `data_acquisition.py` | All external data fetches: statsapi (rates + splits), team RPG, Statcast park factors, sprint speeds, Chadwick names, player handedness, minor-league stat tables (RotoWire); everything cached as parquet |
| `mle_translations.py` | Minor-league translations (MLE): turns a minor-league line into a synthetic MLB-equivalent "prior season" row + per-BIP power profile, so debut rookies with no MLB history still get a baseline |
| `rate_models.py` | K%/BB%/HBP%/SF% projections via PA-weighted recency-decay shrinkage with an adaptive divergence boost |
| `bip_imputation.py` | 3-layer BIP imputation (own player → handedness pool → league) targeting 150 BIPs for hitters / 400 for pitchers |
| `bip_outcomes.py` | XGBoost classifier on 738K real BIPs predicting 1B/2B/3B/HR/Out from launch speed, launch angle, spray angle (~82% test acc) |
| `pa_aggregation.py` | Combines rate-event projections + BIP-event projections into a 9-event PA distribution summing to 1.0 |
| `sb_model.py` | Chained SB projection: attempts/opp (sprint-speed adjusted, k=50) × success rate (heavy league shrinkage, k=200) |
| `runs_rbi_model.py` | Team-context detrended R/PA and RBI/PA via shrinkage, then re-applies the target team's forecast factor; estimates lineup slot 1-9 |
| `park_factors.py` | Statcast 3-yr rolling park factors with handedness-specific factors, applied to each event probability then renormalized via the BIPOut residual |
| `pitcher_outputs.py` | Linear-weights derivation of RA9 from per-PA events; ERA via role-specific ER/RA ratio (starter 0.934, reliever 0.889); TBF/IP and WP/PA |
| `splits_model.py` | Per-side (vL/vR) projection with overall-anchored constraint — uses same shrinkage machinery on side-specific history, then rescales so PA-weighted average matches the main projection |
| `team_context.py` | Canonical team identity (MLBAM id ↔ abbreviation), one shared hitter/pitcher team-assignment rule, roster overrides for players changing teams, and the bottom-up team run environment |
| `season_engine.py` | Season-long projection layer on top of the per-PA CSVs — team alignment, roster-derived team context, R/RBI rescaling, closure diagnostics |
| `run_pipeline.py` | Orchestrator — runs all 12 steps in order, prints progress + validation, and writes the final CSVs |

## Important config knobs (in `pipeline_config.py`)

```python
TARGET_YEAR              = 2027     # year to project
RATE_HIST_START          = 2022     # earliest historical year
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

# Minor-league translations (MLE) for no-MLB-history players
MLE_ENABLE               = True
MLE_LEVELS               = ("AAA", "AA")
MLE_SEASON_OFFSET        = 1        # inject as a (target_year - 1) season row
MLE_HITTER_FACTORS       = {...}    # per-level K%/BB%/HR/2B/3B/BABIP/SB factors
MLE_PITCHER_FACTORS      = {...}    # per-level K%/BB%/HR/BABIP factors (allowed)
MLE_PA_CREDIBILITY       = {"AAA": 0.55, "AA": 0.35}   # sample-size discount
MLE_LOCAL_FEED           = Path("./minors_inputs/minors_<season>.json")
```

All other constants are inline-documented at point of use.

## Season-long projections — the team layer (`season_engine.py`)

The per-PA CSVs are a **skill layer**: rates against a league-average opponent.
Season-long projections need players aligned to teams, and team-level context
(run environment, and eventually wins) built from those rosters.
`season_engine.py` adds that as a layer on top, leaving the per-PA CSVs — and
therefore the daily DFS path — untouched.

```bash
python season_engine.py --target-year 2027
# writes out/season_2027/{hitters,pitchers,team_context}.csv
```

### Team identity

`team_context.TEAM_ABBR_BY_ID` maps all 30 MLBAM team ids to the canonical
abbreviations `slate_config.canonical_team` produces, so projection rows, slate
feeds, and Vegas totals share one vocabulary. **Always join on the numeric
`Pred_target_team_id`** — it survives relocations and rebrands. The `Team`
string is display-only.

### Team assignment, and players who change teams

One rule serves hitters (`volume_col="PA"`) and pitchers (`volume_col="TBF"`):
the team a player accumulated the most volume for in his most recent
qualifying season, with ties broken on the lower team id so the result is
order-independent. The output records `team_assign_source` —
`override` / `history` / `unknown`.

History cannot express an offseason move, so put signings and trades in
`rosters/team_assignments_<year>.json` (copy
`rosters/team_assignments.example.json`):

```json
{"target_year": 2027, "assignments": [
  {"player_id": 592450, "team": "SF",  "note": "signed 2026-12-01"},
  {"player_id": 660271, "team_id": 147, "note": "traded"},
  {"player_id": 111111, "team": null,  "note": "unsigned"}
]}
```

`team` takes any code or full name `canonical_team` understands. A `null` team
means *no team* — the player is carried with neutral context rather than
silently keeping his old club. Overrides are read by both `run_pipeline.py`
(so R/RBI are projected against the right club) and `season_engine.py`.

### Bottom-up team run environment

`runs_rbi_model` scales a hitter's R/RBI by his team's run environment,
forecast from that team's own past runs-per-game. That factor correlates only
**r = 0.67** with the talent of the roster it is applied to, is *more*
dispersed (SD 0.073) than that talent (SD 0.048), and cannot respond to a
roster change at all.

`team_context.bottom_up_team_factors` rebuilds it from the players actually
projected onto each roster, via the same linear weights `pitcher_outputs` uses
for RA9 — so offense and defense stay on one scale. It is then blended with
the historical prior (`TEAM_CONTEXT_BOTTOM_UP_WEIGHT`, default 0.60) and
**normalized so the volume-weighted mean factor is exactly 1.0**.

That normalization is a closure requirement, not a tuning knob: team factors
scale every hitter's R/RBI, so if they don't average to 1, moving players
between teams silently creates or destroys league runs. Because context is
derived from the roster, a team change updates **both** clubs — the player
leaves one aggregate and joins another — and every teammate on both sides is
rescaled off the team-context-free `Pred_R_per_PA_neutral`.

Tune the blend with `--bottom-up-weight`; `0.0` reproduces the historical
prior (but closed), `1.0` ignores it.

## Organizational depth and playing-time tiers (`playing_time.py`)

The projection set covers a whole organization — MLB regulars, September
callups, 4A players, the long-term injured, and MLE-translated minor leaguers
down to Single-A. That makes one distinction load-bearing:

> **Being in the output is not a claim of playing time.**

### Who used to be missing

Three populations fell through *both* gates. `build_inference_panel` dropped
them (under 25 PA, or no season within 2 years), and MLE skipped them because
it only ADDS players with no MLB history — these players have some:

| Population | Why it was dropped |
|---|---|
| September callups | 12 MLB PA is below a 25-PA bar |
| 4A players | last MLB action 3+ years ago, in AAA since |
| Long-term injured | two lost seasons pushes them outside the lookback |

`RATE_MIN_PA_ACTIVE` is now 1 and `RATE_ACTIVE_LOOKBACK` is 4: everyone we
have any evidence for gets a row, and thin evidence is correctly regressed
almost all the way to the league mean by the existing shrinkage (a 12-PA
sample carries almost nothing against `RATE_SHRINK_K_HITTER = 100`).
`MLE_LEVELS` now spans `AAA, AA, A+, A`, with credibility falling steeply
(0.55 → 0.35 → 0.20 → 0.12) so a Single-A line cannot masquerade as a forecast.

### The two columns

| Column | Meaning |
|---|---|
| `pt_tier` | `projected` (expected to accumulate real MLB playing time) or `floor` (carried for organizational completeness) |
| `Proj_PA` / `Proj_IP` | `floor` tier gets exactly **1.0**. `projected` tier gets **NaN** with `pt_source = "unmodeled"` until a playing-time model exists. |

**Why the floor is 1 and not 0.** Zero makes every rate × volume product zero,
so the player silently vanishes from totals while still occupying a row — the
worst of both worlds. NaN propagates through sums. One keeps him present,
ranked and joinable, contributes a rounding error, and reads unambiguously as
a replacement-level placeholder.

**Why `projected` gets NaN rather than a guess.** Filling in a plausible 600 PA
would make every downstream total quietly wrong in a way that is very hard to
notice. A NaN fails loudly at the point of use. A `floor` player's 1.0 is not a
placeholder for a missing number — his volume is genuinely known to be
approximately none.

### Containment

Depth players receive a team factor (so their own R/RBI are contextualized) but
must not **consume** team playing time — a club bats about 6,150 times, so
counting 200 farmhands at the floor would shift its share of the league.
`team_context.roster_volume_weights` zeroes floor-tier rows for exactly this
reason, and it is the basis used to normalize team factors. Verified: adding
400 depth hitters to the real artifacts leaves league R/PA and every team
factor unchanged.

**Daily path.** `matchup.resolve_collisions` treats any name with 2+ projection
rows as needing disambiguation, so a Single-A namesake would have marked a real
MLB player ambiguous and **dropped him from the slate** — the same failure mode
as the truncated-team-code bug. `matchup._mlb_rows` therefore excludes
floor-tier rows from ambiguity counting and collision resolution, while leaving
them available for a direct match so a just-promoted player still gets his own
baseline. Files without `pt_tier` default to `projected`, making the filter a
no-op on legacy output.

## Designing the playing-time model

Not built. This is the intended design, and it is the keystone: every remaining
closure identity and the wins model depend on it.

### The framing that matters

PA and IP are **not player-level quantities**. They are allocations of a fixed
team budget under competition:

```
team hitter PA ≈ 162 × 38   ≈ 6,150
team IP        ≈ 162 × 9    ≈ 1,458
```

A team cannot bat 7,000 times. Regressing each player's PA on his own history
and summing is the common approach and it is wrong: the totals don't close, so
you normalize at the end, and then every player's number moves for reasons that
have nothing to do with him. Model it as **constrained allocation** instead and
the constraint is exact by construction — which is precisely what
`playing_time.PlayingTimeModel` requires.

### Three stages

**Stage 1 — Availability** (per player, unconstrained). What share of the season
is he available?

```
availability = P(on an MLB roster) × (162 − E[games missed]) / 162
```

Inputs: age, IL history, current injury status and expected return, offseason
surgery, service time. This is a survival/hazard problem, not a point estimate —
carry a distribution, because the uncertainty is the useful part.

**Stage 2 — Role** (per player, competitive). Given availability, what job does
he win? A multinomial over {lineup slot 1-9, bench, MiLB} per position group for
hitters; {rotation slot 1-5, swing, bullpen role, MiLB} for pitchers.

The key move: drive role from **projected talent rank within the org at that
position** — which closes the loop with the rate engine already in place. The
org's best projected shortstop gets the shortstop job.

**Stage 3 — Allocation** (team-level, constrained). Each player carries a claim
of `availability × role weight`; normalize claims within each team to the budget
above. Team closure holds by construction, so the league total is 30 × budget,
so runs-scored equals runs-allowed, so **wins become computable**.

### The hard cases

**Injured players.** Two factors, not one:
`Proj_PA = full_time_PA × availability_share`. A player out until June gets
~0.55. Availability comes from an injury-type → recovery-timeline table (TJ
~12-18 months, UCL brace ~8-10, hamstring grade 2 ~3-5 weeks) plus a re-injury
hazard. A February TJ means availability ≈ 0 → **floor tier, 1 PA** — the rule
falls out of the model rather than being bolted on.

Important: do **not** discount the *rate* projection for injury. A hurt player's
per-PA skill is not worse. Post-surgery velocity loss is real but belongs in the
rate model as its own adjustment, not smuggled into playing time.

**Players starting in the minors.**
`Proj_PA = P(promoted) × E[PA | promoted]`. Promotion probability is driven by
projected talent versus the incumbent at his position, option status and 40-man
standing, age and pedigree, and the org's competitive position. A top prospect
blocked by an All-Star should get a real but modest ~150 PA — not 1, and not
550. A Single-A player's role weight rounds to zero → floor tier. Collapse any
allocation below ~5 PA to the floor rather than carrying a falsely precise 2.7.

### The subtle one: PT and rates are not independent

A player only accumulates 600 PA *if he performs*. So conditional on 600 PA, his
rates are better than his unconditional projection; projecting the marginal rate
and the marginal PA and multiplying over-weights bad outcomes. Talent-conditional
allocation in Stage 2 partially handles this. Handling it properly means
simulating: draw a playing-time scenario, then draw rates conditioned on it.
This is also the honest way to express that a 26-year-old with a job battle is a
bimodal outcome, not a 300-PA expectation.

### Data we need and don't have

| Need | Status |
|---|---|
| `primaryPosition` | **Easy** — a statsapi field the fetch simply doesn't request |
| Positions for prospects | **Already there** — the minors feed carries `position` |
| Depth charts | RotoWire has them; the minors feed is already a RotoWire endpoint |
| IL transactions / injury status | statsapi has a transactions endpoint |
| 40-man + option status | Harder; may need a maintained file like `rosters/team_assignments_<year>.json` |
| Contract / service time | Mostly manual |

### Incremental path

Each stage is independently useful, in reverse order of difficulty:

1. **Stage 3 alone** — allocate each team's fixed budget by projected talent
   rank. Implementable today with the rate projections and team assignment that
   already exist, no new data. Immediately closes the team and league identities
   and unlocks wins.
2. **Add positions** → Stage 2 becomes real, and depth charts and batting orders
   become possible.
3. **Add injury/IL data** → Stage 1 becomes real, and the injured and
   minor-league cases stop depending on the evidence proxy.

### Validation

Walk-forward: project 2025 playing time from data through 2024 only.

- MAE on PA among players who actually played
- **Team-total closure error** — should be ~0 by construction; if not, the
  allocation is broken
- Calibration of promotion probability — of players given P ≈ 0.3, did ~30% get
  promoted?
- AUC on the binary "did he play at all" — this is where most systems fail, and
  it is the metric the floor tier exists to serve

### Where the tier comes from until then

`playing_time.classify_tier` uses evidence: did the player clear 25 PA within 2
years? That is a poor proxy for exactly the cases that matter most — a top
prospect about to break camp as the starting shortstop is `floor`, and a
just-retired veteran is `projected`. It is used because there is nothing better
yet, and because it errs toward `floor`, and a floor player cannot distort
anything. When the real model lands it owns the tier.

### Also still missing

An aging curve, league-level anchoring, and R/RBI as an allocation of team runs.
See `deliverables/projection_engine/CURRENT_STATE_ASSESSMENT.md`.

## Players with no MLB history — minor-league translations (MLE)

The rate and BIP models only project players who have prior MLB data — a debut
rookie has none, so without help he is dropped entirely (`build_inference_panel`'s
`len(prior) == 0` gate, plus the ≥25-MLB-PA active filter). Step **1b** closes
that gap using Major-League Equivalencies:

1. **Fetch** the minor-league stat tables (RotoWire minors tables, AAA + AA by
   default) via `data_acquisition.fetch_minors`. Direct fetch → maintainer proxy
   → local-file fallback (`minors_inputs/minors_<season>.json`) so it still works
   where the live host is blocked. See `minors_inputs/sample_minors_2026.json`
   for the feed shape.
2. **Translate** each hitter/pitcher line to an MLB-equivalent per-PA (per-TBF)
   rate profile with published-consensus, per-level, per-stat factors
   (`MLE_HITTER_FACTORS` / `MLE_PITCHER_FACTORS`): K% rises going up a level,
   BB%/HR/BABIP regress, etc. A credibility discount (`MLE_PA_CREDIBILITY`)
   deflates the PA/TBF the synthetic row carries so shrinkage regresses these
   players appropriately hard.
3. **Link** the RotoWire id to an MLBAM id via the Chadwick name lookup
   (ambiguous / unresolved names are skipped, and any player already in the MLB
   history is skipped — MLE only ever *adds* no-history players).
4. **Inject** the result as a synthetic `target_year − 1` "prior season" row in
   the exact statsapi schema, so it flows through the same shrinkage / recency
   decay / SB machinery as everyone else. Model calibration and league means run
   on the real-only frame (`fit_df`), so synthetic rows seed a rookie's own
   projection without contaminating anyone else's.
5. **Power override** — because a rookie has no MLB batted balls, his
   BIP-derived HR/2B/3B/1B/out would otherwise fall back to league average.
   `apply_mle_bip_override` replaces that batted-ball mass with his
   minor-league-translated per-BIP profile, so a slugging prospect keeps his
   power and a slap hitter keeps his lack of it. The 9 events still sum to 1.0.

Every translated player is flagged with an **`mle_source`** column (`"MiLB"`) in
the output so downstream consumers can treat them as low-confidence baselines.
Toggle the whole feature with `MLE_ENABLE`. Known v1 limits: R/RBI fall back to
league average for these players (no clean MiLB translation), platoon vL/vR
splits are absent (no minor-league split feed), and only AAA/AA are translated.

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

## Daily matchup: opponent-quality adjustment for pitchers (`matchup.py`)

The per-PA projection above is "vs a league-average opponent." When a slate is
built, `matchup.py` conditions each player on that day's actual opponent. Hitters
were already conditioned on the opposing pitcher (their `vL/vR` split for his
hand), but pitchers were only conditioned on the *handedness mix* of the lineup —
**not its quality**. That meant an ace facing the best offense on the slate and
the same ace facing a replacement-level lineup projected identically, so pitchers
opposite elite lineups were systematically over-projected.

`_opponent_adjust_pitcher` fixes this with the **log5 / odds-ratio matchup**: for
each event it combines the pitcher's own rate with each opposing hitter's rate on
the log-odds scale and averages over the lineup —

```
logit(rate_allowed) = logit(pitcher_rate) + γ · ( logit(batter_rate) − logit(league) )
```

The batter-side elasticity `γ` is **calibrated out-of-sample on Statcast
batted-ball logs** (`bip_inputs/`): estimate each batter's and pitcher's contact
rate on 2024, then fit how strongly the batter side moves the actual 2025
outcome. Results:

  - **HR:** γ ≈ 1.0 (full log5 — power is a persistent, real skill)
  - **balls-in-play hits:** γ ≈ 0.7 (below full log5 — the DIPS signature, since
    pitchers have limited control over BABIP; the batter drives contact)

K and BB are not present in balls-in-play logs, so they use full log5 (γ = 1.0),
the standard theoretical value (K is strongly batter-driven — a low-strikeout
contact lineup meaningfully suppresses a pitcher's Ks). `RA9`/`ERA`/`TBF_per_IP`
stay as neutral skill anchors; the extra runs against a tough lineup flow through
the now-elevated hit/HR/BB traffic in the simulator. Elasticities live in
`matchup.OPP_MATCHUP_ELASTICITY`.

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
minors_<target-1>.json          # cached RotoWire minors feed (MLE step)
```

Refresh policy: rate + split files should be re-fetched daily during the
season (`--force` invalidates them). Park factors update weekly. Sprint
speeds update weekly. Team RPG can update daily.

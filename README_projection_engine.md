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

### Or run it as a one-off GitHub Action

**Actions → `refresh-projections` → Run workflow.** Manual only, no schedule —
rebuilding the baselines is a deliberate, occasional act. Useful when you can't
reach statsapi locally, or when you want the rebuild verified and recorded.

It does the whole refresh in order: rebuild the baselines, **verify** them,
run the full audit, build the season layer, and regenerate the role workbooks.

| Input | Default | Notes |
|---|---|---|
| `target_year` | `2027` | season to project |
| `scrape_bip` | off | re-scrape Statcast BIP; off reuses committed `bip_inputs/` and needs no baseballsavant access |
| `force` | off | bypass caches, re-fetch every source |
| `commit` | `branch` | push results to `refresh/projections-<year>-<run>` for review, or `none` to leave them on the run artifact |
| `publish` | off | also push to the object store, which feeds the **live** app |

The verification step is the reason this exists rather than just running the
pipeline. `scripts/verify_refresh.py` compares the rebuilt output against real
batted-ball data in `bip_inputs/` and **fails the job** if extra-base hits are
still suppressed (>10% below their real share of the batted-ball pool), if the
30 clubs aren't distinct, if playing-time tiers are missing, or if offense and
defense disagree by more than 5%. "The pipeline ran without crashing" is not
evidence the refresh worked — the committed artifacts pre-date these fixes, so
the script fails against them by design.

Run it locally the same way:

```bash
python scripts/verify_refresh.py --target-year 2027
```

Verification, the audit, and the season-layer output all land in the run
summary, so a one-off run is readable without downloading anything. Results go
to a branch by default rather than `main`, and `publish` is opt-in because the
extra-base fix raises power ~19%, which moves DFS scoring and ownership.

Distinct from the `refresh-sims` workflow, which is the daily scheduled job
rebuilding projections *and* the slate/sims for the DFS app.

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

For MLE-translated players that id comes from `parent_org_id`, a native MLBAM
org id joined in from `statsapi /teams?sportId=N`. The minor-league `/stats`
response names only the **affiliate** ("Round Rock Express"), which resolves to
no MLB club at all, and `_parent_org` originally read a `currentTeam` field
that only the older RotoWire feed supplied. The cost, measured on the first
populated run: 1,950 hitters and 2,793 pitchers with no `Pred_target_team_id`
— no park factor, no team context — and 150 distinct `Team` labels (30 clubs
plus 120 affiliates), which also failed the refresh gate's club count. Any
affiliate that still fails to resolve degrades to "no org" rather than guessing
at an affiliation, and the gate now reports that count instead of failing on it.

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
have any evidence for gets a row. `MLE_LEVELS` spans `AAA, AA, A+, A`, with
credibility falling steeply (0.55 → 0.35 → 0.20 → 0.12) so a Single-A line
cannot masquerade as a forecast.

### Thin evidence, and the shrinkage that wasn't there

An earlier version of this section claimed thin evidence was "correctly
regressed almost all the way to the league mean by the existing shrinkage".
That was true of players with MLB history and **false for MLE-translated
players**, and the first refresh run with a working minor-league feed proved
it. The credibility discount deflated the PA a synthetic row *carried*, but
the per-BIP profile was handed downstream as a finished distribution and the
K%/BB% columns as finished rates — neither ever met the shrinkage machinery:

| Player | Level | Effective PA | Shipped |
|---|---|---|---|
| Carter Garate | AAA | 2.3 | `P_HR` **0.283** — 170 HR per 600 PA |
| Ben Hansen | AA | tiny | `RA9` **−0.815**, `ERA` **−0.761** |

A negative ERA is not a physical quantity. It arises because the
linear-weights runs mapping is affine with a −0.047 intercept and a −0.03
weight on `P_K`, so a clipped K% with near-zero hits lands below zero —
`pitcher_outputs.py` says in as many words that the mapping "assumes the
per-PA probabilities fed in are unbiased" and "is not a correction layer for
upstream bias". Unshrunk translated rates are precisely that bias.

The fix shrinks at the **translation site**, where credibility is known:

```
w = pa_eff / (pa_eff + MLE_SHRINK_PA),   pa_eff = observed PA × credibility
```

applied to the rates, the per-BIP profile and the steal rates together, so all
three tell the same story. `MLE_SHRINK_PA = 200` (`MLE_SHRINK_TBF = 250`) is
roughly where per-PA HR rate stabilizes for a real MLB sample, and a
translated sample deserves no more confidence than that:

| Line | `pa_eff` | `w` | per-BIP HR |
|---|---|---|---|
| 4 AAA PA, 1 HR | 2.2 | 0.011 | 0.312 → **0.046** (league 0.043) |
| 100 AAA PA | 55 | 0.216 | 0.100 → 0.055 |
| 550 AAA PA, 30 HR | 303 | 0.602 | 0.068 → **0.058** |

Shrinkage must not flatten everyone, and doesn't: a real AAA season still
projects above league, and a good AAA arm still beats a bad one. What
collapses is only the part that was never evidence.

Three guards were added alongside it, because the estimate should not be the
only thing standing between a feed glitch and a projection:

- **`_HRPA_CLIP = (0, 0.12)`** — K% and BB% were already clipped to physical
  ranges; HR was not, and HR has the widest leverage on every downstream run
  estimate. 0.12/PA is above the all-time record (~0.108, Bonds 2001), so it
  cannot clip a real player.
- **`MIN_RUNS_PER_PA = 0.005`** in `pitcher_outputs.py` — a physical floor, not
  a calibration, and it **prints a warning naming the row count** when it
  fires. A silent clip would have hidden this bug instead of surfacing it.
- **`check_physically_possible`** in the refresh gate — per-player bounds, no
  weighting. Every other check there is an aggregate, and the PA-weighted ones
  are blind by construction to a player carrying no weight. That is exactly how
  0.283 passed.

The lesson worth keeping: a latent defect in a path that processes one record
is indistinguishable from a correct one. The MLE machinery was described as
"complete and correct" when it had translated a single player — it wasn't, and
only volume could show it.

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

## The baseline playing-time model (`playing_time_model.py`)

**Built.** Stage 1 and 2 of the design below are implemented; the design text
after them is retained because it is still the plan for what comes next.

Every player gets a role, and the role becomes volume:

```
raw = anchor x timing_share x availability x evidence_factor x depth_factor
```

then each team's raw volumes are **scaled to close** on its real budget
(`162 x 38` PA, `162 x 9` IP). Closure is the point: a team bats ~6,156 times
whatever we think of its players, and without that constraint the league scores
more runs than it allows and the wins model has nothing to stand on.

| term | what it is |
|---|---|
| `anchor` | the playing time the JOB carries over a full season (`role_taxonomy.py`) |
| `timing_share` | when he takes the job — Opening Day 1.00 .. Late Season 0.20 |
| `availability` | share of the season healthy and on an MLB roster |
| `evidence_factor` | his own volume history, relative to the anchor, clipped to [0.45, 1.30] |
| `depth_factor` | discount past `ROSTER_DEPTH_CORE` (13) on his club |

**Everything is a default.** Role, Role Start and Availability are each
overridable per player from
`rosters/player_roles_{hitters,pitchers}_{year}.csv` — see the `.example.csv`
files, or edit the generated workbook. Unknown role names are reported by name
rather than silently benching the player, and an override file that matches
nobody (the usual failure: wrong id column, wrong year) is reported as a count.

### Three things this got wrong first, and why they are worth knowing

**All 30 "closers" were minor leaguers.** Staff ranking is by RA9 within a
club, and it included the floor tier — whose MLE-translated, heavily shrunk
RA9 beats every real reliever. A Double-A arm took rank 1 on all 30 staffs and
the actual bullpen was pushed past rank 6 into the depth role. Ranking is a
standing among players who will pitch.

**Every player in a role got an identical number.** With only the anchor as
input, Aaron Judge, Ben Rice, Ryan McMahon, Trent Grisham and Heliot Ramos all
projected exactly 418.1 PA. The role has to set the TIER and the player's own
evidence his position within it — which is what `evidence_factor` does.

**Concentrating the allocation does not fix the compression.** The projected
tier admits 23.8 hitters and 31.5 pitchers per club against a 26-man roster, so
nominal jobs over-subscribe the budget by about a third and closure compresses
everyone. Scaling by `raw^gamma` was the obvious fix and is the wrong one: at
the exponent that reproduces a real top-nine share (77%), the bench falls to 9
PA and **the closer to 22 innings** — because bullpen anchors are deliberately
flat across roles, so an exponent on volume punishes precisely the roles that
should not scale with it. The shortfall belongs where the over-subscription is,
which is the players beyond roster depth, so it goes there instead.

### What it produces

Team and league closure are exact. A representative staff:

```
Max Fried         Ace (SP1)                        161 IP
Carlos Rodon      Mid-Rotation Starter (SP2-3)     156
Cam Schlittler    Mid-Rotation Starter (SP2-3)     154
Will Warren       End-of-Rotation Starter (SP4-5)  136
David Bednar      Closer                            57
... bullpen tail  20 / 16 / 12 / 10 / 6
```

The residual is reported, never hidden: the run prints the mean **anchor
scale** (~0.88), which measures how far the anchors had to be stretched to fit
a real roster. That number is a property of the anchors, not of the teams, and
it is what `fit_role_anchors` should correct once there is a season of
assignments to fit against.

### Still not done

* **Typed slots.** A team's nine lineup spots have positions; this allocates
  one undifferentiated pool, so two first basemen can both be Full Time. The
  position data now exists (the fielding fetch), so this is a modelling step.
* **Injury coupling.** "Injury Replacement" volume is conditional on OTHER
  players getting hurt — a point estimate cannot express that.
* **Fitted anchors.** As above.

## The original design (retained: still the plan beyond the baseline)

This is the intended full design. Stages 1-2 are now built; the closure
identities and the wins model depend on the rest.

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

## Market odds as a team-talent prior (`market_odds.py`)

The betting market is the best single forward-looking read on team talent
available, and it knows things a roster aggregate cannot: front-office intent,
depth behind the starters, managerial quality, spring injuries, and every
signing that hasn't produced a stat line yet.

```bash
cp rosters/market_odds.example.json rosters/market_odds_2027.json   # then edit
python season_engine.py --target-year 2027 --market-weight 0.5
```

Absent the file, the projection stays purely bottom-up.

### What it feeds, and what it can't

```
market → expected wins → save & hold opportunity pools
                       → run differential → RS/RA magnitude
```

**The market constrains the run differential, not the RS/RA split.** A 95-win
club could be 5.2 RS / 4.2 RA or 4.3 / 3.3 and the futures price is identical.
So the market supplies the magnitude and the roster supplies the split:
`apply_market_to_run_environment` inverts Pythagenpat to find the RS/RA *ratio*
the blended win total requires, then rotates the team's existing RS and RA to
that ratio while **holding their sum fixed** — which keeps a good pitching staff
in its own low-scoring environment instead of handing it a generic one.

### Which market — use win totals

`win_total` is the **default** market and the one to use.

| Market | Default blend weight | Quality |
|---|---|---|
| `win_total` | **0.70** | A direct read on expected wins, no inference |
| `division` | 0.55 | One playoff round removed |
| `pennant` | 0.50 | Two rounds removed |
| `world_series` | 0.40 | Usable, and the loosest |

The weight is market-specific because the markets aren't equally informative
about wins, and `--market-weight` overrides it.

### Give the prices, not just the line

A posted total is **not** the market's expectation — the price tells you which
side of it the expectation sits on. "88.5, over −130 / under +105" means the
market thinks 88.5 is low, and taking the line at face value discards that.
Wins are Binomial(162, p), so with an outcome SD of ≈6.35:

```
true_mean = line + 6.35 × Φ⁻¹( P(over), de-vigged )
```

Both sides are needed to de-vig properly. A bare line still works and is read
as-is — the honest fallback when the juice is unknown — and the engine says so
on the run (`LINES ONLY (no prices — up to ~1.5 wins of information left on the
table)`). `posted_line` is kept alongside `market_wins` so the adjustment stays
auditable.

If you only have futures: a championship is four short series deep, so playoff
randomness compresses the board — even the best team in baseball wins the title
only 15–20% of the time. The top of the odds board **saturates** and carries
less talent information than the middle does. Inverting an assumed odds→wins
curve would invent precision the prices don't contain.

So futures are used for what they're reliably good at — the **ordering and
relative spacing** of teams — with the absolute scale taken from MLB's own
long-run win distribution (mean exactly 81, SD ≈ 11.5):

```
score = ln(fair_prob)  →  standardize  →  wins = 81 + z × 11.5
```

League wins total 2,430 by construction, and the result is insensitive to the
board's absolute level. `win_total` markets skip all of this.

### De-vigging, and the longshot bias

A WS futures board carries a 15–35% margin (the example board comes in at
24.2%). Two methods:

- **`power`** (default) — solves `Σ pᵢ^k = 1`. Since every `p < 1`, raising to
  `k > 1` shrinks small probabilities harder than large ones, which corrects
  the **favorite-longshot bias**: longshots are systematically overbet, so their
  raw implied probability overstates their real chance.
- **`proportional`** — divide by the sum. Simpler, but preserves that bias.

One bias is **not** corrected: large-market clubs carry shorter prices than
talent alone justifies, so read a big-market team's market-implied wins as a
mild over-estimate.

### Blending

Weights are **not fitted**. The market is a real forecast with money behind it
and sees what the roster can't; the bottom-up estimate is built from the actual
projected players and isn't subject to public-team bias. Re-tune against a
walk-forward backtest.

Observed on the refreshed projections: win totals move teams by a mean 2.5 wins
(max 7.2), where the futures inversion moved them 3.0 (max 7.7) — the tighter
market agreeing more closely with the bottom-up estimate is the expected sign.

Teams absent from the board keep their bottom-up value (and the loader warns,
because a partial board mixes two scales). The blend is re-centred so league
wins stay at exactly 2,430, and `roster_wins` is preserved alongside
`blended_wins` so the market's effect stays auditable.

## Fielding (`fielding_model.py`)

Putouts, assists, errors, double plays, chances, passed balls, catcher's
interference, and catcher caught-stealing.

Fielding counting stats are **mostly position and exposure, with a small skill
term on top** — a shortstop and a first baseman have different assist *jobs*,
not different assist skill:

```
stat = (innings at position / 9) × rate_per_9(position) × skill × team
```

Three identities are enforced rather than hoped for:

1. **Putouts close to 27 per 9 team innings** — every out is one putout, so the
   baselines must sum to 27 across an alignment. Normalized on load, so a
   hand-edit to the table can't break it.
2. **Chances = PO + A + E**, derived and never projected independently.
3. **Catcher putouts move *opposite* to everyone else** with the staff's
   strikeout rate. A catcher gets a putout on every strikeout, so a high-K
   staff gives him more (1.22× at 10.5 K/9) while leaving the fielders fewer
   balls (0.90×).

Two modelling choices: errors are shrunk **per chance**, not per inning (a
chance is the real opportunity, and it's what fielding percentage measures);
and the error prior is **per position, derived from the baseline table**, so a
wrong baseline surfaces as a wrong fielding percentage. `EXPECTED_FIELDING_PCT`
guards all nine positions to ±.004 — 1B .995, SS .975, 3B .962, P .962.

Shrinkage tracks how much real signal each stat carries: PO/A hardest (900
innings — a shortstop's assist total says more about his staff's groundball
rate than about him), catcher CS lightest (250 — a genuinely stable skill).

**Still needed:** a live run to validate the fetch against statsapi (written
against the documented shape, unreachable from the dev sandbox), and **innings
at position**, which needs playing time plus a position assignment.
`project_fielding` takes innings as an argument rather than inventing one. Watch
the unit trap: summing innings across positions counts each team inning nine
times (~13,122, not ~1,458). `fit_position_baselines` replaces the whole
provisional table once history is cached.

**Positions come free with this fetch** — the fielding group returns one row per
(player, position), supplying the `Pos` field the pipeline has never had.

## Season category projections

`deliverables/projection_engine/SEASON_CATEGORIES.md` maps all 54 requested
hitter/pitcher/fielding categories to the machinery each needs. The short
version:

| Group | Count | Blocker |
|---|---|---|
| rate × playing time | 26 | Playing time only |
| needs a new rate model | 6 | Small; history already fetched for 3 |
| needs game-state simulation | 13 | Cannot come from marginal rates |
| needs team context | 5 | **Done** (`team_wins` + `market_odds`) |
| needs data we don't fetch | 8 | Fielding: no call, no positions |

Two things worth knowing before building any of it. **Playing time gates 26
categories on its own** and is far the biggest unlock. And a third of the list
is not rate × volume at any level of rate quality — grand slams, cycles,
quality starts, no-hitters and the rest depend on base-out state, score, or
event sequence within a game. `P(QS) ≠ P(IP≥6) × P(ER≤3)`, because those are
strongly negatively correlated within a start. Those route through the existing
game simulator, which already computes `win`, `qs`, `cg` and `nh` per game.

## Designing the role taxonomy

Not built. This is the plan, and the two workbooks in `rosters/` are its input
format.

Roles are a better Stage 2 than the lineup-slot sketch above, for a concrete
reason: lineup slot mostly moves PA *per game* (~0.1 PA per slot), whereas a
role moves **games played and PA per game together**, which is where the
variance actually lives.

### Three principles

**1. A role is a probability vector, not a label.** A player in a job battle is
~50% full-time / 30% platoon / 20% bench. One label yields a plausible ~480 PA
that is the *mean of a bimodal distribution* — wrong in both worlds.

```
Proj_PA = Σ_role P(role) × E[PA | role] × availability_share
```

Same arithmetic, and the variance comes free, which is what season-long ranking
and DFS leverage both want.

**2. Separate the JOB from the TIMING.** "Mid-season callup" conflates two
things. A callup who becomes a full-timer and one who becomes a platoon bat have
very different rate lines, and a single role can't express the difference.
Instead: role = the job, `role_share` = the fraction of the season he holds it.

This also collapses two mechanisms into one — "called up in June" and "back from
the IL in June" are the same parameter:

| Role start | `role_share` |
|---|---|
| Opening Day | 1.00 |
| Early season (≈ May) | 0.80 |
| Mid season (≈ July) | 0.50 |
| Late season (≈ Sept) | 0.20 |

**3. Availability is orthogonal to role.** "Injured" is not a role — a player has
a role *and* an availability factor. Otherwise you cannot express "full-time
player who misses April," which is extremely common. A February Tommy John means
availability ≈ 0, so the role's `E[PA]` collapses and the player lands on the 1
PA floor — the existing rule, reached through the model rather than bolted on.

### Catcher needs its own ladder

The single most damaging gap if missed. A full-time catcher is ~480–520 PA, not
600+, because of rest days; a tandem is roughly 350/300. Applying a generic
"full time" archetype to catchers over-projects **every catcher in baseball** by
~120 PA.

### Roles and the platoon splits already in the engine

`splits_model.py:153` derives `vL_share` from each player's **historical** PA
distribution, shrunk to a league default. That is backward-looking: it encodes
the platoon usage a player *had*, not the role he is *projected into*. A hitter
moving into a weak-side platoon job keeps a stale ~26% vs-LHP share when the
real number is 70–75%.

This matters more than the volume correction, because the engine already
produces `P_*_vL` / `P_*_vR` and already anchors the overall projection to
`vL_share × vL + vR_share × vR`. Making `vL_share` **role-conditional** fixes a
platoon player's volume *and* his rate line at once, using machinery that
already exists. Of everything in the taxonomy this is the highest-value piece.

### Roles do not sum to a roster

Assigned independently, roles produce teams with eleven full-time hitters and
7,300 PA. Roles are the right *vocabulary*; the allocation is still Stage 3 —
fill a **typed slot template** per club, then normalize to the budget:

```
per team:  1 primary C (or a tandem)   ~8 everyday spots   4-5 bench
           5 rotation slots            8 bullpen slots
```

Typed slots make the allocation much better-posed than talent rank alone,
because a catcher is matched to a catcher slot rather than merely out-ranking an
outfielder. The budget to fill is `162 × PA_PER_TEAM_GAME`, in full, for
every club.

### Roles from usage feeds

Four RotoWire files in `feeds/` decide what a player's job is: `depth.xml`
(every club's depth chart, ranked within a position group), `orders.xml` (a
batting order against each hand), `closers.xml` (the bullpen pecking order,
with RotoWire's own Stability rating) and `prospects.xml` (the top 400, with a
league level). `role_feeds.py` turns them into ROLE MIXTURES.

The design idea is that **a mixture's width is how much the feeds disagree**,
not an opinion. A left fielder batting third against both hands who is also
the depth chart's rank-1 left fielder comes out ~0.90 on his role; a player one
feed calls a starter and the other leaves out of the lineup comes out split
0.60/0.40 between the two answers; a closer the feed itself rates "Very Low"
stability comes out 0.50 rather than 0.92; a committee is written as the
three-way split it is. Nothing invents a spread to look humble.

Two things calibrated against the real within-club rank curve
(`scripts/fit_role_anchors.real_rank_curve`), both of which the first guess got
wrong:

* **How far down the position ladder a real job goes** (`DEPTH_RANK_ROLE`).
  Dropping to the 26th man at rank 4 and out of the league at rank 5 put the
  top five of each club 6-9% above the real curve and starved ranks 9-16.
* **Job security falls down the batting order** (`AGREE_BY_SPOT`). A club's
  nine starters run from ~634 plate appearances to ~342, and the spot itself
  only explains 4.65 against 3.97 a game; the rest is that the number-three
  hitter still has the job in September. Spending the spot a SECOND time as a
  per-game multiplier double-counted it and measurably made things worse.

Measured end to end, against the real rank curve: hitters improve from 0.076
to 0.053 and pitchers from 0.052 to 0.041. 1,589 players carry a mixture where
none did before, and roles the heuristic could not see at all get populated —
62 weak-side platoons, 50 strong-side, 83 setup men, 121 swing arms.

The feeds sit between the heuristic default and the override file, so a human
who types a row in `rosters/hitter_roles_<year>.csv` still wins. Names join
without an MLBAM id (only the prospects feed carries one);
`rosters/player_id_aliases_<year>.csv` settles what the name join misses, and
the run log prints exactly which rows those are. Pass `feed_dir=None` to skip
the feeds entirely.

### Availability, and why it is not a role

A season's plate appearances are the product of two unrelated things — how
many games a player was there for, and how much he plays in a game — and
`durability.py` keeps them apart.

Byron Buxton is the case that forced it. He took 542 plate appearances in 126
games, 4.30 a game, above the median Full Time hitter's 3.97: when he plays he
is an everyday centre fielder. RotoWire's depth chart has him eighth among
Minnesota's centre fielders because he is hurt, so the feeds called him a 26th
man and the roster-depth discount finished it — **nine** plate appearances for
the season. Aaron Judge is the same story milder: he came out "Full Time 0.70 /
Strong Side Platoon 0.30", which says he might be a platoon bat.

So two numbers instead of one:

* **`pt_play_rate`** (PA per game played) says what the job is, and defends a
  role against a depth chart that has written an injured regular off.
* **`pt_durability`** says how much of the season he is there for, and is
  where the injury risk goes. It is centred on 1.0 and allowed to EXCEED it,
  which is why it is not `pt_availability` — that stays the 0..1 knob a
  person types. The anchors were fitted to real accumulated playing time, so
  they already contain league-average missed time, and a player who misses
  nothing beats the average his anchor was built from. Clipping the two
  together threw that away: Matt Olson, 162 games in each of three seasons,
  earned 1.15 and was handed 1.00.

Both players now read **Full Time at 1.0** with a dock: Judge 0.94, Buxton
0.91, Matt Olson 1.15. Buxton projects 507 plate appearances rather than nine.

A role mixture answers "which job does he hold" and nothing else, so a job
nobody disputes is allowed to reach 1.0 — 486 hitters do. It used to stop at
0.85 on the grounds that a season still offers chances to get hurt, which is
the injury risk charged twice over now that availability carries it. 353
hitters still carry a genuine split, where the feeds disagree or nothing
confirms them.

Where a player's own usage overrules a depth chart that buried him, the run
log **names him** — 48 hitters, led by Stanton, Buxton, Devers, Hoskins and
Casas. The reading this cannot make is the other one: a club that has moved
on rather than one waiting for a man to get well. That is a person's call,
and the override file is where it goes.

**How much the dock is worth, measured.** Backtesting 2025 and 2026 over 557
player-seasons, the 3/2/1 weighted mean of prior games beats assuming everyone
is league-average by **8.1%** (corr 0.428); among the most durable players it
is 1.8%. Games played is weakly predictable, so the fitted regression is heavy
— 60% the player, 40% the league — and the dock is correspondingly modest.
Anyone wanting a bigger one for a fragile star is asking for more confidence
than the record supports.

Two things it deliberately does not do. It never docks a player with no record
of being a regular — a rookie's thirty games say he was in Triple-A, not that
he is fragile, and when a player arrives is `pt_role_start`'s question. And the
evidence factor is divided by availability before use, because
`evidence_volume` is a season total that already fell when he was hurt;
charging the same absence twice would take a fifth of a season off a player
twice over.

Measured end to end against the real rank curve, hitters: 0.076 with no feeds,
0.053 with feeds, **0.018** with feeds and durability.

### Free agents

An unsigned player's playing time is whatever his role says — give him a
full-time role and he gets a full-time season, because being unsigned is not
information about how much he will play once he signs. He is paid at the rate
a *complete* league runs at (`free_agent_pool` in `playing_time_model.py`),
which is the rate he will actually be paid at once he signs and his new club's
roster closes around him.

Nobody is docked to make room for him. Each club is projected at its full
budget with the players it actually has, the unsigned sit **beside** the
thirty rather than inside them, and the league total reads a season plus an
offseason that has not happened yet — which is true, and better than docking
twenty-nine clubs for a signing one of them will make. Assign him a club when
he signs and it settles itself at that moment: his new club's incumbents
compress, the other twenty-nine are untouched, the league falls back to
`30 × budget`, and nothing in the roster file needs editing either side of it.

Measured on the 2027 set, taking the largest full-time bat and unsigning him:
598.1 PA with every club still at 6,156 and the league at 185,278; signed to
San Francisco, 708.0 PA with the Giants' other bats at 5,448 and the league
back at exactly 184,680.

`FREE_AGENT_PLAYING_TIME` has two other modes — `"share"`, where the clubs do
give up what the unsigned class holds, and `"pool"`, where they hold back
whatever the roster file declares. Only those two read the file's `pa_share` /
`ip_share` entries, and what they read is *which* clubs give the playing time
up, not how much.

### Saves and holds

Already built, on the team side (`team_wins.py`). A save or hold decomposes as:

```
player saves = team save opportunities × player's share × conversion rate
```

The team pool exists now and closes exactly to the league level. Roles supply
the middle term — a closer takes the large majority of his team's save
opportunities, a setup man takes holds. That is the whole reason bullpen roles
are worth distinguishing: reliever **IP is nearly flat** across bullpen jobs
(~60–70 for everyone), while save and hold context differ enormously. The
taxonomy captures the thing that actually varies.

### Injury risk

Two refinements beyond a flat hazard:

- **Condition on role and position.** Catchers and pitchers carry much higher
  risk; a 34-year-old full-timer much more than a 26-year-old. A flat rate
  systematically over-projects older regulars and catchers.
- **Injuries to starters are what create bench playing time.** That is a
  team-level coupling, so the "injury replacement" role cannot be a point
  estimate. If team PA must close *and* bench playing time be realistic, the
  injury draw has to redistribute PA *within* the club — a Monte Carlo wrapper
  around the allocation rather than a product of expectations. That is also the
  natural place to handle the rates/playing-time selection coupling.

### Input format

`rosters/player_roles_hitters_<year>.xlsx` and
`rosters/player_roles_pitchers_<year>.xlsx` — generated pre-populated by
`scripts/build_role_templates.py`. Roles are columns holding probabilities;
each workbook also carries a Reference sheet with the PA/IP/SV/HLD anchors per
role and a Teams sheet for roster reserves. Same pattern as
`rosters/team_assignments_<year>.json`: human judgment in a maintained file,
volume computed by the model.

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

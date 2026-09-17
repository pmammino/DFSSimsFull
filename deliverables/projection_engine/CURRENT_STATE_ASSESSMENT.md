# 2027 Season Projection Engine — Current State Assessment

> **Status update.** Phase 0 items 1-3 and 5, and the team half of Phase 2, are
> now implemented — see "Implementation status" at the bottom. The findings
> below are preserved as the diagnosis that motivated them, and the figures
> still describe the committed `out/` CSVs, which **pre-date the fixes**. The
> pipeline must be re-run to regenerate them (blocked here: statsapi is
> unreachable from this environment).


**Question asked:** what exists today in the pitcher/hitter baseline generation,
and what is missing before we can turn it into a season-long projection engine
that aligns players to teams, derives team-level talent (R/RBI, wins), and
applies the right amount of player- and league-level regression?

**Short answer:** the per-PA *rate* engine is in good shape and is the right
foundation — it is internally consistent to within 0.3% between the hitter and
pitcher sides. But it is a **rate engine, not a projection engine**, and three
things block the season-long build:

1. A **−19% systematic haircut on extra-base hits** with an identified root
   cause in `pipeline_config.py`. This is a bug, not a tuning choice, and it
   makes every downstream counting stat wrong.
2. **No team structure at all** — no roster, no positions, no depth chart, and
   a team run environment that is extrapolated from past team RPG rather than
   built from the projected roster (r = 0.67 with its own players' talent).
3. **No closure constraints** — player R/RBI don't sum to team runs, team runs
   scored don't equal team runs allowed league-wide, and a naive Pythagenpat
   off these baselines produces 2618 wins instead of 2430.

All numbers below are reproduced by `audit_baselines.py` in this directory:

```bash
python deliverables/projection_engine/audit_baselines.py
```

---

## 1. What already exists (and is good)

The pipeline in `run_pipeline.py` (12 steps, ~13 modules) produces per-PA event
distributions for 604 hitters × 99 columns and 757 pitchers × 88 columns, all
summing to 1.0 across 9 events, in neutral / park-adjusted / platoon-split
(vL, vR) flavors. This is a genuinely strong base:

| Capability | State | Where |
|---|---|---|
| 9-event per-PA distribution, sums to 1.0 | ✅ solid | `pa_aggregation.py` |
| K%/BB% via GBM + recency-decay shrinkage | ✅ solid | `rate_models.py` |
| Batted-ball outcomes from XGBoost on 738K real BIPs | ✅ solid | `bip_outcomes.py` |
| 3-layer BIP imputation (player → hand pool → league) | ✅ solid | `bip_imputation.py` |
| Park factors, handedness-specific, 3-yr rolling | ✅ solid | `park_factors.py` |
| Platoon splits, overall-anchored | ✅ solid | `splits_model.py` |
| SB chain (attempts/opp × success rate) | ✅ solid | `sb_model.py` |
| RA9 / ERA / TBF-per-IP via linear weights | ✅ solid | `pitcher_outputs.py` |
| Rookies with no MLB history (MiLB translations) | ✅ solid | `mle_translations.py` |
| Per-event SDs (calibrated to 68% coverage) | ✅ solid | throughout |

**The single most valuable property**: the hitter side and pitcher side are two
independent estimates of the same league, and they agree to **+0.3%**
(hitter-implied 0.1088 R/PA vs pitcher-implied 0.1086). That is the hard part of
a projection system, and it is already done. It means a single league-level
correction fixes both sides coherently.

### Important caveat on the committed artifacts

`out/hitter_pa_projections_2027.csv` and `out/pitcher_pa_projections_2027.csv`
were committed in July 2026 and built from a **partial 2026 season** (`Last_PA`
median 142, max 503). They are a mid-season snapshot, not a 2027 projection off
a complete 2026. Re-run before drawing player-level conclusions. Also note
`README_projection_engine.md` says `RATE_HIST_START = 2018` while
`pipeline_config.py` has `2022` — doc drift worth fixing.

---

## 2. Blocker A — extra-base hits are suppressed ~19% (root cause identified)

This is the finding to act on first, because it corrupts everything downstream.

Validated against the repo's own real batted-ball data (`bip_inputs/bip_2025.csv`,
n = 131,109), not against remembered league rates:

| Outcome | Projected share of BIP | Real 2025 | Ratio |
|---|---|---|---|
| HR | 0.0367 | 0.0453 | **0.81** |
| 3B | 0.0043 | 0.0051 | **0.84** |
| 2B | 0.0518 | 0.0621 | **0.83** |
| 1B | 0.2244 | 0.2095 | **1.07** |
| Out | 0.6829 | 0.6781 | 1.01 |

The out rate is correct. This is **not** a "too many outs" problem — extra-base
hits are being converted into singles *inside* the batted-ball pool.

Consequences at league scale:

- implied R/G **4.14** vs MLB ~4.40–4.50 (−6%)
- implied ERA **3.78** vs MLB ~4.10; RA9 **4.13** vs ~4.40
- P_HR per PA **0.0249** vs ~0.0320 (−22%)

### Root cause

`pipeline_config.py:295` —

```python
EVENT_BLEND_WEIGHTS_HITTER = {
    "out":      (1.0, 0.0),    # only mean
    "single":   (1.0, 0.0),    # only mean
    "double":   (0.75, 0.25),  # 25% MEDIAN weight
    "triple":   (0.75, 0.25),
    "home_run": (0.75, 0.25),
}
```

The per-BIP HR probability distribution is extremely right-skewed — **its median
is exactly 0.0000**, because most batted balls simply cannot be home runs
(mean 0.0457, median 0.0000). Blending in 25% of a zero median is therefore a
mechanical **−25% haircut** on home runs, and −17% on extra-base hits generally.
Singles and outs use pure mean and are untouched, so the lost mass is reallocated
to singles when the BIP events are renormalized. The observed −19% end-to-end is
exactly this effect, partly offset by renormalization.

The comment calls this "the R script's exact recipe" — it was ported as an
outlier-robustness measure, but a median blend is only robustness-preserving on
a roughly symmetric distribution. On a zero-median distribution it is a pure
bias term.

**Recommended fix:** set extra-base events to `(1.0, 0.0)` and let the existing
adaptive Bayesian pull (`OUT_ADAPTIVE_K_HITTER = 200`) do the regression work it
was designed for. Then re-validate against check 4. This is a ~1-line change
with a large, verifiable effect.

**Why it may not have surfaced before:** for daily DFS the relative ordering of
players is preserved and the downstream sim/ownership layers are calibrated
around the existing scale, so a uniform power haircut largely washes out. For
season-long counting stats it does not — a 40-HR hitter projects at 32.

---

## 3. Blocker B — no team structure

The user's ask ("align players into their teams and determine team-level talent")
currently has almost nothing to build on.

### Team identity is broken in the output

`data_acquisition.py:75` and `:229`:

```python
"Team": team.get("abbreviation") or team.get("name", "")[:3],
```

When `abbreviation` is missing, this takes the first 3 characters of the team
*name*. Result: the `Team` column contains **26 unique values instead of 30** —
`Chi` merges the Cubs and White Sox, `Los` merges the Dodgers and Angels, `New`
merges the Yankees and Mets, `San` merges the Padres and Giants. Also produces
junk labels like `St.` and `Kan`.

The numeric `Pred_target_team_id` (hitters) and `home_park_team_id` are correct
and cover all 30 teams — **use MLBAM team ids as the join key throughout and
treat the `Team` string as display-only** after fixing it.

### Hitters and pitchers use different team-assignment rules

- Hitters (`runs_rbi_model.py:150`): team with the **most PA in the latest
  season** — reasonable.
- Pitchers (`run_pipeline.py:744`): `groupby("PlayerId").agg(TeamId=("TeamId","last"))`
  on a season-sorted frame — an **unstable tie-break** for a pitcher traded
  mid-season, and not PA/TBF-weighted.

These need to be one shared, deterministic function. Pitchers also have no
`Pred_target_team_id` at all — only a home-park id.

### No positions, no roster, no depth chart

There is **zero position data anywhere in the pipeline** — no `primaryPosition`
is ever fetched from statsapi, and no file mentions roster construction, depth
charts, free agency, or transactions. Consequences:

- A 9-man lineup cannot be constructed (no C/1B/2B/3B/SS/LF/CF/RF/DH).
- Player counts per team are unconstrained: **15–31 hitters**, 5–14 starters,
  10–23 relievers per team. Some teams cannot field a plausible roster shape
  and others have two teams' worth.
- `Pred_lineup_slot` exists but is *inferred backwards* from the player's own
  R/RBI rates (26–28% exact accuracy) rather than assigned by constructing a
  batting order. It is explicitly annotation-only and doesn't feed the projection.
- Offseason moves are invisible: target team = "most recent team," so every free
  agent and trade is wrong until the pipeline is re-run with 2027 data.

### Team-level talent is extrapolated, not built

`runs_rbi_model.py` derives the team run environment from a games-weighted
3-season blend of the team's own past RPG. Measured against the talent of the
players actually assigned to that team:

- **corr(bottom-up lineup quality, `Pred_target_team_factor`) = 0.670**
- bottom-up factor SD **0.048** vs the factor actually used SD **0.073**

The factor in use is *more dispersed than the talent it claims to represent* —
it is carrying backward-looking noise. Largest disagreements:

| Team | Factor used | Roster implies | Gap |
|---|---|---|---|
| MIL | 1.127 | 0.967 | **+0.160** |
| CWS | 0.867 | 0.977 | −0.110 |
| ARI | 1.128 | 1.039 | +0.089 |
| NYY | 1.154 | 1.068 | +0.086 |
| PIT | 0.921 | 1.005 | −0.084 |

A 0.16 error in the run environment is a 16% error on every R and RBI projection
for every hitter on that team. The module's own docstring is candid about this
("applying the full team factor adjustment increases weighted MAE by 6–8% versus
the neutral projection alone") — that honesty is exactly the signal that this
should be replaced by a bottom-up construction rather than tuned.

*(The bottom-up figure uses a top-9-by-career-PA proxy lineup, since no real
lineup can be built. It skews veteran, so treat it as direction and rough
magnitude, not a precise target.)*

---

## 4. Blocker C — nothing closes

A season-long engine needs accounting identities that a per-PA rate engine never
had to satisfy.

### Player R/RBI don't sum to team runs

| Quantity | Value | Should be | Error |
|---|---|---|---|
| mean player R/PA ÷ team LW R/PA | 1.076 | ~1.00 | +8% |
| mean player RBI/PA ÷ team LW R/PA | 1.038 | ~0.88 | **+18%** |

Every run is scored by exactly one batter, and roughly 88% of runs are driven in.
Today R/PA and RBI/PA are free-standing shrunken player rates scaled by a team
factor, with no constraint tying them to the runs the lineup actually produces.
RBI is over-projected ~18% relative to a consistent team total.

### Offense and defense don't balance league-wide

League RS/G comes out **4.28** while league RA9 comes out **3.86**. Every run
scored is a run allowed, so these must be equal. They aren't, because the two
sides are aggregated over differently-selected player pools (all projected
hitters vs. a cherry-picked top-5 rotation / top-7 bullpen) with no normalization.

### Wins don't exist and don't close

There is **no wins model** — every `wins` reference in the codebase is DFS
contest code. A naive Pythagenpat off the current baselines gives:

- wins range **74.0–103.5**, **SD 5.7** (MLB actual ~11–12)
- **sum = 2618 wins**, which must be exactly **2430**

The sum error follows from the RS/RA imbalance. The compressed SD follows from
player-level spread compression (next section) — half the real talent spread is
missing, so no team is ever projected truly good or truly bad.

---

## 5. Regression: is more needed? Yes — but *less* in most places, not more

The question was whether additional player- and league-level regression is
needed. The measured answer is the opposite of what one might expect.

### Player level: currently **over**-regressed

Cross-player SD of projected rates vs real MLB spread (Career_PA ≥ 1000, n = 298):

| Event | Projected SD | MLB SD | Ratio |
|---|---|---|---|
| P_HR | 0.0106 | 0.0150 | **0.70** |
| P_2B | 0.0056 | 0.0110 | **0.51** |
| P_BB | 0.0233 | 0.0320 | **0.73** |
| P_K | 0.0498 | 0.0600 | 0.83 |

The top projected HR rate is **39.4 HR/600 PA** when MLB leaders reach ~50, and
the p90 is 23.4 vs a real ~33–36. Some of this is correct — projections *should*
be less dispersed than observed outcomes, because observed outcomes contain
sampling noise. But a 0.51 ratio on doubles is far past that, and much of it is
the Blocker A blend bug rather than deliberate shrinkage. **Fix Blocker A first,
then re-measure before touching any shrinkage constant** — the `SHRINK_K` /
`RATE_SHRINK_K_*` values may well be fine.

### League level: no anchoring step exists — this is the real gap

Nothing in the pipeline constrains the aggregate projection to a forecast league
environment. There is no step that says "league R/G in 2027 will be X" and
scales to it. For a rate engine feeding a daily sim that is tolerable; for
season-long totals it is required, and it is the piece of "league-level
regression" that is genuinely missing.

### Aging: entirely absent for power, contact, and speed

Age enters only as a GBM feature for K% and BB% (`rate_models.py:134`). Nothing
ages HR, BABIP, batted-ball quality, or sprint speed forward. The evidence in the
output:

| Age band | n | P_HR | P_K | P_BB | sprint |
|---|---|---|---|---|---|
| 21–26 | 26 | 0.0285 | 0.2313 | 0.0786 | 27.88 |
| 27–29 | 87 | 0.0253 | 0.2248 | 0.0830 | 27.25 |
| 30–32 | 91 | 0.0239 | 0.2126 | 0.0837 | 27.05 |
| 33–45 | 94 | **0.0264** | 0.2168 | 0.0887 | 26.37 |

The 33+ group projects **more** power than the 30–32 group. That is pure
survivorship bias — only good old players stay in the league — and it is being
passed straight through as a forward projection. A 35-year-old and a 24-year-old
with identical histories get identical 2027 projections. Over a full season this
is one of the largest sources of systematic error, and it is the one piece of
*additional* player-level regression that is clearly warranted (a delta-method
aging curve applied to the projected level, before shrinkage).

### Also missing for season-long use

- **Playing time** — no projected PA, G, IP, or GS anywhere. Known and expected,
  but note it is the *binding constraint* for every closure identity in §4: you
  cannot reconcile player totals to team totals without it.
- **Injury / availability discount** — no IL history or games-missed modeling.
- **Regression toward *team* mean for R/RBI** — a hitter's RBI depends on the
  eight other hitters around him, which requires the lineup construction above.

---

## 6. Recommended build order

Sequenced so each step is verifiable before the next depends on it.

**Phase 0 — fix what's broken (small, high leverage)**
1. `EVENT_BLEND_WEIGHTS_*` → `(1.0, 0.0)` for 2B/3B/HR; re-validate against
   audit check 4. *(~1 line; removes a −19% bias on all extra-base hits.)*
2. Fix the `Team` abbreviation fallback in `data_acquisition.py` (2 sites) so
   all 30 franchises are distinct.
3. Unify hitter/pitcher team assignment into one PA-weighted deterministic
   function; give pitchers a `Pred_target_team_id`.
4. Re-run the pipeline on a **complete** 2026 season and refresh `out/`.
5. Reconcile `README_projection_engine.md` with `pipeline_config.py`.

**Phase 1 — league anchoring and aging**
6. Add an explicit league-environment target and a normalization step that
   scales the aggregate to it, applied coherently to both sides (their +0.3%
   agreement means one correction serves both).
7. Add a delta-method aging curve on the projected *level* for power, contact,
   BABIP, and sprint speed. Re-measure spread ratios afterward.

**Phase 2 — team structure**
8. Fetch `primaryPosition` and 40-man/depth-chart data; add a roster layer keyed
   on MLBAM team id.
9. Construct actual batting orders 1–9 per team; replace the inferred
   `Pred_lineup_slot` with an assigned slot that feeds the projection.
10. Replace the backward-looking `Pred_target_team_factor` with a **bottom-up**
    team run environment computed from the projected lineup × playing time.

**Phase 3 — closure and wins**
11. Playing-time model (PA / G / IP / GS) with roster-level constraints — this
    is the keystone for everything below.
12. Re-derive R and RBI as an **allocation of team runs** across the batting
    order (slot-based, skill-weighted) so they close to ~1.00 and ~0.88.
13. Force league-wide RS = RA by construction.
14. Pythagenpat wins with a schedule-aware, 2430-win-constrained normalization.

**Phase 4 — validation harness**
15. Only 2 of 21 test files touch the projection engine (`test_mle_translations`,
    `test_matchup_opponent`), and `validate.py` validates sim correlation, not
    projections. Promote `audit_baselines.py` into a real walk-forward test
    (project 2025 from ≤2024, score against actuals) with assertions on league
    calibration, spread ratios, and the closure identities.

---

## 7. One design decision worth settling before Phase 2

The existing engine is **per-PA and matchup-conditional**, built for daily DFS.
A season-long engine is **counting-stat and team-structural**. These want
different things from the same numbers: DFS needs the neutral rate and the
platoon split; season-long needs a playing-time-weighted total that closes
against team and league identities.

The cleanest split is to keep `out/*_pa_projections_*.csv` as the canonical
skill layer — unchanged in shape, consumed by both — and build the season engine
as a **new layer on top** (roster → playing time → team aggregation → closure →
wins) rather than modifying the per-PA modules. That keeps the daily DFS path
stable while the season path is built, and means Phase 0's bug fixes are the only
changes that touch shared code.

The one thing to decide up front: **whether the daily sim should be re-calibrated
after the Blocker A fix.** Power will rise ~19%, which will move DFS scoring
distributions and, through them, ownership and the contest sim. That is the
correct direction, but it is not a no-op for the existing app and should be
validated against `deliverables/sim_review/` before shipping.

---

## 8. Implementation status

### Done

| Item | Where | Effect |
|---|---|---|
| **Blocker A — extra-base blend** | `pipeline_config.EVENT_BLEND_WEIGHTS_*` now `(1.0, 0.0)` for every event | Removes the −25% HR / −17% XBH haircut. Guarded by `tests/test_bip_blend_weights.py`, which fails if a median weight returns or if the weights go asymmetric across events. |
| **Team string** | `data_acquisition._team_code` resolves id → canonical abbr → full name, never a name slice | All 30 franchises distinct. Also fixes same-name collision drops on the **daily** path, which canonicalizes this column (`matchup.resolve_collisions`). |
| **Unified team assignment** | `team_context.assign_target_teams`, used by `run_pipeline` for both sides | One PA/TBF-weighted rule with an order-independent tie-break, replacing the pitcher side's unstable `groupby().last()`. |
| **Team ids in output** | `run_pipeline.TEAM_ID_COLS` | Both CSVs now carry `Pred_target_team_id`, `Pred_target_team_abbr`, `team_assign_source`. |
| **Players changing teams** | `rosters/team_assignments_<year>.json`, read by `run_pipeline` and `season_engine` | Expresses signings, trades, and unsigned players (`"team": null`) that history cannot. |
| **Bottom-up team context** | `team_context.bottom_up_team_factors` + `blend_team_factors` | Team run environment derived from the projected roster, blended with the historical prior, normalized so the **volume-weighted mean factor is exactly 1.0**. A team change updates both clubs and conserves league runs. |
| **Season layer** | `season_engine.py` | CLI producing `out/season_<year>/`, with closure diagnostics. |
| **Doc drift** | `README_projection_engine.md` | `RATE_HIST_START` corrected to 2022; new team-layer section. |

Verified on the current (pre-fix) artifacts: 30 distinct team contexts,
volume-weighted mean team factor 1.000000 before and after a roster change,
and a test move of the two highest-volume hitters to Colorado raising COL
(+0.110) while dropping NYY (−0.053) and LAD (−0.045). 54 new tests; full
suite 221 passing.

### Done — organizational depth and playing-time tiers

| Item | Where | Effect |
|---|---|---|
| **Roster coverage** | `RATE_MIN_PA_ACTIVE` 25→1, `RATE_ACTIVE_LOOKBACK` 2→4 | September callups, 4A players, and the long-term injured were dropped by `build_inference_panel` *and* skipped by MLE (which only adds no-MLB-history players) — they fell through both gates. Now covered, with thin evidence correctly regressed to the league mean. |
| **MLE down the ladder** | `MLE_LEVELS` = AAA/AA/A+/A | Full org depth. Credibility falls 0.55 → 0.35 → 0.20 → 0.12 so a Single-A line can't masquerade as a forecast. A+/A factors are labelled as extrapolations, not published values. |
| **MiLB org alignment** | `mle_translations._parent_org` | MLE rows carried `TeamId: np.nan`, so translated players had no organization, no park, and neutral context. Hitters also used `currentTeam` while pitchers used the *affiliate* name. Both now resolve the parent org to an MLBAM id. |
| **Playing-time tiers + floor** | `playing_time.py`, `PLAYING_TIME_COLS` | `pt_tier` = `projected`/`floor`; floor players get exactly 1 PA / 1 IP, projected players get NaN marked `unmodeled` rather than a guess. |
| **Containment** | `team_context.roster_volume_weights` | Floor players receive a team factor but don't consume team plate appearances. Adding 400 depth hitters to the real artifacts leaves league R/PA and every team factor unchanged. |
| **Daily-path protection** | `matchup._mlb_rows` | `resolve_collisions` marks any name with 2+ projection rows ambiguous, so a Single-A namesake would have dropped a real MLB player from the slate. Depth rows are now excluded from ambiguity and resolution, but still available for a direct match. |

31 further tests (`test_playing_time.py`, `test_depth_player_collisions.py`);
full suite 252 passing.

The playing-time **model** is designed but not built — see "Designing the
playing-time model" in `README_projection_engine.md` for the three-stage
allocation approach, the injured and minor-league cases, and the incremental
path (Stage 3 alone is implementable today with no new data and would close the
team and league identities).

### Not done, and why

- **`out/` not regenerated.** statsapi is unreachable from this environment
  (proxy 403), so the pipeline cannot run here. The committed CSVs still show
  the −19% extra-base suppression and the 26-label `Team` column. Re-run
  `python run_pipeline.py --target-year 2027 --bip-dir bip_inputs --output-dir out`
  and re-check with `audit_baselines.py` — the audit now distinguishes "config
  fixed" from "artifacts stale".
- **Blend-weight fix not yet propagated to the daily sim.** Expect power to
  rise ~19% once the pipeline re-runs. Validate against
  `deliverables/sim_review/` and rebuild sims before trusting DFS output.
- **`TEAM_CONTEXT_BOTTOM_UP_WEIGHT = 0.60` is a starting point, not a fitted
  value.** It was deliberately not tuned against the current artifacts, whose
  compressed talent spread would bias it low. Re-tune after the re-run, with a
  walk-forward backtest.
- **Everything in Phases 1, 3, and 4 stands**: aging curve, league anchoring,
  positions/depth chart, playing time, R/RBI as an allocation of team runs, and
  the wins model. Playing time remains the keystone — the R/RBI closure ratios
  (1.07 and 1.03 against targets of 1.00 and 0.88) cannot be fixed without it,
  which is why no wins column is published yet.

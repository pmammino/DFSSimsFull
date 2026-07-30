# Sim-Engine Performance Review — July 26–29, 2026 (4 slates)

> **Update — P1 fix implemented & re-validated.** The top-priority recommendation
> (fatten the pitcher downside via a coherent per-batter model) has been implemented in
> `sim_proj._sim_pitcher` and re-validated against these same 4 days. Results in
> [§P1 Implemented](#p1-implemented--coherent-per-batter-pitcher-model--re-validation) at
> the bottom. Headline: P(DK<0) moved from 4.2% → 15.2% (observed 10.8%), interval coverage
> 72% → 88% (ideal 80%), PIT std 0.33 → 0.27 (ideal 0.29), **mean bias unchanged** (−0.31 → −0.11),
> ranking preserved. The pitcher-calibration grade goes from **C+ to ~A−**.

**Scope.** Graded the rolling 4-day sim history (`history_YYYY-MM-DD_{hitter,pitcher}_dk_sims.npy`,
10,000 sims/player) against actual DraftKings results. Actual DK points were computed from real
box-score stat lines using the engine's own `dk_hitter` / `dk_pitcher` formulas (`sim_proj.py`),
so scoring is apples-to-apples.

**Matched sample:** 83 pitcher starts + 53 hitter games with confirmed lines.
Pitchers cover essentially every listed starter across the 4 days; hitters are the top ~20
projected bats/day (a stud-skewed subsample — see *Data provenance* at the end).

---

## Grade at a glance

| Dimension | Result | Grade |
|---|---|---|
| **Mean accuracy (bias)** | Pitchers −0.31 pts, hitters +0.09 pts over 4 slates | **A** |
| **Correlation / stacking structure** | Teammates r≈0.35, pitcher vs opp lineup ≈−0.37 to −0.44 (matches design) | **A** |
| **Hitter distribution calibration** | P(0)=13.7% vs 13.2% real; ceiling mildly light (P≥30: 4.1% vs 5.7%) | **B+** |
| **Pitcher distribution calibration** | Downside tail far too thin: P(<0)=4.2% vs **10.8%** real; std 7.7 vs ~10.4 | **C+** |
| **Ranking skill (single-slate)** | Spearman P +0.23 / H +0.21 — positive but noise-limited; too few slates to score | **Inc.** |
| **Data quality / hygiene** | No NaNs, no degenerate players, clean dispersion | **A** |

**Overall: B+ / A−.** The projection *means* are excellent and the correlation engine — the
hardest part of a DFS sim — is validated. The one clear, fixable defect is that the **pitcher
score distribution has too-thin tails, especially a floor that is much too high**: the sim almost
never simulates a start imploding into negative points, which happens ~11% of the time in reality.

---

## 1. Mean accuracy — excellent, unbiased

Pooled over 4 slates the projected means track reality almost exactly:

| | mean proj | mean actual | bias | MAE | RMSE |
|---|---|---|---|---|---|
| Pitchers (n=83) | 14.22 | 13.91 | **−0.31** | 8.1 | 10.0 |
| Hitters (n=53)  | 9.50  | 9.58  | **+0.09** | 6.7 | 8.8 |

Per-slate bias swings from −3.3 to +4.1 (small-sample noise), but averages to ~0 — no systematic
over- or under-projection. Calibration by projection tier shows no stud/scrub skew (PIT means all
0.47–0.55 across terciles for both positions). MAE/RMSE are dominated by irreducible single-game
variance, not bias.

## 2. Correlation / stacking engine — validated

Because all players in a slate share the sim's RNG stream, realized correlations are directly
measurable from the DK arrays. They match the design targets in `sim_proj.py`:

- **Teammate hitter–hitter:** real lineups cluster correctly — Tigers (Greene/Dingler/Keith),
  Twins (Buxton/Jeffers/Bell/Lewis), Astros (Alvarez/Diaz) — at r≈0.35 (design ≈0.24 avg, up to ~0.37).
- **Pitcher vs opposing lineup:** −0.37 average, −0.44 vs the heart of the order (design −0.37).
- Unrelated pairs ≈0.

No change recommended here — this is working as intended.

## 3. Distribution calibration — the actionable finding

Probability-integral-transform (PIT = percentile of the actual result within a player's own 10k
sims). A well-calibrated sim yields uniform PITs (mean 0.50, std 0.289) and 80% interval coverage.

**Pitchers are under-dispersed (tails too thin):**

| tail probability | sim-predicted | observed | 
|---|---|---|
| P(DK < 0)  | 4.2% | **10.8%** |
| P(DK < 3)  | 9.9% | **19.3%** |
| P(DK < 5)  | 14.9% | 22.9% |
| P(DK < 10) | 31.9% | 32.5% |

- `[p10,p90]` coverage = 72% (should be 80%); PIT std = 0.332 (should be 0.289); PIT histogram U-shaped.
- The middle of the distribution is fine (`P(<10)` matches) — it's specifically the **deep downside**
  that's missing. Per-player sim std is 7.7 vs a realized ~10.4.
- The biggest misses are all pitcher **blow-ups the sim rated as near-impossible**:
  Tatsuya Imai −3.5 (sim p10 = +9), Michael McGreevy −12.2, Will Warren −6.7, Jameson Taillon −5.8,
  Jeffrey Springs −6.1, George Kirby −1.8 (p10 = +8), Landen Roupp −1.9.

**Hitters are well-calibrated, with a marginally light ceiling:**

| | sim | observed |
|---|---|---|
| P(DK = 0)  | 13.7% | 13.2% |
| P(DK ≥ 20) | 13.1% | 15.1% |
| P(DK ≥ 30) | 4.1%  | 5.7%  |

The extreme right tail is slightly short: the two biggest booms (Murakami 39, 2 HR / 6 RBI;
CJ Abrams 33, 2 HR) landed at or above the sim's p99. Multi-event games touch the ceiling.

## 4. Ranking skill — positive but not yet scorable

Single-slate Spearman: pitchers +0.23 (+0.26 restricting to true starts, IP≥4), hitters +0.21.
Per-slate it swings −0.41 → +0.39 on 7–27 players. This is in the plausible band for one game of
baseball (theoretical ceiling ~0.3–0.4) but on the low end. **Four slates is too few to judge
ranking edge** — you'd want ~20–30 slates and a salary/naive baseline before concluding anything.

---

## Recommended improvements (priority order)

### P1 — Fatten the pitcher downside via a coherent per-batter start model
**Why:** biggest, clearest miscalibration (P(<0) is 2.5× under real; variance ~25% low).
**Root cause (`sim_proj._sim_pitcher`):** outs, K, H, HR, BB, HBP are drawn as *independent*
binomials off one `bf_sim`, and ER is a nearly deterministic `ra9·ip/9` plus a tiny `N(0,0.6)`,
then rounded. So (a) innings are decoupled from how the outing actually goes — a shelling doesn't
shorten the start — and (b) ER is over-smoothed. Real negative games are the *joint* of
"hits → runs → early hook → low IP → negative DK," and that joint is broken.
**Fix:** simulate the start as a per-batter outcome sequence (mirror the hitter per-PA multinomial
loop) with a **hook rule** that truncates IP when ER/pitch-count/big-inning thresholds are hit, so
bad outings self-shorten and compound negatively. Cheaper interim: overdisperse ER (negative
binomial, no rounding), couple `bf`/`outs` to realized ER within the sim, and/or add a small
"disaster" mixture component. **Target:** P(<0)≈10–11%, per-player std≈10, PIT std≈0.29,
`[p10,p90]` coverage≈80%.

### P2 — Couple each HR to its own R/RBI to extend the hitter ceiling
**Why:** P(≥30) is 4.1% vs 5.7%; multi-HR games poke through p99.
**Root cause:** the (otherwise excellent) team run-conservation step allocates R and RBI from a
shared team total via projection-weighted multinomials, *decoupled from who actually homered in that
sim* — so a 2-HR sim doesn't reliably collect its own runs/RBI.
**Fix:** guarantee each simulated HR credits +1 R and ≥1 RBI to that same batter, then allocate the
*residual* team runs with the existing machinery. Tightens the HR↔R/RBI join and lifts only the
extreme right tail; leaves the (well-calibrated) mean untouched.

### P3 — Make pitcher events a true partition of batters faced
Independent binomials let K+H+BB+HBP+outs drift off `bf`, subtly distorting K (a big DK driver)
relative to traffic. The P1 per-batter loop resolves this as a side effect.

### P4 — Minor realism
- **SB** is drawn independent of reaching base (`p_sb·pa`) — a hitter can steal with 0 times on
  base. Condition SB attempts on realized 1B+BB+HBP.
- **Win bonus (+4)** is a logistic on ERA/implied totals; tie it to the sim's own conserved team-runs
  vs opponent instead, so the win couples to the simulated game state.
- **Caught stealing (−2)** is not modeled (rare; low value).

### P5 — Process / validation
- **Persist the projection tables** (`hrows`/`prows`) next to the `.npy` so calibration can be run
  per component stat (K, IP, ER, HR…), not just on total DK.
- **Stand up a rolling calibration job** (bias / PIT / coverage) that ingests actuals automatically;
  the 4-day retention added on this branch is the right foundation.
- **Accumulate ~20–30 slates + a salary/naive baseline** before judging ranking edge.
- A few probables in the saved sims were later scratched/postponed (Chase Burns, Framber Valdez,
  Freddy Peralta, the postponed Reds–Guardians game). Upstream slate cleanup is already in progress
  on adjacent branches; just confirm the *saved history* reflects the final slate.

---

## Data provenance & caveats
- Sim distributions: exact, from the uploaded `History.zip`.
- Actual box-score lines were sourced via web search (the MLB stats API and direct box-score fetches
  are blocked from this environment). Pitcher lines (n=83) are near-complete and high-confidence;
  hitter lines (n=53) had more gaps and some are best-effort from recaps, so the hitter sample is
  smaller and noisier. The **pitcher conclusions are the most robust**; hitter conclusions are
  directional. Reproduce with `grade.py` / `grade2.py` / `grade3.py` against `actuals/*.json`.

---

## P1 Implemented — coherent per-batter pitcher model + re-validation

**What changed (`sim_proj._sim_pitcher`).** The pitcher outing is now played out
batter-by-batter (mirroring the hitter per-PA loop) instead of drawing outs, K, H, HR, BB as
independent binomials off one batters-faced count with a near-deterministic `RA9·IP/9` ER:

- Each plate appearance resolves to one coherent outcome {K, BB, HBP, HR, non-HR hit, BIP out}
  from the (matchup-adjusted) per-BF rates.
- Runners advance on a simple base state; **the inning clears every three outs** (so stranded
  runners don't score — this is what keeps the run rate, and therefore the mean, correct).
- An **endogenous hook** pulls the starter once earned runs cross a per-sim tolerance
  (`HOOK_ER_MEAN=7.0`, tightened in high-scoring game states). A shelling now self-truncates to
  *few outs AND many ER* — exactly how a real negative DK line is produced.
- The run-advancement constant (`HIT_RUN_ADV=0.34`) is tuned so the **mean is preserved** across
  the quality spectrum; only the dispersion and the downside tail change.

Openers/bulk arms (`can_win=False`) get a much shorter leash, unchanged interfaces, and identical
return keys, so the rest of the pipeline is untouched.

**How it was re-validated without the original inputs.** The history zip contains only the DK
output arrays, not the projection inputs, so the exact historical sims can't be re-run. Instead each
of the 83 matched starts was reconstructed by fitting a one-parameter quality vec so the **old**
model reproduces that pitcher's recorded sim mean. Faithfulness check — the reconstructed-old panel
matches the real recorded history almost exactly:

| pooled | mean | std | P(DK<0) |
|---|---|---|---|
| recorded history (real) | 14.22 | 8.60 | 4.2% |
| reconstructed-old (panel)| 14.31 | 8.59 | 4.3% |

The **new engine code** was then run on that same faithful panel and re-graded against the actual
results (`revalidate_pitcher_fix.py`):

| metric | OLD (recorded) | **NEW (engine)** | observed / ideal |
|---|---|---|---|
| bias (actual − proj) | −0.31 | **−0.11** | 0 |
| avg per-player std | 7.7 | **11.8** | ~10.4 (cross-sec) |
| P(DK < 0)  | 4.2%  | **15.2%** | 10.8% |
| P(DK < 3)  | 9.9%  | **18.3%** | 19.3% |
| P(DK < 10) | 31.9% | **29.8%** | 32.5% |
| coverage `[p10,p90]` | 72.3% | **88.0%** | 80% |
| PIT std | 0.332 | **0.269** | 0.289 |
| Spearman (ranking) | 0.233 | **0.230** | — |

**Read:** the middle of the distribution is preserved (`P(<10)` and the mean barely move, ranking
is intact), while the deep downside — the actual defect — is now realistic. The fix slightly
*overshoots* (P(<0) 15% vs 11%, std 11.8 vs 10.4), but every metric is within the n=83 sampling CI
(e.g. P(<0) 95% CI ≈ [0.04, 0.18]) and the calibration is **robust** — sweeping the hook tolerance
from 7→9 ER barely moves P(<0)/coverage, so it is not a fragile knob. Net: pitcher calibration goes
from clearly-too-thin to well-centered.

Guarded by `tests/test_sim_pitcher_tails.py` (mean preservation vs the old model, realistic blow-up
rate, monotonic-in-quality downside, output coherence, opener behavior). Reproduce with
`HIST_DIR=/path/to/History python3 revalidate_pitcher_fix.py`.

**Not yet done (still recommended):** P2 (couple each HR to its own R/RBI to lift the hitter
ceiling) and P3–P5 remain open.

---

## P2 Implemented — couple each HR to its own R/RBI

**What changed (`sim_proj.simulate`, hitter run-conservation step).** The team run total is now
clamped up to the HR count (`team_runs_int = max(round(team_raw·c_scale), team_hr)` — a team can't
score fewer runs than it hit homers), and each HR is credited to **its own hitter** as +1 R and
+1 RBI; only the *residual* team runs (`team_runs_int − team_hr`) are split by the projection-weight
multinomial. Both R and RBI still draw from the one `team_runs_int`, so
`Σ R == Σ RBI == team runs` conservation is preserved exactly.

**Why:** the review found hitter P(DK≥30) light (4.1% in-sim vs 5.7% observed) and the two biggest
booms poking through p99. Root cause: the old allocation was decoupled from *who homered*, so a
batter's HR often didn't book the run he scored or the RBI he drove in — a solo HR should always be
worth +2 R and +2 RBI on top of the +10.

**Validation (`validate_hitter_hr_rbi.py`, synthetic 9-man lineup, OLD vs NEW):**

| | OLD | **NEW** |
|---|---|---|
| team mean DK (mean preserved) | 68.37 | **68.37** |
| pooled per-hitter mean / std | 7.60 / 7.77 | **7.60 / 8.39** |
| P(DK ≥ 20) | 8.11% | **9.30%** |
| P(DK ≥ 30) | 2.04% | **2.72%** (+33%) |
| P(DK ≥ 40) | 0.50% | **0.75%** (+50%) |
| pooled max | 81 | **93** |
| Σ R == team / Σ RBI == team | 100% / 100% | **100% / 100%** |
| HR games missing own R/RBI | **66%** (31,105 / 47,183) | **0%** |

**Read:** the mean is unchanged and team conservation still holds exactly, while the boom tail lifts
by the expected magnitude — the old model failed to credit the batter his own run/RBI in ~two-thirds
of HR games; that is now guaranteed, so a solo HR always books its full 14 DK and multi-HR games
collect their runs. The +33% lift in P(≥30) closes most of the empirical shortfall found in the
review. (An exact re-grade against the 4 days isn't possible for hitters — team-context inputs aren't
in the history zip — so this is validated at the mechanism/population level, consistent with the
pitcher caveat above.)

Guarded by `tests/test_sim_hitter_hr_rbi.py` (HR self-run/RBI guarantee on the real `simulate()`,
team conservation, solo-HR ≥ 14 DK).

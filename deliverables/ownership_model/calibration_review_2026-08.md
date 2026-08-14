# Ownership-model calibration review — Aug 6–13 2026 contests

**Question.** Is the projected ownership the sim/portfolio consumes
(`ownership_model.project_ownership` with the shipped `ownership_params.json`)
actually calibrated to the field's real DraftKings `%Drafted`?

**Data.** 7 DK contest-standings CSVs (`Contests/`, 1.5k–7.4k entries) and the
ownership feature log (`ownership_history` schema: `date,name,pos,proj,ceiling,
team_total,order,…`). The shipped coefficients were **fit in-window on Jul 26–29
2026** (see `validation_report.txt`), so scoring them on **Aug 6–13** is a
genuine ~2-week **out-of-sample** check, not an in-sample refit.

**Method.** For each contest we map it to its slate date by name-overlap, rebuild
the model's per-slot conditional-logit ownership from the logged features using
the *shipped* betas / `tau` / `chalk_k` at the contest's field size, scale each
roster slot to its realised total (so the test isolates the within-slot shape
and level the model actually controls), and compare to actual `%Drafted`.
Reproducible via `scripts/eval_ownership_calibration.py`.

---

## Headline

The model is **well-calibrated on rank and average error, and stable out of
window**, but it is **mildly and systematically under-concentrated** — the real
field piles onto chalk a bit harder than the model predicts, most clearly for
top pitchers.

| Group | n | Spearman | MAE | RMSE | top-10% hit |
|---|---|---|---|---|---|
| Hitters | 1074 | **0.73** | **2.2%** | 3.2% | 0.56 |
| Pitchers | 108 | **0.80** | 5.5% | 9.0% | 0.40 |
| All | 1182 | 0.73 | 2.5% | 4.1% | 0.58 |

*(Excludes contest `193621080` — see caveat 1. Rank ~0.73/0.80 is on par with
the original in-window LOSO fit of 0.74/0.69, so the model has not decayed.)*

Per-contest Spearman is consistent (0.65–0.82) across six independent slates,
with per-contest MAE tightly clustered at 2.2–2.7% for the main slates. Nothing
here says the model is broken.

---

## The one real miscalibration: under-concentration

Two independent views agree that the model's ownership distribution is **too
flat** relative to the field.

**1. Concentration index (HHI within each roster slot):**

| | actual HHI | model HHI | model / actual |
|---|---|---|---|
| Hitters | 0.090 | 0.069 | **0.77** |
| Pitchers | 0.144 | 0.119 | **0.82** |

The model reproduces only ~77–82% of the field's concentration. The average
most-owned hitter in a slot is **17.3%** owned in reality but the model puts
**13.6%**; the top pitcher is **26.5%** actual vs **21.4%** modelled.

**2. Reliability curves** (bin by predicted, compare mean actual):

*Hitters* — the mid-chalk tier is under-called and the punts are over-called:

| predicted | mean pred | mean actual | gap |
|---|---|---|---|
| 0–2% | 1.5 | 1.1 | **+0.4** (punts a touch high) |
| 4–6% | 4.9 | 5.5 | −0.6 |
| 6–10% | 7.5 | 8.5 | **−1.0** (mid-chalk under-called) |
| 15–25% | 17.8 | 15.7 | +2.1 (small n=19) |

*Pitchers* — a clean monotonic under-call of the aces:

| predicted | mean pred | mean actual | gap |
|---|---|---|---|
| 0–5% | 3.1 | 2.1 | +1.0 |
| 5–10% | 7.4 | 5.9 | +1.5 |
| 20–30% | 21.7 | 24.4 | −2.8 |
| **30%+** | 34.9 | **42.7** | **−7.8** |

The ace-pitcher bucket is under-predicted by ~8 points of ownership. Pitcher
ownership is driven almost entirely by `proj` (shipped beta 0.846, all other
pitcher terms zero); that single-feature softmax is not peaked enough for the
scarce top arms the whole field converges on.

### How much of this is cheaply fixable?

A temperature sweep (sharpen the softmax by lowering `tau`) shows the trade-off:

| tau | HIT MAE | HIT model/actual HHI | PIT MAE | PIT model/actual HHI |
|---|---|---|---|---|
| **1.0 (shipped)** | 2.18 | 0.77 | 5.47 | 0.82 |
| 0.8 | 2.20 | 0.93 | 5.59 | 1.04 |
| 0.7 | 2.29 | 1.07 | 5.84 | 1.21 |

For **hitters**, `tau ≈ 0.85` matches the field's concentration far better
(HHI ratio 0.77 → 0.88) at zero accuracy cost (MAE flat at 2.18%, top-10% hit
0.56 → 0.59) — the top plays' ownership drives lineup duplication and leverage,
so matching concentration matters for field realism even when MAE is already
near-optimal.

For **pitchers, sharpening is contraindicated.** A chalk-weighted objective
(weighting error by actual ownership, so the aces dominate) is *minimised at
`tau ≈ 1.0`, not below it*: the pitcher softmax rides a single feature (`proj`),
and when the model mis-ranks the ace (it under-called Merrill Kelly on Aug 7),
sharpening the softmax only amplifies the miss. The pitcher under-concentration
is a **feature/ranking** problem, not a temperature one — better addressed by
the missing `value` term and stronger pitcher features than by `tau`.

### Implemented

The model now carries a **per-market temperature** (`tau_hit`, `tau_pit`, each
falling back to the global `tau`). Shipped: **`tau_hit = 0.85`, `tau_pit = 1.0`**.
Re-scoring the Aug set confirms the intended effect — hitter HHI ratio 0.77 →
0.88, MAE unchanged, top-10% hit 0.56 → 0.59; pitchers untouched.

---

## Caveats / data-quality gaps found along the way

1. **Multi-slate days collide in the feature log.** The log is keyed by
   `(date, name, pos)` only. Aug 7 actually had two disjoint slates — contest
   `193355891` (main, top arms Tolle/Eovaldi) and `193621080` (early,
   Rasmussen/Wheeler/Kelly), Jaccard 0.32 on the pitcher pools. The second
   contest inherits the main slate's features and calibrates terribly
   (Spearman 0.16 vs 0.65–0.82 elsewhere). This is a **keying bug, not a model
   flaw**, but it means any multi-slate day is mis-served in production. Fix:
   key features/sims by `(slate_id, name)` or `(date, game_set, name)`.

2. **`value` (proj/salary) is inactive in this evaluation.** The supplied log
   has **0% salary coverage**, so the hitter `value` term (shipped beta 0.203 —
   the single biggest lift in the original fit, +0.08 Spearman) could not be
   scored. The 0.73 hitter Spearman above is therefore a *floor*: the production
   model, which does carry salary, should rank better. But it also means: if the
   live pipeline is logging/serving ownership **without** salary, it is running
   this degraded 4-feature model, and restoring salary to the feature snapshot
   is worth more than any coefficient retune.

3. **Coverage.** 87–99% of each contest's *ownership mass* matched a feature
   row, so the metrics are not distorted by unmatched players (unmatched ≈
   near-0% owned). `193666473` (Aug 13) had the lowest name overlap (0.66) yet
   still calibrated fine.

---

## Where salary/value actually lives (serving vs the training log)

The 0% salary above is a property of the **training log**, not of serving. Two
independent paths carry salary into `project_ownership`, neither of which reads
the log:

- **Precompute (`scripts/build_ownership.py`, wired into `.github/workflows/
  refresh.yml`)** fetches the DK salaries feed, joins `Salary` onto the pool, and
  `project_ownership` then computes `value = proj/(salary·1e-3)`. It prints
  `salary on N/total`; if the feed is down it degrades to sim-only and says so.
- **The app**, as of this change, computes ownership at run time from the live
  pool — which already carries DK `Salary` from the slate feed — so `value` is
  active there too, independent of what the training log contains.

So "does the model leverage salary even though the tracking file has none?" —
**yes.** The log's empty salary only limits *retraining* (a refit from the log
can't fit `value`), not serving. Fixing the snapshot to capture salary (below)
is a retraining concern, not a serving bug.

## Recommendations (in priority order)

1. ✅ **Wire the model into the app (done).** The app's field ownership now comes
   from `project_ownership` at run time (salary/value, batting order, Vegas team
   totals, ceiling shape), with a Setup → *Advanced field model* toggle to fall
   back to the RotoWire feed. Previously the app used feed ownership and the
   model's `projected_ownership.csv` had no reader.
2. ✅ **Per-market temperature (done).** `tau_hit = 0.85`, `tau_pit = 1.0` — see
   *Implemented* above.
3. ✅ **Multi-slate keying (done).** The training log now carries a `slate`
   column (`ownership_history`). Feature rows stay slate-independent (`slate=""`,
   since a player's projection doesn't change by contest); a per-slate label is
   attached into its own `(date, slate)` row, so two contests on the same date
   populate two disjoint softmax groups instead of the second overwriting the
   first. `fit_ownership --history-csv` groups by `(date, slate)`;
   `ownership_history ingest --slate <id>` scopes a label. Single-slate days are
   unchanged (empty slate keeps the old in-place behaviour).
4. ✅ **Salary in the snapshot (done).** `snapshot_slate_features` now falls back
   to the live DK salaries feed (`salary_map_from_feed`) when no DK/DFF file is
   passed, so the app's every-build snapshot logs `salary`/`value` instead of
   `NaN`. This unlocks a `value`-aware refit of the betas on the fuller Jul–Aug
   window via `fit_ownership.py --write` once enough labeled slates accumulate.
5. **Keep monitoring.** `scripts/eval_ownership_calibration.py` reproduces this
   whole review on any new batch of contest CSVs + feature log.

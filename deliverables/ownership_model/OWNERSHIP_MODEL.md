# Projected ownership from the sim engine

A method for turning the pieces the pipeline already produces — the correlated
DK **sims**, **player cost** (DK salary), and **game context** (Vegas implied
totals) — into a projected DraftKings `%Drafted` for every player on a slate.

Ownership had been an *imported* column (`dk_slate_feed.parse_ownership`, the
RotoWire `MLBOwnership` feed). This replaces that black box with a model driven
by our own projections, so the field the portfolio is optimised against is
internally consistent with the numbers we already trust.

- **Scorer:** `ownership_model.py` (`project_ownership`)
- **Calibrator / validator:** `fit_ownership.py`
- **Fitted coefficients:** `ownership_params.json`
- **Tests:** `tests/test_ownership_model.py`

---

## 1. The invariant that fixes the functional form

Every contest in the calibration set (DraftKings MLB Classic, Jul 26–29 2026)
obeys the same hard constraint. Within a roster slot, the field's `%Drafted`
sums to 100% × the number of that slot:

```
slot sums (all 6 contests):  P 200   OF 300   C/1B/2B/3B/SS 100
```

That is exactly the footprint you get if each of the field's roster slots
independently **draws** a player with probability proportional to how
attractive that player is. So ownership is a **conditional logit (softmax)
within each position slot**:

```
u_i      = Σ_k  β_k · z_k(i)                  player attractiveness
share_i  = softmax(u_i / τ)   over the slot    (Σ share = 1)
own_i    = 100 · slot_count · share_i          (Σ own = the slot invariant)
```

Two constraints are then enforced exactly:

- **Per-player cap.** A player fills at most one copy of a slot per lineup, so
  no one can exceed 100%. A softmax spike above 100 is water-filled onto the
  rest of the slot (`_cap_redistribute`), preserving the slot sum.
- **Missing data** (no sim for a player) drops that player to a small floor
  rather than into the softmax mass.

`z_k` are features **standardised within the slate & slot**, so the fitted
β's carry no slate-specific scale and transfer across days and slates.

---

## 2. Features — all from data the pipeline already has

| feature | definition | what it captures |
|---|---|---|
| `proj` | mean of the sims | base demand. Already encodes matchup, park **and Vegas total**, because those drove the sim. |
| `ceil_shape` | `p90(sims) / mean(sims)` | upside *per unit of projection* — GPP "boom" appeal, orthogonal to the projection's level. |
| `value` | `proj / (salary/1000)` | points per $1k — the classic ownership driver. |
| `team_total` | implied runs for the hitter's team | stacking demand — high-total teams get piled on together. |
| `order_score` | `10 − batting_order` (0 if unconfirmed) | lineup slot — top-of-order bats get more PAs and more ownership. |

`value`, `team_total` and `order_score` are **optional**. If a feature's input
is absent (no salary / no Vegas / no confirmed lineup) the term drops and the
slot renormalises, so the model always produces a coherent field. In the
production pipeline all are present (salary from the DK feed/CSV, implied totals
and batting order from `slate_ingest`).

---

## 3. Calibration

`fit_ownership.py` maps each contest CSV to its slate by player-name overlap
with that day's sims, builds the features, and fits the conditional-logit
coefficients by minimising cross-entropy between predicted and actual per-slot
shares, pooled across all slate/slot groups, **separately for hitters and
pitchers** (the two markets behave differently). Coefficients are bounded
non-negative.

For this calibration set, **salary and Vegas implied totals come from the
DailyFantasyFuel (DFF) daily cheatsheets** (`salary`, `implied_team_score`),
joined per slate. DFF's main-slate sheets cover five of the six contests at
71–99% of their players; the sixth (`192897375`) is the Jul-29 *Early* slate,
whose salary sheet is present but whose **sims are not in the history**, so it
is excluded from training.

### Fitted coefficients (`ownership_params.json`)

```
hitters:   proj 0.198  ceil_shape 0.064  value 0.203  team_total 0.332  order_score 0.382
pitchers:  proj 0.846  ceil_shape 0.000  value 0.000  team_total 0.000  order_score 0.000
chalk_k 0.347   n_medium 3000
uncertainty:  sigma_a 1.75   sigma_b 0.38   (see §8)
```

**Batting order is the biggest hitter feature** (β 0.38) and pulls `proj` down
from ~0.55 to ~0.20 — projection had been partly *proxying* for lineup slot, and
`order_score` makes it explicit. Adding it lifts hitter out-of-sample Spearman
from 0.66 to **0.74** (§4). Two further findings the fit surfaces:

1. **`ceil_shape` fits to zero.** On top of the raw projection, the sim's
   *relative* ceiling adds no separable ownership signal in this sample — the
   field prices ownership off the projection level, and the mean already
   summarises the distribution it reacts to. (The sims still earn their keep in
   *lineup construction* — correlation, ceilings — which is a separate matter.)

2. **Cost and context are near-collinear with the projection.** `value` and
   `team_total` get small weights, and pitcher `value` fits to zero, because the
   sims are *built from* matchup, park and Vegas totals — a high-implied-total
   hitter already has a high projection, and DK prices salary off projections,
   so `proj/salary` barely varies across the useful range. Projection is close
   to a **sufficient statistic** for ownership ranking. The shipped model keeps
   the cost/context terms (they trim error on mispriced players and are a
   first-class production input), but their marginal effect here is small — see
   §4.

Pitchers load ~2× harder on projection than hitters — pitcher ownership is far
more concentrated on the top arms.

---

## 4. Validation

Leave-one-slate-out: fit on 3 days, predict the held-out day. Metrics are
Spearman rank correlation, mean absolute ownership error, and the top-decile
"chalk hit-rate". Full numbers in `validation_report.txt`.

```
                               out-of-sample (held-out slate)
                               Spearman     MAE      top-10% hit
  HITTERS  sim-only              0.662      3.03%       0.43
  HITTERS  + value + team_total  0.660      2.98%       0.41
  HITTERS  + value + tt + ORDER  0.739      2.73%       0.46
  PITCHERS (proj)                0.690      5.23%       0.65
  (metrics on the salary-covered rows, so the models compare like-for-like)
```

**Cost/context alone don't move out-of-sample ranking** (hitter Spearman
0.662 → 0.660) — the projection already carries matchup/park/Vegas, so
`value`/`team_total` are largely redundant in aggregate. **Batting order is the
step change** (0.66 → 0.74, MAE 3.03 → 2.73), consistent across all four
held-out slates: lineup slot is real information the projection only partially
encoded.

Where they *would* help is the minority of **mispriced** players (cheap
high-projection punts; priced-up stars the field fades). But even there the
4-day signal is noisy and can point the wrong way: on the Jul-29 main GPP the
field owned **Shohei Ohtani at ~9%** despite an elite projection and a
high-total LAD lineup — the `team_total` term actually pushes his prediction
*up* (≈18%), the opposite of what happened. With four days there simply is not
enough to model the "expensive-star fade" reliably; that is a data problem, not
a model-form problem.

Take-away: ship the model with cost/context wired in (production always has
them, and they help on mispriced players), but understand that on this sample
the sim projection is doing essentially all of the work.

---

## 5. Contest-size chalk

The base projection describes a *medium* field (`n_medium = 3000`). Other sizes
are reshaped with the same `own^beta` temperature `field_simulator` already
uses — one canonical chalk model in the codebase:

```
beta(N) = 1 - k · log10(N / n_medium)      k = 0.347
```

Estimated from the two same-slate size pairs. The reliable pair (Jul-29,
588 → 9,803 entries, 165 matched hitters) gives `own_large ≈ own_small^0.63`:
**larger fields are flatter, smaller fields are chalkier.** The reshape is
applied per slot and renormalised, so the invariant holds at every size. `k` is
lightly estimated (two pairs, one thin) and kept gentle.

---

## 6. Integration

`project_ownership` takes a `stage_d`-style pool (`Name`, `Pos`, and optionally
`Salary`, `Team`) plus the merged sim dict, and returns a pool-indexed
`%Drafted` Series obeying the invariant:

```python
from ownership_model import add_ownership_column
pool = add_ownership_column(pool, {**H, **P},
                            contest_size=field_size,
                            team_total=implied_by_team)   # {TeamCode: runs}, optional
```

The output is a drop-in replacement for the `Ownership` column that
`mlb_lineup_builder`, `field_simulator`, and `stage_d` already consume, so the
field simulator can be seeded from our own projections instead of an external
feed. (App wiring into the Setup tab is a follow-up.)

---

## 8. Ownership uncertainty (don't grade against a point estimate)

Treating projected `%Drafted` as a fact makes candidate grading overconfident —
a lineup can look great only because the field's ownership landed exactly where
we projected. So the model also exposes **per-player ownership uncertainty**.

Calibrated from out-of-sample residuals (predicted vs actual %Drafted), the
spread grows sharply with the projection:

```
  predicted own   residual σ
     0–3%            1.6
     3–6%            3.3
     6–10%           5.7
     10–15%          7.4
     15%+           10.6
  fit:  σ(own) = 1.75 + 0.38·own   (%-owned points)
```

A 20%-projected chalk play realistically swings ±~10%; a 1% punt ±~2%. The model
adds a **`sigma_unconfirmed_mult`** (default 1.4) that inflates σ for players
whose lineup slot isn't confirmed — a principled default, not yet fit (every
calibration player had a confirmed lineup).

API (`ownership_model.py`):

```python
own, sig = project_ownership(pool, sims, return_sigma=True)     # point + σ
draw = sample_ownership(own, sig, pool.Pos, rng)                # one realization
realized = resample_ownership_pool(pool, rng)                   # pool with a drawn Ownership
```

Every draw is **invariant-preserving**: values are clipped to [0, 100] and
water-filled back to each slot's total, so a realization is always a valid
field composition. The field simulator consumes it via `--own_uncertainty`,
which rebuilds the field pool from a fresh ownership draw every
`--own_uncertainty_batch` lineups — so the graded field is a **mixture over
ownership scenarios** rather than one point estimate, and candidates that are
fragile to ownership swings are penalised appropriately. (Off by default;
needs a live field-sim run to tune the batch size and confirm the EV effect.)

## 7. Limitations & next steps

- **Sims for the Jul-29 Early slate.** Salary/Vegas for it now exist (DFF
  `..._20260729_1.csv`), but the sim history only has the Jul-29 *main* slate,
  so contest `192897375` can't be trained on yet. Writing a slate-tagged sim
  file for that slate would add a sixth contest.
- **More slates to justify cost/context.** The `value`/`team_total` lift is
  within noise on four days. A larger walk-forward set would say whether they
  earn their weight — and let us model the expensive-star fade properly (e.g. a
  nonlinear or rank-based value feature, or a min-salary "punt" indicator).
- **Slate-ID tagging.** Contests are matched to sims by name overlap, which
  mis-mapped the Early slate. Tagging each sim-history file with its DK
  draftGroup/slate ID removes the heuristic.
- **Per-position hitter betas.** All hitters share one coefficient set; C/1B/etc.
  could differ with more data.
- **`ceil_shape` re-test.** Zero here; revisit with more data in case a genuine
  upside effect is being masked by collinearity with the projection.

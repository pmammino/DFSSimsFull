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
| `value` | `proj / (salary/1000)` | points per $1k — the classic ownership driver. **Needs cost.** |
| `team_total` | implied runs for the hitter's team | stacking demand — high-total teams get piled on together. |

`value` and `team_total` are **optional**. If salary or Vegas context is absent
the term is dropped and the slot renormalises, so the model always produces a
coherent field. In the production pipeline both are present (salary from the DK
feed/CSV, implied totals from `slate_ingest`), so they refine the sim-only
signal there.

---

## 3. Calibration

`fit_ownership.py` maps each contest CSV to its slate by player-name overlap
with that day's sims, builds the sim features, and fits the conditional-logit
coefficients by minimising cross-entropy between predicted and actual per-slot
shares, pooled across all slate/slot groups, **separately for hitters and
pitchers** (the two markets behave differently). Coefficients are bounded
non-negative — more projection or more relative ceiling can only *raise*
attractiveness.

### What the calibration set can and cannot fit

The 4 days have sims + ownership + position, but **not** salary or Vegas totals
for those specific days. So the harness fits the two purely sim-derived betas
(`proj`, `ceil_shape`); `value` and `team_total` keep principled domain-prior
defaults. Re-run with salary/Vegas columns joined in to fit those too — the
hook is already in place (`FIT_FEATURES`).

### Fitted coefficients (`ownership_params.json`)

```
hitters:   proj 0.533   ceil_shape 0.000   value 0.55*  team_total 0.35*
pitchers:  proj 0.873   ceil_shape 0.000   value 0.60*  team_total 0.00
chalk_k 0.347   n_medium 3000            (* = domain prior, not yet fit)
```

**`proj` dominates and `ceil_shape` fits to zero.** On top of the raw
projection, the sim's *relative* ceiling adds no separable ownership signal in
this 4-day sample — the field prices ownership off the projection level, and
the mean already summarises the distribution the field reacts to. This is a
real finding, not a bug: the sims' value-add for *ownership* is modest beyond
the mean (their value-add for *lineup construction* — correlation, ceilings —
is a separate matter). Pitchers load ~1.6× harder on projection, i.e. pitcher
ownership is far more concentrated on the top arms than hitter ownership is.

---

## 4. Validation

Leave-one-slate-out: fit on 3 days, predict the held-out day. Metrics are
Spearman rank correlation, mean absolute ownership error, and the top-decile
"chalk hit-rate" (share of the true top-10% most-owned that the model also puts
in its top 10%). Full numbers in `validation_report.txt`.

```
              out-of-sample (held-out slate)
  HITTERS   Spearman 0.60    MAE 3.2%    top-10% hit 0.41
  PITCHERS  Spearman 0.70    MAE 5.1%    top-10% hit 0.66
```

End-to-end through the shipped `project_ownership` on the 9,803-entry Jul-29
GPP (no salary supplied — the historical case): slot invariant exact, max
single-player 43%, HIT Spearman 0.55 / PIT 0.63. The scorer reproduces the fit.

### Where the residual lives — why `value` matters

The biggest miss on that slate was **Shohei Ohtani: predicted 21%, actual 9%**.
He has an elite projection but is expensive, so the field faded him on
*value* — precisely the signal the salary term carries and that the sim-only
historical fit could not see. This is the single clearest argument for wiring
salary into the production scorer (where it is available): the projection term
gets the ranking right; the value term fixes the level on priced-up stars.

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
lightly estimated (two pairs, one thin) and kept gentle — refine it as more
same-slate pairs at different sizes accumulate.

---

## 6. Integration

`project_ownership` takes a `stage_d`-style pool (`Name`, `Pos`, and optionally
`Salary`, `Team`) plus the merged sim dict, and returns a pool-indexed
`%Drafted` Series obeying the invariant:

```python
from ownership_model import add_ownership_column
pool = add_ownership_column(pool, {**H, **P},
                            contest_size=field_size,
                            team_total=implied_by_team)   # both optional
```

The output is a drop-in replacement for the `Ownership` column that
`mlb_lineup_builder`, `field_simulator`, and `stage_d` already consume, so the
field simulator can be seeded from our own projections instead of an external
feed. (App wiring into the Setup tab is a follow-up.)

---

## 7. Limitations & next steps

- **Fit the cost/context betas.** Join DK salary and implied totals for the
  calibration days and let `value` / `team_total` fit instead of using priors.
  This is the highest-value next step (see the Ohtani residual).
- **More slates for `chalk_k`.** Two size pairs is thin; one is noisy.
- **Small-slate mapping.** Two of the six contests overlap their slate's sims
  at only ~0.66 (likely partial/early sub-slates); a slate-id tag on the sims
  would remove the name-overlap heuristic.
- **Per-position hitter betas.** Currently all hitters share one coefficient
  set; C/1B/etc. could differ with more data.
- **`ceil_shape` re-test.** It is zero here; revisit once `value` is in the
  model, in case cost unmasks an upside effect.

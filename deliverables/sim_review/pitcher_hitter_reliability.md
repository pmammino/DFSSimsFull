# Reliability & selection: pitchers vs hitters, and the portfolio salary-cap question

Backs the pitcher ace-tier change (`ACE_FLOOR_WEIGHT` / `Builder.ace_attr`) and
records two deliberate **non-changes** so they aren't re-litigated: hitters do
**not** get the floor-aware treatment, and the portfolio selector has **no**
salary-cap bias to fix.

Data: Aug-2026 DK contest **actual FPTS** joined to our sim projections
(`proj/ceiling/floor/std`) per player-slate — 109 pitcher rows, 968 hitter rows,
6 slates. (Salary is absent from the available data, so "price" itself is a
proxy throughout — see the ace-tier PR.)

---

## 1. Pitchers — reliability separates similar-ceiling arms (fix shipped)

Among arms with a genuine ace ceiling (p90 ≥ 25), ceiling does **not** separate
the true ace from a volatile mid arm — the **floor** does:

- Same-ceiling pairs (±1.5 p90): the higher-floor arm outscored its twin **57%**
  of the time, **+3.0 DK pts** on average.
- Low- vs high-floor thirds (similar ceilings): bust rate **26% → 12%**,
  ceiling-hit **9% → 29%**.
- `std` (volatility) correlates **negatively** with actual points (Spearman
  −0.03 overall, worse at the top) — a strong variance penalty was rejected
  because it barely helped ranking and *hurt* the top picks by stripping upside.

→ Shipped a **mild, ceiling-dominated** tie-breaker: rank the ace tier by
`exp(tilt·(z(ceiling) + 0.35·z(floor)))`. Stays 0.97 rank-correlated with
ceiling (upside preserved) yet favors the higher-floor arm on a same-ceiling
pair 84% of the time.

## 2. Hitters — the phenomenon does NOT exist (deliberately not changed)

The pitcher logic does not transfer, for structural reasons:

| signal | pitchers (Spearman vs actual) | hitters |
|---|---|---|
| ceiling | +0.283 | +0.159 |
| floor | +0.223 | **undefined — floor is 0 for 100% of hitters** |
| std | −0.03 | **+0.134 (variance is *good*)** |

- **No floor axis.** A hitter's p10 DK score is **0** for every hitter (they go
  0-fer often enough that the 10th percentile is zero), so there is no
  floor/reliability signal to differentiate on — unlike pitchers, whose floor
  ranges roughly −12 to +12.
- **Variance is desirable.** Hitter upside is HR/multi-hit driven; higher `std`
  is *positively* associated with actual points. Penalizing hitter variance
  would remove exactly the boom a GPP stack wants.
- **Nothing separates similar-ceiling bats.** At equal ceiling, higher `proj`
  wins only **51%** (coin flip, +1.1 pts); the only within-ceiling effect is a
  weak mean preference, not a distinct reliability dimension.

→ **Keep hitters on the pure-ceiling `Upside` weight.** A floor-aware nudge for
hitters would be counterproductive (no floor signal to use, and it would fight
the variance you want). This is intentional, not an oversight.

## 3. Portfolio selection — no salary-cap bias to fix

Traced the export path:

- Candidate **shortlist** is sorted by standalone simulated contest outcomes —
  `["Wins","Top10","Top100","AvgPlace"]` — not by salary or projected total.
- `select_portfolio_ev` optimizes **expected payout utility against a simulated
  field** (held-out sim slices); it carries **no salary or projection term**.

So the selector faithfully reflects the sims: a lineup ranks well because it
*places* well in the field sim, full stop. There is no place to "unbias" salary.
The expensive-ace squeeze is entirely upstream — candidate generation
(cap-efficiency, addressed by the ace tier) and whether the **sims** value the
ace's reliability — not portfolio selection.

**One minor, non-salary caveat:** because the shortlist pre-filters by
*standalone* placement before the EV optimizer runs, a lineup that is
individually mediocre but portfolio-valuable for leverage/decorrelation can be
cut before EV sees it. Enlarging `shortlist` mitigates this; it affects all
lineups equally, not expensive ones specifically. Flagged for awareness, not
fixed here.

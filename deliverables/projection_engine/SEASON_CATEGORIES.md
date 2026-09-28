# Season Category Projections — What Each One Needs

Target: project every requested category at a season level. This maps all 54 to
the machinery required, because they do **not** all need the same thing, and
three of the groups below cannot be produced by multiplying a rate by playing
time however good the rates get.

## Summary

| Group | Count | Blocker |
|---|---|---|
| **A** — rate × playing time | 26 | Playing-time model only |
| **B** — needs a new rate model | 6 | Small models; history is already fetched for 3 |
| **C** — needs game-state simulation | 13 | Cannot come from marginal rates. Sim exists |
| **D** — needs team context | 5 | **Done** (`team_wins.py` + `market_odds.py`) |
| **E** — fielding | 8 | Model built; needs a live fetch + innings at position |

**The single biggest unlock is playing time.** 26 categories are waiting only on
it, and it gates the season totals for most of group C too.

---

## Group A — rate × playing time

Every rate already exists in `out/*_pa_projections_*.csv`. These become real the
day `Proj_PA` / `Proj_IP` stop being NaN.

**Hitters:** PA, AB, H, 1B, 2B, 3B, HR, BB, HBP, SF, K, SB, CS, R, RBI
**Pitchers:** IP, BF, H, 1B, 2B, 3B, HR, BB, K, R, ER, HBP, GS

```
PA          = Proj_PA
AB          = PA × (1 − P_BB − P_HBP − P_SF − P_SH)
1B          = PA × P_1B          (same shape for 2B/3B/HR/BB/HBP/SF/K)
H           = 1B + 2B + 3B + HR
R, RBI      = PA × P_R,  PA × P_RBI     ← see the closure note below
SB, CS      = PA × P_SB,  PA × P_CS
BF          = IP × TBF_per_IP
ER          = IP × ERA / 9        R = IP × RA9 / 9
GS          = from the pitcher's role (rotation slot)
```

**R and RBI carry a closure debt.** They are currently free-standing player
rates: summed over a lineup they come to 1.07× and 1.03× of team runs against
targets of ~1.00 and ~0.88 (`season_engine.py` reports this every run). Once
playing time exists they must be re-derived as an **allocation of team runs**
across the batting order rather than scaled independently, or team totals won't
close. This is the one place in group A where PA alone isn't enough.

---

## Group B — needs a new rate model

Small, self-contained shrinkage models in the mould of `sb_model.py`. Three
already have their history fetched by `data_acquisition._fetch_statsapi_one` and
just need a projection; three need the history added too.

| Category | History fetched? | Notes |
|---|---|---|
| SH (sac bunt) | ✅ `SH` | Heavily role-dependent — pitchers and #8/#9 hitters. Nearly extinct post-2020, so shrink hard to a role-conditional mean. |
| IBB (hitters) | ✅ `IBB` | Not a hitter skill so much as a *reputation × context* effect: it tracks the batter's power **and** who hits behind him. Model as team-context-conditional, like R/RBI. |
| IBB (pitchers) | ✅ (same field) | Largely a manager decision. Shrink very hard. |
| Balk | ✅ `BK` | Extremely rare (~1 per 300 IP). Pure league-mean shrinkage; player signal is almost nil. |
| GIDP | ❌ | Needs `groundIntoDoublePlay`. Real skill signal: GB% × contact × speed, and the engine already has sprint speed and the batted-ball mix to build it from. |
| Pickoff | ❌ | Needs `pickoffs`. Handedness-dependent (LHP >> RHP). |

Add the three missing fields to the statsapi fetch — they are in the same
`stat` payload already being parsed, so it is a few lines, not a new source.

---

## Group C — needs game-state simulation

**These cannot be computed from marginal per-PA rates.** Each depends on the
base-out state, the score, or the *sequence* of events within a game. Knowing a
hitter's HR rate tells you nothing about how often he hits one with the bases
loaded, because it doesn't tell you how often the bases are loaded behind him.

The good news: `sim_proj.py` already simulates games with correlated
game/team/pitcher shocks, and already computes `win`, `qs`, `cg` and `nh` per
game. The season versions are **aggregate the existing sim over 162 games**, not
new modelling.

| Category | Why marginals fail | Route |
|---|---|---|
| Grand Slam | Needs P(HR **and** bases loaded) | Sim base-out state |
| Cycle | 1B+2B+3B+HR in the *same game* — a joint event | Sim, per-game |
| GWRBI | Needs the score margin | Closed form, see below |
| GIDP (exact) | Needs runner on first, < 2 outs | Sim (group B gives the rate) |
| Win / Loss | Needs team run support *in his starts* vs runs allowed | Sim + team context |
| Save / Blown Save | Needs a save situation to exist | Team pool × role |
| Hold | Same | Team pool × role |
| Quality Start | Needs IP ≥ 6 **and** ER ≤ 3 *jointly* | Sim per start |
| Complete Game | Needs the full-game leash | Sim |
| Shutout | CG **and** R = 0 | Sim |
| No-hitter | 27 outs, zero hits — a sequence event | Sim, low frequency |
| Perfect Game | No-hitter **and** no walks/HBP/errors | Sim; ~1 per 3-4 seasons league-wide |
| Catcher's Interference | Needs a catcher in the play | Sim + fielding (group E) |

A worked example of why the joint requirement bites: Quality Start. A starter
averaging 6.1 IP and a 3.60 ERA does *not* have `P(QS) = P(IP≥6) × P(ER≤3)` —
those are strongly negatively correlated within a start, because the innings
that push him past 6 are the ones where he's being hit. Multiplying the
marginals materially overstates QS. Only a per-start simulation gets it right.

### GWRBI — the simple formula asked for

Each win has **at most one** game-winning RBI, and not every win produces one
(some are won on a walk, error, wild pitch, or a first-lead run that wasn't
driven in). So the team pool is:

```
team_GWRBI ≈ team_wins × GW_SHARE            GW_SHARE ≈ 0.80
```

`team_wins` now comes from `team_wins.py`, blended with the market prior. Then
allocate across hitters by RBI share:

```
GWRBI_i = team_GWRBI × RBI_i / Σ RBI_j       (baseline)
```

That baseline is slightly **too flat**. Check the magnitude: an 85-win team has
~68 GWRBI against ~750 team RBI, so league-wide GWRBI/RBI ≈ 0.09, which hands a
100-RBI hitter about 9 — where real high-RBI hitters land nearer 12-15. GWRBI
concentrate in the middle of the order, because those RBI arrive with more
runners on and in closer games. One parameter fixes it:

```
GWRBI_i = team_GWRBI × RBI_i^α / Σ RBI_j^α   α ≈ 1.2
```

Both `GW_SHARE` and `α` want calibrating against real GWRBI totals if you have
them; the exponent is the more uncertain of the two. Note GWRBI is not an
official MLB stat (discontinued 1989), so a source may define it slightly
differently — confirm whether yours means "final lead" or "go-ahead in the
winning rally" before fitting.

---

## Group D — needs team context — **done**

| Category | Where |
|---|---|
| Team wins / losses | `team_wins.project_team_wins`, market-blended |
| Save opportunities | `team_wins.project_save_hold_opportunity` |
| Hold opportunities | same |
| Blown saves | save opportunities × (1 − `SAVE_CONVERSION_RATE`) |
| Run environment | `market_odds.apply_market_to_run_environment` |

League wins total exactly 2,430 and the save/hold pools close to their league
levels. What's still missing is the **player's share** of those pools, which is
a bullpen role (closer / setup / middle) — designed, not built.

The save/hold *levels* (`SAVES_PER_WIN`, `HOLDS_PER_WIN`, the elasticities) are
documented priors, not fitted values. `fit_opportunity_rates` replaces them once
`fetch_team_rpg` carries team SV/HLD.

---

## Group E — fielding

**Model built** (`fielding_model.py`), **fetch added** (`fetch_fielding_data`),
**exposure still missing**.

Assists · OF assists · Putouts · DP turned · Chances · Catcher's interference ·
Errors · Catcher caught-stealing

### The governing fact

Fielding counting stats are **mostly position and exposure, with a small skill
term on top**. A shortstop and a first baseman don't have different assist
*skill* so much as different assist *jobs*: a first baseman records a putout on
nearly every infield groundout and almost never an assist, and no amount of
talent changes that. So:

```
stat = (innings at position / 9) × rate_per_9(position) × skill × team
```

### Three identities enforced rather than hoped for

1. **Putouts close to 27 per 9 team innings.** Every out is a putout credited
   to exactly one fielder, so the position baselines must sum to 27 across an
   alignment. The table is normalized on load so a hand-edit can't break it.
2. **Chances = PO + A + E**, definitionally. Derived, never projected.
3. **Catcher putouts move *opposite* to everyone else** with the staff's
   strikeout rate. A catcher is credited with a putout on every strikeout, so
   ~8.6 of his ~9.0 putouts per 9 are Ks — while a high-K staff leaves fewer
   balls for the fielders. At 10.5 K/9 the catcher gets 1.22× and the fielders
   0.90×. This is the one team effect large enough to matter here, and it's
   why the catcher is handled separately.

### Two modelling choices worth knowing

**Errors are shrunk per *chance*, not per inning.** A chance is the actual
opportunity for an error, and it's what fielding percentage measures. Shrinking
in rate space let three errors in thirty innings move a projection 45% off
baseline on what is almost pure noise.

**The error prior is per position, derived from the baseline table.** A flat
league 0.016 (fielding ~.984) is the all-positions average, dominated by first
basemen and outfielders; a shortstop's rate per chance is nearer 0.025. Using
the flat value alongside position-specific baselines made the two contradict
each other, and a shortstop with a terrible error history came out *below* his
own position's baseline. Deriving it from the table means a wrong error
baseline surfaces as a wrong fielding percentage instead of hiding —
`EXPECTED_FIELDING_PCT` guards all nine positions to ±.004.

### Shrinkage, by how much real signal each stat carries

| Stat | Strength | Why |
|---|---|---|
| PO / A | 900 innings | Almost pure position and opportunity. A shortstop's assist total says more about his staff's groundball rate than about him. |
| DP | 700 innings | Needs a partner and a runner on first; mostly context. |
| E | 700 chances | The genuine skill term, but noisy season to season. |
| PB | 1200 innings | Rare enough to be mostly noise. |
| CI | 2000 innings | ~2 per league season. Effectively all prior. |
| **CS** | **250** | A real, reasonably stable catcher skill — the lightest shrinkage, and why catcher CS is worth projecting individually. |

### What's still needed

1. **A live fetch.** `fetch_fielding_data` is written against the documented
   statsapi shape but **not yet exercised against the live API** — statsapi is
   unreachable from the dev sandbox. Every field is read defensively and a
   missing one degrades to 0 rather than raising. The refresh workflow will
   validate it on the next run.
2. **Innings at position** — the exposure term, and the real blocker. Needs the
   playing-time model *and* a position assignment. `project_fielding` takes
   innings as an argument rather than inventing one, so the assumption stays
   visible. Note the unit trap: summing innings across positions counts each
   team inning nine times (~13,122, not ~1,458), and getting it wrong makes the
   putout identity read 3.0 instead of 27.0.
3. **Refit the baselines.** `fit_position_baselines` replaces the whole table
   from real history once the cache is populated. The cross-position ordering
   is the trustworthy part today; the absolute levels are provisional.

**Positions come free with this fetch.** The fielding group returns one row per
(player, position), so it supplies the `Pos` field the pipeline has never had —
the same gap that blocks depth charts and role assignment. It's the cheapest
place to pick positions up, since it arrives with a fetch we want anyway.

## Recommended order

1. **Playing time, Stage 3** — allocate each team's fixed PA/IP budget by
   projected talent rank. Unblocks all 26 of group A, needs no new data, and
   closes the team and league identities. Nothing else comes close on leverage.
2. **Re-derive R/RBI as an allocation of team runs** — the closure debt above.
   Cheap once playing time exists, and it fixes the two ratios the season engine
   currently reports as wrong.
3. **Bullpen roles** — turns the finished group-D pools into player SV/HLD/BS.
4. **Group B rate models** — small and independent; three need only a
   projection, three need fields added to a fetch that already runs.
5. **Season aggregation of the existing sim** — group C. Start with QS, W/L and
   CG, which matter most and are already computed per game; leave cycles and
   perfect games last, since they're rare enough that the sim needs many
   iterations for a stable estimate.
6. **Run the fielding fetch and refit the baselines** — group E. The model and
   the fetch exist; what's missing is a live run to validate the response shape
   and populate history, then `fit_position_baselines` to replace the
   provisional table. Positions arrive free with that same fetch, which also
   unblocks depth charts and role assignment.

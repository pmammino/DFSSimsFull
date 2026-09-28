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
| **E** — needs data we don't fetch | 8 | Fielding: no statsapi call, no positions |

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

## Group E — needs data we don't fetch

**No fielding data enters the pipeline at all.** `data_acquisition` requests the
`hitting` and `pitching` stat groups only; there is no `fielding` call, and zero
references to assists, putouts, errors or double plays anywhere in the repo.

Assists · OF assists · Putouts · DP turned · Chances · Catcher's interference ·
Errors · Catcher caught-stealing

Three things are needed, in order:

1. **A `fielding` stat-group fetch.** statsapi exposes all of these
   (`assists`, `putOuts`, `errors`, `doublePlays`, `chances`, `passedBall`,
   `catcherInterference`, `caughtStealing`) through the same endpoint already in
   use — add `"fielding"` alongside `"hitting"` and `"pitching"`.
2. **Positions.** Fielding stats are meaningless without them: a shortstop's
   assist rate and a first baseman's are different quantities, and putouts are
   dominated by position (a 1B records a putout on nearly every infield
   groundout). `primaryPosition` is a statsapi field the fetch simply doesn't
   request — the same gap that blocks depth charts and role assignment.
3. **Innings at position.** The exposure denominator. A rate per *team defensive
   inning at that position* is the only stable way to project these; per-game
   rates confound playing time with position changes.

Catcher caught-stealing is the exception worth separating: it's a genuine,
reasonably stable catcher skill, and the opportunity side is already partly
modelled — `sb_model.py` projects attempt rates against the league. Pairing the
two gives catcher CS without the full fielding build.

---

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
6. **Positions, then the fielding fetch** — group E. Positions are one field and
   unlock depth charts too, so they're worth doing early even though the
   fielding build behind them is the largest single item on this list.

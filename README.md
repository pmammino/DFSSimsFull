# MLB DFS Pipeline — Refresh → Projections → Correlated Sims

Complete daily pipeline. Pulls the latest data, builds handedness-specific
per-plate-appearance projections from **batted-ball data** (Statcast BIP +
XGBoost expected outcomes), then generates the slate's player projections and
**10,000 correlated DraftKings simulations** per player — matched to the actual
lineups and the handedness of each opposing pitcher.

```
   STAGE A           STAGE B                    STAGE C
 prefetch_bip  ->  run_pipeline.py  ->  run_slate.py
 (Statcast BIP)    (handedness per-PA)  (slate -> matchup -> 10k sims)
        \________________ refresh_and_run.py ________________/
```

## Install
```bash
pip install -r requirements.txt
```

## Run (one command)
```bash
# Full refresh to "as of today" + fetch slate feeds live
python refresh_and_run.py

# Full refresh, slate fed from saved feed files
python refresh_and_run.py --confirmed data/confirmed.xml \
                          --expected  data/expected.xml \
                          --vegas     data/vegas.json

# Fast re-run (projections already built today; lineups changed)
python refresh_and_run.py --skip-refresh \
                          --confirmed data/confirmed.xml \
                          --expected data/expected.xml --vegas data/vegas.json
```
Flags: `--skip-refresh`, `--skip-bip`, `--n-sims` (default 10000), `--seed`.

## IN-SEASON CONVENTION (read this)
The projection engine is walk-forward: it projects `TARGET_YEAR` from strictly
prior seasons. During the season set the target to **current_year + 1** so the
current season is the most-recent, highest-weighted prior. One place controls
it -- `refresh_and_run.py`:
```python
CURRENT_SEASON = 2026                # live season -- bump once per year
TARGET_YEAR    = CURRENT_SEASON + 1  # = 2027 (do NOT set to 2026 in-season)
```
Targeting the literal current year excludes the current season from the rate
panel and silently drops players who debuted this year (rookies). With +1,
current-season call-ups are projected off their current line and returning
players reflect current-year form via recency-decay weighting. Players below the
sample floor (`RATE_MIN_PA_ACTIVE = 25` PA/BF) fall back to league average and
are listed in the manifest's `missing_from_projection`.

## Files

### Top-level runners
- `refresh_and_run.py` -- one-command orchestrator (STAGE A->B->C)
- `prefetch_bip.py`    -- STAGE A: pull Statcast BIP seasons -> bip_inputs/
- `run_pipeline.py`    -- STAGE B: 12-step handedness per-PA projection engine
- `run_slate.py`       -- STAGE C: slate ingest -> matchup -> correlated sims, for EVERY game on the day (not one DK slate's window); a specific slate's players are filtered out of this full-day pool at lineup-build time (stage_d.build_pool)

### Projection engine (STAGE B)
pipeline_config.py, data_acquisition.py, rate_models.py, bip_imputation.py,
bip_outcomes.py, pa_aggregation.py, sb_model.py, runs_rbi_model.py,
park_factors.py, pitcher_outputs.py, splits_model.py, plus R converters
convert_rds_to_csv.R / convert_historical_lean.R (optional, only if you feed
.rds Statcast dumps instead of scraping).

### Integration layer (STAGE C)
- `slate_ingest.py` -- confirmed + expected lineup feeds, opener/primary, Vegas totals
- `slate_config.py` -- park factors, team-code map, DK scoring (slate side)
- `matchup.py`      -- pick hitter vL/vR split by opposing-pitcher hand + park; blend pitcher splits by lineup L/R share; league-avg fallback
- `sim_proj.py`     -- 10k correlated sims from projection per-PA vectors, scored under both DraftKings and Underdog scoring; opener/primary; TBF innings
- `validate.py`     -- correlation + stacking checks (PASS/CHECK)

## Outputs
`out/`:
- hitter_pa_projections_<TARGET_YEAR>.csv  (~600 hitters x 99 cols: neutral+park+vL/vR for 9 events, R/RBI/SB, slot, BatSide)
- pitcher_pa_projections_<TARGET_YEAR>.csv (~750 pitchers x 88 cols: + RA9/ERA/TBF/WP, role, PitchHand)

`deliverables/`:
- hitter_projections_<date>.csv   -- proj, floor_p25, median_p50, ceil_p75, p10/p90/p99, per-stat means, handedness, opp SP, lineup source
- pitcher_projections_<date>.csv  -- same percentiles + role, IP/BF/K/BB/ER/H/HR, win%, QS%
- hitter_dk_sims.npy / pitcher_dk_sims.npy  -- {name: array[N_SIMS]} DraftKings points, for every player on the day (not just one slate)
- hitter_ud_sims.npy / pitcher_ud_sims.npy  -- {name: array[N_SIMS]} Underdog points, same player universe
- hitter_stat_sims.npy            -- {name: {1B,2B,3B,HR,R,RBI,BB,HBP,K,SB,PA: array}}
- sim_manifest_<date>.json        -- provenance, loadings, realized correlations, stack check, validation, missing list

Reload:
```python
import numpy as np
hdk = np.load('deliverables/hitter_dk_sims.npy', allow_pickle=True).item()
pdk = np.load('deliverables/pitcher_dk_sims.npy', allow_pickle=True).item()
hud = np.load('deliverables/hitter_ud_sims.npy', allow_pickle=True).item()
pud = np.load('deliverables/pitcher_ud_sims.npy', allow_pickle=True).item()
```

## Correlation design
Per game, per sim: shared latents L_game (game environment) + L_team (each
lineup). Hitter events + R/RBI scale by exp(0.20*L_game + 0.50*L_team +
0.30*idio); the opposing pitcher's hits/HR/BB/ER use the same lineup shock, so
teammates move together and the opposing pitcher moves opposite. Sim index j is
aligned across all players in a game. Validated each run: teammate ~ +0.23,
unrelated ~ 0, hitter-vs-opposing-SP ~ -0.33.

## Performance
First full refresh ~6-8 min (BIP scrape + XGBoost + statsapi). --skip-refresh
re-runs slate + sims in ~30-60s.

## Freshness caveat
Projections are only as fresh as statsapi/Savant publications; some
environments lag box scores a few days. The manifest records the build.

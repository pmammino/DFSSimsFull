# Stage D — Lineups & Contest Sims (integration layer)

Adds a **Stage D** to the existing projection→sim pipeline: it consumes the
correlated DK sims and turns them into fields, candidate lineups, and simulated
contest results at any contest size.

```
 STAGE A          STAGE B            STAGE C                 STAGE D  (new)
prefetch_bip  ->  run_pipeline  ->  run_slate.py        ->  stage_d.py
(Statcast BIP)    (per-PA proj)     (10k correlated sims)   (fields + candidates
        \____________ refresh_and_run.py ___________/        + contest results)
                                                     \__ run_full.py __/
```

## One command, end to end

```bash
# build today's projections + sims, then simulate contests at three sizes
python3 run_full.py --dk slateplayers.csv --contest-sizes 1000 6000 20000

# reuse the sims already in deliverables/ (skip the heavy rebuild)
python3 run_full.py --from-deliverables --dk slateplayers.csv \
        --contest-sizes 1000 6000 --num-candidates 10000
```

`--dk` is the DraftKings salary/ownership export (the one external input the
projection pipeline doesn't itself produce). `slateplayers.csv` columns used:
`FullName, Team, Position, Salary, Ownership`.

## Stage D alone

```bash
python3 stage_d.py \
    --hitter-sims deliverables/hitter_dk_sims.npy \
    --pitcher-sims deliverables/pitcher_dk_sims.npy \
    --dk slateplayers.csv \
    --contest-sizes 1000 6000 20000 --num-candidates 10000 \
    --outdir stage_d_out
```

### What it does
1. **Pool build** — joins the DK file (salary, ownership, roster position, team)
   with the sim universe, keeping only simmed players so every lineup is
   scorable. Multi-position players are expanded; SP/RP map to `P`; every
   sim-confirmed pitcher is treated as a **starter** (the sims contain only the
   day's starters, which auto-corrects unreliable SP/RP labels). **Matchups are
   inferred from the sims** — a starter's scores anti-correlate hardest with the
   team he faces — so no team-code reconciliation between feeds is needed.
2. **Field** per contest size — ownership-weighted, with the contest-size model
   (chalk sharpens in small fields, flattens in large; light stack-shape tilt).
3. **Candidates** — one uniform, ownership-blind, starters-only pool (the search
   space), scored once and reused across all field sizes.
4. **Contest** — each candidate is inserted into the constant field per sim,
   ranked, removed; records Wins / Top10 / Top100 / AvgPlace.

### Outputs (`--outdir`)
- `candidates_<M>.csv` — the candidate lineups
- `field_<N>.csv` — the field at each contest size
- `candidate_results_<N>.csv` — every candidate ranked, per size

### Optional: DK upload of the best lineups
```bash
python3 stage_d.py ... \
    --select 20 --objective win --from-size 6000 \
    --player-cap 0.60 --team-cap 0.50 \
    --dk-template DKSalaries.csv
```
Greedily selects N lineups by objective (`win`, `top10`, or `top100`) subject to
per-player and primary-stack-team exposure caps, maps to DK player IDs from the
`DKSalaries` template, and writes a ready-to-upload `DK_upload_<N>.csv`
(`P,P,C,1B,2B,3B,SS,OF,OF,OF` header + ID rows). Caps default to 1.0 (off).

## Key knobs
`--contest-sizes` (any list), `--num-candidates`, `--medium` (baseline size where
chalk = projection), `--chalk-sensitivity`, `--stack-tilt`, `--select`,
`--objective`, `--player-cap`, `--team-cap`, seeds.

## New / added files
| file | role |
|---|---|
| `run_full.py` | Stage A–C → Stage D orchestrator |
| `stage_d.py` | Stage D: sims + DK file → fields/candidates/contests/upload |
| `mlb_lineup_builder.py` | rules engine (roster, $50K cap, ≤5 hitters/team, stack shapes, pitcher anti-correlation, uniform/weighted modes) |
| `field_simulator.py` | contest-size model helpers (ownership temperature, slot normalization, stack tilt) |
| `field_params.json` | empirical stack-shape distribution + rule probabilities |

Stages A–C are unchanged. Stage D needs only the sims + a DK file, so it can run
the moment Stage C finishes (or against any prior `deliverables/`).

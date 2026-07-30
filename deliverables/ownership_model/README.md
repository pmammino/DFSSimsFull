# Ownership model — reproducible calibration & validation

`OWNERSHIP_MODEL.md` is the write-up. To reproduce the fit and the validation
numbers in `validation_report.txt`:

1. Put the sim history `.npy` files in a `History/` dir, the contest-standings
   CSVs in a `Contests/` dir, and (optionally) the DailyFantasyFuel cheatsheets
   in a `DFF/` dir. Point the env vars at them:

   ```
   HIST_DIR=/path/to/History CONTEST_DIR=/path/to/Contests DFF_DIR=/path/to/DFF \
       python3 fit_ownership.py            # fit + leave-one-slate-out report
   HIST_DIR=... CONTEST_DIR=... DFF_DIR=... python3 fit_ownership.py --write  # + json
   ```

   Expected inputs:
   - `History/history_<date>_hitter_dk_sims.npy` and `..._pitcher_dk_sims.npy`
     (the rolling 4-day sim history; dict {player name -> 10k DK sim scores}).
   - `Contests/contest-standings-*.csv` (the dual-column DK export; the
     right-hand block Player / Roster Position / %Drafted / FPTS is the target).
   - `DFF/*DFF_MLB_cheatsheet_YYYYMMDD.csv` (optional; supplies `salary` and
     Vegas `implied_team_score` per player, enabling the `value` and
     `team_total` features. Several sheets for one day — e.g. a main and an
     `..._YYYYMMDD_1.csv` early slate — are merged.) Without `DFF_DIR`, only the
     sim-derived features are fitted.

2. `--write` refreshes `ownership_params.json` at the repo root, which
   `ownership_model.load_params()` reads at runtime.

## Files

| file | role |
|---|---|
| `../../ownership_model.py` | production scorer — `project_ownership(pool, sims, ...)` |
| `../../fit_ownership.py` | calibration + leave-one-slate-out validation |
| `../../ownership_history.py` | accumulate a compact per-slate training log |
| `../../ownership_params.json` | fitted coefficients (loaded at runtime) |
| `../../tests/test_ownership_model.py` | structural guarantees (invariant, cap, monotonicity, size chalk) |
| `../../tests/test_ownership_history.py` | log guarantees (dedup, label/slot attach, size) |
| `validation_report.txt` | saved output of the run above |

## Accumulating more slates (grow the training set cheaply)

Raw 10k-sim `.npy` files are heavy and prune to a 4-day window. Ownership
training instead wants *many* slates × a few summary numbers, so
`ownership_history.py` keeps a separate **append-only log** —
`ownership_history/features.csv`, ~a few dozen KB per slate (a season < ~10 MB)
— synced via `shared_store` independently of the sim pruning. Each build
auto-appends the slate's projection + Vegas context (hook in `app.ensure_fresh`);
salary and the actual-ownership label are added from the files you download:

```
# one command per slate — sims + DFF cheatsheet(s) + the settled contest:
python3 ownership_history.py ingest --date 2026-07-29 \
    --sims-dir History \
    --dff DFF/DFF_MLB_cheatsheet_20260729.csv \
    --dff DFF/DFF_MLB_cheatsheet_20260729_1.csv \
    --contest Contests/contest-standings-192897436.csv

python3 ownership_history.py show          # summarise the accumulated log
```

Then retrain from the accumulated log (no sims needed, scales to many slates):

```
python3 fit_ownership.py --history-csv ownership_history/features.csv --write
```

## Calibration data provenance

DraftKings MLB Classic contest standings, Jul 26–29 2026 (6 contests spanning
181 → 9,803 entries), matched to the sim history of the same slate by
player-name overlap. Salary and Vegas implied totals come from the
DailyFantasyFuel daily cheatsheets for those slates, which cover five of the
six contests at 71–99% of their players; the sixth (`192897375`, the Jul-29
Early slate) has a salary sheet but no sims in the history, so it is excluded
from training. See `OWNERSHIP_MODEL.md` §3–4 for the fitted coefficients and
the finding that cost/context add little beyond the projection here.

# Ownership model — reproducible calibration & validation

`OWNERSHIP_MODEL.md` is the write-up. To reproduce the fit and the validation
numbers in `validation_report.txt`:

1. Unzip the sim history so the `.npy` files live in a `History/` dir, and the
   contest-standings CSVs in a `Contests/` dir. Point the env vars at them:

   ```
   HIST_DIR=/path/to/History CONTEST_DIR=/path/to/Contests \
       python3 fit_ownership.py            # fit + leave-one-slate-out report
   HIST_DIR=... CONTEST_DIR=... python3 fit_ownership.py --write   # + save json
   ```

   Expected inputs:
   - `History/history_<date>_hitter_dk_sims.npy` and `..._pitcher_dk_sims.npy`
     (the rolling 4-day sim history; dict {player name -> 10k DK sim scores}).
   - `Contests/contest-standings-*.csv` (the dual-column DK export; the
     right-hand block Player / Roster Position / %Drafted / FPTS is the target).

2. `--write` refreshes `ownership_params.json` at the repo root, which
   `ownership_model.load_params()` reads at runtime.

## Files

| file | role |
|---|---|
| `../../ownership_model.py` | production scorer — `project_ownership(pool, sims, ...)` |
| `../../fit_ownership.py` | calibration + leave-one-slate-out validation |
| `../../ownership_params.json` | fitted coefficients (loaded at runtime) |
| `../../tests/test_ownership_model.py` | structural guarantees (invariant, cap, monotonicity, size chalk) |
| `validation_report.txt` | saved output of the run above |

## Calibration data provenance

DraftKings MLB Classic contest standings, Jul 26–29 2026 (6 contests spanning
181 → 9,803 entries), matched to the sim history of the same slate by
player-name overlap. Salary and Vegas totals were **not** available for these
specific days, so `value` and `team_total` carry domain priors; rerun with
those columns joined in to fit them (`FIT_FEATURES` in `fit_ownership.py`).

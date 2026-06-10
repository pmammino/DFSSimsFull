#!/usr/bin/env python3
"""
refresh_and_run.py — ONE command, full pipeline: data refresh → projections →
slate → matchup → 10k correlated DK sims → deliverables.

This is the top-level entry point. It chains:

  STAGE A  prefetch_bip.py        pull Statcast BIP seasons (imputation + XGB pool)
  STAGE B  run_pipeline.py        handedness per-PA projections (BIP/XGBoost engine)
  STAGE C  run_slate.py           slate ingest → matchup splits → correlated sims

Use this when you want everything refreshed to "as of today". For a faster
re-run that reuses cached projections (e.g. lineups changed but you already
built projections earlier today), pass --skip-refresh and it jumps straight to
STAGE C.

Examples
--------
# Full refresh from scratch, fetch slate feeds live:
python refresh_and_run.py

# Full refresh, but feed the slate from saved files:
python refresh_and_run.py --confirmed data/confirmed.xml \
                          --expected  data/expected.xml \
                          --vegas     data/vegas.json

# Lineups-only re-run (projections already built today):
python refresh_and_run.py --skip-refresh --confirmed data/confirmed.xml \
                          --expected data/expected.xml --vegas data/vegas.json

Important in-season convention
------------------------------
TARGET_YEAR must be current_year + 1 so the current season is included as the
most-recent (highest-weighted) prior. CURRENT_SEASON below is the live season;
the engine target is CURRENT_SEASON + 1. Update CURRENT_SEASON once per year.
"""
import argparse, subprocess, sys, os, datetime

CURRENT_SEASON = 2026               # the live MLB season
TARGET_YEAR    = CURRENT_SEASON + 1 # walk-forward "as-of-now" target
BIP_YEARS      = [CURRENT_SEASON - 2, CURRENT_SEASON - 1, CURRENT_SEASON]  # imputation + XGB pool
PY = sys.executable


def run(cmd, label):
    print(f"\n{'='*72}\n{label}\n{'='*72}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-refresh', action='store_true',
                    help='skip BIP scrape + projection rebuild; reuse cached projections')
    ap.add_argument('--skip-bip', action='store_true',
                    help='skip BIP scrape only (reuse bip_inputs/), still rebuild projections')
    ap.add_argument('--confirmed'); ap.add_argument('--expected'); ap.add_argument('--vegas')
    ap.add_argument('--n-sims', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20260610)
    args = ap.parse_args()

    os.makedirs('bip_inputs', exist_ok=True)

    if not args.skip_refresh:
        # STAGE A — refresh Statcast BIP seasons
        if not args.skip_bip:
            for yr in BIP_YEARS:
                run([PY, 'prefetch_bip.py', str(yr)],
                    f"STAGE A: Statcast BIP refresh — {yr}")
        # STAGE B — rebuild handedness per-PA projections (force = fresh statsapi pulls)
        run([PY, 'run_pipeline.py', '--target-year', str(TARGET_YEAR),
             '--bip-dir', 'bip_inputs', '--skip-2026-scrape',
             '--output-dir', 'out', '--force'],
            f"STAGE B: Handedness per-PA projections (target {TARGET_YEAR})")

    # STAGE C — slate → matchup → correlated sims
    cmd = [PY, 'run_slate.py', '--n-sims', str(args.n_sims), '--seed', str(args.seed)]
    for flag in ('confirmed', 'expected', 'vegas'):
        v = getattr(args, flag)
        if v:
            cmd += [f'--{flag}', v]
    run(cmd, "STAGE C: Slate ingest → matchup splits → correlated sims")

    print(f"\nAll done — deliverables/ holds the projections + sim arrays.")


if __name__ == '__main__':
    main()

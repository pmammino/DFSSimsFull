#!/usr/bin/env python3
"""
run_full.py  —  end-to-end orchestrator:  Stage A-C (projections+sims) -> Stage D (lineups+contests)
=====================================================================================================
One command from raw data to candidate lineups + contest results.

  STAGE A-C : refresh_and_run.py  -> deliverables/{hitter,pitcher}_dk_sims.npy + projections + manifest
  STAGE D   : stage_d.py          -> fields (configurable sizes) + candidates + contest results
                                      (+ optional DK upload)

The only Stage-D-specific input the projection pipeline doesn't produce is the
DraftKings salary/ownership file (an external DK export). Pass it with --dk.

Examples
--------
  # full run: build today's projections+sims, then contests at 3 sizes
  python3 run_full.py --dk slateplayers.csv --contest-sizes 1000 6000 20000

  # skip the heavy projection/sim rebuild; use the sims already in deliverables/
  python3 run_full.py --from-deliverables --dk slateplayers.csv \
          --contest-sizes 1000 6000 --num-candidates 10000 \
          --select 20 --objective win --player-cap 0.6 --team-cap 0.5 \
          --dk-template DKSalaries.csv
"""
import argparse, glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def latest(pattern):
    fs = sorted(glob.glob(os.path.join(HERE, pattern)))
    return fs[-1] if fs else None

def main():
    ap = argparse.ArgumentParser(description="Full pipeline: projections+sims -> contests.")
    ap.add_argument('--dk', required=True, help='DraftKings salary/ownership file (slateplayers.csv)')
    ap.add_argument('--from-deliverables', action='store_true',
                    help='skip Stage A-C; use existing deliverables/*.npy')
    ap.add_argument('--contest-sizes', nargs='+', type=int, default=[1000, 6000, 20000])
    ap.add_argument('--num-candidates', type=int, default=10000)
    ap.add_argument('--outdir', default='stage_d_out')
    # pass-throughs to refresh_and_run (Stage A-C)
    ap.add_argument('--confirmed'); ap.add_argument('--expected'); ap.add_argument('--vegas')
    ap.add_argument('--n-sims', type=int, default=10000); ap.add_argument('--seed', type=int)
    ap.add_argument('--skip-refresh', action='store_true'); ap.add_argument('--skip-bip', action='store_true')
    # pass-throughs to stage_d selection
    ap.add_argument('--select', type=int, default=0)
    ap.add_argument('--objective', choices=['top100','top10','win'], default='top100')
    ap.add_argument('--from-size', type=int)
    ap.add_argument('--player-cap', type=float, default=1.0)
    ap.add_argument('--team-cap', type=float, default=1.0)
    ap.add_argument('--dk-template')
    a = ap.parse_args()

    # ---- STAGE A-C ----
    if not a.from_deliverables:
        cmd = [sys.executable, os.path.join(HERE, 'refresh_and_run.py'), '--n-sims', str(a.n_sims)]
        for flag in ('confirmed','expected','vegas'):
            if getattr(a, flag): cmd += [f'--{flag}', getattr(a, flag)]
        if a.seed is not None: cmd += ['--seed', str(a.seed)]
        if a.skip_refresh: cmd += ['--skip-refresh']
        if a.skip_bip: cmd += ['--skip-bip']
        print(">>> STAGE A-C:", ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=HERE)
    else:
        print(">>> STAGE A-C skipped (--from-deliverables)")

    hsim = latest('deliverables/hitter_dk_sims*.npy')
    psim = latest('deliverables/pitcher_dk_sims*.npy')
    if not hsim or not psim:
        sys.exit("ERROR: no sim .npy found in deliverables/ — run Stage A-C first.")
    print(f">>> using sims: {os.path.basename(hsim)}, {os.path.basename(psim)}")

    # ---- STAGE D ----
    cmd = [sys.executable, os.path.join(HERE, 'stage_d.py'),
           '--hitter-sims', hsim, '--pitcher-sims', psim, '--dk', a.dk,
           '--contest-sizes', *map(str, a.contest_sizes),
           '--num-candidates', str(a.num_candidates), '--outdir', a.outdir]
    if a.select:
        cmd += ['--select', str(a.select), '--objective', a.objective,
                '--player-cap', str(a.player_cap), '--team-cap', str(a.team_cap)]
        if a.from_size: cmd += ['--from-size', str(a.from_size)]
        if a.dk_template: cmd += ['--dk-template', a.dk_template]
    print(">>> STAGE D:", ' '.join(os.path.basename(c) if c.endswith('.npy') else c for c in cmd))
    subprocess.run(cmd, check=True, cwd=HERE)
    print(f">>> DONE. Outputs in {a.outdir}/")

if __name__ == '__main__':
    main()

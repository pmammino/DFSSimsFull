"""
run_slate.py — full daily pipeline, projection-driven.

Chain:
  1. PROJECTIONS  run_pipeline.py (handedness-split per-PA from BIP/XGBoost)
                  -> out/{hitter,pitcher}_pa_projections_<year>.csv
     (skipped automatically if today's projection CSVs already exist; use
      --refresh-proj to force a rebuild from fresh Statcast/statsapi data)
  2. SLATE        slate_ingest.build_slate(): confirmed + expected lineups,
                  opener/primary, FantasyLabs Vegas implied totals
  3. MATCHUP      matchup.build_matchup_inputs(): pick each hitter's vL/vR split
                  by opposing pitcher hand, park-adjust; blend pitcher splits by
                  the lineup's L/R share
  4. SIMULATE     sim_proj.simulate(): 10k correlated DK sims/player
  5. VALIDATE     correlation + stacking checks
  6. WRITE        projection CSVs (floor/median/ceiling), per-sim .npy, manifest

Usage:
  python run_slate.py --confirmed data/confirmed.xml --expected data/expected.xml \
                      --vegas data/vegas.json
  python run_slate.py            # fetch feeds live (if reachable)
  python run_slate.py --refresh-proj   # rebuild projections from fresh data first
"""
import argparse, json, os, subprocess, sys
import numpy as np

import slate_ingest
import matchup as M
import sim_proj
import validate

TARGET_YEAR = 2027
OUT_DIR = "out"
DELIV_DIR = "deliverables"
HFIELDS = ['player','pos','slot','bat','team','side','game','datetime','lineup_source',
           'opp_sp','team_total','proj','floor_p25','median_p50','ceil_p75','p10','p90',
           'ceiling_p99','std','mean_hr','mean_r','mean_rbi','mean_bb','mean_sb','p_2x','p_30']
PFIELDS = ['player','pos','role','team','side','game','datetime','opp','opp_total','proj',
           'floor_p25','median_p50','ceil_p75','p10','p90','ceiling_p99','std','mean_ip',
           'mean_bf','mean_k','mean_bb','mean_er','mean_h','mean_hr','win_pct','p_qs','p_30']


def ensure_projections(refresh):
    hp = os.path.join(OUT_DIR, f"hitter_pa_projections_{TARGET_YEAR}.csv")
    pp = os.path.join(OUT_DIR, f"pitcher_pa_projections_{TARGET_YEAR}.csv")
    if refresh or not (os.path.exists(hp) and os.path.exists(pp)):
        print("[1/6] Building handedness per-PA projections (run_pipeline.py)...")
        cmd = [sys.executable, "run_pipeline.py", "--target-year", str(TARGET_YEAR),
               "--bip-dir", "bip_inputs", "--skip-2026-scrape", "--output-dir", OUT_DIR]
        if refresh:
            cmd.append("--force")
        subprocess.run(cmd, check=True)
    else:
        print("[1/6] Reusing existing projections (use --refresh-proj to rebuild).")
    return hp, pp


def write_csv(path, rows, fields):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirmed'); ap.add_argument('--expected'); ap.add_argument('--vegas')
    ap.add_argument('--refresh-proj', action='store_true')
    ap.add_argument('--n-sims', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20260610)
    ap.add_argument('--team-totals', help='JSON {team_code: implied_runs} to '
                    'override the slate Vegas team totals (scales that team\'s '
                    'offense by edited/original).')
    args = ap.parse_args()

    os.makedirs(DELIV_DIR, exist_ok=True)

    # 1) projections
    hp, pp = ensure_projections(args.refresh_proj)
    hproj, pproj = M.load_projections(OUT_DIR, TARGET_YEAR)
    print(f"   projections: {len(hproj)} hitters, {len(pproj)} pitchers")

    # 2) slate
    print("[2/6] Ingesting slate (confirmed + expected + Vegas)...")
    cx = open(args.confirmed).read() if args.confirmed else None
    ex = open(args.expected).read() if args.expected else None
    vj = open(args.vegas).read() if args.vegas else None
    slate = slate_ingest.build_slate(cx, ex, vegas_json=vj)
    print(f"   date={slate['date']} games={len(slate['games'])}")

    # 2b) optional user override of team Vegas totals -> rescale team offense
    if args.team_totals:
        ov = json.load(open(args.team_totals))
        n_ov = 0
        for g in slate['games'].values():
            g.setdefault('total_scale', {'away': 1.0, 'home': 1.0})
            for side in ('away', 'home'):
                team = g[side]
                if team in ov:
                    orig = float(g['implied'].get(side) or 0.0) or 4.4
                    new = float(ov[team])
                    g['implied'][side] = new
                    g['total_scale'][side] = max(0.2, min(2.5, new / orig))
                    n_ov += 1
            g['implied']['total'] = (float(g['implied'].get('away') or 0.0)
                                     + float(g['implied'].get('home') or 0.0))
        print(f"   applied team-total overrides for {n_ov} team(s)")

    # 3) matchup
    print("[3/6] Building matchup-specific per-PA inputs (vL/vR + park)...")
    matchup = M.build_matchup_inputs(slate, hproj, pproj)
    miss = matchup.get('missing', {})
    if miss.get('hitters'): print(f"   hitters w/o projection ({len(miss['hitters'])}): {miss['hitters'][:8]}{'...' if len(miss['hitters'])>8 else ''}")
    if miss.get('pitchers'): print(f"   pitchers w/o projection ({len(miss['pitchers'])}): {miss['pitchers'][:8]}{'...' if len(miss['pitchers'])>8 else ''}")

    # 4) simulate
    print(f"[4/6] Simulating {args.n_sims} correlated sims/player...")
    hitter_dk, pitcher_dk, hitter_stat, hrows, prows, _ = sim_proj.simulate(
        matchup, n_sims=args.n_sims, seed=args.seed)
    print(f"   hitters={len(hitter_dk)} pitchers={len(pitcher_dk)}")

    # 5) validate
    print("[5/6] Validating correlation structure...")
    rep, sb, ok = validate.print_report(hitter_dk, pitcher_dk, hrows)

    # 6) write
    print("[6/6] Writing deliverables...")
    hrows.sort(key=lambda r: r['proj'], reverse=True)
    prows.sort(key=lambda r: r['proj'], reverse=True)
    stamp = slate['date'] or 'slate'
    write_csv(os.path.join(DELIV_DIR, f"hitter_projections_{stamp}.csv"), hrows, HFIELDS)
    write_csv(os.path.join(DELIV_DIR, f"pitcher_projections_{stamp}.csv"), prows, PFIELDS)
    np.save(os.path.join(DELIV_DIR, 'hitter_dk_sims.npy'), hitter_dk, allow_pickle=True)
    np.save(os.path.join(DELIV_DIR, 'pitcher_dk_sims.npy'), pitcher_dk, allow_pickle=True)
    np.save(os.path.join(DELIV_DIR, 'hitter_stat_sims.npy'), hitter_stat, allow_pickle=True)
    opener_games = [f"{gid}:{s}" for gid, g in slate['games'].items()
                    for s in ('away','home') if g['pitchers'][s].get('opener')]
    manifest = {'date': slate['date'], 'n_sims': args.n_sims, 'seed': args.seed,
                'version': 'projection_driven_v1', 'target_year': TARGET_YEAR,
                'projection_source': 'handedness per-PA (BIP/XGBoost) + park + vL/vR splits',
                'n_games': len(slate['games']),
                'opener_primary': opener_games,
                'realized_correlations': rep, 'stack_check': sb, 'validation_pass': bool(ok),
                'missing_from_projection': miss,
                'hitters': sorted(hitter_dk), 'pitchers': sorted(pitcher_dk)}
    json.dump(manifest, open(os.path.join(DELIV_DIR, f"sim_manifest_{stamp}.json"), 'w'), indent=2)
    print(f"\nDone. Deliverables in {DELIV_DIR}/:")
    for f in sorted(os.listdir(DELIV_DIR)):
        print(f"  {f}")


if __name__ == '__main__':
    main()

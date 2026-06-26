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
# Implied total that maps to a neutral 1.0 offense scale (MLB league-average
# per-team runs). A team's Vegas total is scaled against this fixed anchor, so
# the run environment powers the sim consistently across slates.
LEAGUE_AVG_RUNS = 4.2
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


def apply_vegas_scaling(slate, overrides=None, baseline=0.0, lo=0.5, hi=2.0):
    """Scale every team's offense by its Vegas implied total relative to the
    slate average, so the day's run environment powers the sim even with no
    manual edits. `total_scale[side] = clip(implied/baseline, lo, hi)`.

    - `overrides`: optional {team_code: implied_runs} that REPLACE a team's
      implied total before scaling (a user edit). The baseline is taken from the
      LOADED (pre-override) totals, so an edit moves only the edited teams, not
      the reference everyone is measured against.
    - `baseline`: >0 pins an absolute league-average reference; 0 (default) uses
      the slate's own average implied total (the typical team -> ~1.0, unchanged).

    Mutates `slate` in place (sets g['total_scale'] and applies overrides to
    g['implied']). Returns (baseline_used, n_overridden_sides).
    """
    overrides = overrides or {}
    games = slate['games'].values()
    loaded = [float(g['implied'].get(s) or 0.0)
              for g in games for s in ('away', 'home')
              if float(g['implied'].get(s) or 0.0) > 0]
    base = baseline if baseline > 0 else (sum(loaded) / len(loaded) if loaded else 4.4)

    for g in games:
        for side in ('away', 'home'):
            if g[side] in overrides:
                g['implied'][side] = float(overrides[g[side]])
        g['implied']['total'] = (float(g['implied'].get('away') or 0.0)
                                 + float(g['implied'].get('home') or 0.0))

    n_ov = 0
    for g in games:
        g.setdefault('total_scale', {'away': 1.0, 'home': 1.0})
        for side in ('away', 'home'):
            imp = float(g['implied'].get(side) or 0.0) or base
            g['total_scale'][side] = max(lo, min(hi, imp / base))
            if g[side] in overrides:
                n_ov += 1
    return base, n_ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirmed'); ap.add_argument('--expected'); ap.add_argument('--vegas')
    ap.add_argument('--date', help='Rebuild a historical slate for this date '
                    '(YYYY-MM-DD): fetch that day\'s lineups/matchups/Vegas from '
                    'the feeds and simulate it as if it were that slate.')
    ap.add_argument('--refresh-proj', action='store_true')
    ap.add_argument('--n-sims', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20260610)
    ap.add_argument('--team-totals', help='JSON {team_code: implied_runs} to '
                    'override the slate Vegas team totals (replaces that team\'s '
                    'implied total before the Vegas-vs-slate-average scaling).')
    ap.add_argument('--total-baseline', type=float, default=LEAGUE_AVG_RUNS,
                    help=f'Implied-total that maps to a neutral 1.0 offense scale '
                         f'(default {LEAGUE_AVG_RUNS}, the league average). Each '
                         f'team\'s Vegas total is scaled against this fixed anchor; '
                         f'pass 0 to use the slate average instead.')
    args = ap.parse_args()

    os.makedirs(DELIV_DIR, exist_ok=True)

    # 1) projections
    hp, pp = ensure_projections(args.refresh_proj)
    hproj, pproj = M.load_projections(OUT_DIR, TARGET_YEAR)
    print(f"   projections: {len(hproj)} hitters, {len(pproj)} pitchers")

    # 2) slate
    if args.date:
        print(f"[2/6] Ingesting HISTORICAL slate for {args.date} "
              "(confirmed + expected + Vegas)...")
    else:
        print("[2/6] Ingesting slate (confirmed + expected + Vegas)...")
    cx = open(args.confirmed).read() if args.confirmed else None
    ex = open(args.expected).read() if args.expected else None
    vj = open(args.vegas).read() if args.vegas else None
    slate = slate_ingest.build_slate(cx, ex, vegas_json=vj, date=args.date)
    print(f"   date={slate['date']} games={len(slate['games'])}")
    if args.date and not slate['games']:
        print(f"   WARNING: no games returned for {args.date} — the feed may not "
              "have historical lineups for that date.", file=sys.stderr)

    # 2b) drive every team's offense from its Vegas implied total (see
    #     apply_vegas_scaling) so the day's run environment powers the sim even
    #     with NO manual edits.
    ov = json.load(open(args.team_totals)) if args.team_totals else None
    baseline, n_ov = apply_vegas_scaling(slate, ov, args.total_baseline)
    print(f"   offense scaled by Vegas vs league-average baseline {baseline:.2f} "
          f"({n_ov} user-overridden)")

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

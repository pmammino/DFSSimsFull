#!/usr/bin/env python3
"""
field_simulator.py
==================
Takes a projected-ownership slate pool and simulates a full contest FIELD of a
given size, using the rules learned from the contest-standings analysis plus a
contest-size model.

CONTEST-SIZE MODEL
------------------
The projected ownership is assumed to describe a MEDIUM contest (default 6,000
entries). Two size effects are applied relative to that baseline:

1. OWNERSHIP TEMPERATURE  (the chalk knob — user-specified behavior)
   Within each position group, ownership is re-shaped as  own^beta  and
   renormalized so the per-slot total is preserved (100% per single slot, 300%
   OF, 200% P — the invariant verified in the standings data).
       beta > 1  -> chalk concentrates (small contests lean into chalk)
       beta = 1  -> projections unchanged (medium)
       beta < 1  -> ownership flattens toward uniform (large contests)
   beta(N) = 1 - k * log10(N / N_medium)        [k = chalk_sensitivity]
   This also propagates to stack-team selection and pitcher selection, since
   both are ownership-weighted, so chalk *teams* concentrate in small fields too.

2. STACK-SHAPE TILT  (structural consensus — from the slate analysis)
   Larger contests consolidated onto the optimal 5-man stack (57.9% in the
   21k-entry contest vs ~52% in the smaller ones). The empirical shape
   distribution is mildly tilted toward 5-primary shapes for large fields and
   away for small ones:
       weight *= (1 + s * log10(N/N_medium))   for 5-primary shapes (inverse for <5)
   This is a deliberately light adjustment (only 3 slates were observed).

Everything else (roster, <$50k cap, <=5 hitters/team, avoid-hitter-vs-own-
pitcher, etc.) is inherited unchanged from mlb_lineup_builder.

USAGE
-----
  python3 field_simulator.py --pool slate_pool.csv --sizes 1000 6000 20000 \
          --medium 6000 --outdir fields
"""
import argparse, json, os, math
from collections import Counter
import numpy as np
import pandas as pd

from mlb_lineup_builder import Pool, Builder, DEFAULT_PARAMS, SALARY_CAP

POS_SLOT_SUM = {'P':2,'C':1,'1B':1,'2B':1,'3B':1,'SS':1,'OF':3}


def normalize_to_slots(df, rp_share=0.15):
    """Rescale ownership within each position so it sums to that position's DK
    slot count x100% (P->200, OF->300, others->100). Raw projections are often
    over-subscribed (e.g. SP+RP can exceed 200%); this makes targets feasible
    while preserving each player's relative ownership.

    Pitchers get special handling: real lineups are SP-dominant, so if a 'Role'
    column distinguishes SP/RP, the 200% pitcher allocation is split as
    (1-rp_share) to starters and rp_share to relievers, rather than letting an
    over-projected reliever pool dominate the two P slots."""
    out = df.copy()
    out['Ownership'] = out['Ownership'].astype(float).clip(lower=0.001)
    for pos, idx in out.groupby('Pos').groups.items():
        target = POS_SLOT_SUM.get(pos, 1) * 100.0
        if pos == 'P' and 'Role' in out.columns:
            sp_idx = [i for i in idx if out.loc[i,'Role'] == 'SP']
            rp_idx = [i for i in idx if out.loc[i,'Role'] != 'SP']
            for sub_idx, sub_tgt in [(sp_idx, target*(1-rp_share)),
                                     (rp_idx, target*rp_share)]:
                if sub_idx:
                    s = out.loc[sub_idx, 'Ownership']
                    out.loc[sub_idx, 'Ownership'] = s / s.sum() * sub_tgt
        else:
            s = out.loc[idx, 'Ownership']
            out.loc[idx, 'Ownership'] = s / s.sum() * target
    return out


def beta_for_size(n, n_med, k):
    return 1.0 - k * math.log10(max(n,1) / n_med)


def adjust_ownership(df, beta):
    """Re-shape ownership as own^beta, renormalized within each position so the
    per-slot ownership total is preserved. Multi-position rows share a name; we
    transform on the row's listed position group (consistent with how the field
    fills that slot)."""
    out = df.copy()
    out['Ownership'] = out['Ownership'].astype(float).clip(lower=0.001)
    for pos, idx in out.groupby('Pos').groups.items():
        sub = out.loc[idx, 'Ownership']
        reshaped = sub ** beta
        # preserve the original group total (keeps slot-sum invariant)
        reshaped = reshaped / reshaped.sum() * sub.sum()
        out.loc[idx, 'Ownership'] = reshaped
    return out


def tilt_structures(structs, n, n_med, s):
    factor = 1.0 + s * math.log10(max(n,1) / n_med)
    out = []
    for shape, w in structs:
        primary = max(shape)
        if primary >= 5:
            w = w * factor
        elif primary <= 3:
            w = w / factor
        out.append((shape, max(w, 1e-9)))
    tot = sum(w for _, w in out)
    return [(sh, w / tot) for sh, w in out]


def summarize(lineups, pool_df):
    n = len(lineups)
    prim = Counter(); maxt = Counter(); sals = []; vsp = 0
    player_use = Counter()
    for lu in lineups:
        sizes = sorted(lu['teams'].values(), reverse=True)
        prim[sizes[0]] += 1
        allc = Counter(lu['teams'])
        for p in lu['players']:
            if p.Pos == 'P':
                allc[p.Team] += 1
            player_use[p.Name] += 1
        maxt[max(allc.values())] += 1
        sals.append(lu['salary'])
        hteams = set(lu['teams'])
        if any(p.Pos == 'P' and p.Opp in hteams for p in lu['players']):
            vsp += 1
    # realized ownership of the projection's top-10 chalk (by input ownership)
    top10 = (pool_df.drop_duplicates('Name').nlargest(10, 'Ownership')['Name'].tolist())
    top10_realized = np.mean([100*player_use[nm]/n for nm in top10])
    # concentration: share of all roster spots taken by the top 20 players
    total_spots = sum(player_use.values())
    top20_share = sum(c for _, c in player_use.most_common(20)) / total_spots * 100
    return {
        'n': n,
        'prim5': 100*prim[5]/n, 'prim4': 100*prim[4]/n,
        'prim3': 100*prim[3]/n, 'prim2': 100*prim[2]/n,
        'six_body': 100*maxt.get(6,0)/n, 'seven_body': maxt.get(7,0),
        'vs_own_p': 100*vsp/n,
        'sal_mean': np.mean(sals), 'sal_max': max(sals),
        'top10_chalk_realized': top10_realized,
        'top20_share': top20_share,
        'uniques': len(player_use),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[1000,6000,20000])
    ap.add_argument('--medium', type=int, default=6000)
    ap.add_argument('--chalk_sensitivity', type=float, default=0.35)
    ap.add_argument('--stack_tilt', type=float, default=0.15)
    ap.add_argument('--no_normalize', action='store_true',
                    help='skip per-position normalization to slot counts')
    ap.add_argument('--rp_share', type=float, default=0.15,
                    help='target share of the 2 pitcher slots filled by relievers')
    ap.add_argument('--params', default='field_params.json')
    ap.add_argument('--outdir', default='fields')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    params = json.load(open(a.params)) if os.path.exists(a.params) else DEFAULT_PARAMS
    base_df = pd.read_csv(a.pool)
    # normalize raw projections to DK slot counts (feasible, relative-preserving)
    norm_df = normalize_to_slots(base_df, a.rp_share) if not a.no_normalize else base_df.copy()

    summaries = {}
    for n in a.sizes:
        beta = beta_for_size(n, a.medium, a.chalk_sensitivity)
        adj_df = adjust_ownership(norm_df, beta)
        tilted = tilt_structures([(tuple(s), w) for s, w in params['stack_structures']],
                                 n, a.medium, a.stack_tilt)
        p = dict(params); p['stack_structures'] = [(list(s), w) for s, w in tilted]
        pool = Pool(adj_df)
        b = Builder(pool, p, seed=a.seed + n)

        lineups, fails = [], 0
        while len(lineups) < n and fails < n*20 + 500:
            lu = b.build_one()
            if lu is None:
                fails += 1; continue
            lineups.append(lu)

        # write field
        rows = []
        for i, lu in enumerate(lineups, 1):
            row = {'Lineup': i, 'Salary': lu['salary'],
                   'Stack': '-'.join(map(str, sorted(lu['teams'].values(), reverse=True)))}
            for j, pl in enumerate(lu['players']):
                slot = ['P1','P2','C','1B','2B','3B','SS','OF1','OF2','OF3'][j]
                row[slot] = f"{pl.Name} ({pl.Team})"
            rows.append(row)
        path = os.path.join(a.outdir, f'field_{n}.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        s = summarize(lineups, base_df)
        s['beta'] = beta; s['fails'] = fails; s['path'] = path
        # realized vs (normalized) projected for the medium field's top chalk
        if n == a.medium:
            use = Counter()
            for lu in lineups:
                for pl in lu['players']:
                    use[pl.Name] += 1
            tgt = norm_df.drop_duplicates('Name').nlargest(8, 'Ownership')
            s['calib'] = [(r.Name, r.Ownership, 100*use[r.Name]/len(lineups))
                          for r in tgt.itertuples()]
        summaries[n] = s
        print(f"[{n:>6} entries] beta={beta:.3f}  built={len(lineups)}  fails={fails}  -> {path}")

    # comparison table
    print("\n" + "="*74)
    print(f"{'metric':30}" + "".join(f"{n:>14}" for n in a.sizes))
    print("-"*74)
    def row(label, key, fmt):
        print(f"{label:30}" + "".join(f"{fmt(summaries[n][key]):>14}" for n in a.sizes))
    row("ownership beta (chalk knob)", 'beta', lambda v: f"{v:.3f}")
    print("-"*74)
    row("primary stack = 5", 'prim5', lambda v: f"{v:.1f}%")
    row("primary stack = 4", 'prim4', lambda v: f"{v:.1f}%")
    row("primary stack = 3", 'prim3', lambda v: f"{v:.1f}%")
    print("-"*74)
    row("top-10 chalk avg realized own", 'top10_chalk_realized', lambda v: f"{v:.1f}%")
    row("top-20 players' roster share", 'top20_share', lambda v: f"{v:.1f}%")
    row("unique players used", 'uniques', lambda v: f"{v}")
    print("-"*74)
    row("salary mean", 'sal_mean', lambda v: f"${v:,.0f}")
    row("max bodies/team = 6", 'six_body', lambda v: f"{v:.1f}%")
    row("7 from one team (illegal)", 'seven_body', lambda v: f"{v}")
    row("hitter vs own pitcher", 'vs_own_p', lambda v: f"{v:.1f}%")
    print("="*74)

    if a.medium in summaries and 'calib' in summaries[a.medium]:
        print(f"\nMedium-field ({a.medium}) calibration — realized vs normalized projection:")
        print(f"  {'player':22}{'proj(norm)':>12}{'realized':>12}")
        for name, proj, real in summaries[a.medium]['calib']:
            print(f"  {name:22}{proj:>11.1f}%{real:>11.1f}%")


if __name__ == '__main__':
    main()

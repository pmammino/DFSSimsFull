#!/usr/bin/env python3
"""
test_lineup_builder_upside.py
=============================
Covers the candidate-upside additions to mlb_lineup_builder:
  * candidate_stack_structures  — shape distribution tilts toward 5/4 primaries
  * _hit_weight                 — upside column + batting-order tilt
  * Builder upside_attr         — one-offs/stack members chosen for ceiling
  * Builder bringback_prob      — one-off forced onto primary's opponent
  * Builder game_stack_prob     — secondary stack can be the primary's opponent
  * Builder order_weight        — stacks skew to the top of the batting order

Runs standalone (`python3 tests/test_lineup_builder_upside.py`) or under pytest.
All new controls default OFF, so the field builder is unaffected — verified too.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlb_lineup_builder import (  # noqa: E402
    Pool, Builder, candidate_stack_structures, _hit_weight, DEFAULT_PARAMS)

HIT_POS = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']


def make_pool(*, elite_oneoff=False, order=False):
    """A 4-hitter-team + 4-pitcher pool where every hitter team can field a full
    5-stack. Ownership is flat; Upside is elevated for team HOT and (optionally)
    for one specific one-off bat, so upside-weighting is distinguishable from
    ownership-weighting. Opponent map pairs HOT<->CLD and MID<->FAR."""
    opp = {'HOT': 'CLD', 'CLD': 'HOT', 'MID': 'FAR', 'FAR': 'MID'}
    rows = []
    for team in ['HOT', 'CLD', 'MID', 'FAR']:
        for i, pos in enumerate(HIT_POS):
            up = 5.0
            if team == 'HOT':
                up = 12.0
            rows.append({'Name': f'{team}_{pos}{i}', 'Pos': pos, 'Team': team,
                         'Opp': opp[team], 'Salary': 3000, 'Ownership': 5.0,
                         'Upside': up, 'Order': (i + 1) if order else 0})
    # one clearly-elite one-off bat on a team that is NOT a natural 5-stack anchor
    if elite_oneoff:
        rows.append({'Name': 'STUD_OF', 'Pos': 'OF', 'Team': 'STD', 'Opp': 'ZZZ',
                     'Salary': 3000, 'Ownership': 5.0, 'Upside': 40.0, 'Order': 0})
    # 4 arms with a clear ceiling ranking that is DECORRELATED from ownership:
    # P0 is the ace (top ceiling, low owned); P3 is a middling chalk arm.
    p_up = {'P0': 30.0, 'P1': 12.0, 'P2': 8.0, 'P3': 6.0}
    p_own = {'P0': 3.0, 'P1': 6.0, 'P2': 20.0, 'P3': 25.0}
    for j in range(4):
        nm = f'P{j}'
        rows.append({'Name': nm, 'Pos': 'P', 'Team': f'PT{j}', 'Opp': 'NON',
                     'Salary': 6000, 'Ownership': p_own[nm], 'Upside': p_up[nm],
                     'Order': 0})
    return Pool(pd.DataFrame(rows))


def build_n(pool, n, **kw):
    b = Builder(pool, DEFAULT_PARAMS, seed=7, **kw)
    out = []
    tries = 0
    while len(out) < n and tries < n * 60:
        lu = b.build_one()
        tries += 1
        if lu is not None:
            out.append(lu)
    return out


def primary_and_opp(lu):
    tc = Counter(r.Team for _, r in [(p.Pos, p) for p in lu['players']] if r.Pos != 'P')
    prim = tc.most_common(1)[0][0]
    return prim, tc


# --------------------------------------------------------------------------- #
def test_candidate_stack_structures_tilts_to_five():
    structs = [([5, 3], 0.18), ([5, 2, 1], 0.28), ([4, 3, 1], 0.08),
               ([3, 2, 2, 1], 0.03), ([2, 2, 1, 1, 1, 1], 0.03)]
    base = dict((tuple(s), w) for s, w in structs)
    tilted = dict((tuple(s), w) for s, w in candidate_stack_structures(structs, 1.0))
    assert abs(sum(tilted.values()) - 1.0) < 1e-9
    # 5-primary share rises, 2-primary share falls
    five_base = base[(5, 3)] + base[(5, 2, 1)]
    five_tilt = tilted[(5, 3)] + tilted[(5, 2, 1)]
    assert five_tilt > five_base + 0.15, (five_base, five_tilt)
    assert tilted[(2, 2, 1, 1, 1, 1)] < base[(2, 2, 1, 1, 1, 1)]
    # strength 0 == unchanged (renormalized)
    same = dict((tuple(s), w) for s, w in candidate_stack_structures(structs, 0.0))
    tot = sum(w for _, w in structs)
    for k in base:
        assert abs(same[k] - base[k] / tot) < 1e-9


def test_hit_weight_upside_and_order():
    r = pd.DataFrame([{'Name': 'x', 'Ownership': 5.0, 'Upside': 20.0, 'Order': 1}]
                     ).itertuples(index=False).__next__()
    assert _hit_weight(r) == 5.0                       # base = ownership
    assert _hit_weight(r, 'Upside') == 20.0            # upside column
    lead = _hit_weight(r, 'Upside', order_weight=0.3)  # slot 1 -> boosted
    assert lead > 20.0
    r9 = pd.DataFrame([{'Name': 'y', 'Ownership': 5.0, 'Upside': 20.0, 'Order': 9}]
                      ).itertuples(index=False).__next__()
    assert _hit_weight(r9, 'Upside', order_weight=0.3) < 20.0   # slot 9 -> cut
    # missing Order/Upside degrade gracefully
    r0 = pd.DataFrame([{'Name': 'z', 'Ownership': 3.0}]
                      ).itertuples(index=False).__next__()
    assert _hit_weight(r0, 'Upside', order_weight=0.3) == 3.0


def test_upside_oneoffs_beat_ownership():
    """With an elite (high-Upside, avg-Ownership) one-off available, upside
    weighting rosters it far more than ownership weighting does."""
    pool = make_pool(elite_oneoff=True)
    base = build_n(pool, 300)
    up = build_n(pool, 300, upside_attr='Upside', team_weights={'HOT': 3.0})
    def stud_rate(ls):
        return np.mean([any(p.Name == 'STUD_OF' for p in lu['players']) for lu in ls])
    rb, ru = stud_rate(base), stud_rate(up)
    assert ru > 2.0 * rb and ru > rb + 0.05, (rb, ru)


def test_bringback_puts_opponent_in_lineup():
    pool = make_pool()
    off = build_n(pool, 300, bringback_prob=0.0)
    on = build_n(pool, 300, bringback_prob=1.0)
    def opp_oneoff_rate(ls):
        hits = 0
        for lu in ls:
            prim, tc = primary_and_opp(lu)
            opp = pool.opp.get(prim)
            if opp and tc.get(opp, 0) >= 1:
                hits += 1
        return hits / len(ls)
    assert opp_oneoff_rate(on) > opp_oneoff_rate(off) + 0.15, \
        (opp_oneoff_rate(off), opp_oneoff_rate(on))


def test_game_stack_prob_favours_opponent_secondary():
    """With game_stack_prob high, a secondary 2+ stack is the primary's opponent
    much more often than under the field's suppression."""
    structs = [([4, 4], 1.0)]
    params = dict(DEFAULT_PARAMS); params['stack_structures'] = structs
    pool = make_pool()

    def opp_secondary_rate(gsp):
        b = Builder(pool, params, seed=3, game_stack_prob=gsp)
        n = 0; hit = 0
        tries = 0
        while n < 200 and tries < 200 * 60:
            lu = b.build_one(); tries += 1
            if lu is None:
                continue
            n += 1
            prim, tc = primary_and_opp(lu)
            opp = pool.opp.get(prim)
            secs = [t for t, c in tc.items() if t != prim and c >= 2]
            if opp in secs:
                hit += 1
        return hit / max(n, 1)

    lo = opp_secondary_rate(0.05)   # suppressed
    hi = opp_secondary_rate(0.95)   # favored
    assert hi > lo + 0.20, (lo, hi)


def test_order_weight_prefers_top_of_order():
    pool = make_pool(order=True)
    flat = build_n(pool, 250, order_weight=0.0)
    tilt = build_n(pool, 250, order_weight=0.6)
    def mean_stack_order(ls):
        vals = []
        for lu in ls:
            prim, tc = primary_and_opp(lu)
            for p in lu['players']:
                if p.Pos != 'P' and p.Team == prim and getattr(p, 'Order', 0):
                    vals.append(p.Order)
        return np.mean(vals)
    assert mean_stack_order(tilt) < mean_stack_order(flat) - 0.3, \
        (mean_stack_order(flat), mean_stack_order(tilt))


def test_pitcher_ceiling_weighting_prefers_high_ceiling_arms():
    """Ownership-weighted (field) rarely rosters the low-owned ace P0; ceiling
    (upside) weighting rosters it far more often."""
    pool = make_pool()
    field = build_n(pool, 300)                      # upside_attr None -> Ownership
    cand = build_n(pool, 300, upside_attr='Upside')

    def ace_rate(ls):
        return np.mean([any(p.Name == 'P0' for p in lu['players']) for lu in ls])
    assert ace_rate(cand) > ace_rate(field) + 0.10, (ace_rate(field), ace_rate(cand))


def test_ace_pitcher_prob_forces_a_ceiling_arm():
    """Forcing the first arm from the top-ceiling tier raises the best-arm ceiling
    and all but eliminates the two-middling-arms lineup."""
    pool = make_pool()
    off = build_n(pool, 300, upside_attr='Upside', ace_pitcher_prob=0.0)
    on = build_n(pool, 300, upside_attr='Upside', ace_pitcher_prob=1.0,
                 ace_pool_frac=0.5)

    def best_arm_ceiling(ls):
        return np.mean([max(getattr(p, 'Upside', 0.0)
                            for p in lu['players'] if p.Pos == 'P') for lu in ls])

    def both_middling(ls):
        c = 0
        for lu in ls:
            ups = [getattr(p, 'Upside', 0.0) for p in lu['players'] if p.Pos == 'P']
            if ups and all(v <= 8.0 for v in ups):   # both P2/P3-class arms
                c += 1
        return c / len(ls)

    assert best_arm_ceiling(on) > best_arm_ceiling(off) + 1.0, \
        (best_arm_ceiling(off), best_arm_ceiling(on))
    assert both_middling(on) < both_middling(off), \
        (both_middling(off), both_middling(on))


def test_field_builder_unaffected_by_defaults():
    """A Builder with no new kwargs must behave exactly as before: same lineups
    for the same seed, using ownership weighting and the field's game-stack rate."""
    pool = make_pool()
    a = Builder(pool, DEFAULT_PARAMS, seed=123)
    b = Builder(pool, DEFAULT_PARAMS, seed=123)
    for _ in range(50):
        la, lb = a.build_one(), b.build_one()
        assert [p.Name for p in la['players']] == [p.Name for p in lb['players']]
    # and the defaults leave the upside knobs off
    assert a.upside_attr is None and a.bringback_prob == 0.0
    assert a.game_stack_prob is None and a.order_weight == 0.0
    assert a.ace_pitcher_prob == 0.0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")

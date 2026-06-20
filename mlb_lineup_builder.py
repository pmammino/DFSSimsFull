#!/usr/bin/env python3
"""
mlb_lineup_builder.py
=====================
Builds DraftKings MLB Classic lineups from a player pool + expected ownership,
reproducing the lineup-construction grammar OBSERVED in real GPP contest
standings (three slates: 2026-05-30, 2026-06-04, 2026-06-05).

WHAT WAS LEARNED FROM THE DATA (see analysis report for details)
----------------------------------------------------------------
Roster slots (fixed)        : 2x P, 1x C, 1x 1B, 1x 2B, 1x 3B, 1x SS, 3x OF  (10)
Salary cap                  : < $50,000
Max hitters from one team   : 5  (a pitcher from that team may make a 6th body;
                                  7-from-one-team was never observed)
Hitter-stack structures     : the 8 hitters partition into team groups. The field
                              overwhelmingly builds a 5-man primary stack. Empirical
                              distribution of the full partition is loaded from
                              field_params.json. Top shapes: 5-2-1 (28%), 5-3 (18%),
                              5-1-1-1 (10%), 4-3-1 (8%).
Pitcher anti-correlation    : ~90% of lineups AVOID rostering a hitter who is facing
                              their own pitcher. Encoded as a soft rule.
Two pitchers same game      : rare (~5%) -> avoided by default.
Secondary = game stack      : a 2nd stack being the primary's opponent is uncommon
                              (~12%); secondaries are usually a different game.
Player selection signal     : %Drafted summed to exactly 100% per single slot
                              (300% OF, 200% P), i.e. ownership IS the per-slot
                              pick probability. So expected ownership is used
                              directly as the sampling weight.

INPUT POOL CSV columns (required): Name, Pos, Team, Opp, Salary, Ownership
  Pos in {P,C,1B,2B,3B,SS,OF};  Salary int;  Ownership = projected % (0-100)

USAGE
-----
  python3 mlb_lineup_builder.py --pool sample_pool.csv --n 150 --out lineups.csv
  python3 mlb_lineup_builder.py --pool sample_pool.csv --n 150 --validate
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROSTER = {'P':2,'C':1,'1B':1,'2B':1,'3B':1,'SS':1,'OF':3}
HITTER_SLOTS = {'C':1,'1B':1,'2B':1,'3B':1,'SS':1,'OF':3}   # 8 hitter bodies
SALARY_CAP = 50000
MAX_HITTERS_PER_TEAM = 5

DEFAULT_PARAMS = {  # used if field_params.json is absent
    "stack_structures": [([5,2,1],0.28),([5,3],0.18),([5,1,1,1],0.10),
                         ([4,3,1],0.08),([4,2,1,1],0.06),([4,4],0.04),([3,2,2,1],0.03)],
    "rules": {"avoid_hitter_vs_own_pitcher_prob":0.90,
              "two_pitchers_same_game_prob":0.047,
              "secondary_is_game_stack_prob":0.122},
}


# ----------------------------------------------------------------------------- 
# Pool handling
# ----------------------------------------------------------------------------- 
class Pool:
    def __init__(self, df):
        df = df.copy()
        df['Ownership'] = df['Ownership'].astype(float).clip(lower=0.01)
        df['Salary'] = df['Salary'].astype(int)
        # represent every player as a lightweight, value-comparable namedtuple
        self.rows = list(df.itertuples(index=False))
        self.pitchers = [r for r in self.rows if r.Pos == 'P']
        self.hitters  = [r for r in self.rows if r.Pos != 'P']
        self.team_hitters = defaultdict(list)
        self.team_pos = defaultdict(lambda: defaultdict(list))
        for r in self.hitters:
            self.team_hitters[r.Team].append(r)
            self.team_pos[r.Team][r.Pos].append(r)
        self.team_weight = {t: sum(r.Ownership for r in g)
                            for t, g in self.team_hitters.items()}
        self.opp = {r.Team: r.Opp for r in self.rows}
        self.team_pos_all = defaultdict(list)   # pos -> all hitters at that pos
        for r in self.hitters:
            self.team_pos_all[r.Pos].append(r)

    def teams_that_can_stack(self, k):
        return [t for t, g in self.team_hitters.items() if self._max_distinct(g) >= k]

    @staticmethod
    def _max_distinct(g):
        cap = {'C':1,'1B':1,'2B':1,'3B':1,'SS':1,'OF':3}
        cnt = Counter(r.Pos for r in g)
        return sum(min(cap.get(p,0), cnt.get(p,0)) for p in cap)


# ----------------------------------------------------------------------------- 
# Weighted sampling helpers
# ----------------------------------------------------------------------------- 
def wchoice(rng, items, weights):
    w = np.asarray(weights, float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return items[rng.choice(len(items), p=w / w.sum())]


def fill_team_stack(rng, pool, team, k, open_slots, used_names, jitter=0.0):
    """Pick k hitters from `team` filling k distinct still-open slots,
    weighted by ownership, never reusing a player name. Returns list or None.

    `jitter` (>=0) multiplies each candidate's weight by a per-draw lognormal
    shock exp(jitter * N(0,1)); >0 lets near-equally-weighted teammates trade
    places between lineups, so a team's stack uses varied members across the
    portfolio instead of the same highest-weighted bats every time."""
    slots = dict(open_slots)
    avail = {p: [r for r in rows if r.Name not in used_names]
             for p, rows in pool.team_pos[team].items()}
    chosen = []
    def rec(need):
        if need == 0:
            return True
        cand = [(p, r) for p in slots if slots[p] > 0 for r in avail.get(p, [])
                if r.Name not in {c[1].Name for c in chosen}]
        if not cand:
            return False
        weights = [r.Ownership for _, r in cand]
        if jitter:
            noise = np.exp(jitter * rng.standard_normal(len(weights)))
            weights = [wv * float(nz) for wv, nz in zip(weights, noise)]
        for _ in range(len(cand)):
            idx = rng.choice(len(cand), p=np.array(weights)/sum(weights))
            p, r = cand[idx]
            chosen.append((p, r)); slots[p] -= 1
            if rec(need - 1):
                return True
            chosen.pop(); slots[p] += 1
            del cand[idx]; del weights[idx]
            if not cand:
                break
        return False
    if rec(k):
        return chosen
    return None


# ----------------------------------------------------------------------------- 
# Core builder
# ----------------------------------------------------------------------------- 
class Builder:
    def __init__(self, pool, params, seed=None, uniform=False, team_weights=None,
                 jitter=0.0):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self.uniform = uniform   # if True, pick stack TEAMS uniformly (ignore ownership)
        # optional explicit per-team selection weights (already tilted); when
        # provided they drive stack-TEAM choice regardless of `uniform`, letting
        # candidates favor higher-projected/Vegas teams while player picks stay
        # governed by the pool's Ownership column.
        self.team_weights = team_weights
        # per-draw lognormal shock applied to every weighted selection (stack
        # team, stack members, one-off bats, pitchers). 0 = deterministic
        # weighting (prior behaviour); >0 diversifies the portfolio by letting
        # near-equally-ranked options trade places between lineups, which spreads
        # near-twin players, stack composition, and primary/secondary pairings.
        self.jitter = float(jitter)
        structs = params["stack_structures"]
        self.struct_shapes = [tuple(s) for s, _ in structs]
        p = np.array([w for _, w in structs], float); self.struct_probs = p / p.sum()
        self.rules = params["rules"]

    def _sample_structure(self):
        i = self.rng.choice(len(self.struct_shapes), p=self.struct_probs)
        return list(self.struct_shapes[i])

    def build_one(self, max_tries=400):
        for _ in range(max_tries):
            lu = self._attempt()
            if lu is not None:
                return lu
        return None

    def _attempt(self):
        pool = self.pool; rng = self.rng
        shape = self._sample_structure()                 # e.g. [5,2,1]
        stack_sizes = [s for s in shape if s >= 2]       # real stacks
        ones = [s for s in shape if s == 1]              # one-off hitters
        open_slots = dict(HITTER_SLOTS)
        used_teams = set()
        used_names = set()
        hitters = []

        # ---- assign + fill the multi-hitter stacks (largest first) ----
        for gi, k in enumerate(sorted(stack_sizes, reverse=True)):
            cands = [t for t in pool.teams_that_can_stack(k)
                     if t not in used_teams]
            if not cands:
                return None
            w = []
            for t in cands:
                if self.team_weights is not None:
                    wt = max(self.team_weights.get(t, 1e-6), 1e-6)
                else:
                    wt = 1.0 if self.uniform else pool.team_weight[t]
                # secondary stacks: usually NOT the primary's opponent (game stack rare)
                if gi > 0 and hitters:
                    prim_team = hitters[0][1].Team
                    if t == pool.opp.get(prim_team):
                        wt *= self.rules["secondary_is_game_stack_prob"] / \
                              (1 - self.rules["secondary_is_game_stack_prob"])
                w.append(max(wt, 1e-6))
            if self.jitter:
                noise = np.exp(self.jitter * rng.standard_normal(len(w)))
                w = [wi * float(nz) for wi, nz in zip(w, noise)]
            team = wchoice(rng, cands, w)
            picked = fill_team_stack(rng, pool, team, k, open_slots, used_names,
                                     self.jitter)
            if picked is None:
                return None
            for p, r in picked:
                open_slots[p] -= 1; used_names.add(r.Name)
            hitters.extend(picked)
            used_teams.add(team)

        # ---- fill the one-off hitters into remaining slots ----
        # eligible singletons for open slots, off the existing stack teams
        for _ in range(len(ones)):
            elig = [r for p in open_slots if open_slots[p] > 0
                    for r in pool.team_pos_all.get(p, [])
                    if r.Team not in used_teams and r.Name not in used_names]
            if not elig:
                return None
            weights = np.array([r.Ownership for r in elig], float)
            if self.jitter:
                weights = weights * np.exp(self.jitter * rng.standard_normal(len(weights)))
            r = elig[rng.choice(len(elig), p=weights/weights.sum())]
            hitters.append((r.Pos, r)); open_slots[r.Pos] -= 1; used_names.add(r.Name)

        if any(v != 0 for v in open_slots.values()):
            return None
        hitter_teams = set(r.Team for _, r in hitters)
        # enforce 5-hitter cap (defensive)
        if any(c > MAX_HITTERS_PER_TEAM for c in Counter(r.Team for _, r in hitters).values()):
            return None

        # ---- pick 2 pitchers (anti-correlation + avoid same-game pair) ----
        avoid_vs = rng.random() < self.rules["avoid_hitter_vs_own_pitcher_prob"]
        cand = [p for p in pool.pitchers if p.Name not in used_names]
        if avoid_vs:
            # exclude pitchers whose OPPONENT is one of our hitter teams
            # (that would mean our hitters are facing our own pitcher)
            filt = [p for p in cand if p.Opp not in hitter_teams]
            if len(filt) >= 2:
                cand = filt
        if len(cand) < 2:
            return None
        w1 = np.array([p.Ownership for p in cand], float)
        if self.jitter:
            w1 = w1 * np.exp(self.jitter * rng.standard_normal(len(w1)))
        p1 = cand[rng.choice(len(cand), p=w1/w1.sum())]
        rest = [p for p in cand if p.Name != p1.Name]
        if rng.random() > self.rules["two_pitchers_same_game_prob"]:
            r2 = [p for p in rest if p.Team != p1.Opp]   # avoid P-vs-P same game
            if r2:
                rest = r2
        if not rest:
            return None
        w2 = np.array([p.Ownership for p in rest], float)
        if self.jitter:
            w2 = w2 * np.exp(self.jitter * rng.standard_normal(len(w2)))
        p2 = rest[rng.choice(len(rest), p=w2/w2.sum())]
        pitchers = [p1, p2]

        # ---- never more than 6 total bodies from one team (5 hitters max + at
        #      most that team's pitcher); 7-from-one-team never occurs in the
        #      observed field. Also blocks the freak 2-pitchers-same-team case. ----
        body = Counter(r.Team for _, r in hitters)
        for pp in pitchers:
            body[pp.Team] += 1
        if max(body.values()) > 6:
            return None

        # ---- salary cap ----
        players = [r for _, r in hitters] + pitchers
        total = sum(int(r.Salary) for r in players)
        if total >= SALARY_CAP:
            return None

        return self._format(hitters, pitchers, total)

    def _format(self, hitters, pitchers, total):
        slot_order = ['P','P','C','1B','2B','3B','SS','OF','OF','OF']
        byslot = defaultdict(list)
        for p, r in hitters:
            byslot[p].append(r)
        out, used = [], Counter()
        for s in slot_order:
            if s == 'P':
                out.append(pitchers[used['P']]); used['P'] += 1
            else:
                out.append(byslot[s][used[s]]); used[s] += 1
        return {
            'players': out,
            'salary': total,
            'teams': Counter(r.Team for _, r in hitters),
            'names': [f"{r.Pos if r.Pos!='P' else 'P'} {r.Name}" for r in out],
        }


# ----------------------------------------------------------------------------- 
# Validation: does the GENERATED field match the OBSERVED field?
# ----------------------------------------------------------------------------- 
def validate(lineups, pool):
    n = len(lineups)
    prim = Counter(); maxteam = Counter(); vs_own_p = 0; sals = []
    for lu in lineups:
        sizes = sorted(lu['teams'].values(), reverse=True)
        prim[sizes[0]] += 1
        # max bodies incl pitchers
        allc = Counter(lu['teams'])
        for p in lu['players']:
            if p.Pos == 'P':
                allc[p.Team] += 1
        maxteam[max(allc.values())] += 1
        hteams = set(lu['teams'])
        if any(p.Pos == 'P' and p.Opp in hteams for p in lu['players']):
            vs_own_p += 1
        sals.append(lu['salary'])
    print(f"\n--- VALIDATION of {n} generated lineups (vs observed field) ---")
    print("Primary stack size      gen      observed")
    obs = {5:55.8,4:21.4,3:12.7,2:8.6,1:1.4}
    for s in sorted(prim, reverse=True):
        print(f"   {s} hitters          {100*prim[s]/n:5.1f}%   {obs.get(s,0):5.1f}%")
    print(f"Max bodies one team =6 : {100*maxteam.get(6,0)/n:5.1f}%   (obs 11.5%)  | =7: {maxteam.get(7,0)} (obs 0)")
    print(f"Hitter vs own pitcher  : {100*vs_own_p/n:5.1f}%   (obs ~9.8%)")
    print(f"Salary  mean ${np.mean(sals):.0f}  max ${max(sals)}  (cap $50000, all under: {all(s<50000 for s in sals)})")


# ----------------------------------------------------------------------------- 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--out', default='generated_lineups.csv')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--params', default='field_params.json')
    ap.add_argument('--validate', action='store_true')
    a = ap.parse_args()

    params = DEFAULT_PARAMS
    if os.path.exists(a.params):
        params = json.load(open(a.params))

    pool = Pool(pd.read_csv(a.pool))
    b = Builder(pool, params, seed=a.seed)
    lineups, fails = [], 0
    while len(lineups) < a.n and fails < a.n * 50:
        lu = b.build_one()
        if lu is None:
            fails += 1; continue
        lineups.append(lu)
    print(f"Built {len(lineups)} lineups ({fails} failed attempts).")

    rows = []
    for i, lu in enumerate(lineups, 1):
        row = {'Lineup': i, 'Salary': lu['salary'],
               'Stack': '-'.join(map(str, sorted(lu['teams'].values(), reverse=True)))}
        for j, p in enumerate(lu['players']):
            slot = ['P1','P2','C','1B','2B','3B','SS','OF1','OF2','OF3'][j]
            row[slot] = f"{p.Name} ({p.Team})"
        rows.append(row)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Wrote {a.out}")

    if a.validate:
        validate(lineups, pool)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
showdown_builder.py
===================
Builds DraftKings MLB **Showdown / Captain Mode** lineups from a single-game
player pool + expected ownership. This is the showdown-format counterpart to
``mlb_lineup_builder.py`` (which builds Classic lineups) and is wired into the
pipeline the same way: a ``Pool`` wraps the player rows and a ``Builder`` draws
weighted-random rosters that are then scored by the existing correlated sim.

DRAFTKINGS SHOWDOWN (CAPTAIN MODE) RULES
----------------------------------------
Roster slots               : 1x CPT + 5x UTIL  (6 bodies)
Game scope                 : a SINGLE game — every player is on one of the two
                             teams in that game.
Captain (CPT)              : scores 1.5x DK points AND costs 1.5x salary.
Positions                  : none required — any player (hitter or pitcher) may
                             fill any of the 6 slots.
Salary cap                 : <= $50,000  (captain counted at 1.5x salary)
Both teams                 : a valid roster must include at least one player from
                             EACH of the two teams (5-1 is the max team split;
                             6-0 is illegal).

FIELD GRAMMAR (heuristic, not yet calibrated to standings)
----------------------------------------------------------
Classic's field is reverse-engineered from real GPP standings (field_params.json).
No showdown standings are available yet, so the field here is an ownership-driven
heuristic with two tunable knobs (see ``DEFAULT_PARAMS``):

  * cpt_chalk   — captain-selection weight is ``cpt_ownership ** cpt_chalk``.
                  >1 concentrates on chalk captains (small fields), <1 flattens
                  toward uniform (large fields). Mirrors field_simulator's beta.
  * util_chalk  — same shaping for the 5 UTIL picks' ownership weights.

The structure is deliberately parameterized so that, if real showdown standings
arrive later, the learned captain distribution / team-split shape can be dropped
in exactly the way ``field_params.json`` recalibrates the classic field — no code
change, just data.

INPUT POOL CSV columns (required): Name, Pos, Team, Salary, Ownership
  Salary = the UTIL (base) salary as an int;  Ownership = projected UTIL draft %.
Optional columns:
  CptSalary     — explicit captain salary (else derived as round(1.5 * Salary)).
  CptOwnership  — projected CAPTAIN draft % (else derived from Ownership).

USAGE
-----
  python3 showdown_builder.py --pool game_pool.csv --n 150 --out sd_lineups.csv
  python3 showdown_builder.py --pool game_pool.csv --n 150 --validate
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd

ROSTER_SIZE = 6
N_UTIL = 5
SALARY_CAP = 50000
CPT_MULT = 1.5

DEFAULT_PARAMS = {
    "cpt_chalk": 1.0,   # captain ownership exponent (1.0 = use ownership as-is)
    "util_chalk": 1.0,  # util ownership exponent
}


def cpt_salary_of(base_salary, explicit=None):
    """Captain salary: the feed's explicit value when present, else 1.5x base
    rounded to the nearest dollar (DK lists CPT at 1.5x the UTIL salary)."""
    if explicit not in (None, "", 0):
        try:
            return int(round(float(explicit)))
        except (TypeError, ValueError):
            pass
    return int(round(float(base_salary) * CPT_MULT))


# --------------------------------------------------------------------------- #
# Pool handling
# --------------------------------------------------------------------------- #
class Pool:
    """A single-game showdown pool. Every row is a rosterable player on one of
    the two teams in the game."""

    def __init__(self, df):
        df = df.copy()
        df['Ownership'] = df['Ownership'].astype(float).clip(lower=0.01)
        df['Salary'] = df['Salary'].astype(int)
        if 'CptSalary' not in df.columns:
            df['CptSalary'] = df['Salary'].apply(lambda s: cpt_salary_of(s))
        else:
            df['CptSalary'] = [cpt_salary_of(s, c)
                               for s, c in zip(df['Salary'], df['CptSalary'])]
        if 'CptOwnership' not in df.columns:
            # No separate captain ownership: use the UTIL ownership as the proxy
            # captain weight (chalk hitters/pitchers are also the popular CPTs).
            df['CptOwnership'] = df['Ownership']
        df['CptOwnership'] = df['CptOwnership'].astype(float).clip(lower=0.01)

        self.rows = list(df.itertuples(index=False))
        self.teams = sorted({r.Team for r in self.rows})
        self.by_name = {r.Name: r for r in self.rows}

    def is_two_team_game(self):
        return len(self.teams) == 2


# --------------------------------------------------------------------------- #
# Core builder
# --------------------------------------------------------------------------- #
class Builder:
    def __init__(self, pool, params=None, seed=None, uniform=False,
                 team_weights=None, jitter=0.0):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        # uniform=True -> ownership-blind draws (used to build the CANDIDATE pool,
        # exactly like the classic builder); False -> ownership-weighted (field).
        self.uniform = uniform
        self.team_weights = team_weights
        self.jitter = float(jitter)
        p = dict(DEFAULT_PARAMS)
        p.update(params or {})
        self.cpt_chalk = float(p.get("cpt_chalk", 1.0))
        self.util_chalk = float(p.get("util_chalk", 1.0))

    def _weights(self, rows, attr, chalk):
        if self.uniform:
            w = np.ones(len(rows), float)
        else:
            w = np.array([float(getattr(r, attr)) for r in rows], float)
            if chalk != 1.0:
                w = np.power(np.clip(w, 1e-9, None), chalk)
        if self.team_weights is not None:
            w = w * np.array([max(self.team_weights.get(r.Team, 1e-6), 1e-6)
                              for r in rows], float)
        if self.jitter:
            w = w * np.exp(self.jitter * self.rng.standard_normal(len(w)))
        s = w.sum()
        return w / s if s > 0 else np.ones(len(rows)) / len(rows)

    def _pick(self, rows, attr, chalk):
        w = self._weights(rows, attr, chalk)
        return rows[self.rng.choice(len(rows), p=w)]

    def build_one(self, max_tries=400):
        for _ in range(max_tries):
            lu = self._attempt()
            if lu is not None:
                return lu
        return None

    def _attempt(self):
        pool = self.pool
        rows = pool.rows
        if len(rows) < ROSTER_SIZE or not pool.is_two_team_game():
            return None

        # ---- captain ----
        cpt = self._pick(rows, 'CptOwnership', self.cpt_chalk)
        cpt_cost = int(cpt.CptSalary)
        if cpt_cost >= SALARY_CAP:
            return None

        # ---- 5 UTIL (weighted, no repeats, salary-feasible) ----
        remaining = [r for r in rows if r.Name != cpt.Name]
        util = []
        used = {cpt.Name}
        spent = cpt_cost
        for _ in range(N_UTIL):
            cand = [r for r in remaining if r.Name not in used]
            # keep only players we can still afford (leave room is implicit; a
            # too-expensive pick just gets rejected and the attempt retried)
            if not cand:
                return None
            r = self._pick(cand, 'Ownership', self.util_chalk)
            util.append(r)
            used.add(r.Name)
            spent += int(r.Salary)

        players = [cpt] + util   # index 0 is always the captain

        # ---- both-teams rule (5-1 max split; 6-0 illegal) ----
        teamc = Counter(r.Team for r in players)
        if len(teamc) < 2:
            return None

        # ---- salary cap (captain at 1.5x) ----
        if spent > SALARY_CAP:
            return None

        return self._format(cpt, util, spent, teamc)

    def _format(self, cpt, util, salary, teamc):
        return {
            'captain': cpt,
            'util': list(util),
            'players': [cpt] + list(util),   # captain first
            'salary': salary,
            'teams': dict(teamc),
        }


# --------------------------------------------------------------------------- #
# Scoring: captain scores 1.5x
# --------------------------------------------------------------------------- #
def score_matrix(lineups, score, n_sim, norm=None):
    """(n_sim, n_lineup) DK totals. The captain (players[0]) is scored at 1.5x.

    ``score`` maps a (optionally normalized) player name to its per-sim DK array;
    ``norm`` is the name-normalizer used to key it (defaults to identity)."""
    key = norm or (lambda x: x)
    cols = []
    for lu in lineups:
        t = np.zeros(n_sim, np.float32)
        for i, pl in enumerate(lu['players']):
            arr = score[key(pl.Name)]
            t += (CPT_MULT * arr) if i == 0 else arr
        cols.append(t)
    return np.column_stack(cols)


SD_COLS = ['CPT', 'UTIL1', 'UTIL2', 'UTIL3', 'UTIL4', 'UTIL5']


def lineups_to_df(lineups):
    out = []
    for i, lu in enumerate(lineups, 1):
        r = {'Lineup': i, 'Salary': lu['salary'],
             'Split': '-'.join(map(str, sorted(lu['teams'].values(), reverse=True)))}
        for j, pl in enumerate(lu['players']):
            r[SD_COLS[j]] = f"{pl.Name} ({pl.Team})"
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(lineups):
    n = len(lineups)
    cpt_use = Counter(); split = Counter(); sals = []
    bad_team = 0; bad_size = 0; bad_cap = 0
    for lu in lineups:
        cpt_use[lu['captain'].Name] += 1
        split['-'.join(map(str, sorted(lu['teams'].values(), reverse=True)))] += 1
        sals.append(lu['salary'])
        if len(lu['players']) != ROSTER_SIZE:
            bad_size += 1
        if len(lu['teams']) < 2:
            bad_team += 1
        if lu['salary'] > SALARY_CAP:
            bad_cap += 1
    print(f"\n--- VALIDATION of {n} showdown lineups ---")
    print(f"roster size == 6        : {'OK' if bad_size == 0 else f'{bad_size} BAD'}")
    print(f"both teams present      : {'OK' if bad_team == 0 else f'{bad_team} BAD'}")
    print(f"under ${SALARY_CAP} cap    : {'OK' if bad_cap == 0 else f'{bad_cap} BAD'}")
    print(f"salary mean ${np.mean(sals):.0f}  max ${max(sals)}")
    print("team split distribution :",
          {k: f"{100*v/n:.0f}%" for k, v in split.most_common()})
    print("top captains            :",
          [(nm, f"{100*c/n:.0f}%") for nm, c in cpt_use.most_common(5)])


# --------------------------------------------------------------------------- #
def _selftest():
    """Synthetic two-team pool -> assert every DK showdown rule holds."""
    rng = np.random.default_rng(0)
    rows = []
    for team in ('LAD', 'SD'):
        for i in range(13):
            pos = 'P' if i == 0 else ('OF' if i < 5 else '1B')
            rows.append({'Name': f'{team}_{i}', 'Pos': pos, 'Team': team,
                         'Salary': int(rng.integers(2000, 11000)),
                         'Ownership': float(rng.uniform(1, 45))})
    pool = Pool(pd.DataFrame(rows))
    assert pool.is_two_team_game()

    b = Builder(pool, {'cpt_chalk': 1.2}, seed=7)
    lus = []
    while len(lus) < 300:
        lu = b.build_one()
        if lu:
            lus.append(lu)
    for lu in lus:
        assert len(lu['players']) == ROSTER_SIZE
        assert len({p.Name for p in lu['players']}) == ROSTER_SIZE   # no dupes
        assert len(lu['teams']) == 2                                 # both teams
        assert lu['players'][0].Name == lu['captain'].Name           # cpt first
        exp = cpt_salary_of(lu['captain'].Salary) + sum(int(p.Salary) for p in lu['util'])
        assert exp == lu['salary'] <= SALARY_CAP                     # 1.5x cpt salary

    # captain scores 1.5x
    n_sim = 500
    score = {r.Name: rng.normal(8, 4, n_sim).astype(np.float32) for r in pool.rows}
    mat = score_matrix(lus[:5], score, n_sim)
    lu0 = lus[0]
    manual = 1.5 * score[lu0['captain'].Name].copy()
    for p in lu0['util']:
        manual += score[p.Name]
    assert np.allclose(mat[:, 0], manual, atol=1e-3)

    # a single-team pool is unbuildable (6-0 illegal)
    one = Pool(pd.DataFrame([r for r in rows if r['Team'] == 'LAD']))
    assert Builder(one, seed=2).build_one() is None
    print("showdown_builder.py self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--out', default='showdown_lineups.csv')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--uniform', action='store_true',
                    help='ownership-blind draws (candidate pool)')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest or not a.pool:
        _selftest()
        return

    pool = Pool(pd.read_csv(a.pool))
    b = Builder(pool, seed=a.seed, uniform=a.uniform)
    lineups, fails = [], 0
    while len(lineups) < a.n and fails < a.n * 50:
        lu = b.build_one()
        if lu is None:
            fails += 1
            continue
        lineups.append(lu)
    print(f"Built {len(lineups)} showdown lineups ({fails} failed attempts).")
    lineups_to_df(lineups).to_csv(a.out, index=False)
    print(f"Wrote {a.out}")
    if a.validate:
        validate(lineups)


if __name__ == '__main__':
    main()

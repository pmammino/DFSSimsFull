#!/usr/bin/env python3
"""
showdown_contest.py
===================
Contest wiring for DraftKings MLB **Showdown / Captain Mode**, the showdown
counterpart to the classic contest flow in ``stage_d.py``. It consumes the SAME
correlated per-player DK sims (Stage C ``.npy`` output) — nothing about the
simulation changes for showdown — and produces candidates, an ownership-weighted
opponent field, and per-candidate contest results.

What is showdown-specific here (vs. ``stage_d``):

  * roster              — 1 CPT + 5 UTIL from a single game (``showdown_builder``)
  * scoring             — the captain scores 1.5x (``showdown_builder.score_matrix``)
  * captain field model — CAPTAIN selection is ownership tilted toward ceiling
    (real showdown fields captain studs more than flex ownership implies). The
    per-player captain weight is ``ownership * (ceiling / mean_ceiling) ** tilt``,
    where ceiling is the sim's p90 for that player. ``tilt=0`` reduces to pure
    ownership. This is the un-calibrated heuristic that stands in until real
    showdown standings are available (see module note in ``showdown_builder``).

Everything else — the per-sim placement/EV machinery (``run_contest``,
``portfolio_ev``) — is format-agnostic and reused unchanged.
"""
from collections import Counter

import numpy as np
import pandas as pd

import showdown_builder as sb
from field_simulator import beta_for_size
from stage_d import norm, run_contest

# field-size chalk model, mirroring classic (field_simulator defaults)
MEDIUM_FIELD = 6000
CHALK_SENSITIVITY = 0.35
DEFAULT_CPT_CEILING_TILT = 0.5   # how hard captain weight leans on ceiling


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #
def build_pool(dk_df, score, normfn=norm):
    """DK showdown slate frame + sims -> a single-game pool restricted to simmed
    players. ``dk_df`` has FullName, Team, Position ('UT'), Salary, Ownership
    (from ``dk_slate_feed.to_dk_df``). Returns a DataFrame with the columns the
    showdown ``Pool`` expects (Name, Pos, Team, Salary, Ownership).

    Raises ValueError if the simmed pool doesn't cover exactly two teams (a
    showdown slate is one game)."""
    simset = set(score)
    rows = []
    for r in dk_df.itertuples():
        if normfn(r.FullName) not in simset:
            continue
        rows.append({'Name': r.FullName, 'Pos': str(r.Position) or 'UT',
                     'Team': r.Team, 'Salary': int(r.Salary),
                     'Ownership': float(r.Ownership)})
    pool = pd.DataFrame(rows).drop_duplicates('Name').reset_index(drop=True)
    teams = sorted(pool['Team'].unique())
    if len(teams) != 2:
        raise ValueError(
            f"showdown pool must cover exactly 2 teams, got {teams} — check that "
            "the sims cover this game and the slate is a single matchup")
    return pool


# --------------------------------------------------------------------------- #
# Captain field model: ownership tilted toward ceiling
# --------------------------------------------------------------------------- #
def player_ceiling(pool, score, normfn=norm, q=90):
    """p{q} of each pool player's sim array (their ceiling)."""
    out = {}
    for nm in pool['Name']:
        a = score.get(normfn(nm))
        if a is not None and len(a):
            out[nm] = float(np.percentile(a, q))
    return out

def captain_weights(pool, score, normfn=norm, tilt=DEFAULT_CPT_CEILING_TILT, q=90):
    """Per-player captain-selection weight = ownership * (ceil/mean_ceil)**tilt.
    Returns {name: weight}. tilt=0 -> pure ownership."""
    ceil = player_ceiling(pool, score, normfn, q)
    mc = float(np.mean(list(ceil.values()))) if ceil else 1.0
    mc = mc or 1.0
    out = {}
    for r in pool.itertuples():
        c = ceil.get(r.Name, mc)
        out[r.Name] = max(float(r.Ownership), 1e-6) * (max(c, 1e-9) / mc) ** float(tilt)
    return out


def _field_pool(pool, score, normfn, tilt):
    """Pool frame carrying a ceiling-tilted CptOwnership column for the field."""
    cw = captain_weights(pool, score, normfn, tilt)
    fp = pool.copy()
    fp['CptOwnership'] = fp['Name'].map(cw).astype(float)
    return fp


# --------------------------------------------------------------------------- #
# Build candidates + field, score, run the contest
# --------------------------------------------------------------------------- #
def build_candidates(pool, n, seed, jitter=0.0):
    """Uniform (ownership-blind) candidate showdown lineups."""
    b = sb.Builder(sb.Pool(pool), seed=seed, uniform=True, jitter=jitter)
    out, fails = [], 0
    while len(out) < n and fails < n * 50 + 500:
        lu = b.build_one()
        if lu is None:
            fails += 1
            continue
        out.append(lu)
    return out

def build_field(pool, score, n_field, seed, *, medium=MEDIUM_FIELD,
                chalk_sensitivity=CHALK_SENSITIVITY, tilt=DEFAULT_CPT_CEILING_TILT,
                normfn=norm, jitter=0.0):
    """Ownership-weighted opponent field. Field size sets the chalk exponent
    (beta) the same way the classic field does; beta drives BOTH the util and
    captain ownership shaping."""
    beta = beta_for_size(n_field, medium, chalk_sensitivity)
    fp = _field_pool(pool, score, normfn, tilt)
    b = sb.Builder(sb.Pool(fp), {'cpt_chalk': beta, 'util_chalk': beta},
                   seed=seed, uniform=False, jitter=jitter)
    out, fails = [], 0
    while len(out) < n_field and fails < n_field * 20 + 500:
        lu = b.build_one()
        if lu is None:
            fails += 1
            continue
        out.append(lu)
    return out, beta


def _split(teams):
    return '-'.join(map(str, sorted(teams.values(), reverse=True)))


def simulate(dk_df, score, n_sim, *, contest_size=6000, n_candidates=3000,
             seed_cand=2025, seed_field=101, cpt_tilt=DEFAULT_CPT_CEILING_TILT,
             medium=MEDIUM_FIELD, chalk_sensitivity=CHALK_SENSITIVITY,
             cand_jitter=0.0, field_jitter=0.0, normfn=norm):
    """End-to-end showdown contest for one slate. Returns a dict with the
    candidate results DataFrame, the built candidate/field lineups, the scored
    matrices, and metadata — the shape the app persists in session state."""
    pool = build_pool(dk_df, score, normfn)

    cands = build_candidates(pool, n_candidates, seed_cand, jitter=cand_jitter)
    if not cands:
        raise RuntimeError("could not build any showdown candidate lineup")
    field, beta = build_field(pool, score, contest_size, seed_field,
                              medium=medium, chalk_sensitivity=chalk_sensitivity,
                              tilt=cpt_tilt, normfn=normfn, jitter=field_jitter)
    if not field:
        raise RuntimeError("could not build a showdown field")

    cand_mat = sb.score_matrix(cands, score, n_sim, norm=normfn)
    field_mat = sb.score_matrix(field, score, n_sim, norm=normfn)

    K = cand_mat.shape[0]
    wins, t10, t100, avg = run_contest(field_mat, cand_mat, K, len(field))

    own_map = {normfn(r.FullName): float(r.Ownership) for r in dk_df.itertuples()}
    res = sb.lineups_to_df(cands)
    res.insert(0, 'Candidate', np.arange(1, len(cands) + 1))
    res['Wins'] = wins
    res['Win%'] = np.round(100 * wins / K, 3)
    res['Top10'] = t10
    res['Top10%'] = np.round(100 * t10 / K, 2)
    res['Top100'] = t100
    res['Top100%'] = np.round(100 * t100 / K, 2)
    res['AvgPlace'] = np.round(avg, 1)
    res['Captain'] = [lu['captain'].Name for lu in cands]
    res['CptTeam'] = [lu['captain'].Team for lu in cands]
    res['Split'] = [_split(lu['teams']) for lu in cands]
    res['OwnSum'] = [round(sum(own_map.get(normfn(pl.Name), 0.0)
                               for pl in lu['players']), 1) for lu in cands]
    res = res.sort_values(['Wins', 'Top10', 'Top100', 'AvgPlace'],
                          ascending=[False, False, False, True]).reset_index(drop=True)

    return {
        'format': 'showdown', 'res': res, 'cands': cands, 'field': field,
        'cand_mat': cand_mat, 'field_mat': field_mat,
        'K': K, 'contest_size': len(field), 'beta': beta, 'pool': pool,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    names, teams, sals, owns = [], [], [], []
    for team in ('NYM', 'PHI'):
        for i in range(11):
            names.append(f'{team}_{i}')
            teams.append(team)
            sals.append(int(rng.integers(2500, 11000)))
            owns.append(float(rng.uniform(2, 40)))
    dk_df = pd.DataFrame({'FullName': names, 'Team': teams, 'Position': 'UT',
                          'Salary': sals, 'Ownership': owns})
    n_sim = 400
    # give the two nominal aces a higher ceiling so the tilt can be observed
    score = {}
    for nm, s in zip(names, sals):
        mu = 6 + s / 1500.0
        score[norm(nm)] = rng.normal(mu, 4 + mu * 0.3, n_sim).astype(np.float32)

    pool = build_pool(dk_df, score)
    assert len(pool) == 22 and set(pool['Team']) == {'NYM', 'PHI'}

    # ceiling tilt raises the captain weight of high-ceiling players above their
    # bare ownership share
    cw0 = captain_weights(pool, score, tilt=0.0)
    cw1 = captain_weights(pool, score, tilt=1.5)
    hi = max(pool['Name'], key=lambda n: float(np.percentile(score[norm(n)], 90)))
    share0 = cw0[hi] / sum(cw0.values())
    share1 = cw1[hi] / sum(cw1.values())
    assert share1 > share0, (share0, share1)

    out = simulate(dk_df, score, n_sim, contest_size=800, n_candidates=500)
    res = out['res']
    assert len(res) == 500 and out['cand_mat'].shape == (n_sim, 500)
    assert out['field_mat'].shape == (n_sim, 800)
    assert res['Win%'].max() > 0
    # every candidate is a legal showdown roster
    for lu in out['cands']:
        assert len(lu['players']) == 6 and len(lu['teams']) == 2
        assert lu['salary'] <= sb.SALARY_CAP
    # a wins-ranked, non-empty captain column
    assert res['Captain'].notna().all()
    print("showdown_contest.py self-test passed:",
          f"best Win% {res['Win%'].max():.2f}, beta {out['beta']:.2f}")


if __name__ == '__main__':
    _selftest()

"""
validate.py — verify the realized correlation structure of a sim run.

Targets (from spec):
    teammate hitter-hitter   +0.20 .. +0.50
    unrelated hitter-hitter   ~0
    hitter vs opposing SP    -0.30 .. -0.60

Also runs a stacking behavior check: a correlated N-stack should have a much
fatter combined-score tail than the same number of unrelated hitters with
matched individual means.
"""
import itertools, random
import numpy as np


def correlation_report(hitter_dk, pitcher_dk, hrows, sample_unrelated=400, seed=1):
    rnd = random.Random(seed)
    team_players, pgame, opp_sp = {}, {}, {}
    for r in hrows:
        team_players.setdefault(r['team'], []).append(r['player'])
        pgame[r['player']] = r['game']
        opp_sp[r['player']] = r['opp_sp']

    def c(a, b):
        return float(np.corrcoef(hitter_dk[a], hitter_dk[b])[0, 1])

    tm = [c(a, b) for _, ps in team_players.items()
          for a, b in itertools.combinations([p for p in ps if p in hitter_dk], 2)]

    allh = list(hitter_dk)
    unrel = []
    for _ in range(sample_unrelated):
        a, b = rnd.sample(allh, 2)
        if pgame[a] != pgame[b]:
            unrel.append(c(a, b))

    hvp = [float(np.corrcoef(hitter_dk[p], pitcher_dk[opp_sp[p]])[0, 1])
           for p in hitter_dk if opp_sp.get(p) in pitcher_dk]

    return {
        'teammate_mean': round(float(np.mean(tm)), 3) if tm else None,
        'unrelated_mean': round(float(np.mean(unrel)), 3) if unrel else None,
        'hitter_vs_sp_mean': round(float(np.mean(hvp)), 3) if hvp else None,
        'n_teammate_pairs': len(tm), 'n_unrelated_pairs': len(unrel), 'n_hvp_pairs': len(hvp),
    }


def stack_behavior(hitter_dk, hrows, team=None, k=5, seed=7):
    """Compare a correlated k-stack vs k unrelated matched-mean hitters."""
    rnd = random.Random(seed)
    by_team, by_game, means = {}, {}, {}
    for r in hrows:
        by_team.setdefault(r['team'], []).append(r['player'])
        by_game.setdefault(r['game'], []).append(r['player'])
    for p in hitter_dk:
        means[p] = float(hitter_dk[p].mean())

    if team is None:  # pick the team with the highest top-k projected total
        team = max(by_team, key=lambda t: sum(sorted((means[p] for p in by_team[t] if p in hitter_dk),
                                                      reverse=True)[:k]))
    stack = sorted([p for p in by_team[team] if p in hitter_dk],
                   key=lambda p: -means[p])[:k]

    # matched unrelated set: nearest-mean players from distinct other games
    used_games, rand_set = set(), []
    games = list(by_game)
    for tgt in stack:
        best, bd = None, 1e9
        for gm in games:
            if gm in used_games:
                continue
            for p in by_game[gm]:
                if p not in hitter_dk:
                    continue
                d = abs(means[p] - means[tgt])
                if d < bd:
                    bd, best = d, (gm, p)
        if best:
            used_games.add(best[0]); rand_set.append(best[1])

    cs = np.sum([hitter_dk[p] for p in stack], axis=0)
    rs = np.sum([hitter_dk[p] for p in rand_set], axis=0) if rand_set else cs
    return {
        'team': team, 'stack': stack,
        'corr_mean': round(float(cs.mean()), 1), 'corr_p99': round(float(np.percentile(cs, 99)), 0),
        'unrel_mean': round(float(rs.mean()), 1), 'unrel_p99': round(float(np.percentile(rs, 99)), 0),
        'corr_P(>=2x_mean)': round(float((cs >= 2 * cs.mean()).mean()), 4),
        'unrel_P(>=2x_mean)': round(float((rs >= 2 * rs.mean()).mean()), 4),
    }


def print_report(hitter_dk, pitcher_dk, hrows):
    rep = correlation_report(hitter_dk, pitcher_dk, hrows)
    print("  Correlations:")
    print(f"    teammate H-H:   {rep['teammate_mean']:+.3f}  (target +0.20..+0.50, n={rep['n_teammate_pairs']})")
    print(f"    unrelated H-H:  {rep['unrelated_mean']:+.3f}  (target ~0, n={rep['n_unrelated_pairs']})")
    print(f"    hitter vs SP:   {rep['hitter_vs_sp_mean']:+.3f}  (target -0.30..-0.60, n={rep['n_hvp_pairs']})")
    sb = stack_behavior(hitter_dk, hrows)
    print(f"  Stack check ({sb['team']} {len(sb['stack'])}-stack vs matched unrelated):")
    print(f"    correlated  mean {sb['corr_mean']}  p99 {sb['corr_p99']}  P(>=2x)={sb['corr_P(>=2x_mean)']}")
    print(f"    unrelated   mean {sb['unrel_mean']}  p99 {sb['unrel_p99']}  P(>=2x)={sb['unrel_P(>=2x_mean)']}")
    ok = (rep['teammate_mean'] and rep['teammate_mean'] >= 0.18
          and rep['hitter_vs_sp_mean'] and rep['hitter_vs_sp_mean'] <= -0.25
          and abs(rep['unrelated_mean']) < 0.05)
    print(f"  VALIDATION: {'PASS' if ok else 'CHECK'}")
    return rep, sb, ok

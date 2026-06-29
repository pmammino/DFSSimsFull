#!/usr/bin/env python3
"""
portfolio.py
============
Diversity-aware selection of a lineup PORTFOLIO for export, plus value-group
detection.

The export step has always been "rank the candidates, then greedily take the
top N (under per-player exposure caps)". That funnel is what makes an exported
set look same-y: the single highest-projected build and its near-clones win
every tie, so the same teammates fill every stack and the same primary team is
always paired with the same secondary.

This module keeps the familiar "rank then greedily fill" shape but lets the
caller spread the portfolio along the axes that actually matter:

  * per-player exposure          (hitter_cap / pitcher_cap)   - existing
  * per primary STACK TEAM       (team_cap)                    - existing
  * per (primary, secondary) PAIR(pair_cap)        -> diversifies stack pairings
  * per exact STACK CORE         (core_cap)        -> diversifies stack teammates
  * pairwise lineup OVERLAP      (max_overlap)     -> no near-duplicate lineups
  * per VALUE-GROUP of near-twin players (group_cap)

Every control is OFF by default (caps = 1.0, overlap = 1.0): with the defaults
this reproduces the prior top-N-by-rank behaviour exactly.
"""
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
# Row parsing
# --------------------------------------------------------------------------- #
def _split(cell):
    """'Name (TEAM)' -> ('Name', 'TEAM'); tolerates a missing/odd team tag."""
    s = str(cell)
    if " (" in s and s.endswith(")"):
        name, team = s.rsplit(" (", 1)
        return name, team[:-1]
    return s, ""


def lineup_features(row, cols, hitc):
    """Pull the bits the selector reasons about out of one result row.

    `cols` is the full slot order (2 pitchers then 8 hitters); `hitc` is the
    subset of hitter slot labels. Returns the player names, the primary stack
    team, the secondary stack team (next team that is itself a >=2 stack), the
    (primary, secondary) pair, and the exact primary stack core.
    """
    names, teams = [], []
    for c in cols:
        nm, tm = _split(row[c])
        names.append(nm)
        teams.append(tm)
    hset = set(hitc)
    hteams = [t for c, t in zip(cols, teams) if c in hset and t]
    tc = Counter(hteams)
    ordered = [t for t, _ in tc.most_common()]
    primary = ordered[0] if ordered else ""
    secondary = next((t for t in ordered[1:] if tc[t] >= 2), "")
    core = frozenset(nm for c, nm, tm in zip(cols, names, teams)
                     if c in hset and tm == primary)
    return {
        "names": names,
        "playerset": frozenset(names),
        "primary": primary,
        "secondary": secondary,
        "pair": (primary, secondary),
        "core": (primary, core),
    }


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# --------------------------------------------------------------------------- #
# Diversity-aware greedy selection
# --------------------------------------------------------------------------- #
def select_portfolio(res_df, n_select, sort_cols, *, cols, hitc,
                     eligible=None, hitter_cap=1.0, pitcher_cap=1.0,
                     team_cap=1.0, pair_cap=1.0, core_cap=1.0,
                     max_overlap=1.0, group_of=None, group_cap=1.0,
                     player_caps=None, team_caps=None):
    """Rank `res_df` by `sort_cols` (descending) then greedily accept lineups
    that keep every exposure cap and the pairwise-overlap ceiling satisfied.

    Caps are fractions of `n_select` (0-1); 1.0 means "no cap". `eligible(names)`
    is an optional predicate (e.g. "every player maps to a DK ID"); rows that
    fail it are skipped and counted. `group_of` maps player name -> value-group
    id; a lineup counts once against each group it touches.

    `player_caps` ({player_name: fraction}) and `team_caps` ({team_code:
    fraction}) give per-ENTITY maximums that OVERRIDE the global hitter/pitcher/
    team caps for the named players/teams. A player not in `player_caps` falls
    back to the global hitter_cap (or pitcher_cap for the two pitcher slots); a
    team not in `team_caps` falls back to the global team_cap. This lets the
    caller cap, say, one star hitter at 40% while leaving everyone else at the
    global default.

    Returns (chosen_rows, info).
    """
    N = int(n_select)
    rdf = res_df.sort_values(list(sort_cols), ascending=False).reset_index(drop=True)

    def cap_n(frac):
        # frac<=0 means "exclude" (0 lineups); otherwise at least 1 so a tiny
        # positive cap still admits one lineup.
        f = float(frac)
        return 0 if f <= 0 else max(1, int(round(f * N)))

    hcap, pcap, tcap = cap_n(hitter_cap), cap_n(pitcher_cap), cap_n(team_cap)
    paircap, ccap, gcap = cap_n(pair_cap), cap_n(core_cap), cap_n(group_cap)
    group_of = group_of or {}

    # per-entity overrides -> precomputed cap counts
    player_caps = player_caps or {}
    team_caps = team_caps or {}
    player_capn = {nm: cap_n(fr) for nm, fr in player_caps.items()}
    team_capn = {tm: cap_n(fr) for tm, fr in team_caps.items()}

    def player_cap_for(name, i):
        if name in player_capn:
            return player_capn[name]
        return pcap if i < 2 else hcap

    def team_cap_for(team):
        return team_capn.get(team, tcap)

    expo = Counter()     # per player
    teamc = Counter()    # per primary stack team
    pairc = Counter()    # per (primary, secondary) pair
    corec = Counter()    # per (primary, frozenset of stack members)
    groupc = Counter()   # per value-group id
    pitchers = set()
    chosen, chosen_sets, skipped = [], [], 0

    for _, row in rdf.iterrows():
        f = lineup_features(row, cols, hitc)
        names = f["names"]
        if eligible is not None and not eligible(names):
            skipped += 1
            continue
        # per-player caps (first two slots are pitchers); per-entity overrides
        # apply where set, else the global hitter/pitcher cap.
        if any(expo[n] >= player_cap_for(n, i) for i, n in enumerate(names)):
            continue
        if teamc[f["primary"]] >= team_cap_for(f["primary"]):
            continue
        if f["secondary"] and pairc[f["pair"]] >= paircap:
            continue
        if corec[f["core"]] >= ccap:
            continue
        gids = {group_of[n] for n in names if n in group_of} if group_of else set()
        if any(groupc[g] >= gcap for g in gids):
            continue
        if max_overlap < 1.0 and chosen_sets:
            if max(_jaccard(f["playerset"], s) for s in chosen_sets) > max_overlap:
                continue
        # ---- accept ----
        chosen.append(row)
        chosen_sets.append(f["playerset"])
        for i, n in enumerate(names):
            expo[n] += 1
            if i < 2:
                pitchers.add(n)
        teamc[f["primary"]] += 1
        if f["secondary"]:
            pairc[f["pair"]] += 1
        corec[f["core"]] += 1
        for g in gids:
            groupc[g] += 1
        if len(chosen) == N:
            break

    info = {
        "chosen": len(chosen), "requested": N, "skipped_unmapped": skipped,
        "max_pitcher": max((expo[n] for n in pitchers), default=0),
        "max_hitter": max((expo[n] for n in expo if n not in pitchers), default=0),
        "max_team": max(teamc.values()) if teamc else 0,
        "max_pair": max(pairc.values()) if pairc else 0,
        "max_core": max(corec.values()) if corec else 0,
        "distinct_pairs": len(pairc),
        "distinct_cores": len(corec),
        "distinct_primaries": len(teamc),
    }
    return chosen, info


# --------------------------------------------------------------------------- #
# Value-group detection (near-twin players)
# --------------------------------------------------------------------------- #
def detect_value_groups(meta, *, salary_tol=300, proj_tol=1.5, min_size=2):
    """Cluster near-equivalent players so the portfolio can spread across them.

    Two players are "near-twins" when they share a position, sit within
    `salary_tol` of each other in salary, and within `proj_tol` in projection.
    Often one of a pair projects a hair higher and therefore eats every "best"
    lineup; grouping them lets a group exposure cap share the load.

    `meta` is {name: {"pos":str, "salary":int, "proj":float|None, "team":str}}.
    Returns (group_of, groups): group_of maps name -> group id; groups is a list
    of {"id", "players", "pos", "salary_lo", "salary_hi", "proj_lo", "proj_hi"}
    for display, largest first.
    """
    by_pos = defaultdict(list)
    for nm, m in meta.items():
        if m.get("proj") is None:
            continue
        by_pos[m.get("pos", "?")].append(nm)

    group_of, groups, gid = {}, [], 0
    for pos, names in by_pos.items():
        # anchor on the cheapest, then chain to projection so twins sit adjacent
        names.sort(key=lambda n: (meta[n]["salary"], meta[n]["proj"]))
        used = set()
        for i, anchor in enumerate(names):
            if anchor in used:
                continue
            cluster = [anchor]
            for other in names[i + 1:]:
                if other in used:
                    continue
                if (abs(meta[other]["salary"] - meta[anchor]["salary"]) <= salary_tol
                        and abs(meta[other]["proj"] - meta[anchor]["proj"]) <= proj_tol):
                    cluster.append(other)
            if len(cluster) >= min_size:
                for c in cluster:
                    used.add(c)
                    group_of[c] = gid
                sals = [meta[c]["salary"] for c in cluster]
                prjs = [meta[c]["proj"] for c in cluster]
                groups.append({
                    "id": gid, "players": cluster, "pos": pos,
                    "salary_lo": min(sals), "salary_hi": max(sals),
                    "proj_lo": min(prjs), "proj_hi": max(prjs),
                })
                gid += 1
    groups.sort(key=lambda g: len(g["players"]), reverse=True)
    return group_of, groups


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pandas as pd

    COLS = ['P1', 'P2', 'C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']
    HITC = ['C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']

    def mk(p1, p2, hitters, ranks):
        """hitters: list of (slot, name, team) length 8; ranks dict."""
        row = {'P1': f"{p1} (AAA)", 'P2': f"{p2} (BBB)"}
        for slot, name, team in hitters:
            row[slot] = f"{name} ({team})"
        row.update(ranks)
        return row

    # five CLE(5)+KC(2)+1 lineups that are near-identical, plus a couple variants
    base_h = [('C', 'c1', 'CLE'), ('1B', 'b1', 'CLE'), ('2B', 'b2', 'CLE'),
              ('3B', 'b3', 'CLE'), ('SS', 's1', 'CLE'), ('OF1', 'k1', 'KC'),
              ('OF2', 'k2', 'KC'), ('OF3', 'o1', 'NYY')]
    rows = []
    for i in range(6):
        rows.append(mk('pa', 'pb', base_h, {'Wins': 100 - i, 'Top10': 0, 'Top100': 0}))
    # one different core (swap s1->s2, KC->MIN secondary)
    alt_h = [('C', 'c1', 'CLE'), ('1B', 'b1', 'CLE'), ('2B', 'b2', 'CLE'),
             ('3B', 'b3', 'CLE'), ('SS', 's2', 'CLE'), ('OF1', 'm1', 'MIN'),
             ('OF2', 'm2', 'MIN'), ('OF3', 'o1', 'NYY')]
    rows.append(mk('pc', 'pd', alt_h, {'Wins': 50, 'Top10': 0, 'Top100': 0}))
    df = pd.DataFrame(rows)

    # No caps -> top 5 are the near-identical CLE/KC builds
    chosen, info = select_portfolio(df, 5, ['Wins', 'Top10', 'Top100'],
                                    cols=COLS, hitc=HITC)
    assert info["chosen"] == 5, info
    assert info["max_core"] == 5, info       # all share the same core
    assert info["max_pair"] == 5, info

    # core_cap forces the alternate build in
    chosen, info = select_portfolio(df, 5, ['Wins', 'Top10', 'Top100'],
                                    cols=COLS, hitc=HITC, core_cap=0.8)
    assert info["max_core"] <= 4, info
    assert info["distinct_cores"] >= 2, info

    # overlap ceiling also breaks up the clones
    chosen, info = select_portfolio(df, 5, ['Wins', 'Top10', 'Top100'],
                                    cols=COLS, hitc=HITC, max_overlap=0.9)
    assert info["distinct_cores"] >= 2, info

    # value groups: s1 & s2 are near-twin SS
    meta = {
        's1': {'pos': 'SS', 'salary': 4500, 'proj': 9.0, 'team': 'CLE'},
        's2': {'pos': 'SS', 'salary': 4600, 'proj': 8.4, 'team': 'CLE'},
        'b1': {'pos': '1B', 'salary': 3000, 'proj': 6.0, 'team': 'CLE'},
    }
    g_of, groups = detect_value_groups(meta)
    assert g_of.get('s1') == g_of.get('s2') and 's1' in g_of, (g_of, groups)
    assert 'b1' not in g_of, g_of

    # per-entity caps: cap one player and one team below the global default
    many = [mk('pa', 'pb', base_h, {'Wins': 100 - i, 'Top10': 0, 'Top100': 0})
            for i in range(10)]
    dfm = pd.DataFrame(many)
    _, pinfo = select_portfolio(dfm, 10, ['Wins', 'Top10', 'Top100'],
                                cols=COLS, hitc=HITC, player_caps={'b1': 0.30})
    assert pinfo["max_hitter"] <= 3, pinfo            # b1 limited to 3/10
    _, tinfo = select_portfolio(dfm, 10, ['Wins', 'Top10', 'Top100'],
                                cols=COLS, hitc=HITC, team_caps={'CLE': 0.40})
    assert tinfo["max_team"] <= 4, tinfo              # CLE primary limited to 4/10
    chx, _ = select_portfolio(dfm, 10, ['Wins', 'Top10', 'Top100'],
                              cols=COLS, hitc=HITC, player_caps={'b1': 0.0})
    assert all('b1 (' not in str(r['1B']) for r in chx)   # 0% excludes entirely

    print("portfolio.py self-test passed:", info)

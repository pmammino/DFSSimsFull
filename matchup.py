"""
matchup.py — convert handedness-split per-PA projections into slate-specific
per-PA event vectors for each hitter and pitcher, then hand them to the sim.

This is the bridge between the projection engine (handedness splits, neutral +
park + vL/vR) and the correlated simulator. For each game we know:
  - the batting order for each side (names + ids from the lineup feed)
  - the opposing starter/opener/primary and their throwing hand

For a hitter we select the split that matches the hand of the pitcher he will
see most (the starter/opener), then apply the home-park factor when the hitter
is batting in his own park. For a pitcher we select the split matching the
predominant batting hand of the lineup he faces (PA-weighted L/R share), and
likewise park-adjust at home.

The output dicts use the SAME keys the simulator's event sampler expects, so
simulate.py consumes them with no change to its correlation machinery.
"""
import json, os, unicodedata
import numpy as np
import pandas as pd

from slate_config import canonical_team, TEAM_CODE_MAP

# the 30 canonical MLB codes — used to tell a resolved team ("OAK") from an
# unresolvable projection-feed truncation ("LOS" from "Los", which could be
# either LA club), so collision resolution only excludes rows it's sure about.
_VALID_TEAMS = set(TEAM_CODE_MAP.values())

# Per-PA event columns produced by the projection pipeline
HIT_EVENTS = ['P_1B', 'P_2B', 'P_3B', 'P_HR', 'P_BB', 'P_HBP', 'P_K']  # P_SF/P_BIPOut are residual
PIT_EVENTS = ['P_1B', 'P_2B', 'P_3B', 'P_HR', 'P_BB', 'P_HBP', 'P_K']

# League-average per-PA fallback for players absent from the projection set
# (rookies/call-ups with insufficient sample). Keeps the lineup intact.
LG_HIT_VEC = dict(p_1b=0.150, p_2b=0.045, p_3b=0.004, p_hr=0.030,
                  p_bb=0.083, p_hbp=0.011, p_k=0.225,
                  r_pa=0.11, rbi_pa=0.11, p_sb=0.010, proj_slot=7)
LG_PIT_VEC = dict(k_pct=0.215, bb_pct=0.082, hbp_per_bf=0.010,
                  h_per_bf=0.235, hr_per_bf=0.032, era=4.50, ra9=4.80,
                  tbf_per_ip=4.35, ip_per_g=5.0, wp_per_pa=0.0, hand='R')

# ── Opponent-quality matchup (log5 / odds-ratio) ────────────────────────────────
# A pitcher's allowed rates depend on the QUALITY of the lineup he faces, not just
# its handedness. `_pitcher_vector` only blended the pitcher's own vL/vR splits by
# the lineup's L/R share, so an elite lineup and a replacement-level one with the
# same handedness produced the same projection — a starter facing the best offense
# on the slate was not docked for it. We now combine the pitcher's rate with each
# opposing hitter's rate on the log-odds (odds-ratio) scale — the textbook log5
# matchup — and average over the lineup.
#
# The batter-side elasticity was calibrated OUT-OF-SAMPLE on Statcast batted-ball
# logs (bip_inputs/): estimate each batter's & pitcher's contact rate on 2024,
# then fit how strongly the batter side moves the actual 2025 outcome. Findings:
#   * HR:            elasticity ~1.0  (full log5; power is a persistent, real skill)
#   * balls-in-play hits: ~0.7        (below full log5 — the DIPS signature, since
#                                      pitchers have limited control over BABIP)
# K and BB are not in balls-in-play logs, so they use full log5 (=1.0), the
# standard theoretical value; K especially is strongly batter-driven, so a
# low-strikeout contact lineup meaningfully suppresses a pitcher's Ks.
OPP_MATCHUP_ELASTICITY = dict(k=1.0, bb=1.0, hr=1.0, bip_hit=0.70)


def _logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _log5_rate(p_rate, batter_rates, league, gamma):
    """Odds-ratio (log5) matchup rate for one event: take the pitcher's own rate
    (his rate vs a league-average lineup) and shift it by each opposing batter's
    log-odds deviation from the league average — scaled by the calibrated
    elasticity `gamma` — then average over the lineup he faces. gamma=1 is textbook
    log5; gamma<1 damps the batter influence (e.g. balls-in-play hits)."""
    if not batter_rates:
        return float(p_rate)
    lp, lL = _logit(p_rate), _logit(league)
    return float(np.mean([_sigmoid(lp + gamma * (_logit(b) - lL))
                          for b in batter_rates]))


def _opponent_adjust_pitcher(vec, opp_lineup):
    """Fold the opposing lineup's hitting QUALITY into the pitcher's per-BF event
    rates via the calibrated log5 matchup (see OPP_MATCHUP_ELASTICITY). HR and the
    balls-in-play hit component (h_per_bf minus HR) carry the calibrated batter
    elasticities; K and BB use full log5. ra9/era/tbf stay as neutral skill anchors
    — the extra runs flow through the now-elevated hit/HR/BB traffic in the sim."""
    bats = [p['vec'] for p in opp_lineup if p.get('vec')]
    if not bats:
        return vec
    E = OPP_MATCHUP_ELASTICITY
    out = dict(vec)
    out['k_pct'] = _log5_rate(vec['k_pct'], [b['p_k'] for b in bats],
                              LG_HIT_VEC['p_k'], E['k'])
    out['bb_pct'] = _log5_rate(vec['bb_pct'], [b['p_bb'] for b in bats],
                               LG_HIT_VEC['p_bb'], E['bb'])
    out['hr_per_bf'] = _log5_rate(vec['hr_per_bf'], [b['p_hr'] for b in bats],
                                  LG_HIT_VEC['p_hr'], E['hr'])
    l_bip = LG_HIT_VEC['p_1b'] + LG_HIT_VEC['p_2b'] + LG_HIT_VEC['p_3b']
    bip_p = max(vec['h_per_bf'] - vec['hr_per_bf'], 1e-6)
    bip = _log5_rate(bip_p, [b['p_1b'] + b['p_2b'] + b['p_3b'] for b in bats],
                     l_bip, E['bip_hit'])
    out['h_per_bf'] = bip + out['hr_per_bf']
    return out


# Map each lowercase per-PA event to its Statcast park-factor column on the
# projection rows (1.0 = league-neutral). P_HBP / P_SF have no published factor.
_VENUE_PF_COLS = {'p_k': 'pf_SO', 'p_bb': 'pf_BB', 'p_hr': 'pf_HR',
                  'p_1b': 'pf_1B', 'p_2b': 'pf_2B', 'p_3b': 'pf_3B'}


def _venue_factors(rows):
    """Game-venue park factors = the HOME team's park. A home player's `pf_*`
    columns already describe the home stadium, so we read them off any home-side
    projection row (hitters or pitcher). Returned at full strength so the venue
    applies to every hitter and pitcher in the game. Neutral (all 1.0) if no
    home-side row carries park factors."""
    for r in rows:
        if r is None:
            continue
        pf_hr = r.get('pf_HR') if hasattr(r, 'get') else None
        if pf_hr is not None and pd.notna(pf_hr):
            out = {}
            for ev, col in _VENUE_PF_COLS.items():
                v = r.get(col, 1.0)
                out[ev] = float(v) if pd.notna(v) else 1.0
            return out
    return {ev: 1.0 for ev in _VENUE_PF_COLS}


def _norm(s):
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    s = s.lower().replace('.', '').replace("'", '').replace('-', ' ')
    # strip generational suffixes so "Jazz Chisholm Jr" == "Jazz Chisholm"
    toks = [t for t in s.split() if t not in ('jr', 'sr', 'ii', 'iii', 'iv', 'v')]
    return ' '.join(toks).strip()


def load_projections(out_dir, target_year):
    h = pd.read_csv(os.path.join(out_dir, f'hitter_pa_projections_{target_year}.csv'))
    p = pd.read_csv(os.path.join(out_dir, f'pitcher_pa_projections_{target_year}.csv'))
    h['name_key'] = h['Name'].map(_norm)
    p['name_key'] = p['Name'].map(_norm)
    return h, p


def _row_for(df, name, team=None):
    """Best-effort match a slate name to a projection row by normalized name.

    When several projection rows share the name (two real players — e.g. the
    Dodgers' vs the Athletics' Max Muncy), ``team`` disambiguates by canonical
    team code. If team can't pick a single row it falls back to the first, which
    is only reached for a *single* slate occurrence of the name (a true two-on-
    the-slate collision is resolved up front by :func:`resolve_collisions`, which
    also uses assignment-by-elimination and can fail safe)."""
    k = _norm(name)
    hit = df[df['name_key'] == k]
    if len(hit) > 1 and team is not None:
        ct = canonical_team(team)
        tmatch = hit[hit['Team'].map(canonical_team) == ct]
        if len(tmatch) == 1:
            return tmatch.iloc[0]
    if len(hit):
        return hit.iloc[0]
    # last-name + first-initial fallback
    parts = k.split()
    if len(parts) >= 2:
        cand = df[df['name_key'].str.endswith(parts[-1]) &
                  df['name_key'].str.startswith(parts[0][0])]
        if len(cand) == 1:
            return cand.iloc[0]
    return None


def sim_name(name, team):
    """The identity a colliding player is keyed by through the sim/pool chain:
    ``"<name> (<CANON_TEAM>)"``. Applied ONLY to names that collide on a slate,
    so non-colliding players keep their plain name everywhere (zero blast radius
    for the 99% case). ``dk_ids`` strips the suffix back off for upload id
    resolution, and the canonical team makes the key reconcile between the sim
    side (slate team) and the pool side (DK team)."""
    ct = canonical_team(team) or str(team or "").upper()
    return f"{name} ({ct})"


def resolve_collisions(slate, hproj):
    """Resolve same-name hitter collisions on a slate to distinct projection rows.

    A collision = the same normalized name rostered on 2+ teams on this slate;
    only then do the players later share one sim array, so only then must we
    separate them. Each colliding slate player is matched to its own projection
    row by (1) canonical team, then (2) assignment-by-elimination (2 players / 2
    rows, one resolvable → the other falls out). Anything still ambiguous is
    flagged UNRESOLVED so the caller can fail safe (drop, never mis-score).

    Returns
    -------
    assign      {(name_key, canon_team): projection_row_index}
    unresolved  {(name_key, canon_team)} — colliding players we couldn't place
    collided    {name_key} — names that collide on the slate (need a sim_name)
    """
    from collections import defaultdict
    occ = defaultdict(list)                       # name_key -> ordered canon teams on slate
    for g in slate.get('games', {}).values():
        for side in ('away', 'home'):
            ct = canonical_team(g[side])
            for p in g['lineups'][side]:
                nk = _norm(p['name'])
                if ct not in occ[nk]:
                    occ[nk].append(ct)

    # A name needs disambiguation if a same-name TWIN could exist: either 2+ teams
    # roster it on this slate, OR the projection set holds 2+ rows for it (its twin
    # simply isn't in a posted lineup right now). The latter matters because the DK
    # salary feed still lists BOTH players, so a single lineup occurrence keyed by
    # a plain name would later be shared with its twin in build_pool.
    proj_counts = hproj['name_key'].value_counts()
    ambiguous = set(proj_counts[proj_counts >= 2].index)

    assign, unresolved, collided = {}, set(), set()
    for nk, teams in occ.items():
        if len(teams) < 2 and nk not in ambiguous:
            continue                              # unique name — nothing to separate
        collided.add(nk)
        rows = hproj[hproj['name_key'] == nk]
        if len(rows) < 2:
            if len(rows) == 1 and len(teams) == 1:   # 1 row, 1 slate team → it's them
                assign[(nk, teams[0])] = rows.index[0]
            else:                                 # can't separate → fail safe
                unresolved.update((nk, t) for t in teams)
            continue
        row_team = {idx: canonical_team(r['Team']) for idx, r in rows.iterrows()}
        remaining = set(row_team)
        for t in teams:                           # pass 1: confident team match
            m = [i for i in remaining if row_team[i] == t]
            if len(m) == 1:
                assign[(nk, t)] = m[0]
                remaining.discard(m[0])
        left = [t for t in teams if (nk, t) not in assign]
        # pass 2: drop rows that confidently belong to a team NOT on this slate
        # (the twin who isn't playing) so they don't absorb a real occurrence.
        for i in list(remaining):
            rt = row_team[i]
            if rt in _VALID_TEAMS and rt not in teams:
                remaining.discard(i)
        if len(left) == 1 and len(remaining) == 1:   # pass 3: elimination
            assign[(nk, left[0])] = remaining.pop()
        else:
            unresolved.update((nk, t) for t in left)
    return assign, unresolved, collided


def _hitter_vector(row, opp_hand, venue_pf):
    """Pick the vL/vR split for the opposing pitcher's hand, then apply the
    GAME-VENUE park factors at full strength (every PA in this game is at that
    park, so no 81/81 home/road blend — that blend only belongs in the season
    `_park` columns). `venue_pf` maps a lowercase event ('p_hr', ...) to a
    multiplicative Statcast factor (1.0 = neutral). Applies to BOTH lineups —
    the visiting hitter now inherits the venue exactly like the home hitter.
    Returns the per-PA event dict + R/RBI/SB rates the simulator expects."""
    suffix = '_vL' if opp_hand == 'L' else '_vR'

    def get(ev):
        # prefer the (park-neutral) handedness split; fall back to neutral base.
        # We deliberately do NOT fall back to the season `_park` column, so the
        # venue factor below is never double-applied on top of a baked-in park.
        for s in (suffix, ''):
            col = ev + s
            if col in row and pd.notna(row[col]):
                return max(0.0, float(row[col]))
        return 0.0

    p_1b, p_2b, p_3b, p_hr = get('P_1B'), get('P_2B'), get('P_3B'), get('P_HR')
    p_bb, p_hbp, p_k = get('P_BB'), get('P_HBP'), get('P_K')

    # ── Game-venue park factors (full strength) ──────────────────────────────
    # P_HBP / P_SF have no published factor -> passthrough. The sim treats the
    # leftover PA mass as in-play outs, so we don't renormalize here.
    p_1b *= venue_pf.get('p_1b', 1.0)
    p_2b *= venue_pf.get('p_2b', 1.0)
    p_3b *= venue_pf.get('p_3b', 1.0)
    p_hr *= venue_pf.get('p_hr', 1.0)
    p_bb *= venue_pf.get('p_bb', 1.0)
    p_k  *= venue_pf.get('p_k', 1.0)

    # R / RBI / SB come from the projection (team-context + lineup slot already baked in)
    r_pa  = float(row['P_R'])  if 'P_R'  in row and pd.notna(row['P_R'])  else 0.11
    rbi_pa = float(row['P_RBI']) if 'P_RBI' in row and pd.notna(row['P_RBI']) else 0.11
    p_sb  = float(row['P_SB']) if 'P_SB' in row and pd.notna(row['P_SB']) else 0.0
    slot  = int(round(float(row['Pred_lineup_slot']))) if 'Pred_lineup_slot' in row and pd.notna(row['Pred_lineup_slot']) else 5

    return dict(p_1b=p_1b, p_2b=p_2b, p_3b=p_3b, p_hr=p_hr, p_bb=p_bb, p_hbp=p_hbp, p_k=p_k,
                r_pa=r_pa, rbi_pa=rbi_pa, p_sb=p_sb, proj_slot=slot)


def _pitcher_vector(row, opp_lineup_hand_share, venue_pf):
    """Blend the pitcher's vL/vR splits by the L/R share of the lineup faced,
    then apply the GAME-VENUE park factors at full strength to the per-PA event
    rates (both pitchers in the game get the venue, not just the home arm).
    ra9/era/tbf are left as the pitcher's neutral-skill anchors; the park's
    run effect flows into the sim's ER through the (now park-inflated) hit and
    HR counts in the earned-run traffic term."""
    sL = opp_lineup_hand_share.get('L', 0.4)
    sR = 1.0 - sL

    def get(ev):
        vL = row.get(ev + '_vL'); vR = row.get(ev + '_vR')
        if pd.notna(vL) and pd.notna(vR):
            v = sL * float(vL) + sR * float(vR)
        else:
            v = float(row.get(ev, 0.0) or 0.0)
        return max(0.0, v)

    p_1b, p_2b, p_3b, p_hr = get('P_1B'), get('P_2B'), get('P_3B'), get('P_HR')
    p_bb, p_hbp, p_k = get('P_BB'), get('P_HBP'), get('P_K')
    # ── Game-venue park factors (full strength) on the rate events ───────────
    p_1b *= venue_pf.get('p_1b', 1.0)
    p_2b *= venue_pf.get('p_2b', 1.0)
    p_3b *= venue_pf.get('p_3b', 1.0)
    p_hr *= venue_pf.get('p_hr', 1.0)
    p_bb *= venue_pf.get('p_bb', 1.0)
    p_k  *= venue_pf.get('p_k', 1.0)
    hits_per_bf = p_1b + p_2b + p_3b + p_hr
    return dict(k_pct=p_k, bb_pct=p_bb, hbp_per_bf=p_hbp,
                h_per_bf=hits_per_bf, hr_per_bf=p_hr,
                era=float(row['ERA']) if 'ERA' in row and pd.notna(row['ERA']) else 4.5,
                ra9=float(row['RA9']) if 'RA9' in row and pd.notna(row['RA9']) else 4.7,
                tbf_per_ip=float(row['TBF_per_IP']) if 'TBF_per_IP' in row and pd.notna(row['TBF_per_IP']) else 4.3,
                # The pitcher's established outing length. The sim uses this to
                # govern a "primary"/bulk arm's workload so a short reliever
                # tagged as the primary in a bullpen game is NOT simulated as a
                # stretched-out ~5-IP starter (see sim_proj primary branch).
                ip_per_g=float(row['weighted_IP_per_G']) if 'weighted_IP_per_G' in row and pd.notna(row['weighted_IP_per_G']) else 5.0,
                wp_per_pa=float(row['Pred_WP_per_PA']) if 'Pred_WP_per_PA' in row and pd.notna(row['Pred_WP_per_PA']) else 0.0,
                hand=row.get('PitchHand', 'R'))


def build_matchup_inputs(slate, hproj, pproj):
    """Return per-player projection-derived inputs keyed for the simulator.

    matchup = {
      'date', 'games': { gid: {
          away, home, datetime, implied,
          pitchers: {side: {role: {name, hand, vec}}},
          lineups:  {side: [ {name, slot, pos, hand, vec} ]},
          lineup_hand_share: {side: {'L':x,'R':y}},
      }}}
    Players missing from the projection set are flagged (vec=None) so the
    caller can decide to skip or fall back.
    """
    # hitter hand lookup from projection BatSide
    def hitter_hand(name):
        r = _row_for(hproj, name)
        return (r['BatSide'] if r is not None and pd.notna(r['BatSide']) else 'R')

    def pitcher_hand(name):
        r = _row_for(pproj, name)
        return (r['PitchHand'] if r is not None and pd.notna(r['PitchHand']) else 'R')

    out = {'date': slate['date'], 'games': {}}
    missing = {'hitters': [], 'pitchers': []}

    # Same-name collisions (two real players sharing a name on the slate — e.g.
    # the Dodgers' vs the Athletics' Max Muncy). Resolve each to its own
    # projection row up front so they no longer collapse onto one sim array
    # downstream; unresolved ones are dropped (never mis-scored). Colliding
    # players are keyed by sim_name() through the sim/pool chain.
    assign, unresolved, collided = resolve_collisions(slate, hproj)
    collisions = {'resolved': [], 'dropped': []}

    def _hitter_row_and_key(name, team):
        """(projection row or None, sim_name) for a slate hitter, honouring the
        collision resolution. A dropped collision returns (None, None)."""
        nk = _norm(name)
        if nk not in collided:
            return _row_for(hproj, name, team=team), name
        ct = canonical_team(team)
        if (nk, ct) in assign:
            row = hproj.loc[assign[(nk, ct)]]
            snm = sim_name(name, team)
            collisions['resolved'].append((snm, str(row.get('Team', ''))))
            return row, snm
        collisions['dropped'].append(sim_name(name, team))   # unresolved → fail safe
        return None, None

    for gid, g in slate['games'].items():
        rec = {'away': g['away'], 'home': g['home'], 'datetime': g['datetime'],
               'implied': g['implied'],
               'total_scale': g.get('total_scale', {'away': 1.0, 'home': 1.0}),
               'pitchers': {}, 'lineups': {},
               'lineup_hand_share': {}, 'lineup_source': g['lineup_source']}

        # resolve pitcher hands first (needed for hitter splits)
        opp_hand = {}
        for side in ('away', 'home'):
            ps = g['pitchers'][side]
            face = ps.get('opener') or ps.get('starter')
            opp_hand_side = pitcher_hand(face) if face else 'R'
            # store the hand the OTHER side's hitters will see
            opp_hand['home' if side == 'away' else 'away'] = opp_hand_side

        # Game-venue park factors (the HOME team's park). These apply to BOTH
        # lineups and BOTH pitchers, since every PA in this game is at that park.
        venue_rows = [_row_for(hproj, p['name']) for p in g['lineups']['home']]
        for rk in ('starter', 'opener', 'primary'):
            nm = g['pitchers']['home'].get(rk)
            if nm:
                venue_rows.append(_row_for(pproj, nm))
        venue_pf = _venue_factors(venue_rows)
        rec['venue_pf'] = venue_pf

        for side in ('away', 'home'):
            # lineup hand share (for pitcher split weighting): count L vs R+S→R bats
            hands = []
            lineup_vecs = []
            for p in g['lineups'][side]:
                r, snm = _hitter_row_and_key(p['name'], g[side])
                hand = (r['BatSide'] if r is not None and pd.notna(r['BatSide']) else 'R')
                hands.append(hand)
                vec = _hitter_vector(r, opp_hand[side], venue_pf) if r is not None else dict(LG_HIT_VEC)
                if r is None:
                    missing['hitters'].append(p['name'])
                # snm is the sim identity (plain name, or "<name> (<TEAM>)" for a
                # resolved collision); None means a dropped/unresolved collision.
                lineup_vecs.append({'name': p['name'],
                                    'sim_name': snm if snm is not None else p['name'],
                                    'dropped': snm is None,
                                    'slot': p['slot'], 'pos': p['pos'],
                                    'hand': hand, 'vec': vec})
            rec['lineups'][side] = lineup_vecs
            # switch hitters (S) count as opposite of the pitcher → treat as the platoon-favorable side;
            # for share purposes count S as L when facing RHP, R when facing LHP
            nL = sum(1 for h in hands if h == 'L') + sum(1 for h in hands if h == 'S')
            denom = max(len(hands), 1)
            rec['lineup_hand_share'][side] = {'L': nL / denom, 'R': 1 - nL / denom}

        # pitchers: weight splits by the lineup they face
        for side in ('away', 'home'):
            opp_side = 'home' if side == 'away' else 'away'
            share = rec['lineup_hand_share'][opp_side]
            ps = g['pitchers'][side]
            rec['pitchers'][side] = {}
            for role_key in ('starter', 'opener', 'primary'):
                nm = ps.get(role_key)
                if not nm:
                    continue
                r = _row_for(pproj, nm)
                if r is None:
                    missing['pitchers'].append(nm)
                    vec = dict(LG_PIT_VEC); hand = 'R'
                else:
                    vec = _pitcher_vector(r, share, venue_pf); hand = vec['hand']
                # fold in the QUALITY of the lineup this arm faces (log5 matchup),
                # so facing an elite offense docks him and a weak one lifts him
                vec = _opponent_adjust_pitcher(vec, rec['lineups'][opp_side])
                rec['pitchers'][side][role_key] = {'name': nm, 'hand': hand, 'vec': vec}
        out['games'][gid] = rec

    out['missing'] = {k: sorted(set(v)) for k, v in missing.items()}
    out['collisions'] = {
        'resolved': sorted(set(collisions['resolved'])),
        'dropped': sorted(set(collisions['dropped'])),
    }
    return out

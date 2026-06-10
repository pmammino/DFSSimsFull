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
                  tbf_per_ip=4.35, wp_per_pa=0.0, hand='R')


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


def _row_for(df, name):
    """Best-effort match a slate name to a projection row by normalized name."""
    k = _norm(name)
    hit = df[df['name_key'] == k]
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


def _hitter_vector(row, opp_hand, at_home):
    """Pick the vL/vR split for the opposing pitcher's hand; park-adjust at home.
    Returns the per-PA event dict + R/RBI/SB rates the simulator expects."""
    suffix = '_vL' if opp_hand == 'L' else '_vR'

    def get(ev):
        # prefer handedness split; fall back to park, then neutral
        for s in (suffix, '_park', ''):
            col = ev + s
            if col in row and pd.notna(row[col]):
                v = float(row[col])
                # if using the split, blend a touch of park when batting at home
                if s == suffix and at_home:
                    pcol = ev + '_park'
                    if pcol in row and pd.notna(row[pcol]):
                        # park already baked into neutral->park; nudge split by park/neutral ratio
                        base = row.get(ev, np.nan)
                        if pd.notna(base) and base:
                            v *= float(row[pcol]) / float(base)
                return max(0.0, v)
        return 0.0

    p_1b, p_2b, p_3b, p_hr = get('P_1B'), get('P_2B'), get('P_3B'), get('P_HR')
    p_bb, p_hbp, p_k = get('P_BB'), get('P_HBP'), get('P_K')

    # R / RBI / SB come from the projection (team-context + lineup slot already baked in)
    r_pa  = float(row['P_R'])  if 'P_R'  in row and pd.notna(row['P_R'])  else 0.11
    rbi_pa = float(row['P_RBI']) if 'P_RBI' in row and pd.notna(row['P_RBI']) else 0.11
    p_sb  = float(row['P_SB']) if 'P_SB' in row and pd.notna(row['P_SB']) else 0.0
    slot  = int(round(float(row['Pred_lineup_slot']))) if 'Pred_lineup_slot' in row and pd.notna(row['Pred_lineup_slot']) else 5

    return dict(p_1b=p_1b, p_2b=p_2b, p_3b=p_3b, p_hr=p_hr, p_bb=p_bb, p_hbp=p_hbp, p_k=p_k,
                r_pa=r_pa, rbi_pa=rbi_pa, p_sb=p_sb, proj_slot=slot)


def _pitcher_vector(row, opp_lineup_hand_share, at_home):
    """Blend the pitcher's vL/vR splits by the L/R share of the lineup faced."""
    sL = opp_lineup_hand_share.get('L', 0.4)
    sR = 1.0 - sL

    def get(ev):
        vL = row.get(ev + '_vL'); vR = row.get(ev + '_vR')
        if pd.notna(vL) and pd.notna(vR):
            v = sL * float(vL) + sR * float(vR)
        else:
            v = float(row.get(ev + '_park', row.get(ev, 0.0)) or 0.0)
        return max(0.0, v)

    p_1b, p_2b, p_3b, p_hr = get('P_1B'), get('P_2B'), get('P_3B'), get('P_HR')
    p_bb, p_hbp, p_k = get('P_BB'), get('P_HBP'), get('P_K')
    hits_per_bf = p_1b + p_2b + p_3b + p_hr
    return dict(k_pct=p_k, bb_pct=p_bb, hbp_per_bf=p_hbp,
                h_per_bf=hits_per_bf, hr_per_bf=p_hr,
                era=float(row['ERA']) if 'ERA' in row and pd.notna(row['ERA']) else 4.5,
                ra9=float(row['RA9']) if 'RA9' in row and pd.notna(row['RA9']) else 4.7,
                tbf_per_ip=float(row['TBF_per_IP']) if 'TBF_per_IP' in row and pd.notna(row['TBF_per_IP']) else 4.3,
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

    for gid, g in slate['games'].items():
        rec = {'away': g['away'], 'home': g['home'], 'datetime': g['datetime'],
               'implied': g['implied'], 'pitchers': {}, 'lineups': {},
               'lineup_hand_share': {}, 'lineup_source': g['lineup_source']}

        # resolve pitcher hands first (needed for hitter splits)
        opp_hand = {}
        for side in ('away', 'home'):
            ps = g['pitchers'][side]
            face = ps.get('opener') or ps.get('starter')
            opp_hand_side = pitcher_hand(face) if face else 'R'
            # store the hand the OTHER side's hitters will see
            opp_hand['home' if side == 'away' else 'away'] = opp_hand_side

        for side in ('away', 'home'):
            at_home = (side == 'home')
            # lineup hand share (for pitcher split weighting): count L vs R+S→R bats
            hands = []
            lineup_vecs = []
            for p in g['lineups'][side]:
                r = _row_for(hproj, p['name'])
                hand = (r['BatSide'] if r is not None and pd.notna(r['BatSide']) else 'R')
                hands.append(hand)
                vec = _hitter_vector(r, opp_hand[side], at_home) if r is not None else dict(LG_HIT_VEC)
                if r is None:
                    missing['hitters'].append(p['name'])
                lineup_vecs.append({'name': p['name'], 'slot': p['slot'], 'pos': p['pos'],
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
            at_home = (side == 'home')
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
                    vec = _pitcher_vector(r, share, at_home); hand = vec['hand']
                rec['pitchers'][side][role_key] = {'name': nm, 'hand': hand, 'vec': vec}
        out['games'][gid] = rec

    out['missing'] = {k: sorted(set(v)) for k, v in missing.items()}
    return out

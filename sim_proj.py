"""
sim_proj.py — correlated DK simulator driven by projection-engine per-PA vectors.

Unlike the earlier stats-based sim, the per-PA event probabilities here come
straight from the handedness-split projection pipeline (BIP/XGBoost expected
outcomes), already matchup- and park-adjusted in matchup.py. This module only
adds: the correlated game/team latent shocks, opener/primary workload, TBF-based
pitcher innings, and DraftKings scoring.

Correlation design is unchanged and still validates to:
  teammate H-H ~ +0.24 | unrelated ~0 | hitter vs opposing SP ~ -0.37
"""
import numpy as np

# ── DK scoring ────────────────────────────────────────────────────────────────
def dk_hitter(s, d, t, hr, rbi, r, bb, hbp, sb):
    return s*3 + d*5 + t*8 + hr*10 + rbi*2 + r*2 + bb*2 + hbp*2 + sb*5

def dk_pitcher(outs, k, win, er, h, bb, hbp, cg, cgs, nh):
    return outs*0.75 + k*2 + win*4 - er*2 - h*0.6 - bb*0.6 - hbp*0.6 + cg*2.5 + cgs*2.5 + nh*5

# correlation loadings (validated)
SG, ST, SI, SG_HR_EXTRA = 0.20, 0.50, 0.30, 0.12
OPENER_BF_MEAN, OPENER_BF_SD = 4.6, 1.3


def _pa_per_game(slot):
    return 4.2 - (slot - 1) * 0.055


def _sim_pitcher(vec, bf_sim, m_opp, z, can_win, win_base, outs_cap, rng, n):
    bf_sim = np.clip(bf_sim, 0, 40).astype(int)
    # outs/BF implied by TBF_per_IP (batters per inning -> outs per batter)
    outs_per_bf = max(0.55, min(0.80, 3.0 / max(vec['tbf_per_ip'], 3.2)))
    outs = np.clip(np.round(bf_sim * outs_per_bf).astype(int), 0, outs_cap)
    ip = outs / 3.0
    k  = rng.binomial(bf_sim, min(0.6, vec['k_pct']))
    h_r  = np.clip(vec['h_per_bf']  * m_opp, 0.05, 0.60)
    hr_r = np.clip(vec['hr_per_bf'] * m_opp, 0.003, 0.12)
    bb_r = np.clip(vec['bb_pct']    * np.sqrt(m_opp), 0.02, 0.25)
    h  = rng.binomial(bf_sim, h_r)
    hr = rng.binomial(bf_sim, hr_r)
    bb = rng.binomial(bf_sim, bb_r)
    hbp = rng.binomial(bf_sim, min(0.05, vec['hbp_per_bf']))
    # earned runs anchored to projected RA9 (per-IP), correlated to traffic + shock
    ra9 = vec.get('ra9', 4.6)
    er_mean = ra9 * ip / 9.0
    er = np.round(np.clip(er_mean + (h*0.10 + hr*0.55 + bb*0.07 - 0.9)
                          + z * 0.5 + rng.normal(0, 0.6, n), 0, 15)).astype(int)
    if can_win:
        wp = np.clip(win_base - er * 0.04 - z * 0.03, 0.02, 0.85)
        win = ((ip >= 5.0) & (rng.uniform(0, 1, n) < wp)).astype(int)
    else:
        win = np.zeros(n, int)
    cg = (ip >= 9.0).astype(int); cgs = (cg & (er == 0)).astype(int); nh = (cgs & (h == 0)).astype(int)
    dk = dk_pitcher(outs, k, win, er, h, bb, hbp, cg, cgs, nh)
    return dict(dk=dk, ip=ip, k=k, bb=bb, h=h, hr=hr, er=er, win=win, bf=bf_sim)


def simulate(matchup, n_sims=10000, seed=20260610):
    rng = np.random.default_rng(seed)
    n = n_sims
    hitter_dk, pitcher_dk, hitter_stat = {}, {}, {}
    hrows, prows = [], []

    for gid, g in matchup['games'].items():
        away, home = g['away'], g['home']
        label = f"{away}@{home}"
        Lg = rng.standard_normal(n)
        Lt = {'away': rng.standard_normal(n), 'home': rng.standard_normal(n)}
        shared = {'away': SG*Lg + ST*Lt['away'], 'home': SG*Lg + ST*Lt['home']}

        # ---- hitters ----
        for side in ('away', 'home'):
            sh = shared[side]; sh_var = float(np.var(sh))
            implied = g['implied'][side]
            # user team-total override rescales this team's offense (1.0 = none)
            ts = float(g.get('total_scale', {}).get(side, 1.0) or 1.0)
            for p in g['lineups'][side]:
                vec = p['vec']
                if vec is None:
                    continue  # no projection -> skip (reported in missing list)
                slot = p['slot'] if p['slot'] else vec['proj_slot']
                pa_mean = _pa_per_game(slot)
                pa = np.clip(rng.poisson(pa_mean, n), 1, 7)
                idio = SI * rng.standard_normal(n)
                m_off = ts * np.exp(sh + idio - 0.5*(sh_var + SI**2))
                m_hr  = ts * np.exp(sh + idio + SG_HR_EXTRA*Lg - 0.5*(sh_var + SI**2 + SG_HR_EXTRA**2))

                p_hr = np.clip(vec['p_hr'] * m_hr, 0.001, 0.20)
                p_3b = np.clip(vec['p_3b'] * m_off, 0, 0.05)
                p_2b = np.clip(vec['p_2b'] * m_off, 0, 0.20)
                p_1b = np.clip(vec['p_1b'] * m_off, 0, 0.55)
                p_bb = np.clip(vec['p_bb'] * np.sqrt(m_off), 0, 0.30)
                p_hbp = np.full(n, vec['p_hbp']); p_k = np.full(n, vec['p_k'])
                c_hr = p_hr; c_3b = c_hr+p_3b; c_2b = c_3b+p_2b; c_1b = c_2b+p_1b
                c_bb = c_1b+p_bb; c_hbp = c_bb+p_hbp; c_k = c_hbp+p_k

                sgl=np.zeros(n,int); dbl=np.zeros(n,int); trp=np.zeros(n,int); hr=np.zeros(n,int)
                bb=np.zeros(n,int); hbp=np.zeros(n,int); ks=np.zeros(n,int)
                for i in range(int(pa.max())):
                    a = pa > i
                    if not a.any(): break
                    u = rng.uniform(0,1,a.sum())
                    chr_=c_hr[a];c3=c_3b[a];c2=c_2b[a];c1=c_1b[a];cb=c_bb[a];chb=c_hbp[a];ck=c_k[a]
                    hr[a]+=(u<chr_).astype(int); trp[a]+=((u>=chr_)&(u<c3)).astype(int)
                    dbl[a]+=((u>=c3)&(u<c2)).astype(int); sgl[a]+=((u>=c2)&(u<c1)).astype(int)
                    bb[a]+=((u>=c1)&(u<cb)).astype(int); hbp[a]+=((u>=cb)&(u<chb)).astype(int)
                    ks[a]+=((u>=chb)&(u<ck)).astype(int)

                r_s   = rng.poisson(np.clip(vec['r_pa']   * pa * m_off, 0, 6))
                rbi_s = rng.poisson(np.clip(vec['rbi_pa'] * pa * m_off, 0, 8))
                sb_s  = rng.poisson(np.clip(vec['p_sb']   * pa,         0, 3))

                dk = dk_hitter(sgl, dbl, trp, hr, rbi_s, r_s, bb, hbp, sb_s)
                nm = p['name']
                hitter_dk[nm] = dk
                hitter_stat[nm] = {'1B':sgl,'2B':dbl,'3B':trp,'HR':hr,'R':r_s,'RBI':rbi_s,
                                   'BB':bb,'HBP':hbp,'K':ks,'SB':sb_s,'PA':pa}
                opp_sp = (g['pitchers']['home' if side=='away' else 'away'].get('opener')
                          or g['pitchers']['home' if side=='away' else 'away'].get('starter') or {}).get('name')
                hrows.append(_hrow(nm, p, side, label, g, implied, opp_sp, dk,
                                   dict(hr=hr,r=r_s,rbi=rbi_s,bb=bb,sb=sb_s)))

        # ---- pitchers ----
        for side in ('away','home'):
            opp_side = 'home' if side=='away' else 'away'
            sho = shared[opp_side]; m_opp = np.exp(sho - 0.5*float(np.var(sho)))
            z = (sho - sho.mean())/(sho.std()+1e-9)
            implied = g['implied'][side]; opp_implied = g['implied'][opp_side]
            ps = g['pitchers'][side]
            win_base = lambda vec: max(0.20, min(0.75, 0.500 + (4.20-vec['era'])*0.03 + (implied-opp_implied)*0.04))

            has_op = 'opener' in ps and 'primary' in ps
            if has_op and ps['opener']['vec'] and ps['primary']['vec']:
                vo = ps['opener']['vec']
                bf_o = np.clip(rng.normal(OPENER_BF_MEAN, OPENER_BF_SD, n), 2, 9)
                seg_o = _sim_pitcher(vo, bf_o, m_opp, z, False, 0.0, 9, rng, n)
                vp = ps['primary']['vec']
                bf_p = np.clip(rng.normal(vp['tbf_per_ip']*5.0, 3.0, n) - z*2.5 - (opp_implied-4.5)*0.8, 6, 30)
                seg_p = _sim_pitcher(vp, bf_p, m_opp, z, True, win_base(vp), 24, rng, n)
                for role, info, seg in [('OPENER', ps['opener'], seg_o), ('PRIMARY', ps['primary'], seg_p)]:
                    pitcher_dk[info['name']] = seg['dk']
                    prows.append(_prow(info['name'], role, side, label, g, opp_side, opp_implied, seg))
            else:
                role_key = 'starter' if 'starter' in ps else next(iter(ps), None)
                if role_key is None: continue
                info = ps[role_key]
                if not info or info['vec'] is None: continue
                v = info['vec']
                bf = np.clip(rng.normal(v['tbf_per_ip']*5.6, 3.0, n) - z*3.0
                             - (v['era']-4.20)*0.8 + (opp_implied-4.5)*1.0, 8, 34)
                seg = _sim_pitcher(v, bf, m_opp, z, True, win_base(v), 27, rng, n)
                pitcher_dk[info['name']] = seg['dk']
                prows.append(_prow(info['name'], 'STARTER', side, label, g, opp_side, opp_implied, seg))

    return hitter_dk, pitcher_dk, hitter_stat, hrows, prows, matchup.get('missing', {})


def _pct(a,q): return round(float(np.percentile(a,q)),2)

def _hrow(nm,p,side,label,g,implied,opp_sp,dk,m):
    return dict(player=nm,pos=p['pos'],slot=p['slot'],bat=p['hand'],team=g[side],
        side=side.upper(),game=label,datetime=g['datetime'],
        lineup_source=g['lineup_source'][side],opp_sp=opp_sp,team_total=implied,
        proj=round(float(dk.mean()),3),floor_p25=_pct(dk,25),median_p50=_pct(dk,50),
        ceil_p75=_pct(dk,75),p10=_pct(dk,10),p90=_pct(dk,90),ceiling_p99=_pct(dk,99),
        std=round(float(dk.std()),3),mean_hr=round(float(m['hr'].mean()),3),
        mean_r=round(float(m['r'].mean()),3),mean_rbi=round(float(m['rbi'].mean()),3),
        mean_bb=round(float(m['bb'].mean()),3),mean_sb=round(float(m['sb'].mean()),3),
        p_2x=round(float((dk>=2*dk.mean()).mean()),4) if dk.mean()>0 else 0.0,
        p_30=round(float((dk>=30).mean()),4))

def _prow(nm,role,side,label,g,opp_side,opp_implied,seg):
    dk=seg['dk']
    return dict(player=nm,pos='SP',role=role,team=g[side],side=side.upper(),game=label,
        datetime=g['datetime'],opp=g[opp_side],opp_total=opp_implied,
        proj=round(float(dk.mean()),3),floor_p25=_pct(dk,25),median_p50=_pct(dk,50),
        ceil_p75=_pct(dk,75),p10=_pct(dk,10),p90=_pct(dk,90),ceiling_p99=_pct(dk,99),
        std=round(float(dk.std()),3),mean_ip=round(float(seg['ip'].mean()),3),
        mean_bf=round(float(seg['bf'].mean()),2),mean_k=round(float(seg['k'].mean()),3),
        mean_bb=round(float(seg['bb'].mean()),3),mean_er=round(float(seg['er'].mean()),3),
        mean_h=round(float(seg['h'].mean()),3),mean_hr=round(float(seg['hr'].mean()),3),
        win_pct=round(float(seg['win'].mean()),4),
        p_qs=round(float(((seg['ip']>=6.0)&(seg['er']<=3)).mean()),4),
        p_30=round(float((dk>=30).mean()),4))

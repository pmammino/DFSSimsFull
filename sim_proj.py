"""
sim_proj.py — correlated DK simulator driven by projection-engine per-PA vectors.

Unlike the earlier stats-based sim, the per-PA event probabilities here come
straight from the handedness-split projection pipeline (BIP/XGBoost expected
outcomes), already matchup- and park-adjusted in matchup.py. This module only
adds: the correlated game/team latent shocks, opener/primary workload, TBF-based
pitcher innings, and DraftKings scoring.

Correlation design is unchanged and still validates to:
  teammate H-H ~ +0.24 | unrelated ~0 | hitter vs opposing SP ~ -0.37

TEAM-TOTAL OVERRIDES (total_scale)
----------------------------------
A per-side `total_scale` (from a user team-total override; 1.0 = none) reshapes
that side's offense three ways:
  1. MEAN   — m_off / m_hr / r / rbi scale linearly with the total.
  2. CEILING— the side's shared latent is widened by TOTAL_CEIL_GAIN·(scale-1),
              so a higher total fattens the whole lineup's upside (it booms
              together more on its big days), not just its average.
  3. PITCHER— the opposing starter's hits/HR/runs allowed carry the offense's
              scale, so boosting a lineup also tanks the arm it faces.
scale == 1.0 reproduces the validated baseline above exactly, so the defaults
(and the published correlations) are untouched when no total is overridden.
"""
import numpy as np

# ── DK scoring ────────────────────────────────────────────────────────────────
def dk_hitter(s, d, t, hr, rbi, r, bb, hbp, sb):
    return s*3 + d*5 + t*8 + hr*10 + rbi*2 + r*2 + bb*2 + hbp*2 + sb*5

def dk_pitcher(outs, k, win, er, h, bb, hbp, cg, cgs, nh):
    return outs*0.75 + k*2 + win*4 - er*2 - h*0.6 - bb*0.6 - hbp*0.6 + cg*2.5 + cgs*2.5 + nh*5

# correlation loadings (validated: teammate H-H ~+0.24, hitter vs opposing SP
# ~-0.37). Left at the validated values on purpose. Recalibration testing against
# 5 real DK GPP contest-standings (target winner/median ~2.06x, p99/median
# ~1.70x) showed that shrinking these loadings is NOT an effective tail lever —
# a 40% shrink (which craters teammate correlation to 0.16) only moves a matched
# field's winner/median 2.79x -> 2.39x, and even a mild shrink weakens the
# hitter-vs-SP anti-correlation below its target. The per-lineup ceiling excess
# lives in the MARGINAL per-player score tail, not the correlation structure. The
# R/RBI team-run-conservation fix above is the effective tail change (it cut the
# extreme stack max ~25% and stack p99 ~8%); closing the remaining gap needs a
# per-player PIT calibration against same-slate (walk-forward) sims, not a knob.
SG, ST, SI, SG_HR_EXTRA = 0.20, 0.50, 0.30, 0.12
OPENER_BF_MEAN, OPENER_BF_SD = 4.6, 1.3

# How much a team-total override reshapes UPSIDE on top of shifting the mean.
# An above-average total (scale > 1) widens that team's shared latent so the
# whole lineup booms together on its big days — a higher total buys a fatter
# CEILING, not just a higher average; a below-average total compresses it. The
# per-side team loading becomes ST·clip(1 + TOTAL_CEIL_GAIN·(scale-1), 0.5, 2.0),
# so scale == 1.0 (no override) reproduces the validated baseline exactly.
TOTAL_CEIL_GAIN = 0.75

# ── pitcher per-batter model (coherent outing + endogenous hook) ───────────────
# The old pitcher sim drew outs, K, H, HR, BB independently off one batters-faced
# count and anchored ER to a near-deterministic RA9·IP/9, so innings were
# DECOUPLED from how the outing actually went — a shelling never shortened the
# start and ER carried almost no dispersion. A walk-forward grade against real DK
# results (Jul 2026, 83 starts) showed the downside tail was far too thin:
# P(DK<0) was 4% in-sim vs ~11% observed and per-player std ~7.7 vs ~10.4. This
# model instead plays the outing out batter-by-batter (mirroring the hitter
# per-PA loop): each PA is one coherent outcome, runners advance with an inning
# that clears every three outs, and the manager PULLS the starter once earned
# runs cross a per-sim hook tolerance. Bad outings now self-truncate to few outs
# AND many ER, which is exactly how a real negative DK line is produced. The run
# rate and hook are tuned so the MEAN is preserved across the quality spectrum
# (bias ~0 vs the old model) while std widens to ~10 and P(DK<0) rises into the
# observed band — the middle of the distribution (cruising starts) is unchanged.
HIT_RUN_ADV   = 0.34   # P(a given runner scores) on a non-HR hit (mean-preserving)
HOOK_ER_MEAN  = 7.0    # earned runs a manager tolerates before pulling a starter
HOOK_ER_SD    = 1.8


def _pa_per_game(slot):
    return 4.2 - (slot - 1) * 0.055


def _sim_pitcher(vec, bf_sim, m_opp, z, can_win, win_base, outs_cap, rng, n):
    bf_sim = np.clip(bf_sim, 0, 40).astype(int)
    # Planned workload as a LEASH in outs (what he throws if cruising). Same
    # TBF_per_IP -> outs/BF anchor as before, so the innings ceiling is unchanged.
    outs_per_bf = max(0.55, min(0.80, 3.0 / max(vec['tbf_per_ip'], 3.2)))
    planned_outs = np.clip(np.round(bf_sim * outs_per_bf).astype(int), 0, outs_cap)

    # Per-batter outcome probabilities (same matchup adjustment + clips as before).
    h_r  = np.clip(vec['h_per_bf']  * m_opp, 0.05, 0.60)   # all hits incl HR
    hr_r = np.clip(vec['hr_per_bf'] * m_opp, 0.003, 0.12)
    bb_r = np.clip(vec['bb_pct']    * np.sqrt(m_opp), 0.02, 0.25)
    hbp_r = np.full(n, min(0.05, vec['hbp_per_bf']))
    k_r   = np.full(n, min(0.60, vec['k_pct']))
    hit_nh = np.clip(h_r - hr_r, 0.0, None)               # non-HR hits
    # Keep the six named outcomes + a BIP-out remainder a valid partition.
    tot = hr_r + hit_nh + bb_r + hbp_r + k_r
    scl = np.where(tot > 0.97, 0.97 / tot, 1.0)
    hr_r = hr_r*scl; hit_nh = hit_nh*scl; bb_r = bb_r*scl; hbp_r = hbp_r*scl; k_r = k_r*scl
    c1 = hr_r; c2 = c1+hit_nh; c3 = c2+bb_r; c4 = c3+hbp_r; c5 = c4+k_r  # u>=c5 -> BIP out

    # Per-sim hook tolerance. A shock term (z) tightens the leash in a big game.
    hook = np.round(rng.normal(HOOK_ER_MEAN, HOOK_ER_SD, n) - np.clip(z, 0, None)*0.6).clip(3, 14)
    if not can_win:                       # opener / bulk-reliever: much shorter leash
        hook = np.minimum(hook, rng.integers(3, 6, n))

    outs = np.zeros(n, int); k = np.zeros(n, int); h = np.zeros(n, int); hr = np.zeros(n, int)
    bb = np.zeros(n, int); hbp = np.zeros(n, int); er = np.zeros(n, int); bf = np.zeros(n, int)
    runners = np.zeros(n, int); inn_outs = np.zeros(n, int)
    safety = planned_outs + 22            # absolute batter backstop
    for _ in range(int(planned_outs.max()) + 24):
        active = (outs < planned_outs) & (er < hook) & (bf < safety)
        if not active.any():
            break
        idx = np.where(active)[0]
        u = rng.uniform(0, 1, idx.size)
        c1a, c2a, c3a, c4a, c5a = c1[idx], c2[idx], c3[idx], c4[idx], c5[idx]
        is_hr = u < c1a; is_h = (u >= c1a) & (u < c2a); is_bb = (u >= c2a) & (u < c3a)
        is_hbp = (u >= c3a) & (u < c4a); is_k = (u >= c4a) & (u < c5a); is_out = u >= c5a
        run = runners[idx]; bf[idx] += 1
        # HR clears the bases and plates the batter
        i = idx[is_hr]; er[i] += run[is_hr] + 1; runners[i] = 0; hr[i] += 1; h[i] += 1
        # non-HR hit: existing runners score ~Binom(runners, ADV); batter to first
        i = idx[is_h]; rr = run[is_h]; sc = rng.binomial(rr, HIT_RUN_ADV)
        er[i] += sc; h[i] += 1; runners[i] = np.minimum(3, (rr - sc) + 1)
        # walk / HBP force a run only with the bases loaded
        i = idx[is_bb]; rr = runners[i]; er[i] += (rr >= 3).astype(int); runners[i] = np.minimum(3, rr+1); bb[i] += 1
        i = idx[is_hbp]; rr = runners[i]; er[i] += (rr >= 3).astype(int); runners[i] = np.minimum(3, rr+1); hbp[i] += 1
        # K and ball-in-play out each record one out
        i = idx[is_k]; k[i] += 1; outs[i] += 1; inn_outs[i] += 1
        i = idx[is_out]; outs[i] += 1; inn_outs[i] += 1
        # end of half-inning: three outs strand and clear the bases
        i = idx[inn_outs[idx] >= 3]; runners[i] = 0; inn_outs[i] = 0

    outs = np.minimum(outs, outs_cap); ip = outs / 3.0; er = np.clip(er, 0, 20)
    if can_win:
        wp = np.clip(win_base - er * 0.04 - z * 0.03, 0.02, 0.85)
        win = ((ip >= 5.0) & (rng.uniform(0, 1, n) < wp)).astype(int)
    else:
        win = np.zeros(n, int)
    cg = (ip >= 9.0).astype(int); cgs = (cg & (er == 0)).astype(int); nh = (cgs & (h == 0)).astype(int)
    dk = dk_pitcher(outs, k, win, er, h, bb, hbp, cg, cgs, nh)
    return dict(dk=dk, ip=ip, k=k, bb=bb, h=h, hr=hr, er=er, win=win, bf=bf)


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
        # per-side team-total scale (1.0 = no override). It drives the MEAN
        # (applied at m_off / m_opp below) AND, via TOTAL_CEIL_GAIN, the width of
        # the team's shared latent so a higher total fattens the whole lineup's
        # ceiling. shvar is the analytic variance of that latent (Lg, Lt are
        # independent unit normals), used for the lognormal mean-correction so the
        # mean stays exactly `scale` regardless of how wide the upside gets.
        tscale = g.get('total_scale', {}) or {}
        ts_side = {s: float(tscale.get(s, 1.0) or 1.0) for s in ('away', 'home')}
        shared, shvar = {}, {}
        for s in ('away', 'home'):
            st_eff = ST * min(2.0, max(0.5, 1.0 + TOTAL_CEIL_GAIN * (ts_side[s] - 1.0)))
            shared[s] = SG*Lg + st_eff*Lt[s]
            shvar[s] = SG**2 + st_eff**2

        # ---- hitters ----
        # Two passes per side so runs are CONSERVED at the team level. Pass 1
        # simulates every hitter's PA events. We then derive ONE team-runs total
        # per sim from those realized events (mean-anchored to the projection's
        # team-run expectation) and, in pass 2, allocate R and RBI out of that
        # shared total. This ties run production to the events that actually
        # happened and enforces team R == team RBI == team runs, instead of the
        # old independent per-player Poissons (which let R and RBI float free of
        # the box score and of each other, fattening stack ceilings).
        for side in ('away', 'home'):
            sh = shared[side]; sh_var = shvar[side]
            implied = g['implied'][side]
            ts = ts_side[side]

            plist = []   # per-hitter pass-1 state
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

                sb_s = rng.poisson(np.clip(vec['p_sb'] * pa, 0, 3))
                plist.append(dict(p=p, vec=vec, pa=pa, m_off=m_off,
                                  sgl=sgl, dbl=dbl, trp=trp, hr=hr,
                                  bb=bb, hbp=hbp, ks=ks, sb=sb_s))

            if not plist:
                continue

            # ── team run-conservation ────────────────────────────────────────
            # Event-based team run SHAPE (a positive combination of the realized
            # events; only its relative shape matters — it is rescaled below).
            team_raw = np.zeros(n)
            team_hr = np.zeros(n, int)  # Σ hr — every HR is guaranteed a run + an RBI
            r_wsum = np.zeros(n)      # Σ r_pa·pa  (projection weight for R split)
            rbi_wsum = np.zeros(n)    # Σ rbi_pa·pa (projection weight for RBI split)
            r_exp = np.zeros(n)       # Σ r_pa·pa·m_off (projection team-run mean)
            for d in plist:
                team_raw += (d['hr'] + 0.6*(d['dbl'] + d['trp'])
                             + 0.3*d['sgl'] + 0.2*(d['bb'] + d['hbp']))
                team_hr += d['hr']
                rw = d['vec']['r_pa'] * d['pa']
                bw = d['vec']['rbi_pa'] * d['pa']
                r_wsum += rw
                rbi_wsum += bw
                r_exp += rw * d['m_off']
            # Anchor the event-based total so its MEAN matches the projection's
            # team-run expectation (preserves the Vegas-anchored run level while
            # taking the bounded, event-driven shape). Rounded to an integer team
            # total per sim WITHOUT extra Poisson scale noise, so team runs are a
            # deterministic function of the realized box score — this is what
            # BOUNDS the ceiling to what the events can actually drive in. Clamp up
            # to the HR count: a team can never score fewer runs than it hit homers.
            raw_mean = team_raw.mean()
            c_scale = (r_exp.mean() / raw_mean) if raw_mean > 0 else 1.0
            team_runs_int = np.clip(np.round(team_raw * c_scale), 0, None).astype(int)
            team_runs_int = np.maximum(team_runs_int, team_hr)
            residual = team_runs_int - team_hr   # runs left after each HR self-scores
            r_wsum = np.where(r_wsum > 0, r_wsum, 1.0)
            rbi_wsum = np.where(rbi_wsum > 0, rbi_wsum, 1.0)

            # Allocate the SAME integer team total to R and to RBI via a
            # conditional-binomial decomposition of a multinomial (per-player
            # shares from the projection weights). Each HR is credited to its OWN
            # hitter as +1 R (he scores) and +1 RBI (he drives himself in), and
            # only the RESIDUAL team runs are split by the projection weights. This
            # ties a player's HR to his own run/RBI within the sim (a solo HR always
            # books its R+RBI points, a multi-HR game always collects them) instead
            # of letting the shared pool decide, which fattens the boom tail. Both R
            # and RBI still draw from the one team_runs_int, so
            # Σ R == Σ RBI == team runs exactly (real box-score conservation).
            def _allocate(rate_key, wsum, out_key):
                rem = residual.copy()
                rem_share = np.ones(n)
                shares = [(d['vec'][rate_key] * d['pa']) / wsum for d in plist]
                for k, d in enumerate(plist):
                    if k == len(plist) - 1:
                        take = rem.copy()
                    else:
                        p_cond = np.clip(shares[k] / np.maximum(rem_share, 1e-9), 0.0, 1.0)
                        take = rng.binomial(rem, p_cond)
                    d[out_key] = d['hr'] + take   # guaranteed HR self-run/RBI + residual share
                    rem = rem - take
                    rem_share = rem_share - shares[k]
            _allocate('r_pa', r_wsum, 'R')
            _allocate('rbi_pa', rbi_wsum, 'RBI')

            # ── pass 2: score DK ─────────────────────────────────────────────
            for d in plist:
                p = d['p']; vec = d['vec']
                sgl, dbl, trp, hr = d['sgl'], d['dbl'], d['trp'], d['hr']
                bb, hbp, ks, sb_s, pa = d['bb'], d['hbp'], d['ks'], d['sb'], d['pa']
                r_s   = np.clip(d['R'],   0, 6)
                rbi_s = np.clip(d['RBI'], 0, 8)

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
            sho = shared[opp_side]
            # the offense this pitcher faces carries its own total scale: a boosted
            # opposing lineup puts proportionally more hits/HR (and runs) on him,
            # and a wider latent (high total) gives him a fatter blow-up tail.
            m_opp = ts_side[opp_side] * np.exp(sho - 0.5*shvar[opp_side])
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

"""
Regression guards for the coherent per-batter pitcher model in sim_proj.

Context: a walk-forward grade of the sim against real DK results (Jul 2026,
83 starts) found the OLD pitcher model's downside tail was far too thin —
P(DK<0) was ~4% in-sim vs ~11% observed and per-player std ~7.7 vs ~10.4 —
because outs/K/H/BB were independent binomials off one batters-faced count and
ER was a nearly deterministic RA9*IP/9. The new `_sim_pitcher` plays the outing
out batter-by-batter with an inning that clears every three outs and an
endogenous hook that pulls a struggling starter, which fattens the negative
tail WITHOUT shifting the mean. These tests pin both properties.
"""
import numpy as np
import sim_proj


def _vec(q):
    """A quality-parameterised starter vec (higher q = better pitcher)."""
    return dict(
        tbf_per_ip=4.30 - 0.06 * q,
        k_pct=float(np.clip(0.215 + 0.045 * q, 0.10, 0.38)),
        h_per_bf=float(np.clip(0.230 - 0.020 * q, 0.14, 0.32)),
        hr_per_bf=float(np.clip(0.032 - 0.006 * q, 0.010, 0.060)),
        bb_pct=float(np.clip(0.082 - 0.008 * q, 0.03, 0.14)),
        hbp_per_bf=0.010,
        era=4.20 - 0.55 * q,
        ra9=4.55 - 0.60 * q,
    )


def _workload(vec, rng, n):
    Lg = rng.standard_normal(n); Lt = rng.standard_normal(n)
    sho = 0.20 * Lg + 0.50 * Lt
    m_opp = np.exp(sho - 0.5 * (0.20 ** 2 + 0.50 ** 2))
    z = (sho - sho.mean()) / (sho.std() + 1e-9)
    bf = np.clip(rng.normal(vec['tbf_per_ip'] * 5.6, 3.0, n) - z * 3.0
                 - (vec['era'] - 4.20) * 0.8, 8, 34)
    wb = max(0.20, min(0.75, 0.500 + (4.20 - vec['era']) * 0.03))
    return bf, m_opp, z, wb


def _old_mean(vec, bf, m_opp, z, wb, rng, n):
    """Mean DK under the PRE-CHANGE pitcher model (for mean-preservation check)."""
    bf = np.clip(bf, 0, 40).astype(int)
    opb = max(0.55, min(0.80, 3.0 / max(vec['tbf_per_ip'], 3.2)))
    outs = np.clip(np.round(bf * opb).astype(int), 0, 27); ip = outs / 3.0
    k = rng.binomial(bf, min(0.6, vec['k_pct']))
    h = rng.binomial(bf, np.clip(vec['h_per_bf'] * m_opp, 0.05, 0.60))
    hr = rng.binomial(bf, np.clip(vec['hr_per_bf'] * m_opp, 0.003, 0.12))
    bb = rng.binomial(bf, np.clip(vec['bb_pct'] * np.sqrt(m_opp), 0.02, 0.25))
    hbp = rng.binomial(bf, min(0.05, vec['hbp_per_bf']))
    er = np.round(np.clip(vec['ra9'] * ip / 9.0 + (h * 0.10 + hr * 0.55 + bb * 0.07 - 0.9)
                          + z * 0.5 + rng.normal(0, 0.6, n), 0, 15)).astype(int)
    win = ((ip >= 5.0) & (rng.uniform(0, 1, n) < np.clip(wb - er * 0.04 - z * 0.03, 0.02, 0.85))).astype(int)
    cg = (ip >= 9.0).astype(int); cgs = (cg & (er == 0)).astype(int); nh = (cgs & (h == 0)).astype(int)
    return sim_proj.dk_pitcher(outs, k, win, er, h, bb, hbp, cg, cgs, nh).mean()


def test_mean_preserved_vs_old_model():
    """New per-batter model must reproduce the old model's mean across the
    quality spectrum (the projection level is well-calibrated; only the tail
    was wrong). Allow a small tolerance for the model change."""
    n = 40000
    for q in (-1.5, -0.5, 0.5, 1.5, 2.5):
        vec = _vec(q)
        rng = np.random.default_rng(100 + int(q * 10))
        bf, m_opp, z, wb = _workload(vec, rng, n)
        old_mu = _old_mean(vec, bf, m_opp, z, wb, np.random.default_rng(7), n)
        new_mu = sim_proj._sim_pitcher(vec, bf, m_opp, z, True, wb, 27,
                                       np.random.default_rng(7), n)['dk'].mean()
        assert abs(new_mu - old_mu) < 1.2, (q, old_mu, new_mu)


def test_downside_tail_is_realistic():
    """A league-average starter must blow up into negative DK at a realistic
    rate (the whole point of the fix) and carry realistic dispersion — the old
    model gave P(DK<0) ~ 0.01-0.04 and std ~ 6-7, which was too thin."""
    n = 40000
    vec = _vec(0.0)
    rng = np.random.default_rng(2026)
    bf, m_opp, z, wb = _workload(vec, rng, n)
    out = sim_proj._sim_pitcher(vec, bf, m_opp, z, True, wb, 27, rng, n)
    dk = out['dk']
    assert 0.07 < (dk < 0).mean() < 0.22, (dk < 0).mean()   # realistic blow-up rate
    assert dk.std() > 8.5, dk.std()                         # realistic dispersion
    assert 10.0 < dk.mean() < 16.0, dk.mean()               # sane projection level


def test_worse_pitchers_blow_up_more():
    """P(DK<0) must increase monotonically as pitcher quality falls."""
    n = 40000
    p = []
    for q in (2.0, 0.5, -1.0):
        vec = _vec(q); rng = np.random.default_rng(55)
        bf, m_opp, z, wb = _workload(vec, rng, n)
        dk = sim_proj._sim_pitcher(vec, bf, m_opp, z, True, wb, 27, rng, n)['dk']
        p.append((dk < 0).mean())
    assert p[0] < p[1] < p[2], p


def test_output_coherence():
    """Returned line must be internally coherent and finite."""
    n = 20000
    vec = _vec(0.3); rng = np.random.default_rng(9)
    bf, m_opp, z, wb = _workload(vec, rng, n)
    o = sim_proj._sim_pitcher(vec, bf, m_opp, z, True, wb, 27, rng, n)
    for kk in ('dk', 'ip', 'k', 'bb', 'h', 'hr', 'er', 'win', 'bf'):
        assert kk in o and np.isfinite(np.asarray(o[kk], float)).all()
    assert (o['ip'] <= 9.0 + 1e-9).all()          # outs_cap respected
    assert (o['k'] <= o['bf']).all()              # can't strike out more than faced
    assert (o['h'] >= o['hr']).all()              # HR are a subset of hits
    assert set(np.unique(o['win'])).issubset({0, 1})


def test_opener_is_short_and_cannot_win():
    """The opener path (can_win=False, small outs_cap) yields short outings and
    never records a win."""
    n = 20000
    vec = _vec(0.5); rng = np.random.default_rng(11)
    bf = np.clip(rng.normal(4.6, 1.3, n), 2, 9)
    m_opp = np.ones(n); z = rng.standard_normal(n)
    o = sim_proj._sim_pitcher(vec, bf, m_opp, z, False, 0.0, 9, rng, n)
    assert o['win'].sum() == 0
    assert o['ip'].mean() < 3.0


def test_ceiling_separates_by_opponent_strength():
    """Holding the pitcher fixed, a favorable matchup (whiff-prone / weaker
    offense) must produce a materially higher DK ceiling than a brutal one
    (contact / stronger offense). Guards against a matchup-invariant ceiling.

    The separation comes from the opponent-adjusted mean K rate (log5), the
    higher hit/HR traffic vs a strong offense, and — in the starter branch — the
    removal of the perverse +(opp_implied) workload term (which used to inflate a
    tough matchup's ceiling); NOT from any added per-sim K variance, since the
    shipped pitcher distribution is already fully dispersed (see sim_review)."""
    n = 60000
    soft = dict(_vec(0.5), k_pct=0.275, h_per_bf=0.220, hr_per_bf=0.030)  # high-K, modest
    tough = dict(_vec(0.5), k_pct=0.185, h_per_bf=0.245, hr_per_bf=0.040)  # low-K, power
    rng = np.random.default_rng(404)
    bf, _, z, wb = _workload(_vec(0.5), rng, n)
    Lg = np.random.default_rng(1).standard_normal(n); Lt = np.random.default_rng(2).standard_normal(n)
    sho = 0.20 * Lg + 0.50 * Lt
    m_soft = 0.92 * np.exp(sho - 0.5 * (0.2**2 + 0.5**2))
    m_tough = 1.15 * np.exp(sho - 0.5 * (0.2**2 + 0.5**2))
    zc = (sho - sho.mean()) / (sho.std() + 1e-9)
    dk_soft = sim_proj._sim_pitcher(soft, bf, m_soft, zc, True, wb, 27, np.random.default_rng(5), n)['dk']
    dk_tough = sim_proj._sim_pitcher(tough, bf, m_tough, zc, True, wb, 27, np.random.default_rng(5), n)['dk']
    assert np.percentile(dk_soft, 90) > np.percentile(dk_tough, 90) + 3.0
    assert (dk_soft >= 30).mean() > 2.0 * (dk_tough >= 30).mean()


def _primary_seg(vp, rng, n, opp_implied=4.2):
    """Replicate the primary/bulk-arm branch of sim_proj.simulate so the
    ip_per_g governor can be exercised without building a full matchup."""
    z = rng.standard_normal(n); m_opp = np.ones(n)
    ip_plan = float(np.clip(vp.get('ip_per_g', sim_proj.PRIMARY_FULL_IP) * sim_proj.PRIMARY_IP_STRETCH,
                            1.0, sim_proj.PRIMARY_FULL_IP))
    if ip_plan >= sim_proj.PRIMARY_FULL_IP:
        bf = np.clip(rng.normal(vp['tbf_per_ip'] * 5.0, 3.0, n) - z * 2.5 - (opp_implied - 4.5) * 0.8, 6, 30)
        return sim_proj._sim_pitcher(vp, bf, m_opp, z, True, 0.28, 24, rng, n)
    bf = np.clip(rng.normal(vp['tbf_per_ip'] * ip_plan, 2.0, n) - z * 1.5 - (opp_implied - 4.5) * 0.5, 3, 18)
    oc = int(np.clip(round(ip_plan * 3) + 3, 3, 12))
    return sim_proj._sim_pitcher(vp, bf, m_opp, z, False, 0.0, oc, rng, n)


def test_primary_governor_shortens_a_true_reliever():
    """A short reliever (low weighted_IP_per_G) tagged as the 'primary' in a
    bullpen game must be simulated near his real usage — NOT as a ~5-IP starter.
    Regression for a bulk-reliever (e.g. Will Klein, ~1.15 IP/G) that shipped
    with a full-starter line (~4.8 IP, ~15 DK)."""
    n = 40000
    vec = _vec(0.6)                       # decent per-batter skill
    reliever = dict(vec, ip_per_g=1.15)   # but never goes deep
    o = _primary_seg(reliever, np.random.default_rng(3), n)
    assert o['win'].sum() == 0            # can't vulture a win in ~1-2 IP
    assert o['ip'].mean() < 2.5           # governed to near his real outing length
    assert o['dk'].mean() < 9.0           # not a starter-sized projection


def _starter_seg(v, rng, n, wb=0.5, opp_implied=4.2):
    """Replicate the solo-STARTER branch of sim_proj.simulate (with the IP/G
    governor) so a reliever listed as the starter can be exercised directly."""
    z = rng.standard_normal(n); m_opp = np.ones(n)
    if float(v.get('ip_per_g', sim_proj.PRIMARY_FULL_IP)) >= sim_proj.STARTER_MIN_IP_PER_G:
        bf = np.clip(rng.normal(v['tbf_per_ip']*5.6, 3.0, n) - z*3.0
                     - (v['era']-4.20)*0.8, 8, 34)
        return sim_proj._sim_pitcher(v, bf, m_opp, z, True, wb, 27, rng, n)
    ip_plan = float(np.clip(v['ip_per_g'] * sim_proj.PRIMARY_IP_STRETCH, 1.0, sim_proj.PRIMARY_FULL_IP))
    bf = np.clip(rng.normal(v['tbf_per_ip']*ip_plan, 2.0, n) - z*1.5 - (opp_implied-4.5)*0.5, 3, 18)
    oc = int(np.clip(round(ip_plan*3) + 3, 3, 12))
    return sim_proj._sim_pitcher(v, bf, m_opp, z, False, 0.0, oc, rng, n)


def test_starter_governor_shortens_a_reliever_listed_as_starter():
    """A reliever the slate feed lists as the 'starter' (a spot start or a
    bullpen-game bulk arm) must be simulated near his real usage, NOT as a
    ~5.6-IP full start — the Jonathan Pintaro case (~15 DK, quality-start-capable
    for an arm who never goes deep)."""
    n = 40000
    reliever = dict(_vec(0.6), ip_per_g=1.6)   # decent stuff, but a short arm
    o = _starter_seg(reliever, np.random.default_rng(7), n)
    assert o['win'].sum() == 0                  # can't earn a win in ~2-3 IP
    assert o['ip'].mean() < 3.0                 # governed near his real outing
    assert o['dk'].mean() < 10.0                # not a full-starter projection


def test_starter_governor_leaves_genuine_starter_unchanged():
    """A genuine starter (IP/G >= the starter line, or a missing IP/G that
    defaults to a full start) keeps the prior full-starter treatment exactly —
    same rng path, byte-for-byte."""
    n = 40000
    starter = dict(_vec(0.6), ip_per_g=5.4)
    seed = 11
    got = _starter_seg(starter, np.random.default_rng(seed), n)
    rng = np.random.default_rng(seed); z = rng.standard_normal(n); m_opp = np.ones(n)
    bf = np.clip(rng.normal(starter['tbf_per_ip']*5.6, 3.0, n) - z*3.0
                 - (starter['era']-4.20)*0.8, 8, 34)
    ref = sim_proj._sim_pitcher(starter, bf, m_opp, z, True, 0.5, 27, rng, n)
    assert abs(got['dk'].mean() - ref['dk'].mean()) < 1e-6
    assert got['ip'].mean() > 4.0
    # a pitcher with NO established IP/G defaults to full-starter treatment
    noipg = _vec(0.6)                            # no ip_per_g key
    g2 = _starter_seg(noipg, np.random.default_rng(seed), n)
    assert g2['ip'].mean() > 4.0 and g2['win'].sum() > 0


def test_primary_governor_leaves_bulk_starter_unchanged():
    """A stretched-out bulk starter (weighted_IP_per_G >= PRIMARY_FULL_IP /
    STRETCH) must keep the prior full-starter treatment exactly."""
    n = 40000
    vec = _vec(0.6)
    starter = dict(vec, ip_per_g=4.86)    # genuine bulk starter
    seed = 4
    got = _primary_seg(starter, np.random.default_rng(seed), n)
    # Reference: prior fixed ~5-IP / 24-out / win-eligible treatment.
    rng = np.random.default_rng(seed); z = rng.standard_normal(n); m_opp = np.ones(n)
    bf = np.clip(rng.normal(starter['tbf_per_ip'] * 5.0, 3.0, n) - z * 2.5 - (4.2 - 4.5) * 0.8, 6, 30)
    ref = sim_proj._sim_pitcher(starter, bf, m_opp, z, True, 0.28, 24, rng, n)
    assert abs(got['dk'].mean() - ref['dk'].mean()) < 1e-6
    assert got['ip'].mean() > 4.0

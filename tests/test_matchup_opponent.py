"""Opponent-quality (log5) matchup adjustment for pitchers in matchup.py.

A pitcher's allowed rates must respond to the QUALITY of the lineup he faces:
facing an elite offense should raise HR/hits/walks and cut strikeouts; facing a
weak one should do the reverse; a league-average lineup should leave him ~as-is.
"""
import numpy as np
import matchup
from matchup import (_opponent_adjust_pitcher, _log5_rate, LG_HIT_VEC,
                     OPP_MATCHUP_ELASTICITY)


def _hitter(mult):
    """A hitter of quality `mult` (>1 = better than league). Offense-positive
    events (power, hits, walks) scale UP with quality; strikeouts scale the
    opposite way, since a better offense also makes more contact."""
    v = dict(LG_HIT_VEC)
    for k in ("p_1b", "p_2b", "p_3b", "p_hr", "p_bb"):
        v[k] = LG_HIT_VEC[k] * mult
    v["p_k"] = LG_HIT_VEC["p_k"] * (2.0 - mult)     # better lineup -> fewer Ks
    return v


def _lineup(mult):
    return [{"vec": _hitter(mult)} for _ in range(9)]


def _pitcher():
    # a roughly average-ish arm expressed in per-BF rates
    return dict(k_pct=0.24, bb_pct=0.075, hbp_per_bf=0.010,
                hr_per_bf=0.030, h_per_bf=0.200, era=4.0, ra9=4.2,
                tbf_per_ip=4.3, hand="L")


def test_league_average_lineup_is_a_near_noop():
    base = _pitcher()
    out = _opponent_adjust_pitcher(base, [{"vec": dict(LG_HIT_VEC)} for _ in range(9)])
    for k in ("k_pct", "bb_pct", "hr_per_bf", "h_per_bf"):
        assert abs(out[k] - base[k]) < 1e-3, (k, out[k], base[k])


def test_elite_lineup_docks_the_pitcher():
    base = _pitcher()
    out = _opponent_adjust_pitcher(base, _lineup(1.4))   # 40% better than league
    assert out["hr_per_bf"] > base["hr_per_bf"]          # more homers allowed
    assert out["h_per_bf"] > base["h_per_bf"]            # more hits allowed
    assert out["bb_pct"] > base["bb_pct"]                # more walks
    assert out["k_pct"] < base["k_pct"]                  # fewer strikeouts


def test_weak_lineup_lifts_the_pitcher():
    base = _pitcher()
    out = _opponent_adjust_pitcher(base, _lineup(0.6))   # 40% worse than league
    assert out["hr_per_bf"] < base["hr_per_bf"]
    assert out["h_per_bf"] < base["h_per_bf"]
    assert out["k_pct"] > base["k_pct"]


def test_effect_is_monotonic_in_opponent_strength():
    base = _pitcher()
    hrs = [_opponent_adjust_pitcher(base, _lineup(m))["hr_per_bf"]
           for m in (0.7, 1.0, 1.3)]
    assert hrs[0] < hrs[1] < hrs[2]


def test_bip_hit_elasticity_is_damped_vs_hr():
    # equal +1 log-odds bump on both events; HR (gamma=1) must move more in
    # log-odds than balls-in-play hits (gamma=0.7).
    L = 0.05
    hr_like = _log5_rate(0.05, [0.05 * np.e / (1 - 0.05 + 0.05 * np.e)], L, 1.0)
    # simpler: compare elasticities directly via the shift they induce
    def shift(g):
        return (_log5_rate(0.10, [0.20], 0.10, g))
    assert shift(OPP_MATCHUP_ELASTICITY["hr"]) > shift(OPP_MATCHUP_ELASTICITY["bip_hit"])


def test_log5_matches_closed_form_odds_ratio():
    # single-batter log5 must equal the textbook odds-ratio combination
    P, B, L, g = 0.22, 0.30, 0.225, 1.0
    got = _log5_rate(P, [B], L, g)
    oP, oB, oL = P / (1 - P), B / (1 - B), L / (1 - L)
    o = oP * oB / oL
    assert abs(got - o / (1 + o)) < 1e-9

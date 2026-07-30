"""
Regression guards for the HR->R/RBI coupling in the hitter run-conservation
step of sim_proj.simulate (the "P2" fix).

Context: the team run-conservation step allocates R and RBI out of one integer
team total via projection-weighted multinomials. The old split was decoupled
from who actually homered, so a batter who hit a HR in a given sim frequently
received neither the run he scored nor the RBI he drove in (on a synthetic
9-man lineup this happened in ~66% of HR games). That suppressed the boom tail:
a solo HR should always book its +2 R and +2 RBI. The fix credits each HR to
its own hitter as +1 R and +1 RBI, then splits only the residual team runs.
Team conservation (Sum R == Sum RBI == team runs) is preserved.
"""
import numpy as np
import sim_proj


def _hvec(q, slot):
    return dict(
        p_hr=float(np.clip(0.032 + 0.012 * q, 0.008, 0.06)), p_3b=0.004,
        p_2b=float(np.clip(0.045 + 0.008 * q, 0.02, 0.07)),
        p_1b=float(np.clip(0.150 + 0.010 * q, 0.10, 0.20)),
        p_bb=float(np.clip(0.082 + 0.010 * q, 0.03, 0.13)), p_hbp=0.010,
        p_k=float(np.clip(0.220 - 0.010 * q, 0.12, 0.32)), p_sb=0.02,
        r_pa=float(np.clip(0.130 + 0.020 * q, 0.08, 0.19)),
        rbi_pa=float(np.clip(0.125 + 0.020 * q, 0.07, 0.18)), proj_slot=slot,
    )


def _pvec():
    return dict(tbf_per_ip=4.3, k_pct=0.22, h_per_bf=0.23, hr_per_bf=0.032,
                bb_pct=0.082, hbp_per_bf=0.01, era=4.2, ra9=4.55)


def _lineup(team, grad):
    return [dict(name=f"{team}{i+1}", slot=i + 1, pos="OF", hand="R", vec=_hvec(q, i + 1))
            for i, q in enumerate(grad)]


def _matchup():
    grad_a = [1.2, 1.0, 1.4, 0.8, 0.6, 0.2, -0.2, -0.6, -1.0]
    grad_h = [1.0, 0.8, 1.2, 0.6, 0.4, 0.0, -0.4, -0.8, -1.2]
    g = dict(
        away="AAA", home="HHH",
        implied={"away": 4.6, "home": 4.4},
        datetime="2026-07-30T19:00",
        lineup_source={"away": "test", "home": "test"},
        lineups={"away": _lineup("A", grad_a), "home": _lineup("H", grad_h)},
        pitchers={
            "away": {"starter": {"name": "APitch", "hand": "R", "vec": _pvec()}},
            "home": {"starter": {"name": "HPitch", "hand": "L", "vec": _pvec()}},
        },
    )
    return {"games": {"g1": g}, "missing": {}}


def test_every_home_run_books_its_own_run_and_rbi():
    """In every sim where a hitter has HR>0, he must be credited at least one
    run and at least one RBI per homer (the core P2 guarantee)."""
    _, _, hitter_stat, _, _, _ = sim_proj.simulate(_matchup(), n_sims=4000, seed=1)
    assert hitter_stat
    for nm, st in hitter_stat.items():
        hr = np.asarray(st["HR"]); R = np.asarray(st["R"]); RBI = np.asarray(st["RBI"])
        m = hr > 0
        if m.any():
            assert (R[m] >= hr[m]).all(), f"{nm}: a HR game with R < HR"
            assert (RBI[m] >= hr[m]).all(), f"{nm}: a HR game with RBI < HR"


def test_team_run_rbi_conservation():
    """Per sim, a team's Sum R must equal its Sum RBI (both equal team runs).
    Clips at R<=6 / RBI<=8 can nick a handful of extreme sims, so require the
    identity to hold in the overwhelming majority."""
    _, _, hitter_stat, _, _, _ = sim_proj.simulate(_matchup(), n_sims=4000, seed=2)
    for team in ("A", "H"):
        names = [nm for nm in hitter_stat if nm.startswith(team)]
        sumR = sum(np.asarray(hitter_stat[nm]["R"]) for nm in names)
        sumRBI = sum(np.asarray(hitter_stat[nm]["RBI"]) for nm in names)
        assert (sumR == sumRBI).mean() > 0.98


def test_solo_home_run_scores_at_least_14_dk():
    """A sim with exactly 1 HR, no other hits/walks/steals must score >= 14 DK
    (10 HR + 2 R + 2 RBI) — proof the run/RBI points attach to the homer."""
    _, _, hitter_stat, _, _, _ = sim_proj.simulate(_matchup(), n_sims=6000, seed=3)
    seen = False
    for nm, st in hitter_stat.items():
        hr = np.asarray(st["HR"]); tb = (np.asarray(st["1B"]) + np.asarray(st["2B"])
              + np.asarray(st["3B"]) + np.asarray(st["BB"]) + np.asarray(st["HBP"]) + np.asarray(st["SB"]))
        solo = (hr == 1) & (tb == 0)
        if solo.any():
            seen = True
            dk = (np.asarray(st["1B"])*3 + np.asarray(st["2B"])*5 + np.asarray(st["3B"])*8
                  + hr*10 + np.asarray(st["RBI"])*2 + np.asarray(st["R"])*2
                  + np.asarray(st["BB"])*2 + np.asarray(st["HBP"])*2 + np.asarray(st["SB"])*5)
            assert (dk[solo] >= 14).all()
    assert seen

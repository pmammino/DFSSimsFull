"""
season_sim.py
=============
Season-long Monte Carlo: a floor, a median and a ceiling for every counting
and rate stat the season export carries.

    floor    the 25th-percentile OUTCOME for the player — the bad quartile
    median   the projection already in the export (the simulation's own
             median is written beside it, for reconciliation)
    ceiling  the 75th-percentile outcome — the good quartile

"Good" is a property of the stat, not of the number. For HR, SO-by-a-pitcher
and SB the ceiling is the 75th percentile of the distribution; for ERA, WHIP,
and hits allowed it is the 25th, because a lower ERA is the better season.
`HITTER_STATS` and `PITCHER_STATS` below are the single place that says which
way each stat points, so changing a judgment (hitter strikeouts, say) is one
line. A floor is always the opposite tail from the ceiling.

What is simulated, per player, per season (all draws are independent across
players — this is a per-player range, not a league simulation):

1. **Role.** One role is drawn from the player's role MIXTURE
   (`pt_role_mix`, e.g. "Closer:0.7|Late Inning RP (Setup):0.3"), and volume
   moves with that role's anchor. A settled player (one role at 1.0) has no
   role spread; a job battle has a lot.
2. **Health.** A season-lost fraction with a probability and a size. The
   volume factor is divided by its own expectation, so it widens the range
   without moving the mean — `Proj_PA` / `Proj_IP` already contain league-
   average missed time because the role anchors were fitted to real seasons.
3. **Talent.** Each event rate is drawn from its projection and its own
   `SD_*` column (the posterior spread the rate model already publishes), then
   renormalised. Talent is drawn once per player per season, so a hitter whose
   power is really 10% higher hits more of everything for the whole year.
4. **Luck.** Events are a multinomial over the simulated PA / TBF. Runs and
   RBI scale with the simulated offensive production; pitcher earned runs are
   the projected ERA moved by linear weights on the simulated events, then
   Poisson with an overdispersion term for sequencing.

Everything tunable is a named constant below. They are ASSUMPTIONS: the
injury and overdispersion constants are set from general baseball knowledge,
not fitted to this repo's data. The reconciliation `reconcile()` prints how
far the simulated means land from the projection, which is the check that the
simulation is not quietly moving the projection.

Floor-tier players (1 PA / 1 IP placeholders) are not simulated; their
columns are blank rather than a false range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from role_taxonomy import parse_role_mix, role_anchor

N_SIMS_DEFAULT = 2000
FLOOR_Q = 0.25
CEIL_Q = 0.75
CHUNK = 250                 # players per vectorised block

# ── assumptions ──────────────────────────────────────────────────────────────
HIT_INJ_P, HIT_INJ_LOSS = 0.35, (0.05, 0.45)     # P(miss time), share lost
SP_INJ_P, SP_INJ_LOSS = 0.40, (0.10, 0.60)
RP_INJ_P, RP_INJ_LOSS = 0.25, (0.10, 0.50)
DUR_INJ_SLOPE = 2.0        # extra injury probability per 1.0 of durability dock
HIT_VOL_NOISE = 0.05       # sd of everything else that moves PA (lineup spot)
PIT_VOL_NOISE = 0.08
HIT_PA_CAP = 740.0
PIT_IP_CAP = 240.0
RUN_LINK = 0.5             # elasticity of R/RBI to simulated production
RUN_CV = 0.15              # teammates / context noise on R and RBI
SEQ_CV = 0.12              # sequencing noise on a pitcher's earned runs
OPP_CV = 0.25              # save / hold opportunity noise
# Per-start model for wins and quality starts (starters only). The shape
# constants below were calibrated so a league of projected starters comes out
# near 5.3 IP, ~0.40 QS and ~0.25 W per start. Those targets are benchmarks
# recalled from recent MLB seasons, NOT checked against data in this repo, so
# treat the absolute W/QS level as provisional.
START_IP_SD = 1.2          # sd of planned innings in a start
START_ER_CV = 0.60         # start-to-start run-environment noise on ER
HOOK_SLOPE = 0.25          # share of the outing lost per earned run past 3
IP_PLAN_K = 1.25           # planned-IP scale so the post-hook mean IP/start
                           # lands on the season projection (calibrated)
TEAM_RS_CV = 0.30          # game-to-game noise on the club's runs scored
BULLPEN_RA_SHARE = 1.30    # leverage: relievers pitch the innings where a lead
                           # is given back, so they cost starters more wins
                           # than their ERA says (calibrated, see below)
SP_WIN_SHARE = 0.49        # share of a club's wins the starters take; the rest
                           # is the relief-win pool (checked against the sim)
LEAGUE_RS_PER_GAME = 4.45
# Linear-weight runs of an event over the out it replaces (outs, including
# strikeouts, are the zero point; TBF is fixed so a hit displaces an out).
LW = {"1B": 0.74, "2B": 1.04, "3B": 1.31, "HR": 1.67, "BB": 0.58, "HBP": 0.60}

# True = a bigger number is the better season for that stat.
HITTER_STATS = {
    "PA": True, "AB": True, "H": True, "1B": True, "2B": True, "3B": True,
    "HR": True, "TB": True, "BB": True, "HBP": True,
    "SF": False,            # an out
    "SO": False, "SB": True,
    "CS": False,
    "R": True, "RBI": True,
    "AVG": True, "OBP": True, "SLG": True, "OPS": True,
}
PITCHER_STATS = {
    "IP": True, "GS": True, "G": True, "TBF": True,
    "H": False, "1B": False, "2B": False, "3B": False, "HR": False,
    "BB": False, "HBP": False,
    "SO": True,
    "ER": False, "R": False,
    "SV": True, "HLD": True, "W": True, "QS": True,
    "ERA": False, "WHIP": False,
    "K9": True, "BB9": False, "HR9": False,
}

_H_EVENTS = ["K", "BB", "HBP", "SF", "HR", "3B", "2B", "1B", "BIPOut"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default) \
        .to_numpy(dtype=float)


def _rates(df: pd.DataFrame, events: list[str]):
    P = np.column_stack([_col(df, f"P_{e}") for e in events])
    S = np.column_stack([_col(df, f"SD_{e}") for e in events])
    return P, S


def _draw_rates(rng, P: np.ndarray, S: np.ndarray, n: int) -> np.ndarray:
    """(players, events) -> (players, sims, events): talent draw, renormalised."""
    z = rng.standard_normal((P.shape[0], n, P.shape[1]))
    r = np.clip(P[:, None, :] + S[:, None, :] * z, 1e-6, None)
    return r * (P.sum(1, keepdims=True)[:, None, :] / r.sum(2, keepdims=True))


def _multinomial(rng, trials: np.ndarray, probs: np.ndarray) -> list[np.ndarray]:
    """Counts per event, by sequential conditional binomials (vectorised)."""
    remaining = trials.astype(np.int64).copy()
    rem_p = np.ones(trials.shape)
    out = []
    for j in range(probs.shape[2] - 1):
        p = np.clip(probs[..., j] / np.maximum(rem_p, 1e-12), 0.0, 1.0)
        c = rng.binomial(remaining, p)
        out.append(c.astype(float))
        remaining = remaining - c
        rem_p = rem_p - probs[..., j]
    out.append(remaining.astype(float))
    return out


def _role_ratios(rng, mix_values, role_values, kind: str, keys: tuple,
                 n: int) -> dict[str, np.ndarray]:
    """Per-key volume ratio (players, sims) from a role-mixture draw.

    The ratio has mean exactly 1 over the mixture, so it moves the spread and
    not the level.
    """
    rows = len(mix_values)
    out = {k: np.ones((rows, n)) for k in keys}
    for i in range(rows):
        mix = parse_role_mix(mix_values[i], kind)
        if not mix and isinstance(role_values[i], str):
            mix = {role_values[i]: 1.0}
        if not mix or len(mix) < 2:
            continue
        names = list(mix)
        p = np.array([mix[r] for r in names], dtype=float)
        p /= p.sum()
        anchors = [role_anchor(r, kind) for r in names]
        if any(a is None for a in anchors):
            continue
        idx = rng.choice(len(names), size=n, p=p)
        for k in keys:
            vals = np.array([float(a.get(k, 0.0)) for a in anchors])
            mean = float((p * vals).sum())
            if mean > 0:
                out[k][i] = vals[idx] / mean
    return out


def _health_factor(rng, p_inj: np.ndarray, loss: tuple, n: int) -> np.ndarray:
    """Season-share factor (players, sims) with mean exactly 1."""
    hurt = rng.random((len(p_inj), n)) < p_inj[:, None]
    share = rng.uniform(loss[0], loss[1], (len(p_inj), n))
    factor = np.where(hurt, 1.0 - share, 1.0)
    mean = 1.0 - p_inj * 0.5 * (loss[0] + loss[1])
    return factor / mean[:, None]


def _summ(arr: np.ndarray, higher_is_better: bool) -> dict[str, np.ndarray]:
    """Floor / median / ceiling / mean across the sim axis.

    The floor is the bad tail and the ceiling the good tail, so for a stat
    where lower is better they swap which numeric quantile they read.
    """
    lo = np.nanquantile(arr, FLOOR_Q, axis=1)
    hi = np.nanquantile(arr, CEIL_Q, axis=1)
    return {"floor": lo if higher_is_better else hi,
            "ceil": hi if higher_is_better else lo,
            "med": np.nanmedian(arr, axis=1),
            "mean": np.nanmean(arr, axis=1)}


def _frame(ids, stats: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    cols = {"PlayerId": ids}
    for stat, d in stats.items():
        cols[f"{stat}_floor"] = d["floor"]
        cols[f"{stat}_ceil"] = d["ceil"]
        cols[f"{stat}_simmed"] = d["med"]
        cols[f"{stat}_simmean"] = d["mean"]
    return pd.DataFrame(cols)


def _projected_mask(df: pd.DataFrame) -> np.ndarray:
    if "pt_tier" not in df.columns:
        return np.ones(len(df), dtype=bool)
    return (df["pt_tier"].astype(str) != "floor").to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# hitters
# ─────────────────────────────────────────────────────────────────────────────

def _sim_hitter_block(rng, d: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    k = len(d)
    proj_pa = _col(d, "Proj_PA")
    dur = _col(d, "pt_durability", 1.0)
    p_inj = np.clip(HIT_INJ_P + DUR_INJ_SLOPE * np.maximum(0.0, 1.0 - dur),
                    0.0, 0.8)
    role = _role_ratios(
        rng, d.get("pt_role_mix", pd.Series([None] * k)).to_numpy(),
        d.get("pt_role", pd.Series([None] * k)).to_numpy(), "hitter",
        ("pa",), n)["pa"]
    noise = 1.0 + HIT_VOL_NOISE * rng.standard_normal((k, n))
    pa = np.clip(proj_pa[:, None] * role * _health_factor(
        rng, p_inj, HIT_INJ_LOSS, n) * noise, 0.0, HIT_PA_CAP)
    pa = np.round(pa)

    P, S = _rates(d, _H_EVENTS)
    probs = _draw_rates(rng, P, S, n)
    c = dict(zip(_H_EVENTS, _multinomial(rng, pa, probs)))

    out = {"PA": pa, "BB": c["BB"], "HBP": c["HBP"], "SF": c["SF"],
           "SO": c["K"], "HR": c["HR"], "3B": c["3B"], "2B": c["2B"],
           "1B": c["1B"]}
    out["H"] = c["1B"] + c["2B"] + c["3B"] + c["HR"]
    out["AB"] = pa - c["BB"] - c["HBP"] - c["SF"]
    out["TB"] = c["1B"] + 2 * c["2B"] + 3 * c["3B"] + 4 * c["HR"]

    # production vs what the BASE rates would have produced, for R / RBI
    base = {e: P[:, i] for i, e in enumerate(_H_EVENTS)}
    exp_prod = pa * (base["1B"] + 2 * base["2B"] + 3 * base["3B"]
                     + 4 * base["HR"] + base["BB"] + base["HBP"])[:, None]
    prod = np.where(exp_prod > 0, (out["TB"] + c["BB"] + c["HBP"])
                    / np.maximum(exp_prod, 1e-9), 1.0)
    link = np.clip(prod, 0.2, 5.0) ** RUN_LINK

    def _gamma(shape_cv):
        a = 1.0 / shape_cv ** 2
        return rng.gamma(a, 1.0 / a, (k, n))

    for stat, pcol, scol in (("R", "P_R", "SD_R"), ("RBI", "P_RBI", "SD_RBI")):
        base_r, sd_r = _col(d, pcol), _col(d, scol)
        talent = np.clip(base_r[:, None] + sd_r[:, None]
                         * rng.standard_normal((k, n)), 0.0, None)
        out[stat] = rng.poisson(pa * talent * link * _gamma(RUN_CV))

    sb_p, sb_s = _col(d, "P_SB"), _col(d, "SD_SB")
    sb_rate = np.clip(sb_p[:, None] + sb_s[:, None]
                      * rng.standard_normal((k, n)), 0.0, 1.0)
    out["SB"] = rng.binomial(pa.astype(np.int64), sb_rate).astype(float)
    cs_ratio = np.where(sb_p > 0, _col(d, "P_CS") / np.maximum(sb_p, 1e-9), 0.0)
    out["CS"] = rng.binomial(pa.astype(np.int64),
                             np.clip(sb_rate * cs_ratio[:, None], 0, 1)
                             ).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ab = np.where(out["AB"] > 0, out["AB"], np.nan)
        pa_ = np.where(pa > 0, pa, np.nan)
        out["AVG"] = out["H"] / ab
        out["OBP"] = (out["H"] + c["BB"] + c["HBP"]) / pa_
        out["SLG"] = out["TB"] / ab
        out["OPS"] = out["OBP"] + out["SLG"]
    return out


def simulate_hitters(raw: pd.DataFrame, n_sims: int = N_SIMS_DEFAULT,
                     seed: int = 20270401) -> pd.DataFrame:
    """Floor / ceiling / sim median / sim mean per stat, one row per hitter."""
    mask = _projected_mask(raw)
    idx = np.flatnonzero(mask)
    ss = np.random.SeedSequence(seed).spawn(max(1, -(-len(idx) // CHUNK)))
    parts = []
    for b, lo in enumerate(range(0, len(idx), CHUNK)):
        d = raw.iloc[idx[lo:lo + CHUNK]]
        sims = _sim_hitter_block(np.random.default_rng(ss[b]), d, n_sims)
        parts.append(_frame(d["PlayerId"].to_numpy(),
                            {s: _summ(sims[s], hib)
                             for s, hib in HITTER_STATS.items()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["PlayerId"])


# ─────────────────────────────────────────────────────────────────────────────
# pitchers
# ─────────────────────────────────────────────────────────────────────────────

_P_EVENTS = ["K", "BB", "HBP", "HR", "3B", "2B", "1B", "SF", "BIPOut"]


def _starts_wl(rng, ip_per_gs, er_per_ip, unearned, gs_n, team_rs, team_ra):
    """Wins and quality starts, played out start by start.

    Shapes: everything is (players, sims) except `team_*` and `unearned`,
    which are per player. A start draws planned innings, then earned runs
    (overdispersed Poisson); a bad start is cut short by a hook, which is what
    makes IP >= 6 and ER <= 3 negatively correlated within a start, so QS is
    not P(IP >= 6) x P(ER <= 3). A win needs 5+ IP, the lead when he leaves,
    and the bullpen not giving the lead back. Returns (wins, quality starts,
    innings) per player-sim; the innings are for calibration checks.
    """
    shape = ip_per_gs.shape
    wins = np.zeros(shape)
    qs = np.zeros(shape)
    ip_tot = np.zeros(shape)
    ga = 1.0 / START_ER_CV ** 2
    ra_ = 1.0 / TEAM_RS_CV ** 2
    for j in range(int(gs_n.max()) if gs_n.size else 0):
        active = j < gs_n
        plan = np.clip(ip_per_gs * IP_PLAN_K
                       + START_IP_SD * rng.standard_normal(shape), 1.0, 9.0)
        plan = np.round(plan * 3.0) / 3.0
        er = rng.poisson(er_per_ip * plan * rng.gamma(ga, 1.0 / ga, shape))
        ip = np.round(plan * np.clip(1.0 - HOOK_SLOPE * np.maximum(er - 3, 0),
                                     0.25, 1.0) * 3.0) / 3.0
        qs += active & (ip >= 6.0) & (er <= 3)
        r_p = er + rng.poisson(er * (unearned[:, None] - 1.0))
        rs = rng.poisson(team_rs[:, None] * rng.gamma(ra_, 1.0 / ra_, shape))
        rs_p = rng.binomial(rs, np.clip(ip / 9.0, 0.0, 1.0))
        bp = rng.poisson(team_ra[:, None] * BULLPEN_RA_SHARE
                         * (9.0 - ip) / 9.0
                         * rng.gamma(ra_, 1.0 / ra_, shape))
        ip_tot += np.where(active, ip, 0.0)
        # Credit needs the lead when he leaves AND that the bullpen does not
        # give it back: runs scored while he was in, less the ones he allowed,
        # must exceed what the bullpen then allows.
        wins += active & (ip >= 5.0) & ((rs_p - r_p) > bp)
    return wins, qs, ip_tot


def _sim_pitcher_block(rng, d: pd.DataFrame, sv_mean: np.ndarray,
                       hld_mean: np.ndarray, n: int,
                       team_rs: np.ndarray | None = None,
                       team_ra: np.ndarray | None = None,
                       w_rp_mean: np.ndarray | None = None
                       ) -> dict[str, np.ndarray]:
    k = len(d)
    proj_ip, proj_g, proj_gs = (_col(d, "Proj_IP"), _col(d, "Proj_G"),
                                _col(d, "Proj_GS"))
    is_sp = proj_gs >= 0.5 * np.maximum(proj_g, 1.0)
    dur = _col(d, "pt_durability", 1.0)
    base_p = np.where(is_sp, SP_INJ_P, RP_INJ_P)
    p_inj = np.clip(base_p + DUR_INJ_SLOPE * np.maximum(0.0, 1.0 - dur),
                    0.0, 0.8)
    health = np.where(is_sp[:, None],
                      _health_factor(rng, p_inj, SP_INJ_LOSS, n),
                      _health_factor(rng, p_inj, RP_INJ_LOSS, n))
    roles = _role_ratios(
        rng, d.get("pt_role_mix", pd.Series([None] * k)).to_numpy(),
        d.get("pt_role", pd.Series([None] * k)).to_numpy(), "pitcher",
        ("ip", "gs", "g", "sv", "hld"), n)
    noise = 1.0 + PIT_VOL_NOISE * rng.standard_normal((k, n))
    vol = health * noise
    ip = np.clip(proj_ip[:, None] * roles["ip"] * vol, 0.0, PIT_IP_CAP)
    gs = proj_gs[:, None] * roles["gs"] * vol
    g = np.maximum(proj_g[:, None] * roles["g"] * vol, gs)

    tbf_ip = _col(d, "TBF_per_IP", 4.3)
    tbf_ip = np.where(tbf_ip > 1.0, tbf_ip, 4.3)
    tbf = np.round(ip * tbf_ip[:, None])

    P, S = _rates(d, _P_EVENTS)
    probs = _draw_rates(rng, P, S, n)
    c = dict(zip(_P_EVENTS, _multinomial(rng, tbf, probs)))
    out = {"IP": ip, "GS": gs, "G": g, "TBF": tbf, "SO": c["K"], "BB": c["BB"],
           "HBP": c["HBP"], "HR": c["HR"], "3B": c["3B"], "2B": c["2B"],
           "1B": c["1B"]}
    out["H"] = c["1B"] + c["2B"] + c["3B"] + c["HR"]

    # Earned runs: projected ERA, moved by the simulated events' linear
    # weights against what the BASE rates would have produced in this TBF.
    era = _col(d, "ERA")
    ra9 = _col(d, "RA9")
    lam0 = era[:, None] * ip / 9.0
    base = {e: P[:, i] for i, e in enumerate(_P_EVENTS)}
    delta = np.zeros_like(lam0)
    for ev in ("1B", "2B", "3B", "HR", "BB", "HBP"):
        delta += LW[ev] * (c[ev] - tbf * base[ev][:, None])
    a = 1.0 / SEQ_CV ** 2
    lam = np.clip(lam0 + delta, 0.0, None) * rng.gamma(a, 1.0 / a, (k, n))
    er = rng.poisson(lam).astype(float)
    out["ER"] = er
    unearned = np.where(era > 0, ra9 / np.maximum(era, 1e-9), 1.0)
    out["R"] = er * np.maximum(unearned, 1.0)[:, None]

    # Saves / holds: the club-pool allocation, scaled by the drawn role.
    cap_g = np.floor(g)
    a2 = 1.0 / OPP_CV ** 2
    for stat, mean, key in (("SV", sv_mean, "sv"), ("HLD", hld_mean, "hld")):
        lam_o = (np.nan_to_num(mean)[:, None] * roles[key] * health
                 * rng.gamma(a2, 1.0 / a2, (k, n)))
        out[stat] = rng.poisson(np.clip(lam_o, 0.0, None)).astype(float)
    out["SV"] = np.minimum(out["SV"], cap_g)
    out["HLD"] = np.minimum(out["HLD"], np.maximum(cap_g - out["SV"], 0.0))

    # Wins and quality starts. Starters are played out start by start; a
    # pitcher who does not start earns his wins from the club's relief pool.
    team_rs = (np.full(k, LEAGUE_RS_PER_GAME) if team_rs is None else team_rs)
    team_ra = (np.full(k, LEAGUE_RS_PER_GAME) if team_ra is None else team_ra)
    starter = is_sp | (proj_gs >= 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ip_gs = np.where(gs >= 0.5, ip / np.maximum(gs, 0.5), 0.0)
    lw = np.array([LW.get(e, 0.0) for e in _P_EVENTS])
    adj = tbf_ip[:, None] * ((probs - P[:, None, :]) @ lw)
    er_per_ip = np.clip(era[:, None] / 9.0 + adj, 0.05, None)
    gs_n = np.where(starter[:, None], np.round(gs), 0.0)
    w_sp, qs_sp, _ = _starts_wl(rng, ip_gs, er_per_ip, np.maximum(unearned, 1.0),
                             gs_n, team_rs, team_ra)
    a3 = 1.0 / OPP_CV ** 2
    w_rp = np.nan_to_num(w_rp_mean if w_rp_mean is not None
                         else np.zeros(k))[:, None]
    w_rp = rng.poisson(np.clip(w_rp * roles["ip"] * health
                               * rng.gamma(a3, 1.0 / a3, (k, n)), 0.0, None))
    out["W"] = np.where(starter[:, None], w_sp, w_rp).astype(float)
    out["QS"] = qs_sp

    with np.errstate(divide="ignore", invalid="ignore"):
        ipn = np.where(ip > 0, ip, np.nan)
        out["ERA"] = 9.0 * er / ipn
        out["WHIP"] = (out["H"] + out["BB"]) / ipn
        out["K9"] = 9.0 * out["SO"] / ipn
        out["BB9"] = 9.0 * out["BB"] / ipn
        out["HR9"] = 9.0 * out["HR"] / ipn
    return out


def simulate_pitchers(raw: pd.DataFrame, n_sims: int = N_SIMS_DEFAULT,
                      seed: int = 20270402,
                      sv_hld: pd.DataFrame | None = None,
                      teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Floor / ceiling / sim median / sim mean per stat, one row per pitcher.

    `sv_hld` is a frame with PlayerId, SV, HLD — the club-pool allocation the
    export already computes — because saves and holds are not a per-pitcher
    rate and cannot be rebuilt from the per-PA file alone.

    `teams` (team_id, rs_per_game, ra_per_game, expected_wins) supplies the
    run support and the win pool for W. Without it every club is league
    average and relievers get no wins.
    """
    mask = _projected_mask(raw)
    idx = np.flatnonzero(mask)
    tcol = "team_id" if "team_id" in raw.columns else "Pred_target_team_id"
    tid = (pd.to_numeric(raw[tcol], errors="coerce") if tcol in raw.columns
           else pd.Series(np.nan, index=raw.index))
    rs_all = np.full(len(raw), LEAGUE_RS_PER_GAME)
    ra_all = np.full(len(raw), LEAGUE_RS_PER_GAME)
    w_rp_all = np.zeros(len(raw))
    if teams is not None and len(teams):
        t = teams.drop_duplicates("team_id").set_index("team_id")
        rs_all = tid.map(t["rs_per_game"]).fillna(LEAGUE_RS_PER_GAME).to_numpy()
        ra_all = tid.map(t["ra_per_game"]).fillna(LEAGUE_RS_PER_GAME).to_numpy()
        # Relief wins: what the club's win total leaves after the starters,
        # shared by innings among its non-starters.
        gs_ = _col(raw, "Proj_GS")
        g_ = _col(raw, "Proj_G")
        reliever = mask & ~((gs_ >= 0.5 * np.maximum(g_, 1.0)) | (gs_ >= 1.0))
        w_ip = np.where(reliever, _col(raw, "Proj_IP"), 0.0)
        tot = pd.Series(w_ip).groupby(tid.to_numpy()).transform("sum").to_numpy()
        pool = tid.map(t["expected_wins"]).to_numpy() * (1.0 - SP_WIN_SHARE)
        w_rp_all = np.where((tot > 0) & np.isfinite(pool),
                            pool * w_ip / np.where(tot > 0, tot, 1.0), 0.0)
    sv_map = hld_map = None
    if sv_hld is not None and len(sv_hld):
        s = sv_hld.drop_duplicates("PlayerId").set_index("PlayerId")
        sv_map, hld_map = s["SV"], s["HLD"]
    ss = np.random.SeedSequence(seed).spawn(max(1, -(-len(idx) // CHUNK)))
    parts = []
    for b, lo in enumerate(range(0, len(idx), CHUNK)):
        d = raw.iloc[idx[lo:lo + CHUNK]]
        ids = d["PlayerId"]
        sv = (ids.map(sv_map).to_numpy(dtype=float) if sv_map is not None
              else np.zeros(len(d)))
        hl = (ids.map(hld_map).to_numpy(dtype=float) if hld_map is not None
              else np.zeros(len(d)))
        rows = idx[lo:lo + CHUNK]
        sims = _sim_pitcher_block(np.random.default_rng(ss[b]), d, sv, hl,
                                  n_sims, team_rs=rs_all[rows],
                                  team_ra=ra_all[rows],
                                  w_rp_mean=w_rp_all[rows])
        parts.append(_frame(ids.to_numpy(),
                            {s: _summ(sims[s], hib)
                             for s, hib in PITCHER_STATS.items()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["PlayerId"])


# ─────────────────────────────────────────────────────────────────────────────
# merging into the export
# ─────────────────────────────────────────────────────────────────────────────

def bracket(frame: pd.DataFrame, sim: pd.DataFrame,
            stats: dict[str, bool]) -> tuple[pd.DataFrame, dict]:
    """Attach `<stat>_floor` / `<stat>_ceil` beside the projection.

    The projection is the median. The simulation's own median can sit a hair
    off it (injury loss is skewed, ERA is a ratio), so the bracket is clipped
    to contain the projection on the right side of it: a floor is never
    better than the projection and a ceiling never worse. The number clipped
    and the largest gap are returned, so a simulation that disagrees with the
    projection is visible rather than hidden.
    """
    out = frame.merge(sim[["PlayerId"] + [c for c in sim.columns
                                          if c.endswith(("_floor", "_ceil"))]],
                      on="PlayerId", how="left")
    report: dict[str, tuple[int, float]] = {}
    for stat, hib in stats.items():
        if stat not in out.columns:
            continue
        m = pd.to_numeric(out[stat], errors="coerce")
        lo, hi = out[f"{stat}_floor"], out[f"{stat}_ceil"]
        if hib:
            lo2, hi2 = np.minimum(lo, m), np.maximum(hi, m)
        else:
            lo2, hi2 = np.maximum(lo, m), np.minimum(hi, m)
        moved = (lo2 != lo) & lo.notna() | (hi2 != hi) & hi.notna()
        gap = float(max((lo2 - lo).abs().max(), (hi2 - hi).abs().max()))
        report[stat] = (int(moved.sum()), 0.0 if np.isnan(gap) else gap)
        out[f"{stat}_floor"], out[f"{stat}_ceil"] = lo2, hi2
    return out, report


def reconcile(frame: pd.DataFrame, sim: pd.DataFrame,
              stats: tuple[str, ...]) -> str:
    """Simulated mean vs projection, summed over the simulated players."""
    m = frame.merge(sim, on="PlayerId", how="inner")
    lines = []
    for s in stats:
        if s in m.columns and f"{s}_simmean" in m.columns:
            a, b = float(m[s].sum()), float(m[f"{s}_simmean"].sum())
            lines.append(f"{s} {b / a - 1:+.1%}" if a else f"{s} n/a")
    return "  sim mean vs projection: " + ", ".join(lines)

#!/usr/bin/env python3
"""
ownership_model.py — projected DFS ownership from the sim engine
================================================================

Turns the pieces the pipeline already produces — the correlated DK **sims**
(``hitter_dk_sims.npy`` / ``pitcher_dk_sims.npy``), **player cost** (DK
salary), and **game context** (Vegas implied team totals) — into a projected
DraftKings ownership (``%Drafted``) for every player on a slate.

Why a model instead of the RotoWire feed
-----------------------------------------
Ownership was previously an *imported* column (``dk_slate_feed.parse_ownership``
from RotoWire's ``MLBOwnership`` feed). That is a black box, is not always
available, and does not reflect *our* projections. This module derives
ownership from the same sims that drive lineup construction, so the field the
portfolio is optimised against is internally consistent with the projections.

The generative model (why the numbers add up the way DK's do)
-------------------------------------------------------------
Real contest standings obey a hard invariant, verified across every contest in
the calibration set (Jul 26–29 2026):

    within a roster slot, the field's ``%Drafted`` sums to 100% x (# of that
    slot).  P -> 200,  OF -> 300,  C/1B/2B/3B/SS -> 100.

That is exactly what you get if each of the field's roster slots independently
"draws" a player with probability proportional to how attractive that player
is. So ownership is modelled as a **conditional logit** (softmax) *within each
position slot*:

    u_i      = Σ_k  β_k · z_k(i)                 (player attractiveness)
    share_i  = softmax(u_i / τ)  over the slot   (Σ share = 1)
    own_i    = 100 · slot_count · share_i         (Σ own = slot invariant)

``z_k`` are features standardised **within the slate & slot**, so the fitted
β's carry no slate-specific scale and transfer across days and slates.

Features (all derived from data the pipeline already has)
---------------------------------------------------------
  proj        mean of the sims                 — base demand; already encodes
                                                  matchup + park + Vegas total,
                                                  because those drove the sim.
  ceil_shape  p90(sims) / mean(sims)            — *upside per unit projection*;
                                                  the GPP "boom" appeal that is
                                                  orthogonal to raw projection.
  value       proj / (salary / 1000)            — points per $1k; the classic
                                                  ownership driver. Needs cost.
  team_total  implied runs for the hitter's     — stacking demand; high-total
              team (0 for pitchers)               teams get piled on together.

``value`` and ``team_total`` are *optional*: if salary or Vegas context is
absent the model drops that term and renormalises, so it always produces a
coherent slate. When they are present (the production path) they refine the
sim-only signal.

Contest-size chalk (one canonical knob, shared with field_simulator)
--------------------------------------------------------------------
The base projection describes a *medium* field (``n_medium``). Ownership is
reshaped for other sizes with the same ``own^beta`` temperature the
``field_simulator`` already uses:

    beta(N) = 1 - k · log10(N / n_medium)

Calibration (Jul 26–29, two same-slate size pairs) confirmed the direction and
magnitude: **larger fields are flatter, smaller fields are chalkier**
(own_large ≈ own_small^0.7). ``beta`` is applied per slot and renormalised so
the invariant is preserved at every size.

Public API
----------
    sim_features(scores)                    -> dict of per-player sim features
    project_ownership(pool, sims, ...)      -> pd.Series of %Drafted, pool-indexed
    load_params() / OwnershipParams          coefficients (see fit_ownership.py)

Input contract (matches stage_d.build_pool / the app's pool):
    pool : DataFrame with columns  Name, Pos [, Salary, Team, Opp]
    sims : dict {norm(name) -> np.ndarray of DK sim scores}   (H and P merged)
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# name normalisation — identical rule to stage_d.norm / contest_review._norm
# ---------------------------------------------------------------------------
def norm(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace(",", "").replace("'", "")
    for s in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(s):
            n = n[: -len(s)]
    return n.strip()


# ---------------------------------------------------------------------------
# slot invariant (verified in every calibration contest)
# ---------------------------------------------------------------------------
SLOT_COUNT = {"P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
PITCHER_POS = "P"


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass
class OwnershipParams:
    """Fitted coefficients + structural knobs. See fit_ownership.py.

    Betas are on **within-slate/slot z-scored** features, so they are unit-free
    and slate-transferable. Separate coefficient sets for hitters and pitchers
    because the two markets behave differently (pitcher ownership is far more
    projection-concentrated).
    """
    # hitter attractiveness coefficients (feature -> beta)
    hit: dict = field(default_factory=lambda: {
        "proj": 1.00, "ceil_shape": 0.30, "value": 0.55, "team_total": 0.35,
        "order_score": 0.35,
    })
    # pitcher attractiveness coefficients (order_score is a hitter concept)
    pit: dict = field(default_factory=lambda: {
        "proj": 1.35, "ceil_shape": 0.25, "value": 0.60, "team_total": 0.0,
        "order_score": 0.0,
    })
    # softmax temperature at the medium field (1.0 = betas as-is)
    tau: float = 1.0
    # ownership UNCERTAINTY: σ (in %-owned points) around each point projection,
    # for treating ownership as a distribution rather than a fact in grading.
    # Calibrated heteroskedastic model σ(own) = sigma_a + sigma_b·own — residual
    # spread grows with ownership (a 20%-owned chalk play swings ±~10%, a 1% punt
    # ±~2%). sigma_unconfirmed_mult inflates σ for players whose lineup slot is
    # not confirmed (more news-driven), a principled default (not yet fit — all
    # calibration players had confirmed lineups).
    sigma_a: float = 1.7
    sigma_b: float = 0.41
    sigma_unconfirmed_mult: float = 1.4
    sigma_min: float = 1.0
    sigma_max: float = 15.0
    # contest-size chalk sensitivity  beta(N) = 1 - k*log10(N/n_medium)
    chalk_k: float = 0.20
    n_medium: int = 3000
    # ceiling percentile used for the ceil_shape feature
    ceil_pct: float = 90.0
    # floor for a player with no sim / no signal (%), keeps the field coherent
    min_own: float = 0.05

    def betas(self, is_pitcher: bool) -> dict:
        return self.pit if is_pitcher else self.hit

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "OwnershipParams":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ownership_params.json"
)


def load_params(path: str | None = None) -> OwnershipParams:
    """Load fitted params from JSON, falling back to the built-in defaults."""
    p = path or _PARAMS_PATH
    if os.path.exists(p):
        with open(p) as f:
            return OwnershipParams.from_dict(json.load(f))
    return OwnershipParams()


# ---------------------------------------------------------------------------
# sim -> features
# ---------------------------------------------------------------------------
def sim_features(scores, ceil_pct: float = 90.0) -> dict:
    """Per-player features from a 1-D array of DK sim scores.

    proj        mean outcome (base demand)
    ceiling     the ceil_pct percentile (raw upside)
    ceil_shape  ceiling / proj — upside *relative to* projection (the boom
                appeal that is independent of the projection's level). Clipped
                to a sane band so a near-zero-projection punt can't blow up.
    """
    a = np.asarray(scores, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"proj": np.nan, "ceiling": np.nan, "ceil_shape": np.nan}
    proj = float(a.mean())
    ceiling = float(np.percentile(a, ceil_pct))
    shape = ceiling / proj if proj > 1e-6 else 1.0
    shape = float(np.clip(shape, 1.0, 6.0))
    return {"proj": proj, "ceiling": ceiling, "ceil_shape": shape}


def _z(x: np.ndarray) -> np.ndarray:
    """Standardise, robust to a zero-variance / single-element group."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-9:
        return np.zeros_like(x)
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0)


# ---------------------------------------------------------------------------
# contest-size reshape (shared temperature model with field_simulator)
# ---------------------------------------------------------------------------
def size_beta(contest_size: int, n_medium: int, chalk_k: float) -> float:
    if not contest_size or contest_size <= 0:
        return 1.0
    return float(1.0 - chalk_k * np.log10(contest_size / float(n_medium)))


def _reshape_slot(own: np.ndarray, beta: float) -> np.ndarray:
    """own^beta renormalised to the original slot total (preserves invariant)."""
    if abs(beta - 1.0) < 1e-9:
        return own
    total = own.sum()
    if total <= 0:
        return own
    resh = np.power(np.clip(own, 1e-9, None), beta)
    return resh * (total / resh.sum())


def _cap_redistribute(own: np.ndarray, slot_total: float, cap: float = 100.0,
                      floor: float = 0.0) -> np.ndarray:
    """Water-fill so every player is in [floor, cap] while the slot still sums
    to slot_total. A single player can be on at most 100% of lineups, so a raw
    softmax spike above 100 must spill its excess onto the rest of the slot.
    """
    own = np.clip(own.astype(float), floor, cap)
    if len(own) == 0:
        return own
    # if the cap makes the target infeasible, everyone maxes out (degenerate)
    if slot_total >= cap * len(own):
        return np.full(len(own), cap)
    # Water-fill: repeatedly scale the un-capped players so the slot hits its
    # total, clamp anyone who spills over the cap, and repeat. Converges whether
    # the input sums above OR below slot_total (a random ownership draw can do
    # either), always landing at Σ = slot_total with every player ≤ cap.
    for _ in range(100):
        total = own.sum()
        if abs(total - slot_total) < 1e-9:
            break
        capped = own >= cap - 1e-12
        free = ~capped
        if not free.any():
            break
        deficit = slot_total - own[capped].sum()      # what the free players owe
        cur = own[free].sum()
        if cur > 0:
            own[free] = own[free] * (deficit / cur)    # keep their proportions
        else:
            own[free] = deficit / free.sum()           # all-zero: split evenly
        own = np.clip(own, floor, cap)
    return own


# ---------------------------------------------------------------------------
# core: pool + sims -> projected ownership
# ---------------------------------------------------------------------------
def project_ownership(
    pool: pd.DataFrame,
    sims: dict,
    *,
    contest_size: int | None = None,
    params: OwnershipParams | None = None,
    salary_col: str = "Salary",
    team_total: dict | None = None,
    order: dict | None = None,
    return_sigma: bool = False,
    pos_col: str = "Pos",
    name_col: str = "Name",
) -> pd.Series:
    """Project ownership (%) for every row of ``pool``.

    Parameters
    ----------
    pool          DataFrame, one row per (player, roster position). Must have
                  ``name_col`` and ``pos_col``. ``salary_col`` enables the value
                  term; absent/NaN salary silently drops it.
    sims          {norm(name) -> np.ndarray} of DK sim scores (merge H and P).
    contest_size  field size to reshape for; None = the medium baseline.
    team_total    optional {norm(name) or Team -> implied runs} for the
                  team-total (stacking) feature. Keyed by team code if the pool
                  has a ``Team`` column, else ignored.
    params        OwnershipParams; defaults to load_params().

    Returns
    -------
    pd.Series of projected %Drafted, aligned to ``pool.index``, obeying the
    per-slot invariant (Σ within slot = 100 · slot_count).
    """
    P = params or load_params()
    df = pool.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    if "_orig_idx" not in df.columns:
        df["_orig_idx"] = pool.index

    # --- per-player sim features -------------------------------------------
    feats = {"proj": [], "ceiling": [], "ceil_shape": []}
    for nm in df[name_col]:
        sc = sims.get(norm(nm))
        f = sim_features(sc, P.ceil_pct) if sc is not None else {
            "proj": np.nan, "ceiling": np.nan, "ceil_shape": np.nan}
        for k in feats:
            feats[k].append(f[k])
    for k, v in feats.items():
        df[k] = v

    has_salary = salary_col in df.columns and df[salary_col].notna().any()
    if has_salary:
        sal = pd.to_numeric(df[salary_col], errors="coerce")
        df["value"] = df["proj"] / (sal / 1000.0)
    else:
        df["value"] = np.nan

    if team_total and "Team" in df.columns:
        df["team_total"] = df["Team"].map(lambda t: team_total.get(t, np.nan))
    else:
        df["team_total"] = np.nan

    # batting order: earlier in the order => more PAs => more owned. A confirmed
    # slot 1-9 scores 10-slot (9 best); an unconfirmed/absent order scores 0
    # (worst), so late-swap/unconfirmed bats sink relative to the confirmed field.
    # Source: an `order` dict {norm(name)->slot} or a pool "Order" column.
    def _order_score(nm, row_order):
        o = None
        if order is not None:
            o = order.get(norm(nm))
        if o is None and row_order is not None and pd.notna(row_order):
            o = row_order
        try:
            o = float(o)
        except (TypeError, ValueError):
            return 0.0
        return (10.0 - o) if 1.0 <= o <= 9.0 else 0.0
    row_orders = df["Order"] if "Order" in df.columns else [None] * len(df)
    df["order_score"] = [_order_score(nm, ro)
                         for nm, ro in zip(df[name_col], row_orders)]

    # --- conditional logit within each roster slot -------------------------
    own = np.full(len(df), np.nan)
    for pos, gidx in df.groupby(pos_col).groups.items():
        gidx = list(gidx)
        g = df.loc[gidx]
        slot_total = 100.0 * SLOT_COUNT.get(str(pos), 1)
        is_pitcher = str(pos) == PITCHER_POS
        betas = P.betas(is_pitcher)

        proj = g["proj"].to_numpy(dtype=float)
        have = np.isfinite(proj)
        u = np.zeros(len(g))
        # only features with a fitted, non-zero beta AND available data are used
        for feat in ("proj", "ceil_shape", "value", "team_total", "order_score"):
            b = betas.get(feat, 0.0)
            if b == 0.0:
                continue
            col = g[feat].to_numpy(dtype=float)
            if not np.isfinite(col).any():
                continue
            u = u + b * _z(np.where(np.isfinite(col), col, np.nanmean(col)))

        u = u / max(P.tau, 1e-6)
        # players with no sim get pushed to the floor, not the softmax mass
        u = np.where(have, u, -1e9)
        u = u - np.nanmax(u[np.isfinite(u)]) if np.isfinite(u).any() else u
        ex = np.exp(np.clip(u, -50, 50))
        s = ex.sum()
        share = ex / s if s > 0 else np.full(len(g), 1.0 / len(g))
        slot_own = share * slot_total

        # size reshape (per slot, invariant-preserving)
        if contest_size:
            beta = size_beta(contest_size, P.n_medium, P.chalk_k)
            slot_own = _reshape_slot(slot_own, beta)

        # per-player 100% cap + floor, water-filled to hold the slot invariant
        slot_own = _cap_redistribute(slot_own, slot_total, cap=100.0,
                                     floor=P.min_own)
        for j, gi in enumerate(gidx):
            own[df.index.get_loc(gi)] = slot_own[j]

    out = pd.Series(own, index=df["_orig_idx"].to_numpy())
    out.index = pool.index
    out = out.round(3)
    if return_sigma:
        order_in = df["order_score"].map(lambda s: 10.0 - s if s else np.nan)
        sig = ownership_sigma(out.to_numpy(), order=order_in.to_numpy(), params=P)
        return out, pd.Series(np.round(sig, 3), index=pool.index)
    return out


# ---------------------------------------------------------------------------
# convenience: attach an Ownership column to a stage_d-style pool
# ---------------------------------------------------------------------------
def add_ownership_column(pool: pd.DataFrame, sims: dict, **kw) -> pd.DataFrame:
    """Return a copy of ``pool`` with the projected ``Ownership`` column set."""
    out = pool.copy()
    out["Ownership"] = project_ownership(out, sims, **kw)
    return out


# ---------------------------------------------------------------------------
# ownership uncertainty — treat %Drafted as a distribution, not a fact
# ---------------------------------------------------------------------------
def ownership_sigma(own, order=None, params: OwnershipParams | None = None,
                    confirmed=None) -> np.ndarray:
    """Per-player ownership standard deviation (in %-owned points).

    σ(own) = clip(sigma_a + sigma_b·own, sigma_min, sigma_max), inflated by
    ``sigma_unconfirmed_mult`` for players whose lineup slot is not confirmed.

    own         array-like of projected %Drafted.
    order       optional array-like batting order (1-9 confirmed; NaN/0/None =
                unconfirmed → σ inflated). Ignored if ``confirmed`` is given.
    confirmed   optional boolean array (True = lineup confirmed) overriding
                ``order`` as the confirmation signal.
    """
    P = params or load_params()
    own = np.asarray(own, dtype=float)
    sig = np.clip(P.sigma_a + P.sigma_b * np.clip(own, 0, None),
                  P.sigma_min, P.sigma_max)
    if confirmed is None and order is not None:
        o = pd.to_numeric(pd.Series(order), errors="coerce").to_numpy()
        confirmed = (o >= 1) & (o <= 9)
    if confirmed is not None:
        conf = np.asarray(confirmed, dtype=bool)
        sig = np.where(conf, sig, sig * P.sigma_unconfirmed_mult)
    return np.clip(sig, P.sigma_min, P.sigma_max * P.sigma_unconfirmed_mult)


def sample_ownership(own, sigma, pos, rng, *, cap: float = 100.0,
                     floor: float = 0.0, corr: float = 0.0) -> np.ndarray:
    """Draw ONE ownership realization, invariant-preserving per roster slot.

    Each player is drawn ~ own + σ·z (clipped to [floor, cap]); the draw is then
    renormalised within each position slot back to that slot's total (100 ×
    slot_count), so a realization is always a valid field composition. Feed the
    result into the field simulator per simulation to propagate ownership
    uncertainty into candidate grading.

    corr in [0,1] mixes in a shared per-slate chalk shock (correlated
    up/down-weighting of the whole slate); 0 = purely idiosyncratic draws.
    """
    own = np.asarray(own, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    pos = np.asarray(pos)
    z = rng.standard_normal(len(own))
    if corr > 0:
        g = rng.standard_normal()
        z = np.sqrt(1 - corr) * z + np.sqrt(corr) * g
    draw = np.clip(own + sigma * z, floor, cap)
    out = draw.copy()
    for p in np.unique(pos):
        idx = np.where(pos == p)[0]
        tgt = 100.0 * SLOT_COUNT.get(str(p), 1)
        # renormalise to the slot total while respecting the per-player cap
        # (water-fill), so the draw stays a valid field composition.
        out[idx] = _cap_redistribute(draw[idx], tgt, cap=cap, floor=floor)
    return out


def resample_ownership_pool(pool: pd.DataFrame, rng, *,
                            params: OwnershipParams | None = None,
                            own_col: str = "Ownership", pos_col: str = "Pos",
                            order_col: str = "Order",
                            sigma_col: str | None = None) -> pd.DataFrame:
    """Return a copy of ``pool`` with ``own_col`` replaced by one uncertainty
    draw. σ is taken from ``sigma_col`` if present, else computed from the point
    ownership (and ``order_col`` if present). Drop-in for a field-sim loop:

        for _ in range(n_contests):
            realized = resample_ownership_pool(pool, rng)
            field = build_field(realized)   # existing field build
    """
    P = params or load_params()
    out = pool.copy()
    own = pd.to_numeric(out[own_col], errors="coerce").to_numpy()
    if sigma_col and sigma_col in out.columns:
        sig = out[sigma_col].to_numpy(dtype=float)
    else:
        order = out[order_col] if order_col in out.columns else None
        sig = ownership_sigma(own, order=order, params=P)
    out[own_col] = sample_ownership(own, sig, out[pos_col].to_numpy(), rng)
    return out


if __name__ == "__main__":  # tiny smoke test
    rng = np.random.default_rng(0)
    names = [f"H{i}" for i in range(20)] + [f"P{i}" for i in range(6)]
    pos = ["OF"] * 6 + ["1B"] * 3 + ["2B"] * 3 + ["3B"] * 3 + ["SS"] * 3 + \
          ["C"] * 2 + ["P"] * 6
    sims = {}
    for n in names:
        base = rng.uniform(5, 12)
        sims[norm(n)] = rng.gamma(2.0, base / 2.0, size=5000)
    pool = pd.DataFrame({"Name": names, "Pos": pos,
                         "Salary": rng.integers(3000, 9000, len(names))})
    own = project_ownership(pool, sims, contest_size=5000)
    pool["Own"] = own.values
    print(pool.groupby("Pos")["Own"].sum().round(1).to_dict())
    print(pool.sort_values("Own", ascending=False).head(8).to_string(index=False))

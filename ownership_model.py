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
    })
    # pitcher attractiveness coefficients
    pit: dict = field(default_factory=lambda: {
        "proj": 1.35, "ceil_shape": 0.25, "value": 0.60, "team_total": 0.0,
    })
    # softmax temperature at the medium field (1.0 = betas as-is)
    tau: float = 1.0
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
    own = np.clip(own.astype(float), floor, None)
    if len(own) == 0:
        return own
    # if the cap makes the target infeasible, just clip (degenerate slot)
    if slot_total >= cap * len(own):
        return np.full(len(own), cap)
    for _ in range(100):
        over = own > cap
        if not over.any():
            break
        excess = (own[over] - cap).sum()
        own[over] = cap
        room = (~over) & (own < cap)
        if not room.any():
            break
        w = own[room]
        own[room] = w + excess * (w / w.sum())
    own = np.clip(own, floor, cap)
    # renormalise the uncapped mass back to the exact slot total
    capped = own >= cap - 1e-9
    resid = slot_total - own[capped].sum()
    free = own[~capped]
    if free.sum() > 0 and resid > 0:
        own[~capped] = free * (resid / free.sum())
    return np.clip(own, floor, cap)


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
        for feat in ("proj", "ceil_shape", "value", "team_total"):
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
    return out.round(3)


# ---------------------------------------------------------------------------
# convenience: attach an Ownership column to a stage_d-style pool
# ---------------------------------------------------------------------------
def add_ownership_column(pool: pd.DataFrame, sims: dict, **kw) -> pd.DataFrame:
    """Return a copy of ``pool`` with the projected ``Ownership`` column set."""
    out = pool.copy()
    out["Ownership"] = project_ownership(out, sims, **kw)
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

"""
playing_time.py
===============
Playing-time tiers and the 1 PA / 1 IP floor.

The projection set now spans a whole organization — MLB regulars, September
callups, 4A players, long-term injured, and MLE-translated minor leaguers down
to Single-A. That is deliberate: a season engine that projects "the org" cannot
stop at the 26-man roster. But it creates a distinction the pipeline previously
never had to make:

    **Being in the output is not a claim of playing time.**

This module carries that distinction explicitly, so a Double-A catcher's rate
line can sit in the same file as an MVP's without polluting a team total, a
league aggregate, or a DFS slate.

Two columns
-----------
`pt_tier`   "projected" — expected to accumulate real MLB playing time
            "floor"     — carried for organizational completeness only

`Proj_PA` / `Proj_IP`
            volume. `floor` tier gets exactly PT_FLOOR_PA / PT_FLOOR_IP (1.0).
            `projected` tier gets NaN with `pt_source = "unmodeled"` until a
            playing-time model exists — deliberately NOT a guess, so nobody
            mistakes a placeholder for a forecast.

Why 1 and not 0: zero makes every rate-times-volume product zero, so the player
silently vanishes from totals while still occupying a row. NaN propagates
through sums. One keeps him present, ranked, and joinable, contributes a
rounding error to any aggregate, and reads unambiguously as
"replacement-level placeholder".

What decides the tier today
---------------------------
Evidence, not opinion: did the player clear the bar that used to decide whether
he got projected at all (PT_PROJECTED_MIN_PA within PT_PROJECTED_LOOKBACK
years)? That is a proxy, and a poor one for exactly the cases that matter most
— a top prospect about to break camp as the starting shortstop is `floor`, and
a veteran who just retired is `projected`.

It is used because without a playing-time model there is nothing better, and
because it is *conservative in the right direction*: it errs toward `floor`,
and a `floor` player cannot distort anything. When the real model arrives it
owns `classify_tier` and this proxy reverts to a data-quality note. The
docstring on `PlayingTimeModel` below is the interface it should implement.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from pipeline_config import (
    PT_FLOOR_IP,
    PT_FLOOR_PA,
    PT_PROJECTED_LOOKBACK,
    PT_PROJECTED_MIN_PA,
)

TIER_PROJECTED = "projected"
TIER_FLOOR = "floor"

# `pt_source` values
SOURCE_FLOOR = "floor"          # the 1 PA / 1 IP placeholder
SOURCE_UNMODELED = "unmodeled"  # would get a real projection; no model yet
SOURCE_MODEL = "model"          # a playing-time model produced this


class PlayingTimeModel(Protocol):
    """The interface a real playing-time model should implement.

    Kept as a Protocol rather than a base class so an implementation can live
    anywhere — a module here, a fitted sklearn pipeline, or a hand-maintained
    depth-chart table — without inheriting from this file.

    It must return a frame keyed by PlayerId with, per player:

        Proj_PA / Proj_IP   expected volume
        Proj_G              expected games (needed to convert IP to starts)
        pt_tier             "projected" or "floor"
        pt_source           "model"

    and it must satisfy two accounting constraints the season engine depends on,
    because neither can be recovered downstream:

      1. **Team closure.** Each team's hitter PA must sum to about
         `162 x PA_PER_TEAM_GAME` (~6,150), and its pitcher IP to about
         `162 x 9` (~1,458). A team cannot bat 7,000 times.
      2. **League closure.** Summed across teams, those totals must match the
         league's, which is what finally lets runs-scored equal runs-allowed
         and makes a wins model possible.

    See README_projection_engine.md for the full modelling design.
    """

    def project(self, players: pd.DataFrame) -> pd.DataFrame: ...


# ─────────────────────────────────────────────────────────────────────────────
# Tier classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_tier(
    history: pd.DataFrame,
    target_year: int,
    *,
    id_col: str = "PlayerId",
    volume_col: str = "PA",
    min_volume: float = PT_PROJECTED_MIN_PA,
    lookback: int = PT_PROJECTED_LOOKBACK,
    mle_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Assign every player in `history` a playing-time tier.

    `projected` requires REAL MLB evidence: at least `min_volume` in a season
    within `lookback` years of the target. Everyone else is `floor`.

    MLE-translated players are always `floor` regardless of the synthetic
    volume their injected row carries — that row is a translated minor-league
    line, not MLB playing time, and letting it clear an MLB evidence bar would
    defeat the whole point of the distinction.

    Returns [id_col, pt_tier, evidence_volume, evidence_season].
    """
    mle_ids = mle_ids or set()
    required = {id_col, "Season", volume_col}
    missing = required - set(history.columns)
    if missing:
        raise KeyError(f"history frame missing columns: {sorted(missing)}")

    recent = history[
        (history["Season"] < target_year)
        & (history["Season"] >= target_year - lookback)
    ].copy()
    recent["_vol"] = pd.to_numeric(recent[volume_col], errors="coerce").fillna(0.0)

    # Best single-season volume inside the window, and which season it was.
    if recent.empty:
        evidence = pd.DataFrame(columns=[id_col, "evidence_volume",
                                         "evidence_season"])
    else:
        idx = recent.groupby(id_col)["_vol"].idxmax()
        evidence = (recent.loc[idx, [id_col, "_vol", "Season"]]
                    .rename(columns={"_vol": "evidence_volume",
                                     "Season": "evidence_season"}))

    rows = []
    for pid in pd.unique(history[id_col].dropna()):
        pid = int(pid)
        match = evidence[evidence[id_col] == pid]
        vol = float(match["evidence_volume"].iloc[0]) if len(match) else 0.0
        season = (int(match["evidence_season"].iloc[0]) if len(match)
                  else None)
        projected = (vol >= min_volume) and (pid not in mle_ids)
        rows.append({
            id_col: pid,
            "pt_tier": TIER_PROJECTED if projected else TIER_FLOOR,
            "evidence_volume": vol,
            "evidence_season": season,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Applying playing time
# ─────────────────────────────────────────────────────────────────────────────

def apply_playing_time(
    proj: pd.DataFrame,
    tiers: pd.DataFrame,
    *,
    id_col: str = "PlayerId",
    role: str = "hitter",
    model: PlayingTimeModel | None = None,
) -> pd.DataFrame:
    """Attach `pt_tier`, `pt_source`, and `Proj_PA` (or `Proj_IP`).

    With no `model` (the current state):
      * `floor` tier   -> Proj = PT_FLOOR_PA / PT_FLOOR_IP, source "floor"
      * `projected`    -> Proj = NaN, source "unmodeled"

    The NaN is intentional. Filling it with a plausible-looking 600 PA would
    make every downstream total silently wrong in a way that is very hard to
    notice, whereas a NaN fails loudly at the point of use. `floor` players,
    by contrast, have a genuinely known volume — approximately none — so 1.0
    is a real answer rather than a placeholder for a missing one.

    A player absent from `tiers` is treated as `floor`: unknown provenance is
    exactly the case where we must not assert playing time.
    """
    out = proj.copy()
    vol_col = "Proj_PA" if role == "hitter" else "Proj_IP"
    floor_value = PT_FLOOR_PA if role == "hitter" else PT_FLOOR_IP

    keep = [c for c in (id_col, "pt_tier", "evidence_volume", "evidence_season")
            if c in tiers.columns]
    out = out.merge(tiers[keep], on=id_col, how="left")
    out["pt_tier"] = out["pt_tier"].fillna(TIER_FLOOR)

    if model is not None:
        modelled = model.project(out)
        out = out.drop(columns=[c for c in (vol_col, "pt_source")
                                if c in out.columns])
        out = out.merge(modelled, on=id_col, how="left", suffixes=("", "_m"))
        # Anything the model declined to project still gets the floor.
        out[vol_col] = out[vol_col].fillna(floor_value)
        out["pt_source"] = out.get("pt_source", pd.Series(index=out.index)).fillna(
            SOURCE_FLOOR)
        return out

    is_floor = out["pt_tier"].eq(TIER_FLOOR)
    out[vol_col] = np.where(is_floor, floor_value, np.nan)
    out["pt_source"] = np.where(is_floor, SOURCE_FLOOR, SOURCE_UNMODELED)
    return out


def playing_time_weights(df: pd.DataFrame, *,
                         column: str = "Proj_PA",
                         fallback: str = "Career_PA") -> np.ndarray:
    """Weights for aggregating players into a team or league total.

    Prefers real projected volume, falling back to `fallback` for rows the
    playing-time model has not filled. Crucially, a `floor` player's 1.0 stays
    1.0 rather than falling back to his career total — that is the mechanism
    that stops a thousand depth players from dragging team aggregates around.

    Satisfies `team_context.PlayingTimeWeights`.
    """
    n = len(df)
    if column in df.columns:
        w = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
        if fallback in df.columns:
            fb = pd.to_numeric(df[fallback], errors="coerce").to_numpy(float)
            # Only fill where playing time is genuinely unknown (NaN). A
            # floor row's explicit 1.0 must survive.
            w = np.where(np.isnan(w), fb, w)
        w = np.nan_to_num(w, nan=0.0)
        if np.nansum(w) > 0:
            return np.maximum(w, 0.0)

    if fallback in df.columns:
        w = pd.to_numeric(df[fallback], errors="coerce").fillna(0.0).to_numpy(float)
        if w.sum() > 0:
            return np.maximum(w, 1.0)
    return np.ones(n)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def tier_report(df: pd.DataFrame, *, role: str = "hitter") -> str:
    """Summary of tier composition, for pipeline output."""
    if "pt_tier" not in df.columns:
        return "no pt_tier column"
    vol_col = "Proj_PA" if role == "hitter" else "Proj_IP"
    counts = df["pt_tier"].value_counts().to_dict()
    lines = [f"  tiers: {counts}"]
    if "mle_source" in df.columns:
        mle = df[df["mle_source"].astype(str).eq("MiLB")]
        if len(mle):
            lines.append(f"  MLE-translated: {len(mle)} "
                         f"(all tier={sorted(set(mle['pt_tier']))})")
    if vol_col in df.columns:
        floored = int(df[vol_col].eq(PT_FLOOR_PA if role == "hitter"
                                     else PT_FLOOR_IP).sum())
        unmodeled = int(df[vol_col].isna().sum())
        lines.append(f"  {vol_col}: {floored} at floor, "
                     f"{unmodeled} awaiting a playing-time model")
    return "\n".join(lines)

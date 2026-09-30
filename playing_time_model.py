"""
playing_time_model.py
=====================
The baseline playing-time model: a default role for every player, turned into
PA / IP by a constrained team allocation.

This implements the `PlayingTimeModel` protocol in playing_time.py and fills
the hole that module was written around — 715 hitters and 946 pitchers whose
`Proj_PA` / `Proj_IP` were deliberately NaN rather than a guess. Roughly 26 of
the 54 requested season categories are rate x volume, so nothing downstream
could produce a counting stat until this existed.

The shape of the problem
------------------------
Playing time is **not** a per-player regression. A team bats about
`162 x PA_PER_TEAM_GAME` (~6,156) times and throws `162 x 9` (1,458) innings,
and those totals do not care how many players we like. Project every player
independently and the sum lands wherever it lands; the league then scores more
runs than it allows and the wins model has nothing to stand on.

So it is an **allocation**:

    1. every player gets a ROLE                     (a job)
    2. role -> a raw volume                         anchor x timing x availability
    3. each team's raw volumes are SCALED to close  on its real budget
    4. the scale factor is reported, not hidden     it measures anchor error

Step 4 matters as much as the other three. A team whose assigned roles sum to
1,100 innings against a 1,458 budget gets everyone scaled up 33%, and that is
not a fact about the team — it is a fact about the anchors being too small.
Printing the factor turns a silent distortion into a number someone can fit
against (`fit_role_anchors`).

Everything here is a default
----------------------------
Every role, timing and availability value is overridable per player, from
`rosters/player_roles_{hitters,pitchers}_{year}.csv` or the generated .xlsx
workbook. The model's job is to make sure nobody starts from nothing, not to
have opinions that cannot be corrected. `load_role_overrides` reports how many
defaults each file replaced, so an override file that silently matched nothing
(the usual failure: wrong id column, wrong year) is visible.

What it does NOT do yet
-----------------------
* **Typed slots.** A team's nine lineup spots have positions; this allocates a
  single undifferentiated PA pool. Two first basemen can both be Full Time
  here, which no real team does. Position data now exists (the fielding fetch),
  so this is the next step and it is a modelling step, not a data gap.
* **Injury coupling.** "Injury Replacement" volume is conditional on OTHER
  players getting hurt. A point estimate cannot express that; it needs a
  simulation over the roster.
* **Fitted anchors.** See above.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import PT_FLOOR_IP, PT_FLOOR_PA
from playing_time import SOURCE_FLOOR, SOURCE_MODEL, TIER_FLOOR, TIER_PROJECTED
from role_taxonomy import (
    DEFAULT_AVAILABILITY,
    DEFAULT_TIMING,
    DEPTH_HITTER_ROLE,
    DEPTH_PITCHER_ROLE,
    is_depth_role,
    full_time_reference,
    primary_positions,
    role_anchor,
    role_names,
    suggest_hitter_role,
    suggest_pitcher_role,
    timing_share,
)
from team_context import FREE_AGENT_TEAM_ID, PA_PER_TEAM_GAME

# Team budgets over a 162-game season.
TEAM_PA_BUDGET = 162.0 * PA_PER_TEAM_GAME     # ~6,156
TEAM_IP_BUDGET = 162.0 * 9.0                  # 1,458

# Per-player ceilings, as physical bounds rather than opinions. The most PA
# anyone has taken in a season is ~778 (Jimmy Rollins 2007); the most IP in the
# modern era is ~250. These exist so a shallow roster cannot scale one player
# to an impossible workload during closure — without them, a team with few
# projected players hands its whole budget to whoever is there.
PT_MAX_PA = 760.0
PT_MAX_IP = 230.0

# Closure is an iterative proportional fit: scale, clip anyone over the
# ceiling, redistribute the remainder, repeat. Converges in two or three passes
# because the ceiling binds for very few players; the cap stops a pathological
# roster from looping.
_CLOSURE_PASSES = 6


# ─────────────────────────────────────────────────────────────────────────────
# Overrides
# ─────────────────────────────────────────────────────────────────────────────

def role_override_path(kind: str, target_year: int,
                       base: str | Path = "rosters") -> Path:
    plural = "hitters" if kind == "hitter" else "pitchers"
    return Path(base) / f"player_roles_{plural}_{target_year}.csv"


def load_role_overrides(path: str | Path, kind: str) -> pd.DataFrame:
    """Per-player role / timing / availability overrides.

    A CSV keyed by `PlayerId` with any of:

        Role            a name from the taxonomy
        Role Start      a Timing label ("Opening Day", "Mid Season (~July)", ..)
        Availability    0..1, share of the season healthy and on a roster

    Only the columns present are applied, so a file that sets nothing but
    Availability for three players is valid and leaves every other default
    alone. Unknown role names are dropped WITH A WARNING naming them rather
    than silently benching the player — a typo in a hand-edited file is the
    most likely failure here, and the second most likely is a file that matches
    nobody, which `apply_role_overrides` reports as a count.

    Returns an empty frame when the file is absent: overrides are optional by
    design, and the baseline must work with none of them.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["PlayerId"])
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  role overrides {p.name}: UNREADABLE ({type(e).__name__}); "
              "using defaults")
        return pd.DataFrame(columns=["PlayerId"])
    if "PlayerId" not in df.columns:
        print(f"  role overrides {p.name}: no PlayerId column; ignored")
        return pd.DataFrame(columns=["PlayerId"])

    df = df.dropna(subset=["PlayerId"]).copy()
    df["PlayerId"] = pd.to_numeric(df["PlayerId"], errors="coerce")
    df = df.dropna(subset=["PlayerId"])
    df["PlayerId"] = df["PlayerId"].astype(int)

    valid = set(role_names(kind))
    if "Role" in df.columns:
        bad = sorted(set(df.loc[df["Role"].notna(), "Role"]) - valid)
        if bad:
            print(f"  role overrides {p.name}: {len(bad)} unknown role name(s) "
                  f"ignored: {bad[:5]}")
            df.loc[df["Role"].isin(bad), "Role"] = np.nan
    if "Availability" in df.columns:
        df["Availability"] = pd.to_numeric(
            df["Availability"], errors="coerce").clip(0.0, 1.0)
    keep = ["PlayerId"] + [c for c in ("Role", "Role Start", "Availability")
                           if c in df.columns]
    return df[keep].drop_duplicates("PlayerId", keep="last")


def apply_role_overrides(players: pd.DataFrame,
                         overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Let explicit assignments win over the derived defaults."""
    out = players.copy()
    stats = {"matched": 0, "role": 0, "timing": 0, "availability": 0,
             "unmatched": 0}
    if overrides is None or overrides.empty:
        return out, stats

    known = set(out["PlayerId"])
    stats["unmatched"] = int((~overrides["PlayerId"].isin(known)).sum())
    ov = overrides[overrides["PlayerId"].isin(known)].set_index("PlayerId")
    stats["matched"] = len(ov)

    for col, dest, key in (("Role", "pt_role", "role"),
                           ("Role Start", "pt_role_start", "timing"),
                           ("Availability", "pt_availability", "availability")):
        if col not in ov.columns:
            continue
        vals = ov[col].dropna()
        if vals.empty:
            continue
        idx = out["PlayerId"].map(vals)
        mask = idx.notna()
        out.loc[mask, dest] = idx[mask].to_numpy()
        out.loc[mask, "pt_role_source"] = "override"
        stats[key] = int(mask.sum())
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# Default role assignment
# ─────────────────────────────────────────────────────────────────────────────

def _staff_ranks(players: pd.DataFrame, *, team_col: str) -> pd.Series:
    """RA9 rank within each staff, 1 = best.

    A bullpen role is a within-team standing — the best arm on a staff is the
    likeliest closer, and "best reliever in baseball" is not a job. Ranking
    league-wide instead produced 30 closers on a handful of teams the first
    time this was tried.
    """
    if "RA9" not in players.columns or team_col not in players.columns:
        return pd.Series(np.nan, index=players.index)
    ra9 = pd.to_numeric(players["RA9"], errors="coerce")
    relievers = players["role"].astype(str).str.lower() != "starter" \
        if "role" in players.columns else pd.Series(True, index=players.index)
    rank = pd.Series(np.nan, index=players.index)
    # PROJECTED TIER ONLY. Floor-tier arms are MLE-translated minor leaguers,
    # and a translated line shrunk toward the league mean can out-rank every
    # real reliever on the staff. Including them put a Double-A arm at rank 1
    # on all 30 clubs — 30 floor-tier "closers" — and pushed the actual
    # bullpen past rank 6 into the depth role. Ranking is a standing among
    # players who will pitch.
    eligible = relievers & ra9.notna() & players[team_col].notna()
    if "pt_tier" in players.columns:
        eligible &= players["pt_tier"].astype(str) == TIER_PROJECTED
    rank[eligible] = (ra9[eligible]
                      .groupby(players.loc[eligible, team_col])
                      .rank(method="first", ascending=True))
    return rank


def assign_default_roles(players: pd.DataFrame, kind: str, *,
                         fielding: pd.DataFrame | None = None,
                         team_col: str = "Pred_target_team_id",
                         ) -> pd.DataFrame:
    """Give every player a role, a start time and an availability.

    Defaults only — `apply_role_overrides` replaces any of them. Floor-tier
    players get the depth role, which is what routes them to the 1 PA / 1 IP
    floor instead of into the team's allocation.
    """
    out = players.copy()
    vol_col = "evidence_volume" if "evidence_volume" in out.columns else (
        "Last_PA" if "Last_PA" in out.columns else None)
    reference = full_time_reference(out[vol_col]) if vol_col else 1.0

    if kind == "hitter":
        pos = primary_positions(fielding) if fielding is not None else {}
        out["pt_position"] = out["PlayerId"].map(pos)
        roles = [
            suggest_hitter_role(r, reference, position=r.get("pt_position"))
            for _, r in out.iterrows()
        ]
    else:
        ranks = _staff_ranks(out, team_col=team_col)
        roles = [
            suggest_pitcher_role(
                r, reference,
                staff_rank=(int(ranks.iloc[i]) if pd.notna(ranks.iloc[i])
                            else None))
            for i, (_, r) in enumerate(out.iterrows())
        ]

    out["pt_role"] = roles
    out["pt_role_start"] = DEFAULT_TIMING
    out["pt_availability"] = DEFAULT_AVAILABILITY
    out["pt_role_source"] = "default"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Role -> raw volume
# ─────────────────────────────────────────────────────────────────────────────

# How far a player's own history may move him within his role. The role sets
# the TIER; evidence sets his position inside it. Without this every player in
# a role receives an identical number — the first run of this model gave Aaron
# Judge, Ben Rice, Ryan McMahon, Trent Grisham and Heliot Ramos exactly 418.1
# PA each, which is not a projection anyone can use.
#
# Bounded on both sides on purpose. Below the floor the evidence would be
# overriding the role rather than modulating it (a player assigned Full Time is
# a full-time player, whatever last year looked like); above the ceiling the
# player is probably mis-roled, and the override file is the right fix for that
# rather than letting one season's volume silently promote him.
EVIDENCE_FACTOR_MIN = 0.45
EVIDENCE_FACTOR_MAX = 1.30


def _evidence_factor(players: pd.DataFrame, kind: str) -> pd.Series:
    """Within-role differentiation from the player's own volume history.

    Hitter evidence is already PA, directly comparable to a PA anchor. Pitcher
    evidence is BATTERS FACED, which is not comparable to an IP anchor at all
    — converting through each player's own `TBF_per_IP` keeps a reliever who
    faces many batters per inning from being credited with a starter's innings.
    """
    ev = pd.to_numeric(players.get("evidence_volume"), errors="coerce") \
        if "evidence_volume" in players.columns else pd.Series(
            np.nan, index=players.index)
    if kind == "pitcher":
        tbf_per_ip = pd.to_numeric(players.get("TBF_per_IP"),
                                   errors="coerce") \
            if "TBF_per_IP" in players.columns else pd.Series(
                np.nan, index=players.index)
        # 4.3 is the league rate; used only where a player has no own value.
        tbf_per_ip = tbf_per_ip.where(tbf_per_ip > 1.0, 4.3)
        ev = ev / tbf_per_ip
    anchor = pd.to_numeric(players["pt_anchor"], errors="coerce")
    factor = (ev / anchor.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    # No evidence -> 1.0, i.e. take the role at face value.
    return factor.fillna(1.0).clip(EVIDENCE_FACTOR_MIN, EVIDENCE_FACTOR_MAX)


# A 26-man roster is 13 position players and 13 pitchers. The projected tier
# is far more generous than that — it admits anyone with 25+ PA inside the
# lookback, which came to 23.8 hitters and 31.5 pitchers per club. Those extra
# players are real major leaguers who will play, but NOT all of them for this
# team and not all season, and their nominal jobs over-subscribe the budget by
# a third.
#
# Left alone, closure spreads that shortfall evenly and compresses everybody:
# Aaron Judge to 460 PA, an ace to 117 innings, a team's top nine to 61% of its
# plate appearances against a real 77%. Concentrating the allocation instead
# (scaling by raw^gamma) fixes the top and wrecks the bottom — at the exponent
# that reproduces the top-nine share, the bench falls to 9 PA and the CLOSER to
# 22 innings, because bullpen anchors are deliberately flat and an exponent on
# volume punishes precisely the roles that should not scale with it.
#
# So the shortfall belongs where the over-subscription is: the players beyond
# roster depth. Each rank past the core keeps `ROSTER_DEPTH_DECAY` of the
# previous one's volume, which is what "he is up and down from Triple-A" means
# expressed as playing time. Overridable per player like every other default.
ROSTER_DEPTH_CORE = {"hitter": 13, "pitcher": 13}
ROSTER_DEPTH_DECAY = 0.78
ROSTER_DEPTH_FLOOR = 0.04


def apply_roster_depth(players: pd.DataFrame, kind: str, *,
                       team_col: str = "Pred_target_team_id") -> pd.DataFrame:
    """Discount players ranked beyond their club's core roster.

    Ranked among the PROJECTED TIER ONLY: the floor tier is organizational
    depth that never enters the allocation, and including it pushed real
    relievers past rank 30 on every staff.
    """
    out = players.copy()
    out["pt_depth_rank"] = np.nan
    out["pt_depth_factor"] = 1.0
    if team_col not in out.columns:
        return out
    core = ROSTER_DEPTH_CORE.get(kind, 13)
    tier = out.get("pt_tier", pd.Series(TIER_PROJECTED, index=out.index))
    eligible = (tier.astype(str) == TIER_PROJECTED) & out[team_col].notna()
    if not eligible.any():
        return out
    rank = (out.loc[eligible, "pt_raw"]
            .groupby(out.loc[eligible, team_col])
            .rank(ascending=False, method="first"))
    out.loc[eligible, "pt_depth_rank"] = rank
    beyond = rank[rank > core]
    if len(beyond):
        factor = np.maximum(ROSTER_DEPTH_DECAY ** (beyond - core),
                            ROSTER_DEPTH_FLOOR)
        out.loc[beyond.index, "pt_depth_factor"] = factor.to_numpy()
    out["pt_raw"] = out["pt_raw"] * out["pt_depth_factor"]
    return out


def raw_volumes(players: pd.DataFrame, kind: str,
                team_col: str = "Pred_target_team_id") -> pd.DataFrame:
    """anchor x timing x availability x evidence, then roster-depth discount."""
    out = players.copy()
    key = "pa" if kind == "hitter" else "ip"
    anchors, gs, g, sv, hld, vl = [], [], [], [], [], []
    unknown: set[str] = set()

    bats_col = "BatSide" if "BatSide" in out.columns else None
    for _, r in out.iterrows():
        a = role_anchor(str(r.get("pt_role")), kind)
        if a is None:
            unknown.add(str(r.get("pt_role")))
            a = role_anchor(DEPTH_HITTER_ROLE if kind == "hitter"
                            else DEPTH_PITCHER_ROLE, kind)
        anchors.append(float(a[key]))
        if kind == "pitcher":
            gs.append(float(a["gs"])); g.append(float(a["g"]))
            sv.append(float(a["sv"])); hld.append(float(a["hld"]))
        else:
            left = str(r.get(bats_col, "") or "").upper().startswith("L") \
                if bats_col else False
            vl.append(float(a["vl_lhb"] if left else a["vl_rhb"]))

    if unknown:
        print(f"  WARNING: {len(unknown)} unrecognized role name(s) fell back "
              f"to the depth anchor: {sorted(unknown)[:5]}")

    share = out["pt_role_start"].map(timing_share).astype(float)
    avail = pd.to_numeric(out["pt_availability"],
                          errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["pt_anchor"] = np.asarray(anchors, dtype=float)
    out["pt_season_share"] = share * avail
    out["pt_evidence_factor"] = _evidence_factor(out, kind)
    out["pt_raw"] = (out["pt_anchor"] * out["pt_season_share"]
                     * out["pt_evidence_factor"])
    out = apply_roster_depth(out, kind, team_col=team_col)
    if kind == "pitcher":
        out["pt_anchor_GS"] = gs
        out["pt_anchor_G"] = g
        out["pt_save_share"] = sv
        out["pt_hold_share"] = hld
    else:
        out["Proj_vL_share"] = vl
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Team closure
# ─────────────────────────────────────────────────────────────────────────────

def _close_one_team(raw: np.ndarray, target: float,
                    ceiling: float) -> np.ndarray:
    """Scale `raw` so it sums to `target`, with nobody above `ceiling`.

    Iterative proportional fit: scale everyone, clip whoever exceeds the
    ceiling, then re-scale only the unclipped to absorb the remainder. Without
    the redistribution a clipped player's surplus would simply vanish and the
    team would under-close.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0 or target <= 0:
        return np.zeros_like(raw)
    total = raw.sum()
    if total <= 0:
        # No role information at all: split the budget evenly rather than
        # returning zeros, which would delete the team's offense.
        return np.full_like(raw, min(target / raw.size, ceiling))

    out = raw * (target / total)
    for _ in range(_CLOSURE_PASSES):
        over = out > ceiling
        if not over.any():
            break
        out[over] = ceiling
        slack = target - out[over].sum()
        free = ~over
        if not free.any() or slack <= 0:
            break
        s = out[free].sum()
        if s <= 0:
            out[free] = min(slack / free.sum(), ceiling)
            break
        out[free] = out[free] * (slack / s)
    return out


def allocate_playing_time(players: pd.DataFrame, kind: str, *,
                          team_col: str = "Pred_target_team_id",
                          reserves: dict | None = None,
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close each team's raw volumes on its real budget.

    Returns (players with Proj_PA/Proj_IP, per-team diagnostics).

    Floor-tier and depth-role players are held at exactly PT_FLOOR_PA /
    PT_FLOOR_IP and their total is SUBTRACTED from the budget before the
    projected players are scaled, so the team still closes exactly and the
    verifier's "0 not at 1.0" check keeps passing.

    Free agents and players with no club are excluded from closure and keep
    their raw role volume: they are expected to sign somewhere, and folding
    them into a team they are not on would take that team's plate appearances
    away from the players who will actually bat.
    """
    out = players.copy()
    is_pa = kind == "hitter"
    vol_out = "Proj_PA" if is_pa else "Proj_IP"
    budget = TEAM_PA_BUDGET if is_pa else TEAM_IP_BUDGET
    floor_v = PT_FLOOR_PA if is_pa else PT_FLOOR_IP
    ceiling = PT_MAX_PA if is_pa else PT_MAX_IP
    share_key = "pa_share" if is_pa else "ip_share"
    reserves = reserves or {}

    tier = out.get("pt_tier", pd.Series(TIER_PROJECTED, index=out.index))
    at_floor = (tier.astype(str) == TIER_FLOOR) | \
        out["pt_role"].astype(str).map(is_depth_role)

    out[vol_out] = np.nan
    out.loc[at_floor, vol_out] = floor_v
    out["pt_source"] = np.where(at_floor, SOURCE_FLOOR, SOURCE_MODEL)
    out["pt_tier"] = np.where(at_floor, TIER_FLOOR, TIER_PROJECTED)

    teams = out[team_col] if team_col in out.columns else pd.Series(
        np.nan, index=out.index)
    on_a_club = teams.notna() & (teams != FREE_AGENT_TEAM_ID)

    rows = []
    for team_id, idx in out[on_a_club].groupby(teams[on_a_club]).groups.items():
        block = out.loc[idx]
        floor_rows = at_floor.loc[idx]
        reserve = float(reserves.get(int(team_id), {}).get(share_key, 0.0) or 0.0)
        reserved = budget * reserve
        floor_total = float(floor_rows.sum()) * floor_v
        target = budget - reserved - floor_total
        proj_idx = block.index[~floor_rows]
        if len(proj_idx) == 0 or target <= 0:
            rows.append({"team_id": int(team_id), "n": len(block),
                         "n_projected": 0, "raw": 0.0, "target": target,
                         "scale": np.nan, "reserved_share": reserve})
            continue
        raw = pd.to_numeric(out.loc[proj_idx, "pt_raw"],
                            errors="coerce").fillna(0.0).to_numpy()
        closed = _close_one_team(raw, target, ceiling)
        out.loc[proj_idx, vol_out] = closed
        rows.append({
            "team_id": int(team_id), "n": len(block),
            "n_projected": len(proj_idx), "raw": float(raw.sum()),
            "target": float(target),
            "scale": float(target / raw.sum()) if raw.sum() > 0 else np.nan,
            "reserved_share": reserve,
        })

    # Off-roster players keep their raw role volume, capped.
    off = ~on_a_club & ~at_floor
    if off.any():
        out.loc[off, vol_out] = pd.to_numeric(
            out.loc[off, "pt_raw"], errors="coerce").fillna(floor_v).clip(
                upper=ceiling)

    # Save and hold shares are per-ROLE WEIGHTS, not an allocation: a closer's
    # 0.65 says "a closer takes about 65% of a save pool", which is a fact
    # about the role and not about how many pitchers a club happens to carry.
    # Summed over a real staff they come to 1.16 (saves) and 1.85 (holds), so
    # a consumer who multiplies them straight into a team pool over-allocates
    # — league saves read 1,412 against a pool of 1,215, and holds 4,270
    # against 2,308. Normalising per team turns the weights into shares that
    # sum to 1, which is what a pool needs. The raw role weight is kept beside
    # them because it is still the role's own attribute.
    if not is_pa:
        for src, dest in (("pt_save_share", "Proj_SV_share"),
                          ("pt_hold_share", "Proj_HLD_share")):
            w = pd.to_numeric(out[src], errors="coerce").fillna(0.0)
            # Only players who will pitch compete for the pool.
            w = w.where(~at_floor, 0.0)
            tot = w.groupby(teams).transform("sum")
            out[dest] = np.where(tot > 0, w / tot, 0.0)

    # Games / starts follow the same scaling as volume, so a player scaled up
    # 30% is credited with proportionally more appearances rather than pitching
    # 250 innings across 32 starts.
    if not is_pa:
        ratio = (pd.to_numeric(out[vol_out], errors="coerce")
                 / pd.to_numeric(out["pt_anchor"], errors="coerce")
                 .replace(0, np.nan))
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["Proj_GS"] = (out["pt_anchor_GS"] * ratio).clip(0, 40).round(1)
        out["Proj_G"] = (out["pt_anchor_G"] * ratio).clip(0, 82).round(1)
    else:
        pa = pd.to_numeric(out[vol_out], errors="coerce")
        out["Proj_G"] = (pa / PA_PER_TEAM_GAME * 9.0).clip(0, 162).round(1)

    return out, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def project_playing_time(players: pd.DataFrame, kind: str, *,
                         target_year: int,
                         fielding: pd.DataFrame | None = None,
                         roster_path: str | Path = "rosters",
                         reserves: dict | None = None,
                         team_col: str = "Pred_target_team_id",
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Defaults -> overrides -> raw volume -> team closure.

    Satisfies `playing_time.PlayingTimeModel`: returns Proj_PA/Proj_IP, Proj_G,
    pt_tier and pt_source per player, closing on the team and hence the league
    budget.
    """
    stats: dict = {}
    out = assign_default_roles(players, kind, fielding=fielding,
                               team_col=team_col)
    ov = load_role_overrides(role_override_path(kind, target_year, roster_path),
                             kind)
    out, stats["overrides"] = apply_role_overrides(out, ov)
    out = raw_volumes(out, kind, team_col=team_col)
    out, team_diag = allocate_playing_time(out, kind, team_col=team_col,
                                           reserves=reserves)
    stats["roles"] = out["pt_role"].value_counts().to_dict()
    return out, team_diag, stats


def playing_time_report(out: pd.DataFrame, team_diag: pd.DataFrame,
                        stats: dict, kind: str) -> str:
    """What the allocation did, and how much the anchors had to be stretched."""
    is_pa = kind == "hitter"
    vol = "Proj_PA" if is_pa else "Proj_IP"
    budget = TEAM_PA_BUDGET if is_pa else TEAM_IP_BUDGET
    lines = [f"  {kind}s: {len(out)} players"]

    ov = stats.get("overrides", {})
    if ov.get("matched") or ov.get("unmatched"):
        lines.append(f"    overrides: {ov.get('matched', 0)} matched "
                     f"({ov.get('role', 0)} role, {ov.get('timing', 0)} timing, "
                     f"{ov.get('availability', 0)} availability), "
                     f"{ov.get('unmatched', 0)} unmatched")
    else:
        lines.append("    overrides: none (all roles are defaults)")

    v = pd.to_numeric(out[vol], errors="coerce")
    lines.append(f"    {vol}: sum {v.sum():,.0f}  "
                 f"max {v.max():,.1f}  median {v.median():,.1f}")

    if not team_diag.empty:
        sc = pd.to_numeric(team_diag["scale"], errors="coerce").dropna()
        if len(sc):
            lines.append(f"    team closure: {len(team_diag)} teams, "
                         f"budget {budget:,.0f} each")
            lines.append(f"    anchor scale: mean {sc.mean():.3f}  "
                         f"min {sc.min():.3f}  max {sc.max():.3f}")
            if abs(sc.mean() - 1.0) > 0.15:
                lines.append(
                    f"    ^ mean scale is {sc.mean():.2f}, not ~1.0: the role "
                    f"ANCHORS are mis-sized for a real roster, not the teams. "
                    f"This is what fit_role_anchors should correct.")
        lines.append(f"    reserved for signings: "
                     f"{(team_diag['reserved_share'] > 0).sum()} team(s)")
        empty = team_diag[team_diag["n_projected"] == 0]
        if len(empty):
            # Such a club cannot reach its budget — there is nobody to give it
            # to — so the league total will fall short. That is a roster or
            # tier problem, not an allocation one, and it must not be silent.
            lines.append(
                f"    WARNING: {len(empty)} team(s) have no projected players "
                f"and cannot close: {sorted(empty['team_id'].tolist())[:5]}")

    top = stats.get("roles", {})
    if top:
        shown = sorted(top.items(), key=lambda kv: -kv[1])[:6]
        lines.append("    roles: " + ", ".join(f"{k} {v}" for k, v in shown))
    return "\n".join(lines)

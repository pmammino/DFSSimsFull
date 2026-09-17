"""
team_context.py
===============
Team identity, roster assignment, and team run environment for the season-long
projection engine.

This is the first layer of the season engine. It sits ON TOP of the per-PA
projections in `out/` — it does not change how any per-PA rate is projected.
That separation is deliberate: the per-PA CSVs stay the single canonical skill
layer consumed by both the daily DFS path and the season path, so the daily
pipeline is unaffected by anything here.

Three jobs:

1. **Canonical team identity.** MLBAM team id <-> canonical abbreviation, with
   `slate_config.canonical_team` as the single source of truth for what a
   canonical code is. Replaces the `team["name"][:3]` fallback that collapsed
   30 franchises into 26 labels (Chi/Los/New/San each merged two teams).

2. **One team-assignment rule for hitters AND pitchers**, with support for
   players changing teams. Previously hitters used "most PA in the latest
   season" while pitchers used an unstable `groupby().last()`, and neither
   could express an offseason move.

3. **Bottom-up team run environment.** The team context a hitter's R/RBI is
   scaled by is derived from the talent of the players actually projected onto
   that roster, instead of extrapolating the team's own past runs-per-game.
   Because it is built from the roster, moving a player automatically updates
   BOTH the team he left and the team he joined.

Why bottom-up
-------------
The historical-RPG factor correlates only r = 0.67 with the talent of its own
roster, and is MORE dispersed (SD 0.073) than that talent (SD 0.048) — it
carries backward-looking noise. `runs_rbi_model`'s own docstring concedes that
applying it increases weighted MAE by 6-8% versus the neutral projection. It
also cannot represent a roster change at all: a team that lost its three best
hitters still carries last year's run environment.

A roster-derived factor fixes all three, and is the only formulation in which
"player changes teams" and "team context adjusts accordingly" are the same
operation.

What is deliberately still missing
----------------------------------
**Playing time.** There is no projected PA/G/IP model yet, so the roster
weighting used to aggregate a team's talent is a documented proxy (see
`PlayingTimeWeights`). Every function that needs playing time takes it as an
injectable weight source rather than computing one internally, so the real
model drops in without touching this module's logic. Until then, treat the
team factors as structurally correct but provisionally weighted.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from slate_config import canonical_team

# ─────────────────────────────────────────────────────────────────────────────
# Canonical team identity
# ─────────────────────────────────────────────────────────────────────────────
# MLBAM team ids are the durable join key — they survive relocations and
# rebrands that break abbreviation strings. The abbreviations here are the
# canonical codes `slate_config.canonical_team` produces, so ids, projection
# rows, and slate feeds all reconcile on one vocabulary.
#
# Note OAK: MLBAM id 133 is the Athletics, whose feeds now emit "ATH". The
# canonical code in this repo is "OAK" (see slate_config._CANON_ALIASES), so
# that is what we emit. Do not "fix" this to ATH without also updating
# slate_config — `team_alias_fingerprint` exists precisely because changing
# canonical codes invalidates cached sim keys.
TEAM_ABBR_BY_ID: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",  119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT", 135: "SD",  136: "SEA",
    137: "SF",  138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

TEAM_ID_BY_ABBR: dict[str, int] = {v: k for k, v in TEAM_ABBR_BY_ID.items()}

assert len(TEAM_ABBR_BY_ID) == 30, "expected 30 MLB franchises"
assert len(TEAM_ID_BY_ABBR) == 30, "team abbreviations must be unique"


def abbr_for_team_id(team_id) -> str | None:
    """Canonical abbreviation for an MLBAM team id, or None if unknown.

    Unknown ids return None rather than raising: minor-league affiliates and
    All-Star/exhibition team ids do appear in statsapi responses, and they
    should degrade to "no team" rather than kill a pipeline run.
    """
    if team_id is None or (isinstance(team_id, float) and np.isnan(team_id)):
        return None
    try:
        return TEAM_ABBR_BY_ID.get(int(team_id))
    except (TypeError, ValueError):
        return None


def team_id_for_abbr(abbr) -> int | None:
    """MLBAM team id for any team code or full name the repo understands.

    Routes through `slate_config.canonical_team`, so Rotowire codes ("NY-A"),
    FantasyLabs variants ("CHW"), and full names ("New York Yankees") all
    resolve. Returns None for anything unrecognized.
    """
    if abbr is None:
        return None
    canon = canonical_team(abbr)
    return TEAM_ID_BY_ABBR.get(canon) if canon else None


def team_label(team_id, fallback: str = "") -> str:
    """Display abbreviation for a team id, for output columns and printouts."""
    return abbr_for_team_id(team_id) or fallback


# ─────────────────────────────────────────────────────────────────────────────
# Roster assignment — one rule for hitters and pitchers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TeamAssignment:
    """Where a player is projected to play, and how confident we are.

    source is one of:
      "override"  — an explicit entry in the roster-override file (a signing,
                    trade, or manual correction). Highest authority.
      "history"   — inferred from the player's most recent playing time.
      "unknown"   — no signal; the player gets neutral team context.
    """
    player_id: int
    team_id: int | None
    source: str

    @property
    def team_abbr(self) -> str | None:
        return abbr_for_team_id(self.team_id)


ROSTER_DIR = Path(__file__).resolve().parent / "rosters"


def TEAM_OVERRIDE_PATH(target_year: int) -> Path:
    """Conventional location of the roster-override file for a target year.

    `rosters/team_assignments_<year>.json`. Absent by default — see
    `rosters/team_assignments.example.json` for the format.
    """
    return ROSTER_DIR / f"team_assignments_{int(target_year)}.json"


def load_team_overrides(path: str | Path) -> dict[int, int | None]:
    """Load a roster-override file: {PlayerId -> MLBAM team id}.

    The file expresses everything history cannot — offseason signings, trades,
    and manual corrections. Format (see rosters/team_assignments.example.json):

        {
          "target_year": 2027,
          "assignments": [
            {"player_id": 592450, "team": "SF",  "note": "signed 2026-12-01"},
            {"player_id": 660271, "team": "NYY", "note": "traded"},
            {"player_id": 111111, "team": null,  "note": "retired / unsigned"}
          ]
        }

    `team` accepts any code or full name `canonical_team` understands, or an
    explicit MLBAM id via `team_id`. A null team means "no team" — the player
    is carried with neutral context rather than silently keeping his old club.

    A missing file returns {} so the override layer is strictly optional.
    """
    p = Path(path)
    if not p.exists():
        return {}

    payload = json.loads(p.read_text())
    entries = payload.get("assignments", payload) if isinstance(payload, dict) else payload

    out: dict[int, int | None] = {}
    for entry in entries:
        try:
            pid = int(entry["player_id"])
        except (KeyError, TypeError, ValueError):
            warnings.warn(f"team override with no usable player_id: {entry!r}")
            continue

        if "team_id" in entry and entry["team_id"] is not None:
            tid = int(entry["team_id"])
            if tid not in TEAM_ABBR_BY_ID:
                warnings.warn(f"override for {pid} has unknown team_id {tid}; skipped")
                continue
            out[pid] = tid
            continue

        raw = entry.get("team")
        if raw is None:
            out[pid] = None          # explicit "no team"
            continue
        tid = team_id_for_abbr(raw)
        if tid is None:
            warnings.warn(f"override for {pid} has unresolvable team {raw!r}; skipped")
            continue
        out[pid] = tid
    return out


def assign_target_teams(
    history: pd.DataFrame,
    target_year: int,
    *,
    id_col: str = "PlayerId",
    volume_col: str = "PA",
    min_volume: float = 25.0,
    overrides: Mapping[int, int | None] | None = None,
) -> pd.DataFrame:
    """Assign every player in `history` to a target-year team.

    ONE rule, used for hitters (volume_col="PA") and pitchers
    (volume_col="TBF"). Previously the two sides disagreed: hitters took the
    team with the most PA in the latest season, pitchers took a
    `groupby().last()` on a season-sorted frame — an unstable tie-break that
    could pick either club for a mid-season trade depending on row order.

    The rule:
      1. An override wins outright (including an explicit "no team").
      2. Otherwise use the player's most recent season with `volume_col >=
         min_volume`, and within it the team he accumulated the most volume
         for. Ties break on the LOWER team id, so the result is deterministic
         regardless of input ordering.
      3. No qualifying season -> team_id None, source "unknown".

    Returns one row per player: [id_col, team_id, team_abbr, assign_source].
    """
    overrides = dict(overrides or {})
    required = {id_col, "Season", "TeamId", volume_col}
    missing = required - set(history.columns)
    if missing:
        raise KeyError(f"history frame missing columns: {sorted(missing)}")

    qualifying = history[
        (history["Season"] < target_year)
        & history["TeamId"].notna()
        & (history[volume_col].fillna(0) >= min_volume)
    ]

    inferred: dict[int, int] = {}
    if not qualifying.empty:
        latest = qualifying.groupby(id_col)["Season"].transform("max")
        final_season = qualifying[qualifying["Season"] == latest]
        volume = (final_season.groupby([id_col, "TeamId"])[volume_col]
                  .sum().reset_index())
        # Deterministic: most volume first, then lowest team id.
        volume = volume.sort_values(
            [id_col, volume_col, "TeamId"], ascending=[True, False, True],
        )
        picked = volume.groupby(id_col).first().reset_index()
        inferred = {int(r[id_col]): int(r["TeamId"]) for _, r in picked.iterrows()}

    rows = []
    for pid in pd.unique(history[id_col].dropna()):
        pid = int(pid)
        if pid in overrides:
            team_id, source = overrides[pid], "override"
        elif pid in inferred:
            team_id, source = inferred[pid], "history"
        else:
            team_id, source = None, "unknown"
        rows.append({
            id_col: pid,
            "team_id": team_id,
            "team_abbr": abbr_for_team_id(team_id),
            "assign_source": source,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Playing-time weights — the injection point for the future PA/IP model
# ─────────────────────────────────────────────────────────────────────────────

# A weight source maps a projection frame to per-row playing-time weights,
# used to aggregate individual talent into a team total. Anything satisfying
# this signature can be passed to `bottom_up_team_factors`, so the real
# playing-time model replaces the proxy without changing team logic.
PlayingTimeWeights = Callable[[pd.DataFrame], np.ndarray]


def career_pa_weights(df: pd.DataFrame, *, column: str = "Career_PA",
                      floor: float = 1.0) -> np.ndarray:
    """PROVISIONAL playing-time proxy: career volume.

    Known bias: skews veteran. A 33-year-old part-timer with 4000 career PA
    outweighs a 24-year-old everyday starter with 600. It is used only because
    there is no playing-time model yet, and it is at least stable — `Last_PA`
    is worse, since the committed artifacts were built from a partial season.

    Replace with the real projected-PA model as soon as it exists; that is the
    single highest-value upgrade to team-level accuracy.
    """
    if column not in df.columns:
        return np.ones(len(df))
    w = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(float)
    return np.maximum(w, floor)


def depth_weights(df: pd.DataFrame, *, column: str = "Career_PA",
                  top_n: int = 9) -> np.ndarray:
    """Weights that keep only a team's top `top_n` players by `column`.

    Approximates "who is actually in the lineup" for a team aggregate, so deep
    benches don't dilute a team's projected offense. Applied per-frame, so pass
    a single team's rows. Still a proxy — a real depth chart needs the position
    data the pipeline does not yet fetch.
    """
    base = career_pa_weights(df, column=column)
    if len(base) <= top_n:
        return base
    cutoff = np.sort(base)[-top_n]
    return np.where(base >= cutoff, base, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Bottom-up team run environment
# ─────────────────────────────────────────────────────────────────────────────

# Weight on the roster-derived factor when blending against the historical
# team-RPG prior.
#
# The roster-derived factor is causally correct — it is built from the players
# actually projected onto the team, and it is the only one that responds to a
# trade. The historical prior captures real things the roster aggregate
# currently misses: park, bullpen/defense quality, coaching, and the playing
# time we cannot yet project.
#
# 0.60 is a deliberate starting point, not a fitted value. With the extra-base
# blend bug fixed and a real playing-time model in place, the roster side
# should carry more weight; re-tune against a walk-forward backtest rather than
# trusting this number.
TEAM_CONTEXT_BOTTOM_UP_WEIGHT = 0.60

# Linear weights for runs per PA. Imported from pitcher_outputs so offense and
# defense are always measured on ONE scale — that agreement (+0.3% between the
# hitter and pitcher sides) is the property that makes league-level closure
# possible at all.
from pitcher_outputs import (  # noqa: E402
    LINEAR_WEIGHTS_RUNS,
    RUNS_INTERCEPT_DEFAULT,
)

# League-average PA per team-game, for converting R/PA to R/G.
PA_PER_TEAM_GAME = 38.0


def runs_per_pa(df: pd.DataFrame, weights: np.ndarray | None = None,
                suffix: str = "") -> float:
    """Volume-weighted R/PA implied by a group's per-PA event distribution."""
    if len(df) == 0:
        return float("nan")
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    if w.sum() <= 0:
        w = np.ones(len(df))
    total = 0.0
    for col, lw in LINEAR_WEIGHTS_RUNS.items():
        c = f"{col}{suffix}"
        if c in df.columns:
            total += lw * np.average(
                pd.to_numeric(df[c], errors="coerce").fillna(0.0), weights=w,
            )
    return float(total + RUNS_INTERCEPT_DEFAULT)


def _normalize_to_unit_mean(factors: np.ndarray,
                            team_weights: np.ndarray) -> np.ndarray:
    """Scale factors so their VOLUME-WEIGHTED mean is exactly 1.0.

    This is the league-closure constraint, and it must be volume-weighted
    rather than a simple average across teams. Total league runs are

        sum over players of  neutral_rate x team_factor x PA

    so preserving league runs requires the PA-weighted mean factor to be 1,
    not the unweighted mean across 30 clubs. Those differ whenever teams carry
    different roster volume — using the unweighted mean leaves a systematic
    league-wide drift (measured at +3% on the current artifacts).

    It is what makes the factors safe to apply after a roster change: players
    moving between teams redistribute talent, they do not create it.
    """
    w = np.asarray(team_weights, float)
    f = np.asarray(factors, float)
    if len(f) == 0:
        return f
    if w.sum() <= 0 or not np.isfinite(w).all():
        w = np.ones_like(f)
    mean = float(np.average(f, weights=w))
    return f / mean if mean > 0 else f


def bottom_up_team_factors(
    hitters: pd.DataFrame,
    *,
    team_col: str = "team_id",
    weight_fn: PlayingTimeWeights = depth_weights,
) -> pd.DataFrame:
    """Team offensive run-environment factor, built from the roster.

    For each team: aggregate its hitters' per-PA event distributions into a
    team R/PA via linear weights, then index against the league so 1.0 is
    average. `weight_fn` is applied PER TEAM, so a depth-chart-style weighting
    selects that team's top players rather than the league's.

    Normalized so the volume-weighted mean factor is exactly 1.0 — see
    `_normalize_to_unit_mean` for why that is a requirement and not a choice.

    Two different weightings are at play, and conflating them leaves a
    residual league-wide drift:

      `team_RPA`     uses `weight_fn` — WHO DEFINES the team's offense. A
                     depth-chart weighting should consider the lineup, not the
                     30th man.
      `team_weight`  uses total roster volume — WHO RECEIVES the factor. Every
                     hitter on the roster gets it, so normalization must be
                     weighted by all of them or league runs shift.

    Returns [team_col, team_abbr, n_hitters, team_weight, team_RPA,
    bottom_up_factor].
    """
    rows = []
    for team_id, g in hitters.dropna(subset=[team_col]).groupby(team_col):
        rows.append({
            team_col: int(team_id),
            "team_abbr": abbr_for_team_id(team_id),
            "n_hitters": len(g),
            "team_weight": float(np.sum(career_pa_weights(g))),
            "team_RPA": runs_per_pa(g, weight_fn(g)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(team_weight=[], team_RPA=[], bottom_up_factor=[])

    league = float(np.average(out["team_RPA"], weights=out["team_weight"])) \
        if out["team_weight"].sum() > 0 else float(out["team_RPA"].mean())
    raw = (out["team_RPA"] / league).to_numpy(float) if league > 0 \
        else np.ones(len(out))
    out["bottom_up_factor"] = _normalize_to_unit_mean(raw, out["team_weight"])
    return out.sort_values("bottom_up_factor", ascending=False).reset_index(drop=True)


def blend_team_factors(
    bottom_up: pd.DataFrame,
    prior: Mapping[int, float] | None = None,
    *,
    team_col: str = "team_id",
    bottom_up_weight: float = TEAM_CONTEXT_BOTTOM_UP_WEIGHT,
) -> pd.DataFrame:
    """Blend the roster-derived factor with a historical team-RPG prior.

    `prior` maps team id -> historical factor (e.g. the existing
    `Pred_target_team_factor`). Teams absent from it fall back to the
    roster-derived value alone.

    The blend is renormalized to a volume-weighted mean of exactly 1.0, so
    blending can never shift league total runs regardless of the weight
    chosen — including `bottom_up_weight=0.0`, which reproduces the historical
    prior but closed.
    """
    out = bottom_up.copy()
    if out.empty:
        return out.assign(prior_factor=[], team_factor=[])

    w = float(np.clip(bottom_up_weight, 0.0, 1.0))
    out["prior_factor"] = (
        out[team_col].map(lambda t: (prior or {}).get(int(t), np.nan))
        if prior else np.nan
    )
    blended = np.where(
        out["prior_factor"].notna(),
        w * out["bottom_up_factor"] + (1.0 - w) * out["prior_factor"].fillna(1.0),
        out["bottom_up_factor"],
    )
    weights = (out["team_weight"] if "team_weight" in out.columns
               else pd.Series(np.ones(len(out))))
    out["team_factor"] = _normalize_to_unit_mean(blended, weights.to_numpy(float))
    return out


def apply_team_context(
    hitters: pd.DataFrame,
    team_factors: pd.DataFrame,
    *,
    team_col: str = "team_id",
    factor_col: str = "team_factor",
) -> pd.DataFrame:
    """Rescale each hitter's R/RBI to his assigned team's run environment.

    This is what makes a team change take effect. The per-PA pipeline already
    exposes `Pred_R_per_PA_neutral` / `Pred_RBI_per_PA_neutral` — the
    team-context-FREE skill estimates — so re-deriving the team-dependent
    values is exact rather than an approximate rescale of an already-scaled
    number:

        P_R   = Pred_R_per_PA_neutral   x team_factor
        P_RBI = Pred_RBI_per_PA_neutral x team_factor

    Adds `team_factor` and rewrites P_R / P_RBI / SD_R / SD_RBI. The SDs are
    scaled by the same factor, matching how `runs_rbi_model` derives them.

    Players with no team get factor 1.0 — neutral context, which is the honest
    answer for an unsigned player, and never silently last year's club.
    """
    out = hitters.copy()
    lookup = dict(zip(team_factors[team_col].astype(int),
                      team_factors[factor_col].astype(float)))

    out["team_factor"] = [
        lookup.get(int(t), 1.0) if pd.notna(t) else 1.0
        for t in out[team_col]
    ]

    for stat in ("R", "RBI"):
        neutral = f"Pred_{stat}_per_PA_neutral"
        if neutral not in out.columns:
            warnings.warn(
                f"{neutral} missing — cannot re-derive P_{stat} for a new team "
                "context; leaving it as projected."
            )
            continue
        base = pd.to_numeric(out[neutral], errors="coerce")
        out[f"P_{stat}"] = base * out["team_factor"]
        # SDs scale with the factor, as in runs_rbi_model.project_runs_and_rbi.
        sd = f"SD_{stat}"
        if sd in out.columns and "Pred_target_team_factor" in out.columns:
            old = pd.to_numeric(out["Pred_target_team_factor"],
                                errors="coerce").replace(0, np.nan)
            out[sd] = pd.to_numeric(out[sd], errors="coerce") / old * out["team_factor"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def team_context_report(
    factors: pd.DataFrame, *, team_col: str = "team_id",
) -> str:
    """Human-readable summary of how much the roster disagrees with the prior."""
    if factors.empty:
        return "no teams to report"
    lines = [
        f"{'team':<6}{'roster':>9}{'prior':>9}{'blended':>9}{'gap':>8}{'n':>5}",
    ]
    f = factors.copy()
    f["gap"] = f.get("prior_factor", np.nan) - f["bottom_up_factor"]
    for _, r in f.iterrows():
        prior = r.get("prior_factor", np.nan)
        lines.append(
            f"{str(r.get('team_abbr') or r[team_col]):<6}"
            f"{r['bottom_up_factor']:>9.3f}"
            f"{(f'{prior:.3f}' if pd.notna(prior) else '-'):>9}"
            f"{r.get('team_factor', np.nan):>9.3f}"
            f"{(f'{r.gap:+.3f}' if pd.notna(r['gap']) else '-'):>8}"
            f"{int(r['n_hitters']):>5}"
        )
    return "\n".join(lines)


def describe_moves(
    assignments: pd.DataFrame, *, id_col: str = "PlayerId",
    names: Mapping[int, str] | None = None,
) -> str:
    """List the players whose team came from an override rather than history."""
    moved = assignments[assignments["assign_source"] == "override"]
    if moved.empty:
        return "no team overrides applied"
    lines = [f"{len(moved)} team override(s) applied:"]
    for _, r in moved.iterrows():
        pid = int(r[id_col])
        who = (names or {}).get(pid, str(pid))
        lines.append(f"  {who:<24} -> {r['team_abbr'] or '(no team)'}")
    return "\n".join(lines)

"""
fielding_model.py
=================
Season fielding projections: putouts, assists, errors, double plays, chances,
passed balls, catcher's interference, and catcher caught-stealing.

The governing fact about fielding counting stats is that they are **mostly
position and exposure, with a small skill term on top**. A shortstop and a
first baseman do not have different assist *skill* so much as different assist
*jobs*: a first baseman records a putout on nearly every infield groundout and
almost never an assist, and no amount of talent changes that. So the model is

    stat = (innings at position / 9) x rate_per_9(position) x skill x team

where

    rate_per_9  a position baseline — the "assumption" layer
    skill       the player's own history at that position, shrunk hard
    team        an adjustment for how many balls the staff lets reach fielders

Three structural properties are enforced rather than hoped for:

1. **Putouts close to 27 per 9 team innings.** Every out is a putout credited
   to exactly one fielder, so the position baselines must sum to 27 across a
   full defensive alignment. `POSITION_BASELINES` is normalized to satisfy
   this, and it is tested.
2. **Chances = PO + A + E**, definitionally. Chances is derived, never
   projected independently, so the three can't drift apart.
3. **Catcher putouts are driven by strikeouts.** A catcher is credited with a
   putout on every strikeout, so his putout rate is a near-direct function of
   the pitching staff's K rate — roughly 8.6 of a catcher's ~9.0 putouts per 9
   innings are strikeouts. This is why `team_strikeout_adjustment` exists and
   why the catcher moves in the OPPOSITE direction to everyone else: a
   high-strikeout staff gives its catcher more putouts and every other fielder
   fewer chances.

Calibration status
------------------
The baselines are documented starting points assembled from the structure of
the game (27 outs, ~9.9 assists per 9, league fielding ~.984), not fitted
values. `fit_position_baselines` replaces them from real fielding history the
moment `data_acquisition.fetch_fielding_data` has run with network access.
Treat the cross-position ORDERING as reliable and the absolute levels as
provisional.

What this needs that does not exist yet
---------------------------------------
**Innings at position.** The exposure term. It requires the playing-time model
and a position assignment (the role workbooks carry a `Pos` column for exactly
this). Until both land, `project_fielding` takes innings as an argument and
callers supply an estimate; nothing here invents one.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Position baselines — per 9 defensive innings at the position
# ─────────────────────────────────────────────────────────────────────────────
# PROVISIONAL. Assembled from the arithmetic of the game rather than fitted:
#
#   * The nine PO values sum to 27, because every out is exactly one putout.
#   * Catcher PO is dominated by strikeouts (~8.6/9 at modern K rates).
#   * First-base PO is dominated by receiving infield throws: of the ~11
#     infield outs per game, the first baseman takes most.
#   * Assists concentrate at SS/3B/2B and are almost nil in the outfield,
#     where an "assist" means a throw that retires a runner.
#   * Outfield PO is flyball catches, so centre field leads.
#
# `fit_position_baselines` should replace all of it once real history is
# available. The ordering across positions is the trustworthy part.
POSITION_BASELINES: dict[str, dict[str, float]] = {
    # The E column is set so that the IMPLIED fielding percentage,
    # 1 - E/(PO+A+E), lands on the real per-position figure — see
    # EXPECTED_FIELDING_PCT and the test that checks it. Deriving the error
    # prior from this table (baseline_e_per_chance) means a wrong E here shows
    # up as a wrong fielding percentage rather than hiding.
    #        PO      A      E     DP    PB     CI
    "P":  dict(po=0.55, a=1.55, e=0.078, dp=0.09, pb=0.0,   ci=0.0),
    "C":  dict(po=8.60, a=0.45, e=0.045, dp=0.06, pb=0.060, ci=0.0022),
    "1B": dict(po=8.00, a=0.65, e=0.045, dp=0.66, pb=0.0,   ci=0.0),
    "2B": dict(po=1.85, a=2.55, e=0.070, dp=0.60, pb=0.0,   ci=0.0),
    "3B": dict(po=0.72, a=1.70, e=0.093, dp=0.25, pb=0.0,   ci=0.0),
    "SS": dict(po=1.45, a=2.85, e=0.110, dp=0.62, pb=0.0,   ci=0.0),
    "LF": dict(po=1.75, a=0.06, e=0.027, dp=0.01, pb=0.0,   ci=0.0),
    "CF": dict(po=2.30, a=0.05, e=0.030, dp=0.01, pb=0.0,   ci=0.0),
    "RF": dict(po=1.80, a=0.07, e=0.029, dp=0.01, pb=0.0,   ci=0.0),
}

# Positions that make up one defensive alignment. DH is excluded — a
# designated hitter records no defensive innings, and including him would
# break the 27-putout identity.
ALIGNMENT = ("P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")
OUTS_PER_9 = 27.0

# Real per-position fielding percentages, as a guard on the E column above.
# These are the published figures, and the baseline table is set to reproduce
# them. Third base and pitcher are the two that are easy to get wrong: a third
# baseman handles hard-hit balls with no time to set, and a pitcher fields
# comebackers and bunts off balance, so both err far more often per chance than
# their infield neighbours.
EXPECTED_FIELDING_PCT = {
    "P": 0.962, "C": 0.995, "1B": 0.995, "2B": 0.984, "3B": 0.962,
    "SS": 0.975, "LF": 0.985, "CF": 0.988, "RF": 0.986,
}

# Outfield assists are worth separating from total assists: they are a
# distinct, commonly-tracked category and are almost entirely "threw out a
# runner trying to advance", not part of a routine play.
OUTFIELD = ("LF", "CF", "RF")

# Shrinkage strength, in defensive innings. A player's own rate at a position
# only outweighs the baseline once he has this much exposure there.
#
# Deliberately different per statistic, because the amount of real per-player
# signal differs enormously:
#   PO/A    almost pure position and opportunity. Shrink very hard — a
#           shortstop's assist total says more about his team's groundball
#           rate than about him.
#   E       the genuine skill term (this is what fielding percentage measures),
#           but still noisy season to season. Moderate.
#   DP      needs a partner and a runner on first; mostly context.
#   PB/CI   rare enough to be almost all noise. Shrink extremely hard.
#   CS      a real, reasonably stable catcher skill — the lightest shrinkage
#           here, and the reason catcher CS is worth projecting individually.
SHRINK_INNINGS = {
    "po": 900.0,
    "a":  900.0,
    "e":  450.0,
    "dp": 700.0,
    "pb": 1200.0,
    "ci": 2000.0,
    "cs": 250.0,
}

# League error rate PER CHANCE, i.e. 1 - fielding percentage. Errors are
# modelled in this space rather than per inning because a chance is the actual
# opportunity for an error, and it is what fielding percentage measures.
#
# Per-inning shrinkage gets this wrong in a way that matters: three errors in
# thirty innings is an absurd 0.9 per 9, and shrinking that in rate space still
# moved a projection 45% off baseline on what is almost pure noise. Shrinking
# (errors / chances) against a league prior with a strength measured in CHANCES
# handles a thin sample correctly, because a thin sample has few chances.
# The prior is derived PER POSITION from the baseline table rather than being
# a single league constant. A flat 0.016 (league fielding ~.984) is the
# all-positions average, dominated by first basemen and outfielders who rarely
# err; a shortstop's rate per chance is nearer 0.025. Using the flat value
# alongside a position-specific error baseline made the two constants
# contradict each other, and a shortstop with a terrible error history came out
# BELOW his own position's baseline.
SHRINK_CHANCES_E = 700.0


def baseline_e_per_chance(pos: str) -> float:
    """Position-specific error rate per chance, derived from the baselines.

    Self-consistent by construction: a player with no history comes out at
    exactly his position's baseline error rate, because the prior he is shrunk
    to is that rate expressed per chance.
    """
    base = BASELINES.get(str(pos).upper())
    if base is None:
        return 0.016
    chances = base["po"] + base["a"] + base["e"]
    return base["e"] / chances if chances > 0 else 0.016

# League caught-stealing rate against an average catcher. The opportunity side
# (attempts) comes from `sb_model`, which already projects attempt rates.
LEAGUE_CS_RATE = 0.22

# League strikeouts per 9 innings, the reference point for the team adjustment.
LEAGUE_K_PER_9 = 8.60


def _normalized_baselines() -> dict[str, dict[str, float]]:
    """Baselines with putouts rescaled so an alignment sums to exactly 27.

    The identity is not optional: 27 outs per 9 innings are recorded as 27
    putouts, so if the table does not sum to 27 every team total built from it
    is wrong by that factor. Rescaling here means hand-edits to the table
    cannot silently break it.
    """
    out = {k: dict(v) for k, v in POSITION_BASELINES.items()}
    total = sum(out[p]["po"] for p in ALIGNMENT)
    if total > 0:
        for p in ALIGNMENT:
            out[p]["po"] *= OUTS_PER_9 / total
    return out


BASELINES = _normalized_baselines()


# ─────────────────────────────────────────────────────────────────────────────
# Team adjustment
# ─────────────────────────────────────────────────────────────────────────────

def team_strikeout_adjustment(k_per_9: float) -> dict[str, float]:
    """How a staff's strikeout rate scales its fielders' opportunity.

    Strikeouts and fielding chances are the same finite pool of outs viewed two
    ways. A staff that strikes out 10 per 9 leaves ~1.4 fewer balls in play per
    9 than a league-average one, so its non-catcher fielders get fewer plays —
    while its CATCHER gets more putouts, because he is credited with one on
    every strikeout.

    Returns multipliers keyed `po_fielder`, `po_catcher`, `a`, where:

        po_catcher  scales with the strikeout rate directly
        po_fielder  scales with the REMAINING outs, (27 - K)/9
        a           same as po_fielder: an assist requires a ball in play

    This is the one team-level effect large enough to matter for fielding
    counting stats, and it is why the catcher must be handled separately from
    everyone else.
    """
    k = float(np.clip(k_per_9, 2.0, 15.0))
    lg_bip_outs = OUTS_PER_9 - LEAGUE_K_PER_9
    bip_outs = OUTS_PER_9 - k
    return {
        "po_catcher": k / LEAGUE_K_PER_9,
        "po_fielder": bip_outs / lg_bip_outs,
        "a": bip_outs / lg_bip_outs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-player rates
# ─────────────────────────────────────────────────────────────────────────────

def project_fielding_rates(
    history: pd.DataFrame,
    target_year: int,
    *,
    id_col: str = "PlayerId",
    max_history_years: int = 4,
    decay: float = 0.85,
) -> pd.DataFrame:
    """Per-(player, position) fielding rates per 9 innings, shrunk to baseline.

    `history` is `data_acquisition.fetch_fielding_data` output: one row per
    (player, season, position) with `Innings` and the counting stats.

    Rates are innings-weighted with recency decay, then shrunk toward the
    position baseline with a per-statistic strength (see `SHRINK_INNINGS`).
    A player with no history at a position comes out at exactly the baseline,
    which is the right answer — a shortstop moving to second base should be
    projected as an average second baseman until he plays there.

    Returns [id_col, Pos, innings_history, rate_po, rate_a, rate_e, rate_dp,
    rate_pb, rate_ci, cs_rate].
    """
    required = {id_col, "Season", "Pos", "Innings"}
    missing = required - set(history.columns)
    if missing:
        raise KeyError(f"fielding history missing columns: {sorted(missing)}")

    prior = history[
        (history["Season"] < target_year)
        & (history["Season"] >= target_year - max_history_years)
        & history["Pos"].notna()
    ].copy()
    if prior.empty:
        return pd.DataFrame(columns=[id_col, "Pos", "innings_history"])

    prior["_w"] = (pd.to_numeric(prior["Innings"], errors="coerce").fillna(0.0)
                   * decay ** (target_year - prior["Season"].astype(int)))

    rows = []
    for (pid, pos), g in prior.groupby([id_col, "Pos"]):
        pos = str(pos).upper()
        base = BASELINES.get(pos)
        if base is None:
            continue                       # DH and anything unrecognized
        innings = float(pd.to_numeric(g["Innings"], errors="coerce")
                        .fillna(0.0).sum())
        w = float(g["_w"].sum())
        rec = {id_col: int(pid), "Pos": pos, "innings_history": innings}

        for stat in ("po", "a", "dp", "pb", "ci"):
            col = {"po": "PO", "a": "A", "dp": "DP",
                   "pb": "PB", "ci": "CI"}[stat]
            if col not in g.columns or innings <= 0:
                rec[f"rate_{stat}"] = base[stat]
                continue
            observed = (float(pd.to_numeric(g[col], errors="coerce")
                              .fillna(0.0).sum()) / innings * 9.0)
            k = SHRINK_INNINGS[stat]
            rec[f"rate_{stat}"] = (innings * observed + k * base[stat]) / (innings + k)

        # Errors live in chance space, not inning space — see
        # LEAGUE_E_PER_CHANCE. `rate_e` is still emitted per 9 innings so the
        # column set stays uniform, but it is DERIVED from the per-chance rate
        # and the projected chance volume.
        po_a = 0.0
        if {"PO", "A"} <= set(g.columns):
            po_a = float(pd.to_numeric(g["PO"], errors="coerce").fillna(0).sum()
                         + pd.to_numeric(g["A"], errors="coerce").fillna(0).sum())
        errs = (float(pd.to_numeric(g["E"], errors="coerce").fillna(0).sum())
                if "E" in g.columns else 0.0)
        chances = po_a + errs
        if chances > 0:
            observed_epc = errs / chances
            prior_epc = baseline_e_per_chance(pos)
            rec["e_per_chance"] = (
                (chances * observed_epc + SHRINK_CHANCES_E * prior_epc)
                / (chances + SHRINK_CHANCES_E))
        else:
            rec["e_per_chance"] = baseline_e_per_chance(pos)
        # Convert back to a per-9 error rate using the projected chance volume
        # implied by this player's own putout and assist rates.
        proj_chances_per_9 = rec["rate_po"] + rec["rate_a"]
        rec["rate_e"] = (proj_chances_per_9 * rec["e_per_chance"]
                         / max(1.0 - rec["e_per_chance"], 1e-6))

        # Caught stealing is a rate per ATTEMPT, not per inning — the
        # opportunity is the runner's decision, not the catcher's exposure.
        if {"CS", "SB_allowed"} <= set(g.columns):
            cs = float(pd.to_numeric(g["CS"], errors="coerce").fillna(0).sum())
            sb = float(pd.to_numeric(g["SB_allowed"], errors="coerce")
                       .fillna(0).sum())
            attempts = cs + sb
            k = SHRINK_INNINGS["cs"] / 9.0     # innings -> rough attempt scale
            rec["cs_rate"] = ((attempts * (cs / attempts) + k * LEAGUE_CS_RATE)
                              / (attempts + k)) if attempts > 0 else LEAGUE_CS_RATE
            rec["cs_attempts_history"] = attempts
        else:
            rec["cs_rate"] = LEAGUE_CS_RATE
        rows.append(rec)
    return pd.DataFrame(rows)


def baseline_rates(pos: str) -> dict[str, float]:
    """Baseline rates for a position, for a player with no history there."""
    base = BASELINES.get(str(pos).upper())
    if base is None:
        return {}
    out = {f"rate_{k}": v for k, v in base.items()}
    out["cs_rate"] = LEAGUE_CS_RATE
    out["e_per_chance"] = baseline_e_per_chance(pos)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Season totals
# ─────────────────────────────────────────────────────────────────────────────

def project_fielding(
    rates: pd.DataFrame,
    innings_at_position: Mapping[tuple[int, str], float] | pd.DataFrame,
    *,
    id_col: str = "PlayerId",
    team_k_per_9: Mapping[int, float] | None = None,
    team_col: str = "team_id",
    sb_attempts_against: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Season fielding counting stats.

    `innings_at_position` is the exposure — either a {(player_id, pos):
    innings} mapping or a frame with [id_col, Pos, Innings]. It is an ARGUMENT
    rather than something computed here because it needs the playing-time model
    and a position assignment, neither of which exists yet; inventing one
    inside this module would bury the assumption.

    `team_k_per_9` optionally supplies each team's projected strikeout rate so
    fielder opportunity can be scaled (see `team_strikeout_adjustment`). Absent
    it, every team is treated as league-average.

    Chances is DERIVED as PO + A + E, never projected, so the identity holds by
    construction. Outfield assists are broken out separately since they are a
    distinct tracked category.

    Returns one row per (player, position) with PO, A, E, DP, Chances, PB, CI,
    OF_A, and (for catchers) CS.
    """
    if isinstance(innings_at_position, pd.DataFrame):
        exposure = {
            (int(r[id_col]), str(r["Pos"]).upper()):
                float(r.get("Innings", 0.0) or 0.0)
            for _, r in innings_at_position.iterrows()
        }
    else:
        exposure = {(int(k[0]), str(k[1]).upper()): float(v)
                    for k, v in innings_at_position.items()}

    out_rows = []
    for _, r in rates.iterrows():
        pid, pos = int(r[id_col]), str(r["Pos"]).upper()
        innings = exposure.get((pid, pos))
        if innings is None or innings <= 0:
            continue
        games9 = innings / 9.0

        adj = {"po_catcher": 1.0, "po_fielder": 1.0, "a": 1.0}
        if team_k_per_9 is not None and team_col in r.index:
            tid = r.get(team_col)
            if pd.notna(tid) and int(tid) in team_k_per_9:
                adj = team_strikeout_adjustment(team_k_per_9[int(tid)])
        po_adj = adj["po_catcher"] if pos == "C" else adj["po_fielder"]

        po = games9 * float(r.get("rate_po", 0.0)) * po_adj
        a = games9 * float(r.get("rate_a", 0.0)) * adj["a"]
        e = games9 * float(r.get("rate_e", 0.0)) * adj["a"]
        dp = games9 * float(r.get("rate_dp", 0.0)) * adj["a"]

        rec = {
            id_col: pid, "Pos": pos, "Innings": innings,
            "PO": po, "A": a, "E": e, "DP": dp,
            "Chances": po + a + e,              # definitional
            "PB": games9 * float(r.get("rate_pb", 0.0)),
            "CI": games9 * float(r.get("rate_ci", 0.0)),
            "OF_A": a if pos in OUTFIELD else 0.0,
        }
        # Carry the team through so team_putout_check can verify the 27-putout
        # identity; without it that diagnostic silently finds nothing to check.
        if team_col in r.index and pd.notna(r.get(team_col)):
            rec[team_col] = int(r[team_col])
        if pos == "C":
            attempts = None
            if sb_attempts_against is not None and team_col in r.index:
                tid = r.get(team_col)
                if pd.notna(tid):
                    team_att = sb_attempts_against.get(int(tid))
                    if team_att is not None:
                        # Share of the team's attempts proportional to innings
                        # caught, over a full 1458-inning team season.
                        attempts = team_att * innings / 1458.0
            rec["CS_attempts"] = attempts if attempts is not None else np.nan
            rec["CS"] = (attempts * float(r.get("cs_rate", LEAGUE_CS_RATE))
                         if attempts is not None else np.nan)
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def team_putout_check(projected: pd.DataFrame, *,
                      team_col: str = "team_id") -> pd.DataFrame:
    """Verify putouts close to 27 per 9 innings for each team.

    The diagnostic that catches an exposure error. If a team's projected
    putouts per 9 defensive innings are not ~27, either the innings-at-position
    allocation is wrong or the baselines have drifted — and both are silent
    failures otherwise, because each individual player's total looks plausible.
    """
    if projected.empty or team_col not in projected.columns:
        return pd.DataFrame()
    g = projected.groupby(team_col).agg(
        PO=("PO", "sum"), A=("A", "sum"), E=("E", "sum"),
        position_innings=("Innings", "sum"))
    # Summing innings ACROSS POSITIONS counts each team inning nine times, once
    # per fielder on the diamond. A team plays ~1458 defensive innings but its
    # position-innings total is ~13122, so dividing by 9 is what converts one
    # to the other. Getting this wrong makes the identity read 3.0 instead of
    # 27.0 — which looks like a broken model rather than a unit error.
    g["team_innings"] = g["position_innings"] / len(ALIGNMENT)
    g["po_per_9"] = g["PO"] / (g["team_innings"] / 9.0).replace(0, np.nan)
    g["po_per_9_error"] = g["po_per_9"] - OUTS_PER_9
    return g.reset_index()


def fit_position_baselines(history: pd.DataFrame, *,
                           min_innings: float = 200.0) -> dict[str, dict]:
    """Fit the position baselines from real fielding history.

    Replaces `POSITION_BASELINES`, which is assembled from the arithmetic of
    the game rather than measured. Pools every player-season with at least
    `min_innings` at the position and takes innings-weighted rates per 9.

    Run this as soon as `fetch_fielding_data` has populated the cache from a
    machine with network access.
    """
    cols = {"po": "PO", "a": "A", "e": "E", "dp": "DP", "pb": "PB", "ci": "CI"}
    pool = history[pd.to_numeric(history["Innings"], errors="coerce")
                   .fillna(0) >= min_innings]
    fitted: dict[str, dict] = {}
    for pos, g in pool.groupby("Pos"):
        pos = str(pos).upper()
        if pos not in BASELINES:
            continue
        innings = float(pd.to_numeric(g["Innings"], errors="coerce").sum())
        if innings <= 0:
            continue
        fitted[pos] = {
            stat: float(pd.to_numeric(g[col], errors="coerce").fillna(0).sum())
            / innings * 9.0
            for stat, col in cols.items() if col in g.columns
        }
    return fitted


def fielding_report(projected: pd.DataFrame, *, top: int = 10) -> str:
    """Readable summary of a fielding projection."""
    if projected.empty:
        return "no fielding projections"
    lines = [f"{'pos':<5}{'n':>5}{'PO':>9}{'A':>9}{'E':>7}{'DP':>8}{'Chances':>10}"]
    for pos, g in projected.groupby("Pos"):
        lines.append(f"{pos:<5}{len(g):>5}{g['PO'].sum():>9.0f}"
                     f"{g['A'].sum():>9.0f}{g['E'].sum():>7.0f}"
                     f"{g['DP'].sum():>8.0f}{g['Chances'].sum():>10.0f}")
    if "Innings" in projected.columns:
        # Same unit care as team_putout_check: position-innings / 9 = real
        # defensive innings.
        team_innings = projected["Innings"].sum() / len(ALIGNMENT)
        if team_innings > 0:
            lines.append(f"  PO per 9 defensive innings: "
                         f"{projected['PO'].sum() / (team_innings / 9.0):.2f} "
                         f"(must be ~{OUTS_PER_9:.0f})")
    return "\n".join(lines)

"""
league_reconcile.py
===================
Make the offense and the defense describe the same league.

Every plate appearance has exactly one batter and exactly one pitcher, so the
hitter frame and the pitcher frame are two views of ONE event stream. Summed
over the league with the right volumes they must agree, event by event:

    sum_h  e_h[i,j] * PA_i   ==   sum_p  e_p[i,j] * TBF_i        for every j

Nothing in the pipeline made that true. The two sides are projected from
separate panels, shrunk toward separately-computed league means, and given
playing time out of two separate budgets, so they land wherever they land.
Measured on the first clean refresh (run 36895277181):

    event        hitters/PA   pitchers/PA   ratio
    P_K             0.22017       0.22876   0.962
    P_BB            0.08609       0.08293   1.038
    P_HBP           0.01140       0.01089   1.047
    P_3B            0.00364       0.00331   1.099
    P_1B            0.14176       0.13879   1.021
    P_HR            0.03004       0.02952   1.018

The pitchers struck out 3.9% more batters than the hitters struck out, and
let 3.8% fewer of them walk. Run it through the linear weights and the league
scores 21,661 runs while allowing 20,761 — 900 runs, +4.3%, which is roughly
one extra team's offense appearing from nowhere.

This module closes that by moving BOTH sides to a common target:

    target[j] = w * hitters[j] + (1 - w) * pitchers[j]     (then normalized)

and rescaling each side's per-PA probabilities by `target[j] / side[j]`.
Because each player's nine events must still sum to 1, the rows are
renormalized after scaling, which pulls the league aggregate slightly back
off target — so it iterates, the same iterative proportional fit the
playing-time model uses to close team PA.

What it does NOT do is decide which side was right. It has no way to know,
and `LEAGUE_HITTER_WEIGHT` says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_config import (
    LEAGUE_HITTER_WEIGHT, LEAGUE_RECONCILE_PASSES, PROB_EVENTS,
)

# Column families that carry a full set of per-PA probabilities. The
# correction is a statement about the LEAGUE's event levels, so it applies to
# every family alike — rescaling the neutral vector and leaving the
# park-adjusted and platoon-split vectors alone would just make the frame
# disagree with itself.
SUFFIXES = ("", "_park", "_vL", "_vR")


def _volume(df: pd.DataFrame, col: str) -> np.ndarray:
    # `df.get(col)` on an absent column returns None, and pd.to_numeric then
    # hands back a bare scalar rather than a Series. Check membership instead.
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    return _drop_floor(df, v)


def _drop_floor(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Zero the floor tier's weight in a league aggregate.

    Depth players are carried at 1 PA / 1 IP so they are present, ranked and
    joinable. That is not a projection that they will play, so it must not
    be a vote on what the league looks like — and at 7.1% of the pitcher
    weight it is more than enough to matter. They are still CORRECTED by the
    reconciliation; they just do not help decide what to correct toward.
    """
    if "pt_tier" not in df.columns:
        return weights
    floor = df["pt_tier"].astype(str).to_numpy() == "floor"
    out = np.where(floor, 0.0, weights)
    return out if out.sum() > 0 else weights


def batters_faced(pitchers: pd.DataFrame) -> np.ndarray:
    """Projected batters faced — the denominator of a pitcher's per-PA rates.

    Innings are NOT that denominator and are not proportional to it: a
    high-strikeout, high-walk pitcher faces more batters per inning than a
    contact pitcher who works around them. `TBF_per_IP` is the pipeline's own
    conversion and already carries the 0.971 calibration for the extra outs
    that double plays and caught stealings supply without a plate appearance.
    """
    ip = _volume(pitchers, "Proj_IP")
    if "TBF_per_IP" in pitchers.columns:
        rate = pd.to_numeric(pitchers["TBF_per_IP"], errors="coerce")
    else:
        rate = pd.Series(np.nan, index=pitchers.index)
    if rate.isna().all():
        # Fall back to the naive identity if the pitcher summary step has not
        # run yet: each out finishes a third of an inning, so a pitcher faces
        # 3 / (his out rate) batters per inning.
        outs = sum(pd.to_numeric(pitchers[c], errors="coerce").fillna(0.0)
                   for c in ("P_K", "P_BIPOut", "P_SF") if c in pitchers.columns)
        if np.isscalar(outs):
            return np.zeros(len(pitchers), dtype=float)
        rate = 3.0 / pd.Series(outs).replace(0.0, np.nan)
    return ip * np.nan_to_num(rate.to_numpy(float), nan=0.0).clip(min=0.0)


def league_vector(df: pd.DataFrame, weights: np.ndarray,
                  suffix: str = "") -> dict[str, float]:
    """The volume-weighted per-PA event distribution of a whole side."""
    cols = [f"{e}{suffix}" for e in PROB_EVENTS]
    if any(c not in df.columns for c in cols):
        return {}
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(df))
    out = {}
    for e, c in zip(PROB_EVENTS, cols):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(v) & (w > 0)
        out[e] = float(np.average(v[ok], weights=w[ok])) if ok.any() else 0.0
    return out


def _rescale(df: pd.DataFrame, factors: dict[str, float]) -> pd.DataFrame:
    """Scale every probability family by `factors`, then renormalize rows."""
    out = df.copy()
    for suffix in SUFFIXES:
        cols = [f"{e}{suffix}" for e in PROB_EVENTS]
        if any(c not in out.columns for c in cols):
            continue
        block = out[cols].apply(pd.to_numeric, errors="coerce")
        scaled = block.multiply([factors[e] for e in PROB_EVENTS], axis=1)
        total = scaled.sum(axis=1)
        # A row that is all-NaN or sums to zero carries no distribution to
        # correct; leave it exactly as it was rather than inventing one.
        ok = total.to_numpy() > 0
        normalized = scaled.div(total, axis=0)
        for c in cols:
            out.loc[ok, c] = normalized.loc[ok, c]
    return out


def runs_per_pa(vector: dict[str, float]) -> float:
    """League runs per plate appearance implied by an event distribution."""
    from pitcher_outputs import LINEAR_WEIGHTS_RUNS as LW
    from pitcher_outputs import RUNS_INTERCEPT_DEFAULT as ICPT
    return sum(lw * vector.get(e, 0.0) for e, lw in LW.items()) + ICPT


def _scale_runs_to_events(hitters: pd.DataFrame, pa: np.ndarray,
                          vector: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """Make the R/RBI model agree with the league it is scoring runs in.

    `P_R` is not derived from the event vector. It comes from its own model,
    which projects a batter's runs from his own history and his team's run
    environment, and nothing tied its league total to the events. So even
    with the nine events reconciled, the league could score one number of
    runs by its batting lines and a different one by its R column.

    Every run is scored by exactly one batter, so those totals are the same
    number. This scales `P_R` to the total the reconciled events imply, and
    `P_RBI` by the same factor so the RBI-per-run ratio set in
    `season_engine.LEAGUE_RBI_PER_RUN` survives untouched.

    It is a LEAGUE total correction: it moves everybody by one factor and
    changes no player's share of the league, which is what the R/RBI model
    is actually for. What it does not do is make R an allocation of each
    team's projected runs — that is still a free-standing rate per player,
    and still the right next thing to build.
    """
    if "P_R" not in hitters.columns or pa.sum() <= 0:
        return hitters, {"applied": False}
    r = pd.to_numeric(hitters["P_R"], errors="coerce").to_numpy(float)
    ok = np.isfinite(r) & (pa > 0)
    if not ok.any():
        return hitters, {"applied": False}
    current = float(np.average(r[ok], weights=pa[ok]))
    target = runs_per_pa(vector)
    if current <= 0 or target <= 0:
        return hitters, {"applied": False}
    factor = target / current
    out = hitters.copy()
    # The NEUTRAL rates have to move too, not just P_R: the season layer
    # rebuilds P_R as `Pred_R_per_PA_neutral * team_factor`, so scaling only
    # the output column would be silently undone the next time it runs. The
    # SDs scale with them to hold each player's coefficient of variation.
    for col in ("P_R", "P_RBI",
                "Pred_R_per_PA_neutral", "Pred_RBI_per_PA_neutral",
                "SD_R", "SD_RBI"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * factor
    return out, {
        "applied": True,
        "factor": factor,
        "before_R_per_PA": current,
        "after_R_per_PA": target,
        "league_runs": target * float(pa.sum()),
    }


def _weights(hitter_weight) -> dict[str, float]:
    """Accept either one weight for every event or a per-event mapping."""
    if isinstance(hitter_weight, dict):
        return {e: float(np.clip(hitter_weight.get(e, 0.5), 0.0, 1.0))
                for e in PROB_EVENTS}
    w = float(np.clip(hitter_weight, 0.0, 1.0))
    return {e: w for e in PROB_EVENTS}


def reconcile_league(hitters: pd.DataFrame, pitchers: pd.DataFrame, *,
                     hitter_weight=LEAGUE_HITTER_WEIGHT,
                     passes: int = LEAGUE_RECONCILE_PASSES,
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Move both sides onto one league event distribution.

    `hitter_weight` is a per-event mapping (or one float for all nine). It
    decides where the agreed value sits between the two sides, event by
    event, because they are not equally good at the same things — see
    LEAGUE_HITTER_WEIGHT.

    Returns `(hitters, pitchers, report)`. The report carries the before and
    after league vectors and the per-event factors applied to each side, so
    the size of the correction is visible rather than silent — a correction
    quietly absorbing more than a percent or two is evidence of an upstream
    problem, not a thing to be glad about.
    """
    w = _weights(hitter_weight)
    pa = _volume(hitters, "Proj_PA")
    tbf = batters_faced(pitchers)
    if pa.sum() <= 0 or tbf.sum() <= 0:
        return hitters, pitchers, {"applied": False,
                                   "reason": "no projected volume on one side"}

    before_h = league_vector(hitters, pa)
    before_p = league_vector(pitchers, tbf)
    if not before_h or not before_p:
        return hitters, pitchers, {"applied": False,
                                   "reason": "missing probability columns"}

    target = {e: w[e] * before_h[e] + (1.0 - w[e]) * before_p[e]
              for e in PROB_EVENTS}
    total = sum(target.values())
    if total <= 0:
        return hitters, pitchers, {"applied": False, "reason": "empty target"}
    target = {e: v / total for e, v in target.items()}

    h_out, p_out = hitters, pitchers
    for _ in range(max(1, int(passes))):
        cur_h = league_vector(h_out, pa)
        cur_p = league_vector(p_out, tbf)
        h_out = _rescale(h_out, {e: (target[e] / cur_h[e] if cur_h[e] > 0 else 1.0)
                                 for e in PROB_EVENTS})
        p_out = _rescale(p_out, {e: (target[e] / cur_p[e] if cur_p[e] > 0 else 1.0)
                                 for e in PROB_EVENTS})

    after_h = league_vector(h_out, pa)
    after_p = league_vector(p_out, tbf)
    h_out, runs = _scale_runs_to_events(h_out, pa, after_h)
    return h_out, p_out, {
        "runs": runs,
        "applied": True,
        "hitter_weight": w,
        "hitter_weight_scalar": float(np.mean(list(w.values()))),
        "passes": int(passes),
        "total_PA": float(pa.sum()),
        "total_TBF": float(tbf.sum()),
        "before_hitters": before_h,
        "before_pitchers": before_p,
        "target": target,
        "after_hitters": after_h,
        "after_pitchers": after_p,
        "hitter_factors": {e: (after_h[e] / before_h[e] if before_h[e] else 1.0)
                           for e in PROB_EVENTS},
        "pitcher_factors": {e: (after_p[e] / before_p[e] if before_p[e] else 1.0)
                            for e in PROB_EVENTS},
    }


def reconcile_report(report: dict) -> str:
    """Human-readable summary for the pipeline log."""
    if not report.get("applied"):
        return f"  league reconciliation SKIPPED — {report.get('reason', 'unknown')}"
    bh, bp = report["before_hitters"], report["before_pitchers"]
    ah, ap = report["after_hitters"], report["after_pitchers"]
    lines = [
        f"  hitter weight per event: "
        + ", ".join(f"{e.replace('P_', '')} {v:.2f}"
                    for e, v in report["hitter_weight"].items())
        + f"  ({report['passes']} passes)",
        f"  {report['total_PA']:,.0f} PA vs {report['total_TBF']:,.0f} batters "
        f"faced ({report['total_TBF'] / report['total_PA'] - 1:+.2%})",
        "",
        f"  {'event':<10}{'hit before':>12}{'pit before':>12}{'gap':>8}"
        f"{'  ':>4}{'both after':>12}{'gap':>8}",
    ]
    for e in PROB_EVENTS:
        gap0 = bh[e] / bp[e] - 1 if bp[e] else 0.0
        gap1 = ah[e] / ap[e] - 1 if ap[e] else 0.0
        lines.append(f"  {e:<10}{bh[e]:>12.5f}{bp[e]:>12.5f}{gap0:>+8.1%}"
                     f"{'  ':>4}{ah[e]:>12.5f}{gap1:>+8.2%}")
    worst = max(abs(report["hitter_factors"][e] - 1) for e in PROB_EVENTS)
    lines.append("")
    lines.append(f"  largest correction applied to either side: {worst:.2%}")
    runs = report.get("runs", {})
    if runs.get("applied"):
        lines.append(
            f"  R/RBI scaled by {runs['factor']:.4f} to the runs the "
            f"reconciled events imply: R/PA {runs['before_R_per_PA']:.5f} "
            f"-> {runs['after_R_per_PA']:.5f} "
            f"({runs['league_runs']:,.0f} league runs)")
    return "\n".join(lines)

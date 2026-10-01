"""
fit_role_anchors.py
===================
Fit the per-role playing-time anchors in `role_taxonomy` against real
playing time, instead of guessing them.

The anchors were judgement calls — "an ace throws 195 innings" — and the
playing-time report has been warning that they are mis-sized ever since the
model existed. This replaces the judgement with a measurement.

Ground truth is `out/fielding_history_<year>.csv`, which carries five
seasons of real (player, season, position) rows with games and innings:

  * pitchers       innings, directly.
  * position players  innings / 9 at each fielding position, plus games at
                   DH. That proxy is worth stating because it is load
                   bearing: summed over a club it comes to 1,459 a season
                   against the 1,458 that 162 games of nine lineup slots
                   must produce, so it is measuring what it claims to.

The fit works on the WITHIN-CLUB RANK curve rather than on role labels,
because the historical rows carry no roles. For each club-season, players
are ranked by playing time and averaged at each rank, which gives the real
shape of a roster. A role's target is then the mean of that curve over the
ranks its players actually occupy in the current projections, and the
anchor moves toward it. Changing an anchor changes who lands at which rank,
so it iterates.

Run from the repo root:

    python scripts/fit_role_anchors.py                  # report only
    python scripts/fit_role_anchors.py --kind pitcher   # one side

It prints the fitted anchors and the before/after fit. It does NOT edit
`role_taxonomy.py` — the numbers go in by hand, with this script named
beside them, so a change to the anchors is always a reviewed change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import role_taxonomy as RT  # noqa: E402
from playing_time_model import (  # noqa: E402
    TEAM_IP_BUDGET, TEAM_PA_BUDGET, project_playing_time,
)

MIN_SEASON = 2024          # the regime the projections are for
MAX_STEP = 0.25            # damp each pass; the rank curve shifts underfoot


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────

def real_rank_curve(fielding: pd.DataFrame, kind: str,
                    min_season: int = MIN_SEASON) -> tuple[pd.Series, float]:
    """Mean real playing time at each within-club rank, and the club total."""
    f = fielding[fielding["Season"] >= min_season]
    if kind == "pitcher":
        f = f[f["Pos"] == "P"]
        vol = f["Innings"]
    else:
        f = f[f["Pos"] != "P"]
        # Innings at a fielding position convert at nine to the game; a DH
        # appearance is a game outright.
        vol = np.where(f["Pos"] == "DH", f["G"], f["Innings"] / 9.0)
    g = (f.assign(_v=vol)
          .groupby(["Season", "TeamId", "PlayerId"])["_v"].sum().reset_index())
    g["rank"] = g.groupby(["Season", "TeamId"])["_v"].rank("first",
                                                           ascending=False)
    per_club = g.groupby(["Season", "TeamId"])["_v"].sum().mean()
    curve = g.groupby("rank")["_v"].mean()
    # Express the hitter curve in plate appearances by holding each rank's
    # SHARE of its club and scaling to the PA budget.
    budget = TEAM_IP_BUDGET if kind == "pitcher" else TEAM_PA_BUDGET
    return curve / per_club * budget, per_club


# ─────────────────────────────────────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────────────────────────────────────

def _projected(players: pd.DataFrame, kind: str, fielding, reserves):
    out, _, _ = project_playing_time(players, kind, target_year=2027,
                                     fielding=fielding, reserves=reserves)
    vol = "Proj_PA" if kind == "hitter" else "Proj_IP"
    out = out[out["pt_tier"].astype(str) != "floor"].copy()
    out["rank"] = out.groupby("Pred_target_team_id")[vol].rank("first",
                                                               ascending=False)
    return out, vol


def fit(players: pd.DataFrame, kind: str, curve: pd.Series, fielding,
        reserves, passes: int = 6, verbose: bool = True) -> dict[str, float]:
    key = "pa" if kind == "hitter" else "ip"
    defs = RT.HITTER_ROLES if kind == "hitter" else RT.PITCHER_ROLES
    original = {r["role"]: float(r[key]) for r in defs}

    for p in range(passes):
        out, vol = _projected(players, kind, fielding, reserves)
        out["_target"] = out["rank"].map(curve)
        # Ranks past the end of the real curve mean the model is carrying
        # more players than a real club uses. Their target is the thinnest
        # real rank, which pushes the surplus back toward the front.
        out["_target"] = out["_target"].fillna(curve.iloc[-1])
        grp = out.groupby(out["pt_role"].astype(str))
        moved = 0.0
        for role, g in grp:
            cur = float(g[vol].mean())
            tgt = float(g["_target"].mean())
            if cur <= 0 or not np.isfinite(tgt) or tgt <= 0:
                continue
            row = RT.role_anchor(role, kind)
            if row is None or float(row[key]) <= 1.0:
                continue       # the depth anchor is the 1-unit floor
            step = np.clip(tgt / cur, 1 - MAX_STEP, 1 + MAX_STEP)
            row[key] = float(row[key]) * step
            moved = max(moved, abs(step - 1.0))
        if verbose:
            print(f"    pass {p + 1}: largest anchor move {moved:+.1%}")
        if moved < 0.005:
            break

    fitted = {r["role"]: float(r[key]) for r in defs}
    for r in defs:                      # leave the module as we found it
        r[key] = original[r["role"]]
    return fitted


def report(kind: str, original: dict, fitted: dict, curve: pd.Series,
           before: pd.DataFrame, after: pd.DataFrame, vol: str) -> None:
    unit = "IP" if kind == "pitcher" else "PA"
    print(f"\n  {'role':<32}{'was':>8}{'fitted':>9}{'change':>9}")
    for role, was in original.items():
        now = fitted.get(role, was)
        if abs(now - was) < 0.05:
            continue
        print(f"  {role:<32}{was:>8.0f}{now:>9.0f}{now / was - 1:>+9.1%}")

    def err(frame):
        t = frame["rank"].map(curve).fillna(curve.iloc[-1])
        ok = t > 0
        return float(np.mean(np.abs(frame.loc[ok, vol] / t[ok] - 1)))

    print(f"\n  mean absolute error against the real rank curve")
    print(f"    before {err(before):.1%}    after {err(after):.1%}")
    print(f"\n  {'rank':>4}{'real':>9}{'before':>9}{'after':>9}")
    b = before.groupby("rank")[vol].mean()
    a = after.groupby("rank")[vol].mean()
    for r in range(1, 21):
        rv = curve.get(float(r))
        if rv is None or np.isnan(rv):
            continue
        print(f"  {r:>4}{rv:>9.1f}{b.get(float(r), np.nan):>9.1f}"
              f"{a.get(float(r), np.nan):>9.1f}   {unit}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--kind", choices=["hitter", "pitcher", "both"],
                    default="both")
    ap.add_argument("--passes", type=int, default=6)
    a = ap.parse_args(argv)

    fpath = a.out_dir / f"fielding_history_{a.target_year}.csv"
    if not fpath.exists():
        print(f"missing {fpath} — the fit needs real playing time")
        return 1
    fielding = pd.read_csv(fpath, low_memory=False)

    from team_context import TEAM_OVERRIDE_PATH, load_roster_reserves
    reserves = load_roster_reserves(TEAM_OVERRIDE_PATH(a.target_year))

    kinds = ["hitter", "pitcher"] if a.kind == "both" else [a.kind]
    for kind in kinds:
        stem = "hitter" if kind == "hitter" else "pitcher"
        src = a.out_dir / f"{stem}_pa_projections_{a.target_year}.csv"
        if not src.exists():
            print(f"missing {src}")
            return 1
        players = pd.read_csv(src, low_memory=False)
        curve, per_club = real_rank_curve(fielding, kind)
        budget = TEAM_IP_BUDGET if kind == "pitcher" else TEAM_PA_BUDGET

        print("=" * 72)
        print(f"{kind.upper()}S")
        print("=" * 72)
        print(f"  real: {per_club:,.0f} per club per season "
              f"(the budget in use is {budget:,.0f}, "
              f"{per_club / budget - 1:+.1%}), "
              f"{int(curve.index.max())} players deep")

        key = "pa" if kind == "hitter" else "ip"
        defs = RT.HITTER_ROLES if kind == "hitter" else RT.PITCHER_ROLES
        original = {r["role"]: float(r[key]) for r in defs}

        before, vol = _projected(players, kind, fielding, reserves)
        fitted = fit(players, kind, curve, fielding, reserves, a.passes)

        for r in defs:                  # apply, measure, then restore
            r[key] = fitted[r["role"]]
        after, _ = _projected(players, kind, fielding, reserves)
        for r in defs:
            r[key] = original[r["role"]]

        report(kind, original, fitted, curve, before, after, vol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

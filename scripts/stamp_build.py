#!/usr/bin/env python3
"""stamp_build.py — write out/.build_stamp.json after a CI / scheduled rebuild.

The Streamlit app (and the worker) read this stamp to decide, on each Run,
whether the heavy pipeline stages are already fresh for today and can be
skipped. The freshness check keys on:

  * ``projections_date`` — when the per-PA projections (Stage B, the SLOW step)
    were last built. If this is today, an interactive Run skips Stage B.
  * ``slate_date`` / ``slate_sig`` — the slate the correlated sims (Stage C)
    were built from, so a Run only re-sims when the live lineups/matchups/totals
    actually moved.

The pipeline scripts themselves don't write this stamp, so a build published by
the scheduled job would otherwise look "unstamped" and the app would fall back
to file mtimes (unreliable across object-store downloads). Running this once
after the rebuild — and before ``push_artifacts.py`` — records the stamp so the
published build is recognized as fresh everywhere.

Usage
-----
    # after a Stage B (projections) rebuild — marks projections fresh for today:
    python scripts/stamp_build.py --projections

    # after a sims-only (Stage C) rebuild — refreshes the slate signature only:
    python scripts/stamp_build.py
"""
import argparse
import datetime
import json
import os
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = os.path.join(HERE, "out", ".build_stamp.json")
SLATE = os.path.join(HERE, "data", "slate.json")


def slate_change_signature(slate):
    """A comparable fingerprint of the slate the sims were built from — game
    date, each team's batting order, its starting pitcher, and its Vegas implied
    total (rounded to 0.25). Kept in sync with app.py.slate_change_signature so
    the app recognizes a CI-built slate as matching the live one when nothing
    moved."""
    if not slate:
        return None
    teams = {}
    for g in slate.get("games", {}).values():
        for side in ("away", "home"):
            tcode = g.get(side)
            if not tcode:
                continue
            order = [p.get("name") for p in g.get("lineups", {}).get(side, [])]
            pit = g.get("pitchers", {}).get(side, {}) or {}
            sp = pit.get("starter") or pit.get("primary") or pit.get("opener")
            imp = (g.get("implied", {}) or {}).get(side)
            total = round(float(imp) * 4) / 4 if imp not in (None, "") else None
            teams[tcode] = {"order": order, "sp": sp, "total": total}
    return {"date": slate.get("date"), "teams": teams}


def _load_slate():
    if os.path.exists(SLATE):
        try:
            return json.load(open(SLATE))
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projections", action="store_true",
                    help="mark projections (Stage B) as built today — set this "
                         "after a projection rebuild so interactive Runs skip it")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()

    # merge onto any existing stamp so we don't drop unrelated fields
    data = {}
    if os.path.exists(STAMP):
        try:
            data = json.load(open(STAMP)) or {}
        except Exception:
            data = {}

    if args.projections:
        # both the "built" date and the once-a-day "attempt" guard, so the app
        # neither rebuilds nor re-attempts projections again today
        data["projections_date"] = today
        data["proj_attempt_date"] = today

    slate = _load_slate()
    if slate:
        sig = slate_change_signature(slate)
        if sig is not None:
            data["slate_sig"] = sig
        if slate.get("date"):
            data["slate_date"] = slate.get("date")

    data["ts"] = time.time()

    os.makedirs(os.path.dirname(STAMP), exist_ok=True)
    with open(STAMP, "w") as f:
        json.dump(data, f)

    print(f"Wrote {os.path.relpath(STAMP, HERE)}: "
          f"projections_date={data.get('projections_date')}, "
          f"slate_date={data.get('slate_date')}, "
          f"slate_sig={'set' if data.get('slate_sig') else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

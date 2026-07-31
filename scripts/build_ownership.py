#!/usr/bin/env python3
"""
build_ownership.py — precompute projected (expected) ownership as a build artifact
=================================================================================

Runs the ``ownership_model.project_ownership`` conditional-logit model over the
day's correlated sims + projection deliverables and writes a single
``deliverables/projected_ownership.csv`` the app / downstream steps can read
instead of recomputing ownership on every interactive run.

Why a separate, gated step
--------------------------
Projected ownership only changes when the PLAYER POOL changes (who is on the
slate and their sim distributions / order). It does NOT need to be recomputed on
an unchanged pool, so this script fingerprints the pool and SKIPS the recompute
when nothing moved — that's the resource-constraining part. It is wired into the
scheduled GitHub Actions refresh (see .github/workflows/refresh.yml) so the
expensive-to-assemble inputs are read once, in CI, right after the sims rebuild.

Limitation: CI has no DraftKings salary/eligibility feed, so this uses the
sim-derived features (proj, ceiling shape, batting order, team total) only — the
``value`` (proj/salary) term is left out and outfield-eligible bats are grouped
by their listed defensive position. It is a projection to seed/compare against,
not a replacement for a live DK ownership feed.

Usage
-----
    python scripts/build_ownership.py                 # gated on pool change
    python scripts/build_ownership.py --force         # always recompute
    python scripts/build_ownership.py --contest-size 20000
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

# run from the repo root regardless of where it's invoked
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from ownership_model import norm, project_ownership   # noqa: E402

DELIV = os.path.join(HERE, "deliverables")
OUT_CSV = os.path.join(DELIV, "projected_ownership.csv")
SIG_PATH = os.path.join(DELIV, ".ownership_pool_sig.json")

# defensive position -> DK roster slot group used by the ownership softmax.
# LF/CF/RF (and DH, which CI can't map to a real DK slot) collapse to OF.
POS_TO_SLOT = {
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "OF",
}


def _latest_date() -> str | None:
    """Slate date of the newest sim_manifest in deliverables/, or None."""
    manifests = glob.glob(os.path.join(DELIV, "sim_manifest_*.json"))
    if not manifests:
        return None
    newest = max(manifests, key=os.path.getmtime)
    try:
        return json.load(open(newest)).get("date")
    except Exception:
        base = os.path.basename(newest)
        return base[len("sim_manifest_"):-len(".json")] or None


def _load_pool(date: str) -> pd.DataFrame:
    """Build the (player, slot) pool from the dated projection CSVs."""
    hp = os.path.join(DELIV, f"hitter_projections_{date}.csv")
    pp = os.path.join(DELIV, f"pitcher_projections_{date}.csv")
    hd = pd.read_csv(hp)
    pd_ = pd.read_csv(pp)

    hitters = pd.DataFrame({
        "Name": hd["player"],
        "Pos": hd["pos"].map(lambda p: POS_TO_SLOT.get(str(p), "OF")),
        "Team": hd["team"],
        # batting order (1-9) drives the order feature; 0/absent = unconfirmed
        "Order": pd.to_numeric(hd.get("slot"), errors="coerce").fillna(0),
    })
    pitchers = pd.DataFrame({
        "Name": pd_["player"],
        "Pos": "P",
        "Team": pd_["team"],
        "Order": 0.0,
    })
    pool = pd.concat([hitters, pitchers], ignore_index=True)
    return pool, hd


def _load_sims(date: str) -> dict:
    """Merged {norm(name) -> sim array} for hitters and pitchers."""
    h = np.load(os.path.join(DELIV, "hitter_dk_sims.npy"), allow_pickle=True).item()
    p = np.load(os.path.join(DELIV, "pitcher_dk_sims.npy"), allow_pickle=True).item()
    sims = {}
    for d in (h, p):
        for k, v in d.items():
            sims[norm(k)] = v
    return sims


def _pool_signature(pool: pd.DataFrame, contest_size) -> str:
    """A fingerprint that changes only when the slate's players / order / team /
    the requested contest size change — the inputs that move projected ownership."""
    rows = sorted(
        f"{norm(r.Name)}|{r.Pos}|{r.Team}|{float(r.Order):.0f}"
        for r in pool.itertuples()
    )
    payload = json.dumps({"cs": contest_size, "rows": rows}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contest-size", type=int, default=None,
                    help="field size to reshape ownership for (default: the "
                         "model's medium baseline).")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the player pool is unchanged.")
    args = ap.parse_args()

    date = _latest_date()
    if not date:
        print("build_ownership: no sim_manifest found in deliverables/ — nothing "
              "to do (run the sim pipeline first).")
        return 0

    pool, hd = _load_pool(date)
    sig = _pool_signature(pool, args.contest_size)

    # gate: skip the recompute when the pool (and contest size) is unchanged and
    # a prior output is already present.
    if not args.force and os.path.exists(OUT_CSV) and os.path.exists(SIG_PATH):
        try:
            prev = json.load(open(SIG_PATH))
        except Exception:
            prev = {}
        if prev.get("sig") == sig:
            print(f"build_ownership: player pool unchanged for {date} — keeping "
                  f"{os.path.relpath(OUT_CSV, HERE)} (use --force to recompute).")
            return 0

    sims = _load_sims(date)
    # team total (stacking) feature: per-team implied runs from the hitter rows
    team_total = {t: float(v) for t, v in
                  hd.dropna(subset=["team_total"])
                    .groupby("team")["team_total"].first().items()}

    own = project_ownership(pool, sims, contest_size=args.contest_size,
                            team_total=team_total)
    out = pool.copy()
    out["ProjOwnership"] = own.round(2).to_numpy()
    out["Date"] = date
    out = out.sort_values("ProjOwnership", ascending=False).reset_index(drop=True)
    os.makedirs(DELIV, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    json.dump({"sig": sig, "date": date}, open(SIG_PATH, "w"))
    print(f"build_ownership: wrote {os.path.relpath(OUT_CSV, HERE)} "
          f"({len(out)} players, slate {date}, "
          f"contest_size={args.contest_size or 'medium-baseline'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

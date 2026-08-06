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

Salary comes from the DraftKings salaries feed (the same RotoWire proxy the app
uses, ``dk_slate_feed``), joined onto the pool by name so the ``value``
(proj/salary) term is fitted — the biggest single ownership driver after order.
If the feed is unreachable (offline CI), it degrades to the sim-only features and
says so. Outfield-eligible bats are grouped by their listed defensive position.

Usage
-----
    python scripts/build_ownership.py                 # gated on pool change
    python scripts/build_ownership.py --force         # always recompute
    python scripts/build_ownership.py --no-fetch      # skip the DK salary feed
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


def _salary_map(fetch: bool = True):
    """(by_name, by_name_team, by_pid) DK salary lookups from the salaries feed.

    Keyed by ``norm(name)``, by ``(norm(name), team)`` so a same-named player on
    two teams resolves to the right salary, and by RotoID. Returns empty maps
    (salary-blind) if the feed can't be reached — the projection still runs on
    the sim features.

    The RotoID map matters because the salaries feed occasionally ships a mangled
    name — a "De La Cruz" surname truncated to "De", or an accent garbled by a
    mis-encode ("Jose Ramírez" -> "Jose RamÃ­rez"). Those names never match the
    projection pool (which is keyed by the full lineup-feed name), so a
    name-only join silently loses the player's salary. RotoID is stable across
    feeds, so :func:`_attach_salary` can bridge through it (see ``_pid_bridge``).
    """
    if not fetch:
        return {}, {}, {}
    try:
        import dk_slate_feed as feed
        _date, slates = feed.parse_salaries(feed._http_get(feed.FEED_SALARIES))
    except Exception as e:                       # offline / feed hiccup
        print(f"build_ownership: DK salary feed unavailable "
              f"({type(e).__name__}: {e}) — projecting without the value term.")
        return {}, {}, {}
    by_n, by_nt, by_pid = {}, {}, {}
    for s in slates.values():
        for pl in s.get("players", []):
            sal = int(pl.get("salary") or 0)
            if sal <= 0:
                continue
            nm = norm(pl["name"])
            by_n[nm] = sal
            by_nt[(nm, pl.get("team"))] = sal
            pid = str(pl.get("roto_id") or "")
            if pid:
                by_pid[pid] = sal
    return by_n, by_nt, by_pid


def _pid_bridge(fetch: bool = True) -> dict:
    """``norm(full name) -> RotoID`` from the live lineup feed, which carries the
    canonical full name AND the pid for every player in a posted lineup. This is
    what lets us recover a salary the salaries feed filed under a mangled name.
    Empty (a no-op) if the feed is unreachable."""
    if not fetch:
        return {}
    try:
        import slate_ingest
        live = slate_ingest.build_slate(write=False)
    except Exception as e:
        print(f"build_ownership: lineup feed unavailable for RotoID bridge "
              f"({type(e).__name__}: {e}) — name-only salary match.")
        return {}
    bridge = {}
    for g in (live.get("games", {}) or {}).values():
        for side in ("away", "home"):
            for pl in g.get("lineups", {}).get(side, []) or []:
                pid, nm = str(pl.get("pid") or ""), pl.get("name")
                if pid and nm:
                    bridge[norm(nm)] = pid
    return bridge


def _attach_salary(pool: pd.DataFrame, by_n: dict, by_nt: dict,
                   by_pid: dict = None, name_pid: dict = None) -> pd.DataFrame:
    """Add a ``Salary`` column: (name, team) exact, then a RotoID bridge (which
    recovers salaries the feed filed under a mangled name), then name-only."""
    by_pid, name_pid = by_pid or {}, name_pid or {}

    def look(r):
        nn = norm(r.Name)
        return (by_nt.get((nn, r.Team))
                or by_pid.get(name_pid.get(nn))
                or by_n.get(nn)
                or np.nan)
    pool = pool.copy()
    pool["Salary"] = [look(r) for r in pool.itertuples()]
    return pool


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
    salary / the requested contest size change — the inputs that move projected
    ownership."""
    has_sal = "Salary" in pool.columns
    rows = sorted(
        f"{norm(r.Name)}|{r.Pos}|{r.Team}|{float(r.Order):.0f}|"
        f"{'' if not has_sal or pd.isna(r.Salary) else int(r.Salary)}"
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
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the DK salary feed (project without the value term).")
    args = ap.parse_args()

    date = _latest_date()
    if not date:
        print("build_ownership: no sim_manifest found in deliverables/ — nothing "
              "to do (run the sim pipeline first).")
        return 0

    pool, hd = _load_pool(date)
    by_n, by_nt, by_pid = _salary_map(fetch=not args.no_fetch)
    name_pid = _pid_bridge(fetch=not args.no_fetch) if by_pid else {}
    pool = _attach_salary(pool, by_n, by_nt, by_pid, name_pid)
    n_sal = int(pool["Salary"].notna().sum())
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
    out = pool[["Name", "Pos", "Team", "Order", "Salary"]].copy()
    out["ProjOwnership"] = own.round(2).to_numpy()
    out["Date"] = date
    out = out.sort_values("ProjOwnership", ascending=False).reset_index(drop=True)
    os.makedirs(DELIV, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    json.dump({"sig": sig, "date": date}, open(SIG_PATH, "w"))
    print(f"build_ownership: wrote {os.path.relpath(OUT_CSV, HERE)} "
          f"({len(out)} players, slate {date}, "
          f"salary on {n_sal}/{len(out)}, "
          f"contest_size={args.contest_size or 'medium-baseline'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

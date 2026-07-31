#!/usr/bin/env python3
"""
ownership_history.py — accumulate a compact per-slate ownership training log
============================================================================

The 10k-sim `.npy` history is heavy (~25 MB/day) and prunes to a 4-day window.
Training an ownership model wants the *opposite*: many slates, but only a
handful of summary numbers per player. So this module keeps a separate,
**append-only, tiny** log — one row per (slate, player) with just the features
the ownership model uses plus DK salary and Vegas context — that can grow for a
whole season without meaningfully touching storage (~a few dozen KB per slate).

    date, name, team, opp, pos, role, salary,
    proj, ceiling, floor, std,           # from the sims / projection engine
    team_total,                          # Vegas implied runs (hitter's team)
    value,                               # proj / (salary/1000)
    own                                  # actual %Drafted — filled in later

Sources it can build a slate's rows from (whichever is at hand):
  * a stage_d-style pool (Name/Pos/Team/Salary) + the merged sim dict, or
  * the pipeline's archived projection CSVs (proj/p90/p10/std/team_total already
    computed) + a salary lookup.
Salary and context are optional: a row is logged even if salary is missing, so
context still accumulates; the label `own` is attached later from a contest CSV.

Typical lifecycle
------------------
    # at build/snapshot time (salary from the DK pool, context from sims):
    append_slate_from_pool(date, pool, sims, log_path=LOG)

    # when the contest settles:
    attach_ownership(date, "contest-standings-XXXX.csv", log_path=LOG)

    # to train:
    df = load_log(LOG)          # -> feed straight into fit_ownership --history-csv
"""

from __future__ import annotations

import os
import re
import glob

import numpy as np
import pandas as pd

from ownership_model import norm, sim_features

DEFAULT_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ownership_history", "features.csv"
)

COLUMNS = ["date", "name", "team", "opp", "pos", "role", "salary",
           "proj", "ceiling", "floor", "std", "team_total", "value", "own"]
_KEY = ["date", "name", "pos"]

# projection-engine baseball position -> DK roster slot. Multi-eligibility isn't
# recoverable from a projection CSV; the authoritative DK slot for a *labeled*
# row is set from the contest's Roster Position in attach_ownership().
_DK_SLOT = {"SP": "P", "RP": "P", "P": "P",
            "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF",
            "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
            "DH": "OF"}   # DH players are most often OF-eligible on DK; a guess


def _dk_slot(pos: str) -> str:
    return _DK_SLOT.get(str(pos).upper().strip(), str(pos))


# --------------------------------------------------------------------------- #
# salary / context lookups
# --------------------------------------------------------------------------- #
def salary_lookup_from_dk(dk) -> dict:
    """{norm(name): salary} from a DK pool: a DataFrame (FullName/Name + Salary)
    or a path to such a CSV."""
    if dk is None:
        return {}
    df = pd.read_csv(dk, encoding="latin-1") if isinstance(dk, str) else dk
    namecol = "FullName" if "FullName" in df.columns else "Name"
    out = {}
    for r in df.itertuples():
        nm = getattr(r, namecol, None)
        sal = getattr(r, "Salary", None)
        if nm is not None and pd.notna(sal):
            out[norm(nm)] = float(sal)
    return out


def context_from_dff(dff_csv: str) -> dict:
    """{norm(name): (salary, implied_team_score)} from a DailyFantasyFuel sheet."""
    x = pd.read_csv(dff_csv)
    nn = (x["first_name"].astype(str) + " " + x["last_name"].astype(str)).map(norm)
    sal = pd.to_numeric(x["salary"], errors="coerce")
    itt = pd.to_numeric(x["implied_team_score"], errors="coerce")
    return {n: (s, t) for n, s, t in zip(nn, sal, itt)}


# --------------------------------------------------------------------------- #
# build a slate's feature rows
# --------------------------------------------------------------------------- #
def slate_rows_from_pool(date, pool, sims, *, salary=None, team_total=None,
                         ceil_pct=90.0) -> pd.DataFrame:
    """Compact feature rows from a pool + sim dict.

    pool        DataFrame with Name, Pos [, Team, Opp, Role, Salary].
    sims        {norm(name) -> sim scores}.
    salary      optional {norm(name) -> salary}; falls back to pool.Salary.
    team_total  optional {TeamCode -> implied runs} or {norm(name) -> runs}.
    """
    salary = salary or {}
    team_total = team_total or {}
    rows = []
    for r in pool.itertuples():
        nm = getattr(r, "Name")
        nn = norm(nm)
        sc = sims.get(nn)
        f = sim_features(sc, ceil_pct) if sc is not None else {
            "proj": np.nan, "ceiling": np.nan, "ceil_shape": np.nan}
        sd = float(np.std(sc)) if sc is not None else np.nan
        floor = float(np.percentile(sc, 10)) if sc is not None else np.nan
        sal = salary.get(nn)
        if sal is None:
            ps = getattr(r, "Salary", None)
            sal = float(ps) if ps is not None and pd.notna(ps) else np.nan
        team = getattr(r, "Team", "")
        tt = team_total.get(team, team_total.get(nn, np.nan))
        value = (f["proj"] / (sal / 1000.0)
                 if sal and not pd.isna(sal) and sal > 0 else np.nan)
        rows.append({
            "date": date, "name": nn, "team": team,
            "opp": getattr(r, "Opp", ""), "pos": getattr(r, "Pos", ""),
            "role": getattr(r, "Role", ""), "salary": sal,
            "proj": f["proj"], "ceiling": f["ceiling"], "floor": floor,
            "std": sd, "team_total": tt, "value": value, "own": np.nan,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


# --------------------------------------------------------------------------- #
# append-only log I/O (dedup by date+name+pos; last write wins)
# --------------------------------------------------------------------------- #
def _read(log_path) -> pd.DataFrame:
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame(columns=COLUMNS)


def _write(df: pd.DataFrame, log_path) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    df.to_csv(log_path, index=False)


def append_rows(rows: pd.DataFrame, log_path=DEFAULT_LOG,
                keep_label=True) -> pd.DataFrame:
    """Merge `rows` into the log, deduped by (date, name, pos).

    New rows win on the feature columns, but an existing `own` label is
    preserved when the incoming row has none (so re-logging a slate's features
    never wipes a label already attached from its contest)."""
    cur = _read(log_path)
    both = pd.concat([cur, rows], ignore_index=True)
    both = both.reindex(columns=COLUMNS)
    if keep_label and len(cur):
        # backfill missing incoming labels from any prior row for the same key
        lab = (both.dropna(subset=["own"]).drop_duplicates(_KEY, keep="first")
               .set_index(_KEY)["own"])
        miss = both["own"].isna()
        both.loc[miss, "own"] = both.loc[miss].set_index(_KEY).index.map(lab).to_numpy()
    both = both.drop_duplicates(_KEY, keep="last").sort_values(_KEY)
    _write(both, log_path)
    return both


def append_slate_from_pool(date, pool, sims, *, salary=None, team_total=None,
                           log_path=DEFAULT_LOG) -> pd.DataFrame:
    """Convenience: build a slate's rows from a pool+sims and append them."""
    rows = slate_rows_from_pool(date, pool, sims, salary=salary,
                                team_total=team_total)
    return append_rows(rows, log_path=log_path)


def slate_rows_from_projection_csvs(date, deliv_dir="deliverables",
                                    salary=None) -> pd.DataFrame:
    """Compact feature rows from the pipeline's own projection CSVs.

    Reads `deliverables/{hitter,pitcher}_projections_{date}.csv` — which already
    carry the sim summaries (`proj`, `p90`, `p10`, `std`) and Vegas context
    (`team_total` for hitters) — so a slate can be logged at build time with no
    sims or pool object in scope. `salary` is an optional {norm(name)->salary}.
    """
    salary = salary or {}
    frames = []
    for kind in ("hitter", "pitcher"):
        f = os.path.join(deliv_dir, f"{kind}_projections_{date}.csv")
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f)
        nn = x["player"].map(norm)
        sal = nn.map(salary)
        proj = pd.to_numeric(x.get("proj"), errors="coerce")
        tt = pd.to_numeric(x["team_total"], errors="coerce") if "team_total" in x \
            else pd.Series(np.nan, index=x.index)
        frames.append(pd.DataFrame({
            "date": date, "name": nn, "team": x.get("team", ""),
            "opp": x.get("opp", ""),
            "pos": x["pos"].map(_dk_slot) if "pos" in x else "",
            "role": x.get("role", ""), "salary": sal,
            "proj": proj,
            "ceiling": pd.to_numeric(x.get("p90"), errors="coerce"),
            "floor": pd.to_numeric(x.get("p10"), errors="coerce"),
            "std": pd.to_numeric(x.get("std"), errors="coerce"),
            "team_total": tt,
            "value": np.where((sal.notna()) & (sal > 0), proj / (sal / 1000.0), np.nan),
            "own": np.nan,
        }))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)


def snapshot_slate_features(date, deliv_dir="deliverables", *, dk=None,
                            dff_csv=None, log_path=DEFAULT_LOG) -> pd.DataFrame:
    """Build a slate's rows from the archived projection CSVs (+ a salary source)
    and append them to the log. Salary comes from `dk` (a DK pool DataFrame or
    CSV path) or `dff_csv` if given; otherwise rows log with salary NaN. Safe to
    call every build — features re-log, an attached `own` label is preserved."""
    salary = {}
    if dk is not None:
        salary = salary_lookup_from_dk(dk)
    elif dff_csv:
        salary = {k: v[0] for k, v in context_from_dff(dff_csv).items()}
    rows = slate_rows_from_projection_csvs(date, deliv_dir, salary=salary)
    if rows.empty:
        return _read(log_path)
    return append_rows(rows, log_path=log_path)


# --------------------------------------------------------------------------- #
# attach the actual ownership label from a DK contest-standings CSV
# --------------------------------------------------------------------------- #
def parse_contest_ownership(contest_csv: str) -> pd.DataFrame:
    """(name, pos, own) from a DK contest-standings CSV right block. `pos` is the
    authoritative DK Roster Position the field actually drafted the player at."""
    df = pd.read_csv(contest_csv).dropna(subset=["Player"])
    return pd.DataFrame({
        "name": df["Player"].map(norm),
        "pos": df["Roster Position"].astype(str),
        "own": df["%Drafted"].astype(str).str.rstrip("%").astype(float),
    })


def attach_ownership(date, contest_csv, log_path=DEFAULT_LOG) -> pd.DataFrame:
    """Fill `own` (and correct `pos` to the DK Roster Position) for `date`'s rows
    from a contest CSV, matched by name. This makes labeled rows carry the exact
    slot the field drafted at, regardless of what position was logged at build."""
    con = parse_contest_ownership(contest_csv)
    own_by = dict(zip(con["name"], con["own"]))
    pos_by = dict(zip(con["name"], con["pos"]))
    df = _read(log_path)
    m = df["date"] == date
    matched = df.loc[m, "name"].isin(own_by)
    idx = df.loc[m][matched].index
    df.loc[idx, "own"] = df.loc[idx, "name"].map(own_by)
    df.loc[idx, "pos"] = df.loc[idx, "name"].map(pos_by)
    df = df.drop_duplicates(_KEY, keep="last")
    _write(df, log_path)
    return df


# --------------------------------------------------------------------------- #
# load for training
# --------------------------------------------------------------------------- #
def load_log(log_path=DEFAULT_LOG, labeled_only=False) -> pd.DataFrame:
    df = _read(log_path)
    if labeled_only:
        df = df[df["own"].notna()]
    return df.reset_index(drop=True)


def _sims_for_date(sims_dir, date) -> dict:
    out = {}
    for kind in ("hitter", "pitcher"):
        f = os.path.join(sims_dir, f"history_{date}_{kind}_dk_sims.npy")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True).item()
            out.update({norm(k): np.asarray(v, float) for k, v in d.items()})
    return out


def ingest_slate(date, *, sims_dir=None, deliv_dir="deliverables", dff=None,
                 contest=None, log_path=DEFAULT_LOG) -> pd.DataFrame:
    """Add one slate to the log from the files you already download.

    Feature source: the sims in `sims_dir` (history_<date>_*_dk_sims.npy) if
    given, else the pipeline projection CSVs in `deliv_dir`. Salary + Vegas come
    from the DFF cheatsheet(s) in `dff` (one or more paths). If `contest` is
    given, the actual %Drafted label (and DK slot) is attached from it.
    """
    ctx = {}
    for f in (dff or []):
        for k, v in context_from_dff(f).items():
            ctx.setdefault(k, v)
    salary = {k: v[0] for k, v in ctx.items()}
    team_total = {k: v[1] for k, v in ctx.items()}

    if sims_dir:
        sims = _sims_for_date(sims_dir, date)
        # a minimal pool: every simmed player (DK slot corrected later by the
        # contest); pitchers flagged P so is_pitcher is right pre-label.
        h = os.path.join(sims_dir, f"history_{date}_pitcher_dk_sims.npy")
        pnames = set()
        if os.path.exists(h):
            pnames = {norm(k) for k in np.load(h, allow_pickle=True).item()}
        pool = pd.DataFrame([{"Name": n, "Pos": "P" if n in pnames else "OF",
                              "Team": "", "Salary": salary.get(n, np.nan)}
                             for n in sims])
        rows = slate_rows_from_pool(date, pool, sims, salary=salary,
                                    team_total=team_total)
    else:
        rows = slate_rows_from_projection_csvs(date, deliv_dir, salary=salary)
        # team_total from DFF where the projection CSV lacked it
        miss = rows["team_total"].isna()
        rows.loc[miss, "team_total"] = rows.loc[miss, "name"].map(team_total)

    append_rows(rows, log_path=log_path)
    if contest:
        attach_ownership(date, contest, log_path=log_path)
    return load_log(log_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("ingest", help="add a slate to the log from DFF + contest")
    sp.add_argument("--date", required=True)
    sp.add_argument("--sims-dir", default="", help="dir of history_<date>_*_dk_sims.npy")
    sp.add_argument("--deliv-dir", default="deliverables",
                    help="dir of <kind>_projections_<date>.csv (used if no --sims-dir)")
    sp.add_argument("--dff", action="append", default=[],
                    help="DFF cheatsheet CSV (repeat for main + early)")
    sp.add_argument("--contest", default="", help="DK contest-standings CSV for the label")
    sp.add_argument("--log", default=DEFAULT_LOG)

    sh = sub.add_parser("show", help="summarise the log")
    sh.add_argument("--log", default=DEFAULT_LOG)

    args = ap.parse_args()
    if args.cmd == "ingest":
        df = ingest_slate(args.date, sims_dir=args.sims_dir or None,
                          deliv_dir=args.deliv_dir, dff=args.dff,
                          contest=args.contest or None, log_path=args.log)
        d = df[df.date == args.date]
        print(f"ingested {args.date}: {len(d)} rows, "
              f"salary {d.salary.notna().mean():.0%}, labeled {d.own.notna().mean():.0%}"
              f"  -> {args.log}")
    else:
        df = load_log(getattr(args, "log", DEFAULT_LOG))
        if df.empty:
            print("empty log")
        else:
            lab = df["own"].notna().sum()
            print(f"{len(df)} rows, {df['date'].nunique()} slates, "
                  f"{lab} labeled ({lab/len(df):.0%}); salary on "
                  f"{df['salary'].notna().mean():.0%} of rows")
            print(df.groupby("date").size().to_string())

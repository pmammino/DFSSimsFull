"""slate_parse.py — parse an uploaded slate CSV into the (dk_df, id_map) shape
the runner expects. Ported from app.py (parse_dk_export / alias_columns /
ids_from_clean) so the upload path matches the RotoWire-feed path exactly.

Accepts either a raw DraftKings salaries export (the DKSalaries.csv with the
Position-header row) or a "clean" CSV with FullName/Team/Position/Salary
[/Ownership][/ID] columns under any of the accepted aliases.
"""
import csv
import io
import re

import pandas as pd

import dk_ids

COL_ALIASES = {
    "FullName": ["fullname", "name", "player", "player name", "playername"],
    "Team": ["team", "teamabbrev", "team abbrev", "tm"],
    "Position": ["position", "pos", "roster position"],
    "Salary": ["salary", "sal"],
    "Ownership": ["ownership", "own", "own%", "owned", "pown", "proj own",
                  "projected ownership", "projown", "%drafted", "drafted%",
                  "ownership%", "proj. own", "ros own"],
}
ID_NAMES = {"id", "playerid", "player id", "dk id", "dkid", "player_id",
            "playerid#", "contest id", "contestid", "draftkings id",
            "draftkingsid", "dkplayerid", "player id #"}


def _norm_col(c):
    return re.sub(r"\s+", " ", str(c).strip().lower())


def _clean_id(v):
    v = str(v).strip()
    if not v or v.lower() == "nan":
        return None
    m = re.search(r"(\d{4,})", v)
    return m.group(1) if m else None


def parse_dk_export(text):
    """Raw DKSalaries export -> (dk_df, id_map) or None. dk_df has FullName,
    Team, Position, Salary (no Ownership — DK exports don't carry it)."""
    rows = list(csv.reader(io.StringIO(text)))
    hdr = next((i for i, r in enumerate(rows)
                if len(r) >= 20 and r[11] == "Position"), None)
    if hdr is None:
        return None
    recs, idmap = [], {}
    for r in rows[hdr + 1:]:
        if len(r) >= 20 and r[13].strip() and r[14].strip():
            nm = r[13].strip()
            try:
                sal = int(float(r[16])) if r[16].strip() else 0
            except ValueError:
                sal = 0
            recs.append({"FullName": nm, "Team": r[18].strip(),
                         "Position": r[11].strip(), "Salary": sal})
            dk_ids.add_id(idmap, nm, r[18], r[14], pos=r[11], salary=sal)
    if not recs:
        return None
    return pd.DataFrame(recs), idmap


def alias_columns(df):
    lut = {_norm_col(c): c for c in df.columns}
    rename = {}
    for canon, aliases in COL_ALIASES.items():
        if canon in df.columns:
            continue
        for a in aliases:
            if a in lut:
                rename[lut[a]] = canon
                break
    return df.rename(columns=rename) if rename else df


def ids_from_clean(df):
    """name -> DK upload id from a clean CSV, preferring a contest/draftable id."""
    if "FullName" not in df.columns:
        return {}, None
    cols = {_norm_col(c): c for c in df.columns}
    has_team = "Team" in df.columns
    has_pos = "Position" in df.columns
    has_sal = "Salary" in df.columns

    def harvest(orig):
        m = {}
        for _, r in df.iterrows():
            cid = _clean_id(r[orig])
            if cid:
                dk_ids.add_id(m, r["FullName"], r["Team"] if has_team else "", cid,
                              pos=r["Position"] if has_pos else "",
                              salary=r["Salary"] if has_sal else None)
        return m

    for token in ("contest", "draftable"):
        for key, orig in cols.items():
            if token in key and "id" in key:
                m = harvest(orig)
                if m:
                    return m, orig
    for key, orig in cols.items():
        nospace = key.replace(" ", "")
        if key in ID_NAMES or nospace in {k.replace(" ", "") for k in ID_NAMES}:
            m = harvest(orig)
            if m:
                return m, orig
    for key, orig in cols.items():
        if "name" in key and "id" in key:
            m = harvest(orig)
            if m:
                return m, orig
    return {}, None


def parse_slate_csv(text: str):
    """Parse an uploaded slate CSV (raw DK export OR clean CSV) into
    (dk_df, id_map). Raises ValueError with a user-facing message on failure.

    The returned dk_df always has FullName/Team/Position/Salary and, when the
    file carried it, Ownership. Ownership defaults to 0 when absent (the runner
    still works; the ownership-weighted field just treats everyone equally)."""
    # Try the raw DK export first (fixed-column layout).
    export = parse_dk_export(text)
    if export is not None:
        df, idmap = export
        df["Ownership"] = 0.0
        return df, idmap

    # Otherwise treat it as a clean CSV with a normal header row.
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise ValueError(f"Could not read the CSV: {e}")
    df = alias_columns(df)
    required = {"FullName", "Team", "Position", "Salary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Slate CSV is missing required column(s): "
            + ", ".join(sorted(missing))
            + ". Expected FullName, Team, Position, Salary (Ownership optional), "
            "or a raw DraftKings salaries export.")
    if "Ownership" not in df.columns:
        df["Ownership"] = 0.0
    idmap, _ = ids_from_clean(df)
    keep = ["FullName", "Team", "Position", "Salary", "Ownership"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
    df["Ownership"] = pd.to_numeric(df["Ownership"], errors="coerce").fillna(0.0)
    return df, idmap

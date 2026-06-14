#!/usr/bin/env python3
"""
app.py — Streamlit interface for the DFS contest simulator (Stage D)
====================================================================
A point-and-click front end over stage_d.py. It uses the correlated player DK
sims already built into deliverables/ (Stage C) and asks the user for the only
things that aren't fixed by the pipeline:

  * a DraftKings salary/ownership CSV  (FullName, Team, Position, Salary, Ownership)
  * the contest size (field entries)
  * the number of sim runs to score against
  * the number of candidate lineups to develop

Nothing runs until ALL four decisions are made and the user clicks Run. On Run
the app first ensures the underlying data is current — if the projections aren't
from today it rebuilds projections + correlated sims (Stage A–C); otherwise if
the confirmed lineups have changed it reruns the correlated sims (Stage C) —
then builds an ownership-weighted field + a uniform candidate pool and reports
the simulated contest outcomes (Win% / Top10% / Top100% / AvgPlace) for every
candidate lineup. Each lineup can be inspected (clean player table) and its
finishing-place distribution across all sim runs shown.

Launch:
    streamlit run app.py
"""
import csv, datetime, glob, io, json, os, re, subprocess, sys, tempfile, time
from collections import Counter
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from stage_d import (load_sims, build_pool, lineups_to_df, score_matrix,
                     norm as normname, COLS, HITC, SLOT)
from mlb_lineup_builder import Pool, Builder
from field_simulator import (normalize_to_slots, adjust_ownership,
                             beta_for_size, tilt_structures)
import slate_ingest
import shared_store

# per-position place charts can exceed Altair's default 5000-row cap
try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DELIV = os.path.join(HERE, "deliverables")
ASSETS = os.path.join(HERE, "assets")
PARAMS_PATH = os.path.join(HERE, "field_params.json")
REQ_COLS = ["FullName", "Team", "Position", "Salary", "Ownership"]
SIZE_PRESETS = [150, 1000, 6000, 20000, 50000, 150000]

# Brand palette
BRAND = "#A020FE"
INK = "#000000"
PAPER = "#FFFFFF"


def _find_logo():
    # a user-provided raster logo wins over the bundled svg placeholder
    for ext in ("png", "jpg", "jpeg", "webp", "svg"):
        p = os.path.join(ASSETS, f"logo.{ext}")
        if os.path.exists(p):
            return p
    return None


LOGO = _find_logo()
# Browser favicons need a raster image; an SVG logo falls back to an emoji icon.
_PAGE_ICON = LOGO if (LOGO and not LOGO.endswith(".svg")) else "⚾"


def render_logo(width=72):
    """Render the brand logo, handling SVG (inline, scaled to `width`) and
    raster (st.image)."""
    if not LOGO:
        return
    if LOGO.endswith(".svg"):
        try:
            svg = open(LOGO, encoding="utf-8").read()
            st.markdown(
                f'<style>.brandlogo svg{{width:100%;height:auto;display:block}}</style>'
                f'<div class="brandlogo" style="width:{width}px">{svg}</div>',
                unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.image(LOGO, width=width)


st.set_page_config(page_title="DFS Contest Simulator",
                   page_icon=_PAGE_ICON, layout="wide")

# Persistent top-left brand logo (Streamlit ≥1.35) + brand accents.
if LOGO:
    try:
        st.logo(LOGO)
    except Exception:
        pass
st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {BRAND}; }}
  [data-testid="stMetricValue"] {{ color: {BRAND}; }}
  .stProgress > div > div > div > div {{ background-color: {BRAND}; }}
  div[data-baseweb="tab-highlight"] {{ background-color: {BRAND}; }}
  a, a:visited {{ color: {BRAND}; }}
  hr {{ border-top: 2px solid {BRAND}33; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def cached_sims(hpath, hmtime, ppath, pmtime):
    """Load + cache the sim universe. mtimes are part of the key so editing the
    .npy files busts the cache."""
    return load_sims(hpath, ppath)


@st.cache_data(show_spinner=False)
def cached_params(path, mtime):
    if os.path.exists(path):
        return json.load(open(path))
    return None


@st.cache_data(show_spinner=False)
def cached_player_table(hpath, hmtime, ppath, pmtime):
    """Per-player DK-point threshold table from the current sims (mean, floor/
    median/ceiling, min/max, std, bust & boom rates). Cached on file mtimes."""
    H, P, _, _ = cached_sims(hpath, hmtime, ppath, pmtime)
    rows = []
    for typ, D in (("Hitter", H), ("Pitcher", P)):
        for nm, v in D.items():
            a = np.asarray(v, float)
            m = float(a.mean())
            rows.append({
                "Player": nm, "Type": typ,
                "Proj": round(m, 1),
                "Floor (p10)": round(float(np.percentile(a, 10)), 1),
                "Median": round(float(np.percentile(a, 50)), 1),
                "Ceiling (p90)": round(float(np.percentile(a, 90)), 1),
                "p99": round(float(np.percentile(a, 99)), 1),
                "Min": round(float(a.min()), 1),
                "Max": round(float(a.max()), 1),
                "Std": round(float(a.std()), 1),
                "Bust% (≤0)": round(100 * float((a <= 0).mean()), 1),
                "2x%": round(100 * float((a >= 2 * m).mean()), 1) if m > 0 else 0.0,
                "30+%": round(100 * float((a >= 30).mean()), 1)})
    return (pd.DataFrame(rows).sort_values("Proj", ascending=False)
            .reset_index(drop=True))


def find_sims():
    """Locate the two DK-point sim files in deliverables/."""
    h = os.path.join(DELIV, "hitter_dk_sims.npy")
    p = os.path.join(DELIV, "pitcher_dk_sims.npy")
    return (h if os.path.exists(h) else None, p if os.path.exists(p) else None)


def build_many(builder, target, label, hard_cap_mult=60):
    """Build `target` lineups from a Builder, showing a progress bar. Returns
    (lineups, attempts). Stops early (with a warning) if it can't keep up."""
    out, attempts = [], 0
    cap = target * hard_cap_mult + 500
    bar = st.progress(0.0, text=f"{label}: 0 / {target}")
    while len(out) < target and attempts < cap:
        lu = builder.build_one()
        attempts += 1
        if lu is not None:
            out.append(lu)
            if len(out) % max(1, target // 100) == 0 or len(out) == target:
                bar.progress(len(out) / target, text=f"{label}: {len(out)} / {target}")
    bar.progress(1.0, text=f"{label}: {len(out)} / {target}")
    return out, attempts


def parse_dk_template(text):
    """Map norm(player name) -> DraftKings player ID from a DKSalaries CSV.
    The player table header is the row whose column 11 == 'Position', with
    Name at col 13 and ID at col 14 (DK's standard export/upload template)."""
    rows = list(csv.reader(io.StringIO(text)))
    hdr = next((i for i, r in enumerate(rows)
                if len(r) >= 20 and r[11] == "Position"), None)
    if hdr is None:
        return None
    dkid = {}
    for r in rows[hdr + 1:]:
        if len(r) >= 20 and r[14].strip():
            dkid[normname(r[13])] = r[14].strip()
    return dkid


def parse_dk_export(text):
    """Parse a raw DKSalaries export into (slate_df, id_map) or None.
    slate_df has FullName, Team, Position, Salary; id_map is name -> DK ID.
    Lets the user upload ONE DraftKings slate file that drives both the
    simulation pool and the upload export (no separate template needed)."""
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
            idmap[normname(nm)] = r[14].strip()
    if not recs:
        return None
    return pd.DataFrame(recs), idmap


def _norm_col(c):
    return re.sub(r"\s+", " ", str(c).strip().lower())


# canonical column -> accepted aliases (matched case/space-insensitively)
COL_ALIASES = {
    "FullName": ["fullname", "name", "player", "player name", "playername"],
    "Team": ["team", "teamabbrev", "team abbrev", "tm"],
    "Position": ["position", "pos", "roster position"],
    "Salary": ["salary", "sal"],
    "Ownership": ["ownership", "own", "own%", "owned", "pown", "proj own",
                  "projected ownership", "projown", "%drafted", "drafted%",
                  "ownership%", "proj. own", "ros own"],
}
# column-name tokens that denote a DraftKings player ID
ID_NAMES = {"id", "playerid", "player id", "dk id", "dkid", "player_id",
            "playerid#", "contest id", "contestid", "draftkings id",
            "draftkingsid", "dkplayerid", "player id #"}


def _clean_id(v):
    v = str(v).strip()
    if not v or v.lower() == "nan":
        return None
    m = re.search(r"(\d{4,})", v)          # handles 12345, 12345.0, "Name (12345)"
    return m.group(1) if m else None


def alias_columns(df):
    """Rename a slate dataframe's columns to canonical names where a known alias
    is found (case/whitespace-insensitive). Returns a new dataframe; original
    columns without a known alias are left as-is."""
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
    """Pull a name -> DK upload ID map from a clean CSV. Prefers a CONTEST /
    draftable ID (e.g. PlayerContestID, Contest ID) over a generic player ID,
    because the DraftKings upload needs the slate-specific contest ID. Falls
    back to a generic ID column or a DK 'Name + ID' column. Returns
    (idmap, source_col)."""
    if "FullName" not in df.columns:
        return {}, None
    cols = {_norm_col(c): c for c in df.columns}

    def harvest(orig):
        m = {}
        for _, r in df.iterrows():
            cid = _clean_id(r[orig])
            if cid:
                m[normname(r["FullName"])] = cid
        return m

    # priority 1 — a contest / draftable ID column (what DK uploads require)
    for token in ("contest", "draftable"):
        for key, orig in cols.items():
            if token in key and "id" in key:
                m = harvest(orig)
                if m:
                    return m, orig

    # priority 2 — an explicit generic ID column (ID, Id, Player ID, DK ID, …)
    for key, orig in cols.items():
        nospace = key.replace(" ", "")
        if key in ID_NAMES or nospace in {k.replace(" ", "") for k in ID_NAMES}:
            m = harvest(orig)
            if m:
                return m, orig

    # priority 3 — a DK-style "Name + ID" column ("Player Name (1234567)")
    for key, orig in cols.items():
        if "name" in key and "id" in key:
            m = harvest(orig)
            if m:
                return m, orig

    # priority 4 — any 'id'-token column whose values are mostly long integers
    for key, orig in cols.items():
        if "id" in key.split() or key.endswith(" id") or key == "id":
            vals = [_clean_id(v) for v in df[orig].head(50)]
            if sum(v is not None for v in vals) >= max(3, 0.5 * len(vals)):
                m = harvest(orig)
                if m:
                    return m, orig
    return {}, None


def build_dk_upload(res_df, dkid, n_select, sort_by, player_cap=1.0, team_cap=1.0):
    """Greedily pick n_select lineups from the ranked candidate results under
    exposure caps, map players to DK IDs, and emit a ready-to-upload CSV
    (header P,P,C,1B,2B,3B,SS,OF,OF,OF + one ID row per lineup)."""
    keymap = {"Win%": ["Wins", "Top10", "Top100"],
              "Top10 Rate": ["Top10", "Top100", "Wins"],
              "Top100 Rate": ["Top100", "Top10", "Wins"]}[sort_by]
    rdf = res_df.sort_values(keymap, ascending=False).reset_index(drop=True)
    N = int(n_select)
    pcap = max(1, int(round(player_cap * N)))
    tcap = max(1, int(round(team_cap * N)))

    def names_of(row):
        return [str(row[c]).rsplit(" (", 1)[0] for c in COLS]

    def prim(row):
        c = Counter(str(row[x]).rsplit(" (", 1)[1][:-1]
                    for x in HITC if " (" in str(row[x]))
        return c.most_common(1)[0][0]

    expo, teamc, chosen, skipped = Counter(), Counter(), [], 0
    for _, row in rdf.iterrows():
        nms = names_of(row)
        if any(normname(n) not in dkid for n in nms):
            skipped += 1
            continue
        if all(expo[n] < pcap for n in nms) and teamc[prim(row)] < tcap:
            chosen.append(row)
            for n in nms:
                expo[n] += 1
            teamc[prim(row)] += 1
        if len(chosen) == N:
            break

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(SLOT)
    for row in chosen:
        w.writerow([dkid[normname(n)] for n in names_of(row)])
    info = {"chosen": len(chosen), "requested": N, "skipped_unmapped": skipped,
            "max_player": max(expo.values()) if expo else 0,
            "max_team": max(teamc.values()) if teamc else 0}
    return out.getvalue(), info


def rows_to_upload_csv(rows_df, dkid):
    """Emit a DK upload CSV for an explicit, already-ordered set of lineup rows
    (used by the 'export my marked selections' path). Lineups whose players
    aren't all mapped to a DK ID are skipped."""
    def names_of(row):
        return [str(row[c]).rsplit(" (", 1)[0] for c in COLS]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(SLOT)
    chosen = skipped = 0
    for _, row in rows_df.iterrows():
        nms = names_of(row)
        if any(normname(n) not in dkid for n in nms):
            skipped += 1
            continue
        w.writerow([dkid[normname(n)] for n in nms])
        chosen += 1
    return out.getvalue(), {"chosen": chosen, "skipped_unmapped": skipped}


# --------------------------------------------------------------------------- #
# Freshness: rebuild projections / sims before a run when they're stale
# --------------------------------------------------------------------------- #
BUILD_STAMP = os.path.join(HERE, "out", ".build_stamp.json")


def read_build_stamp():
    if os.path.exists(BUILD_STAMP):
        try:
            return json.load(open(BUILD_STAMP))
        except Exception:
            return {}
    return {}


def write_build_stamp(**kw):
    """Record what was built and when, so a same-day re-run recognizes it as
    current regardless of file-timestamp quirks. The stamp file is replaced on
    every successful build."""
    data = read_build_stamp()
    data.update(kw)
    data["ts"] = time.time()
    try:
        os.makedirs(os.path.dirname(BUILD_STAMP), exist_ok=True)
        json.dump(data, open(BUILD_STAMP, "w"))
    except Exception:
        pass


def projections_built_date():
    """Date (YYYY-MM-DD) the projections were last built. Prefers the explicit
    build stamp the app writes; falls back to the projection files' mtime. The
    later of the two wins, so a fresh build is always recognized."""
    dates = []
    stamped = read_build_stamp().get("projections_date")
    if stamped:
        dates.append(stamped)
    fs = glob.glob(os.path.join(HERE, "out", "*pa_projections*.csv"))
    if fs:
        newest = max(os.path.getmtime(f) for f in fs)
        dates.append(datetime.date.fromtimestamp(newest).isoformat())
    return max(dates) if dates else None


def slate_change_signature(slate):
    """A comparable fingerprint of the slate that captures everything a refresh
    cares about: the game date, each team's batting order (lineups/players), and
    each team's starting pitcher (matchups). A change in any of these means the
    correlated sims are stale."""
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
            teams[tcode] = {"order": order, "sp": sp}
    return {"date": slate.get("date"), "teams": teams}


def diff_slate(stored_sig, live_sig):
    """Human-readable list of what changed between two slate signatures."""
    if not stored_sig:
        return ["no prior build recorded"]
    out = []
    if stored_sig.get("date") != live_sig.get("date"):
        out.append(f"game day {stored_sig.get('date')}→{live_sig.get('date')}")
    a, b = stored_sig.get("teams", {}), live_sig.get("teams", {})
    for t in sorted(set(a) | set(b)):
        ta, tb = a.get(t), b.get(t)
        if ta == tb:
            continue
        if ta and tb and ta.get("sp") != tb.get("sp"):
            out.append(f"{t} SP {ta.get('sp')}→{tb.get('sp')}")
        else:
            out.append(f"{t} lineup")
    return out


def sims_present():
    h, p = find_sims()
    return bool(h and p)


def load_stored_slate():
    p = os.path.join(HERE, "data", "slate.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def _tail(text, n=15):
    return "\n".join((text or "").strip().splitlines()[-n:])


def run_script(args, label, status):
    """Run a pipeline script as a subprocess. Returns (ok, combined_output);
    never raises, so callers decide how to react to a failure."""
    status.write(f"⏳ {label} …")
    try:
        p = subprocess.run([sys.executable] + args, cwd=HERE,
                           capture_output=True, text=True)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    if p.returncode == 0:
        status.write(f"✓ {label} complete.")
    return p.returncode == 0, out


# Stage B (projections) — run directly so a failure here never blocks Stage C.
PROJ_CMD = ["run_pipeline.py", "--target-year", "2027", "--bip-dir", "bip_inputs",
            "--skip-2026-scrape", "--output-dir", "out", "--force"]
# Heavy packages Stage B needs that the app itself doesn't import.
STAGE_B_DEPS = ["sklearn", "xgboost", "pybaseball", "pyarrow"]


def stage_b_missing_deps():
    """Names of Stage-B packages not installed for this interpreter (the common
    Python-3.14 failure: no scikit-learn/xgboost wheels yet)."""
    import importlib.util
    missing = []
    for mod in STAGE_B_DEPS:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except Exception:
            missing.append(mod)
    return missing


def _stamp_after_sims(today, live_sig):
    """Record the sim build: read the freshly written slate so the stored
    signature reflects exactly what the sims were built from."""
    fresh = load_stored_slate()
    fsig = slate_change_signature(fresh) or live_sig
    write_build_stamp(sims_date=today, slate_sig=fsig,
                      slate_date=(fresh or {}).get("date") or (live_sig or {}).get("date"))


def ensure_fresh(status, force=False):
    """Refresh only what's stale, with the daily SIM rebuild decoupled from the
    heavier projection rebuild. When `force` is set, rebuild both projections
    (bypassing the once-a-day attempt guard) and sims regardless of staleness.

      * Projections are rebuilt best-effort when stale — if that fails, the run
        continues on the existing projections (they change little day to day),
        and the attempt is not repeated again the same day.
      * The correlated SIMS are rebuilt from TODAY'S live slate (lineups,
        starting pitchers, Vegas game totals) whenever the slate moved, a new
        game day started, projections were rebuilt, or no sims exist — this is
        what actually pulls the new lineups/matchups/totals into the sims.

    Returns (notes, sims_changed, live_playable) where live_playable is the
    set of normnames eligible to be rostered today (lineup hitters + starting/
    opener/primary pitchers), or None if the live feed couldn't be read."""
    notes, sims_changed = [], False
    today = datetime.date.today().isoformat()
    # sync down the latest shared build before deciding anything
    if shared_store.enabled():
        try:
            shared_store.pull()
        except Exception:
            pass
    stamp = read_build_stamp()
    proj_date = projections_built_date()

    # --- live feed (lineups + matchups + starters), compared against the API ---
    live = live_sig = live_starters = live_playable = None
    try:
        status.write("Reading the live lineup/matchup feed…")
        live = slate_ingest.build_slate(write=False)
        live_sig = slate_change_signature(live)
        live_starters = {normname(tg["sp"]) for tg in live_sig["teams"].values()
                         if tg.get("sp")}
        # everyone eligible to be rostered today: hitters in a posted (confirmed
        # or expected) batting order + each game's starter/opener/primary pitcher
        live_playable = set(live_starters)
        for g in live.get("games", {}).values():
            for side in ("away", "home"):
                for pl in g.get("lineups", {}).get(side, []):
                    if pl.get("name"):
                        live_playable.add(normname(pl["name"]))
                for role in ("starter", "opener", "primary"):
                    nm = (g.get("pitchers", {}).get(side, {}) or {}).get(role)
                    if nm:
                        live_playable.add(normname(nm))
        n_lu = sum(1 for tg in live_sig["teams"].values() if tg["order"])
        status.write(f"Live feed: slate {live_sig.get('date')}, "
                     f"{len(live_sig['teams'])} teams, {n_lu} lineups posted, "
                     f"{len(live_starters)} starting pitchers, "
                     f"{len(live_playable)} rosterable players.")
    except Exception as e:
        notes.append(f"⚠️ Couldn't reach the live lineup feed ({type(e).__name__}); "
                     "using the existing sims. Check your network and re-run.")

    slate_day = live.get("date") if live else None
    stored_sig = stamp.get("slate_sig")
    if stored_sig is None and sims_present():
        base = load_stored_slate()
        if base:
            stored_sig = slate_change_signature(base)
            write_build_stamp(slate_sig=stored_sig, slate_date=base.get("date"))

    # serialize the heavy rebuild so concurrent users/instances don't all run it
    _lock = shared_store.RefreshLock()
    _lock.acquire()
    if not _lock.acquired:
        if shared_store.enabled():
            try:
                shared_store.pull()
            except Exception:
                pass
        notes.append("A refresh is already running (another user/session) — "
                     "using the latest shared sims.")
        return notes, True, live_playable

    try:
        # --- 1) projections: ensure present; refresh best-effort when stale ----
        projections_rebuilt = False
        missing_deps = stage_b_missing_deps()
        dep_msg = (f"Stage B can't run under this Python ({sys.version.split()[0]} at "
                   f"{sys.executable}) — these packages aren't importable: "
                   f"{', '.join(missing_deps)}. Install them with "
                   "`python -m pip install -r requirements.txt`. Note: on Python 3.14 "
                   "scikit-learn/xgboost may not have wheels yet — running the app on "
                   "Python 3.11–3.12 is the most reliable fix.")
        want_proj = force or proj_date is None or proj_date != today
        if want_proj:
            if missing_deps:
                if proj_date is None:
                    st.error("No projections exist and " + dep_msg)
                    notes.append("Projection deps missing; cannot build projections.")
                    return notes, False, live_playable
                if stamp.get("proj_warn_date") != today:
                    st.warning("Skipping the projection refresh — " + dep_msg +
                               f"\n\nContinuing with the existing projections from "
                               f"{proj_date}; the sims below are still rebuilt from "
                               "today's lineups.")
                    write_build_stamp(proj_warn_date=today)
                notes.append(f"⚠️ Stage B deps not importable ({', '.join(missing_deps)}); "
                             f"using existing projections from {proj_date}.")
            elif (not force and proj_date is not None and proj_date != today
                  and stamp.get("proj_attempt_date") == today):
                notes.append(f"Using existing projections from {proj_date} "
                             "(today's projection rebuild was already attempted — "
                             "tick “Force full refresh” to retry).")
            else:
                write_build_stamp(proj_attempt_date=today)
                label = "Projection build (Stage B)"
                status.write(f"{'Forcing a ' if force else ''}projection "
                             f"{'build' if proj_date is None else 'refresh'} (Stage B)…")
                ok, out = run_script(PROJ_CMD, label, status)
                if ok:
                    proj_date = today; projections_rebuilt = True
                    write_build_stamp(projections_date=today)
                    notes.append("Rebuilt projections.")
                elif proj_date is None:
                    st.error("No projections exist and the projection build failed — "
                             f"cannot continue.\n\n```\n{_tail(out)}\n```")
                    notes.append("Projection build failed.")
                    return notes, False, live_playable
                else:
                    st.warning("Projection rebuild (Stage B) failed — continuing with "
                               f"the existing projections from {proj_date}. The sims "
                               "below are still rebuilt from today's lineups.\n\n"
                               f"```\n{_tail(out)}\n```")
                    notes.append(f"⚠️ Projection rebuild failed; using existing "
                                 f"projections from {proj_date}.")
        elif stamp.get("projections_date") != today:
            write_build_stamp(projections_date=today)

        # --- 2) sims: rebuild from today's live slate whenever it moved --------
        new_game_day = bool(slate_day and stamp.get("slate_date")
                            and slate_day != stamp.get("slate_date"))
        lineups_changed = live_sig is not None and (stored_sig is None
                                                    or live_sig != stored_sig)
        need_sims = (force or not sims_present() or projections_rebuilt
                     or new_game_day or lineups_changed)
        if need_sims:
            if not sims_present() and live_sig is None:
                st.error("No sims on disk and the live feed is unreachable — can't "
                         "build sims. Check your network and re-run.")
                return notes, False, live_playable
            if not sims_present():
                why = "no sims on disk"
            elif new_game_day:
                why = f"new game day ({stamp.get('slate_date')}→{slate_day})"
            elif lineups_changed and stored_sig is not None:
                why = "lineup/matchup change: " + ", ".join(diff_slate(stored_sig, live_sig)[:8])
            elif projections_rebuilt:
                why = "rebuilt projections"
            elif force:
                why = "forced refresh"
            else:
                why = "first run"
            status.write("Rebuilding correlated sims from today's live slate "
                         f"(lineups + matchups + Vegas totals) — {why}…")
            ok, out = run_script(["run_slate.py"], "Correlated sims (Stage C)", status)
            if ok:
                _stamp_after_sims(today, live_sig)
                notes.append(f"Rebuilt correlated sims from today's slate — {why}.")
                sims_changed = True
            else:
                st.error("Sim rebuild (Stage C) failed — using existing sims.\n\n"
                         f"```\n{_tail(out)}\n```")
                notes.append("⚠️ Sim rebuild failed; using existing sims.")
        else:
            notes.append(f"No lineup/matchup changes since the last build "
                         f"(slate {stamp.get('slate_date') or slate_day}); "
                         "using the existing sims.")

        # publish the refreshed build so every user/instance shares it
        if shared_store.enabled() and (projections_rebuilt or sims_changed):
            try:
                with st.spinner("Publishing the refreshed build to the shared store…"):
                    shared_store.push()
                notes.append("Published the refreshed build to the shared store.")
            except Exception as e:
                notes.append(f"⚠️ Could not publish to the shared store "
                             f"({type(e).__name__}); it stays local to this instance.")
        return notes, sims_changed, live_playable
    finally:
        _lock.release()


# --------------------------------------------------------------------------- #
# Contest scoring that also captures each candidate's finishing-place
# distribution (compact per-candidate histogram + exact best/mean/worst)
# --------------------------------------------------------------------------- #
def run_contest_dist(field_mat, cand_mat, n_sim, n_field, max_cells=40_000_000):
    """Score each candidate against the field per sim and capture its
    finishing-place distribution at PER-POSITION resolution (one bin per place)
    when feasible. To bound memory the bin count adapts so N_candidates × bins
    stays under `max_cells`; for typical runs that still means one bin per place.
    Returns (wins, t10, t100, avg, dist)."""
    N = cand_mat.shape[1]
    wins = np.zeros(N, np.int64); t10 = np.zeros(N, np.int64)
    t100 = np.zeros(N, np.int64); ps = np.zeros(N, np.int64)
    best = np.full(N, n_field + 1, np.int64); worst = np.zeros(N, np.int64)

    # one bin per finishing place when feasible; cap at 6000 so the chart stays
    # renderable (and N×bins memory stays bounded) on very large fields
    bins = min(int(n_field), max(120, min(6000, max_cells // max(1, N))))
    if bins >= n_field:                      # one bin per finishing place
        edges = np.arange(1, n_field + 2, dtype=np.int64)
    else:
        edges = np.unique(np.linspace(1, n_field + 1, bins + 1).astype(np.int64))
    nb = len(edges) - 1
    cdt = np.int16 if n_sim < 32767 else np.int32
    counts = np.zeros((N, nb), cdt)
    idx = np.arange(N)
    for s in range(n_sim):
        fs = np.sort(field_mat[s]); cv = cand_mat[s]
        pl = (n_field - np.searchsorted(fs, cv, side="right")) + 1
        wins += (pl == 1); t10 += (pl <= 10); t100 += (pl <= 100); ps += pl
        best = np.minimum(best, pl); worst = np.maximum(worst, pl)
        b = np.clip(np.searchsorted(edges, pl, side="right") - 1, 0, nb - 1)
        np.add.at(counts, (idx, b), 1)
    dist = {"edges": edges, "counts": counts, "best": best, "worst": worst,
            "mean": ps / n_sim, "per_place": bool(bins >= n_field)}
    return wins, t10, t100, ps / n_sim, dist


def place_distribution_chart(dist, i, n_field, n_sim):
    """Per-finishing-position distribution for candidate i. Each x value is an
    actual finishing place (1 = win); the y value is the share of sims that
    finished at that place. Rendered as a faithful step so no detail is hidden
    by smoothing, and the x-axis is clamped to valid places (never below 1)."""
    edges = dist["edges"].astype(np.int64)
    counts = dist["counts"][i].astype(float)
    place = edges[:-1]                     # the (left) finishing place of each bin
    width = np.maximum(1, np.diff(edges))  # places covered by each bin (1 when per-place)
    pct = 100 * counts / max(1, n_sim) / width   # density: % of sims per place
    df = pd.DataFrame({"place": place, "pct": pct,
                       "sims": counts.astype(np.int64)})
    xmax = max(2, int(n_field))
    xscale = alt.Scale(domain=[1, xmax], nice=False, zero=False, clamp=True)
    per_place = dist.get("per_place", False)

    area = alt.Chart(df).mark_area(
        interpolate="step-after", color=BRAND, opacity=0.35,
        line={"color": BRAND, "strokeWidth": 1}).encode(
        x=alt.X("place:Q", title="Finishing place  (1 = win)", scale=xscale,
                axis=alt.Axis(format="~s")),
        y=alt.Y("pct:Q", title=("% of sims at each place" if per_place
                                else "% of sims (per place)")),
        tooltip=[alt.Tooltip("place:Q", title="place", format=",.0f"),
                 alt.Tooltip("sims:Q", title="sims here", format=","),
                 alt.Tooltip("pct:Q", title="% per place", format=".3f")])

    rules = []
    for x, label, color in [(1, "1st", "#000000"), (10, "Top-10", "#7a13c4"),
                            (100, "Top-100", "#c08bff"),
                            (float(dist["mean"][i]), "Mean", BRAND)]:
        if 1 <= x <= xmax:
            rdf = pd.DataFrame({"x": [x], "label": [label]})
            rules.append(alt.Chart(rdf).mark_rule(
                color=color, strokeDash=[4, 3], size=1.5, opacity=0.85).encode(
                x=alt.X("x:Q", scale=xscale),
                tooltip=[alt.Tooltip("label:N", title="marker"),
                         alt.Tooltip("x:Q", title="place", format=",.0f")]))
    return alt.layer(area, *rules).properties(height=220).configure_view(
        strokeOpacity=0)


def player_score_chart(arr, nbins=40):
    """Histogram of one player's DK-point outcomes across the sims."""
    a = np.asarray(arr, float)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, nbins + 1)
    cnt, _ = np.histogram(a, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    df = pd.DataFrame({"pts": centers, "pct": 100 * cnt / max(1, len(a))})
    mean = float(a.mean())
    bars = alt.Chart(df).mark_bar(color=BRAND, opacity=0.85).encode(
        x=alt.X("pts:Q", title="DK points", scale=alt.Scale(domain=[lo, hi],
                nice=False, zero=False)),
        y=alt.Y("pct:Q", title="% of sims"),
        tooltip=[alt.Tooltip("pts:Q", title="DK pts", format=".1f"),
                 alt.Tooltip("pct:Q", title="% of sims", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"x": [mean]})).mark_rule(
        color="#000000", strokeDash=[4, 3], size=1.5).encode(x="x:Q")
    return alt.layer(bars, rule).properties(height=220).configure_view(
        strokeOpacity=0)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
_t1, _t2 = st.columns([1, 9], vertical_alignment="center")
with _t1:
    render_logo(72)
_t2.title("DFS Contest Simulator")
st.caption(
    "Simulate DraftKings MLB contest outcomes for machine-developed candidate "
    "lineups, against an ownership-weighted field, using the day's correlated "
    "player sims. You provide the expected ownership; you choose the contest "
    "size, the number of sim runs, and how many candidate lineups to develop."
)

# ---- shared store: pull the latest shared sims/projections once per session ----
if shared_store.enabled() and not st.session_state.get("_shared_pulled"):
    try:
        with st.spinner("Syncing shared sims from the team store…"):
            shared_store.pull()
    except Exception as e:
        st.caption(f"(shared-store sync skipped: {type(e).__name__})")
    st.session_state["_shared_pulled"] = True

# ---- sim universe (from deliverables/) ----
hpath, ppath = find_sims()
if not hpath or not ppath:
    st.error(
        "No player sims found in `deliverables/`. Expected "
        "`hitter_dk_sims.npy` and `pitcher_dk_sims.npy`. Run Stage C "
        "(`python3 run_slate.py ...`) first."
    )
    st.stop()

with st.spinner("Loading player sims…"):
    H, P, score, n_sim = cached_sims(hpath, os.path.getmtime(hpath),
                                     ppath, os.path.getmtime(ppath))
params = cached_params(PARAMS_PATH, os.path.getmtime(PARAMS_PATH)
                       if os.path.exists(PARAMS_PATH) else 0)
if params is None:
    st.error("field_params.json not found — needed for the field stack model.")
    st.stop()

n_hit = len(H)
n_pit = len(P)
c1, c2, c3 = st.columns(3)
c1.metric("Hitters simmed", n_hit)
c2.metric("Pitchers simmed", n_pit)
c3.metric("Sims available", f"{n_sim:,}")

with st.expander("📊 Players — projected ranges & thresholds", expanded=False):
    st.caption("Per-player DK-point distribution from the current sims: projected "
               "mean, floor (p10) / median / ceiling (p90), min/max, std, and "
               "bust (≤0) / 2× / 30+ rates. Reflects the latest refreshed sims. "
               "These are also the players your CSV must cover.")
    ptable = cached_player_table(hpath, os.path.getmtime(hpath),
                                 ppath, os.path.getmtime(ppath))
    fc1, fc2 = st.columns([1, 3])
    ptype = fc1.selectbox("Show", ["All", "Hitters", "Pitchers"], index=0)
    psearch = fc2.text_input("Search player", "", placeholder="filter by name…")
    view = ptable
    if ptype != "All":
        view = view[view["Type"] == ptype[:-1]]   # "Hitters"->"Hitter"
    if psearch.strip():
        view = view[view["Player"].str.contains(psearch.strip(), case=False, na=False)]
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        view, use_container_width=True, height=420, hide_index=True,
        column_config={"Bust% (≤0)": pct, "2x%": pct, "30+%": pct})
    st.download_button("Download player thresholds (CSV)",
                       ptable.to_csv(index=False).encode(),
                       file_name="player_thresholds.csv", mime="text/csv")

    if len(view):
        psel = st.selectbox("Inspect a player's outcome distribution",
                            list(view["Player"]))
        arr = H.get(psel)
        if arr is None:
            arr = P.get(psel)
        if arr is not None:
            r = ptable[ptable["Player"] == psel].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Proj (mean)", f"{r['Proj']:.1f}")
            m2.metric("Floor (p10)", f"{r['Floor (p10)']:.1f}")
            m3.metric("Median", f"{r['Median']:.1f}")
            m4.metric("Ceiling (p90)", f"{r['Ceiling (p90)']:.1f}")
            st.altair_chart(player_score_chart(arr), use_container_width=True)
            st.caption(f"{psel}: min {r['Min']:.1f} · max {r['Max']:.1f} · "
                       f"std {r['Std']:.1f} · bust {r['Bust% (≤0)']:.0f}% · "
                       f"30+ {r['30+%']:.0f}%. Dashed line = mean.")

st.divider()

# --------------------------------------------------------------------------- #
# Step 1 — slate upload (outside the form so we can validate/preview)
# --------------------------------------------------------------------------- #
st.subheader("1 · Upload your slate file")
st.caption(
    "Upload either a **DraftKings salaries export** (the `DKSalaries.csv` with "
    "the player table — it already carries salary, position, team **and player "
    "IDs**, so it powers both the simulation and the upload export), or a "
    "**clean CSV** with columns " + ", ".join(f"`{c}`" for c in REQ_COLS) +
    " (add an `ID` column to enable the DK upload without a separate template). "
    "Ownership is the projected draft % (0–100).")
upload = st.file_uploader("Slate file", type=["csv"], label_visibility="collapsed")

dk_df = None
csv_ok = False
id_map = {}
id_col = None
if upload is not None:
    try:
        raw = upload.getvalue().decode("latin-1", "replace")
        export = parse_dk_export(raw)
        if export is not None:
            # raw DKSalaries export: has salary/pos/team/ID but no ownership
            base_df, id_map = export
            st.info(f"Detected a DraftKings salaries export — "
                    f"{len(base_df)} players, {len(id_map)} IDs captured for the "
                    "upload export. Now add ownership for these players.")
            own_up = st.file_uploader(
                "Ownership CSV (columns: FullName, Ownership)", type=["csv"],
                key="ownership_for_export")
            if own_up is None:
                st.warning("Upload an ownership CSV to continue.")
            else:
                own = pd.read_csv(own_up, encoding="latin-1")
                own.columns = [c.strip() for c in own.columns]
                if "FullName" not in own.columns or "Ownership" not in own.columns:
                    st.error("Ownership CSV needs `FullName` and `Ownership` columns.")
                else:
                    omap = {normname(r.FullName): float(r.Ownership)
                            for r in own.itertuples()
                            if str(r.Ownership).strip() not in ("", "nan")}
                    base_df["Ownership"] = base_df["FullName"].map(
                        lambda n: omap.get(normname(n)))
                    dk_df = base_df.dropna(subset=["Ownership"]).copy()
        else:
            # clean CSV — accept DK/own column-name variants via aliasing
            raw_df = pd.read_csv(io.StringIO(raw))
            raw_df.columns = [str(c).strip() for c in raw_df.columns]
            dk_df = alias_columns(raw_df)
            id_map, id_col = ids_from_clean(dk_df)
            missing = [c for c in REQ_COLS if c not in dk_df.columns]
            if "Ownership" in missing and len([m for m in missing if m != "Ownership"]) == 0:
                # everything but ownership is present (e.g. a DK salaries export)
                st.info("This looks like a salaries/slate file without ownership. "
                        "Add ownership for these players below.")
                own_up = st.file_uploader(
                    "Ownership CSV (columns: FullName, Ownership)", type=["csv"],
                    key="ownership_for_clean")
                if own_up is None:
                    st.warning("Upload an ownership CSV to continue.")
                    dk_df = None
                else:
                    own = alias_columns(pd.read_csv(own_up, encoding="latin-1"))
                    own.columns = [str(c).strip() for c in own.columns]
                    own = alias_columns(own)
                    if "FullName" not in own.columns or "Ownership" not in own.columns:
                        st.error("Ownership CSV needs `FullName` and `Ownership` columns.")
                        dk_df = None
                    else:
                        omap = {normname(r.FullName): float(r.Ownership)
                                for r in own.itertuples()
                                if str(r.Ownership).strip() not in ("", "nan")}
                        dk_df["Ownership"] = dk_df["FullName"].map(
                            lambda n: omap.get(normname(n)))
                        dk_df = dk_df.dropna(subset=["Ownership"]).copy()
            elif missing:
                st.error("CSV is missing required column(s): " + ", ".join(missing)
                         + f". Columns found: {', '.join(map(str, raw_df.columns))}")
                dk_df = None

        if dk_df is not None:
            simset = set(score)
            covered = int(dk_df["FullName"].map(lambda n: normname(n) in simset).sum())
            csv_ok = covered > 0
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Players in slate", len(dk_df))
            cc2.metric("Matched to sims", covered)
            cc3.metric("DK IDs available", len(id_map) if id_map else 0)
            if covered == 0:
                st.error("None of the players in this slate matched the sim "
                         "universe — check names/teams. Nothing to simulate.")
            else:
                st.dataframe(dk_df.head(15), use_container_width=True)
                if id_map:
                    src = f"column **{id_col}**" if id_col else "the slate file"
                    st.caption(f"✓ Player IDs detected (from {src}) — "
                               "the DK upload export will use them automatically.")
                else:
                    st.caption("No player IDs detected in this file. Columns found: "
                               f"`{', '.join(map(str, dk_df.columns))}`. To enable "
                               "the one-file DK export, include a player-ID column "
                               "(named e.g. `ID`, `Player ID`, `DK ID`, or a DK "
                               "`Name + ID` column); otherwise you can supply a "
                               "DKSalaries template at export time.")
    except Exception as e:
        st.error(f"Could not read slate file: {e}")

# --------------------------------------------------------------------------- #
# Step 2 — forced decisions, gated behind a Run button
# --------------------------------------------------------------------------- #
st.subheader("2 · Make your selections, then run")

with st.form("contest_form", clear_on_submit=False):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        size_sel = st.selectbox(
            "Contest size (field entries)", SIZE_PRESETS, index=None,
            placeholder="Choose a contest size…",
            help="How many entries the simulated field contains.")
        size_custom = st.number_input(
            "…or a custom size", min_value=2, max_value=2_000_000,
            value=None, step=100, placeholder="optional")
    with fc2:
        sim_runs = st.number_input(
            f"Number of sim runs (max {n_sim:,})", min_value=100,
            max_value=int(n_sim), value=None, step=500,
            placeholder="e.g. 10000",
            help="How many of the available correlated sims to score the "
                 "contest over. More = smoother estimates, slower.")
    with fc3:
        num_candidates = st.number_input(
            "Number of candidate lineups", min_value=10, max_value=200_000,
            value=None, step=500, placeholder="e.g. 5000",
            help="How many machine-built lineups to develop and evaluate.")

    with st.expander("Advanced field model (optional)"):
        a1, a2, a3 = st.columns(3)
        medium = a1.number_input("Medium baseline size", min_value=100,
                                 value=6000, step=500,
                                 help="Field size at which projected ownership "
                                      "is taken as-is (chalk neither sharpened "
                                      "nor flattened).")
        chalk = a2.number_input("Chalk sensitivity", min_value=0.0, max_value=2.0,
                                value=0.35, step=0.05,
                                help="How hard chalk concentrates in small "
                                     "fields / flattens in large ones.")
        tilt = a3.number_input("Stack-shape tilt", min_value=0.0, max_value=1.0,
                               value=0.15, step=0.05,
                               help="How much large fields consolidate onto "
                                    "5-man primary stacks.")
        s1, s2 = st.columns(2)
        seed_field = s1.number_input("Field seed", value=101, step=1)
        seed_cand = s2.number_input("Candidate seed", value=2025, step=1)
        talent_tilt = st.slider(
            "Candidate talent tilt (players)", min_value=0.0, max_value=2.0,
            value=0.7, step=0.1,
            help="How strongly candidate lineups favor higher-projected PLAYERS "
                 "(incl. one-off bats on any team) when filling stacks, one-off "
                 "slots, and pitchers. Applied as exp(tilt·z) on each player's "
                 "projected value, so it's a temperature: 0 = projection-blind "
                 "uniform; ~0.7 moderate; higher = sharply favor elite players.")
        team_tilt = st.slider(
            "Candidate stack-team tilt (Vegas/talent)", min_value=0.0,
            max_value=2.0, value=0.0, step=0.1,
            help="How strongly candidates STACK higher-projected TEAMS (team "
                 "scoring power from the sims, which embed Vegas/park/matchup). "
                 "0 (default) = every team equally likely to be stacked, so teams "
                 "stay diverse; higher = concentrate stacks on the best offenses. "
                 "Separate from the player tilt above.")

    force_refresh = st.checkbox(
        "Force full refresh (rebuild projections + sims now)", value=False,
        help="Rebuild projections (Stage B) and correlated sims (Stage C) "
             "regardless of staleness — bypasses the once-a-day retry guard. "
             "Use after fixing a data/connection issue.")
    submitted = st.form_submit_button("▶ Run simulation", type="primary",
                                      use_container_width=True)

contest_size = int(size_custom) if size_custom else (int(size_sel) if size_sel else None)

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if submitted:
    # ---- hard-gate on every decision ----
    errs = []
    if not csv_ok:
        errs.append("Upload a valid slate file that matches the sim universe (step 1).")
    if contest_size is None:
        errs.append("Choose a contest size.")
    if sim_runs is None:
        errs.append("Enter the number of sim runs.")
    if num_candidates is None:
        errs.append("Enter the number of candidate lineups.")
    if errs:
        for e in errs:
            st.warning(e)
        st.stop()

    if contest_size > 100_000 or num_candidates > 50_000:
        st.info("Large request — field/candidate construction and scoring may "
                "take a while and use significant memory.")

    t0 = time.time()
    with st.status("Running contest simulation…", expanded=True) as status:
        # ---- 0) freshness: rebuild projections / sims when stale ----
        notes, sims_changed, live_playable = ensure_fresh(status, force=force_refresh)
        for n in notes:
            st.write("• " + n)
        H_, P_, score_, n_sim_ = H, P, score, n_sim
        if sims_changed:
            hp_, pp_ = find_sims()
            if hp_ and pp_:
                H_, P_, score_, n_sim_ = cached_sims(
                    hp_, os.path.getmtime(hp_), pp_, os.path.getmtime(pp_))

        K = min(int(sim_runs), int(n_sim_))
        # sim index is aligned across players, so a prefix slice preserves the
        # correlation structure
        score_k = {k: v[:K] for k, v in score_.items()}
        simnames = set(score_k)
        if int(dk_df["FullName"].map(lambda n: normname(n) in simnames).sum()) == 0:
            status.update(label="No players match the sims", state="error")
            st.error("After the freshness check, none of your slate players "
                     "matched the sim universe. Check the slate file.")
            st.stop()

        # ---- build the pool (write CSV to a temp path for build_pool) ----
        st.write("Building player pool from your slate + sims…")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="") as tf:
            dk_df.to_csv(tf.name, index=False)
            tmp_csv = tf.name
        try:
            pool = build_pool(tmp_csv, H_, P_, score_k)
        finally:
            os.unlink(tmp_csv)

        # ---- hard guard: restrict the pool to today's PLAYABLE players only ----
        # Field + candidates may only use hitters in a projected/confirmed batting
        # order and the starting/opener/primary pitchers (per the live slate).
        # Anyone not in today's lineups is dropped even if the sims lag.
        if live_playable:
            keep = pool["Name"].map(lambda n: normname(n) in live_playable)
            drop_df = pool[~keep]
            if len(drop_df):
                dh = drop_df[drop_df.Pos != "P"].Name.nunique()
                dp = drop_df[drop_df.Pos == "P"].Name.nunique()
                pool = pool[keep].reset_index(drop=True)
                st.write(f"⛔ Restricted to projected/confirmed lineups + starting "
                         f"pitchers — excluded **{dh} hitter(s)** and **{dp} "
                         f"pitcher(s)** not in today's lineups"
                         + ("" if sims_changed else " (sims may be a build behind; "
                            "they'll catch up on the next refresh)") + ".")

        nh = pool[pool.Pos != "P"].Name.nunique()
        npi = pool[pool.Pos == "P"].Name.nunique()
        nt = pool.Team.nunique()
        st.write(f"Pool: **{nh} hitters + {npi} starters** across **{nt} teams** "
                 "— only players in today's projected/confirmed lineups (+ starting "
                 "pitchers) with sims. Ownership is renormalized over this pool for "
                 "the field. Both the field and candidate lineups use only these.")
        dropped = len(dk_df) - int(dk_df["FullName"].map(
            lambda n: normname(n) in simnames).sum())
        if dropped:
            st.caption(f"{dropped} player(s) in your slate had no sim and were "
                       "excluded (they can't be scored).")
        if npi < 2 or nh < 8:
            status.update(label="Pool too small to build lineups", state="error")
            st.error("Need at least 2 starting pitchers and 8 hitters in the "
                     "matched pool to fill a roster. Check your slate file.")
            st.stop()

        # ---- candidate lineups: players tilted to projected value (effective,
        #      via a z-score softmax); stack TEAMS uniform unless team tilt > 0 ----
        st.write(f"Developing {int(num_candidates):,} candidate lineups…")
        cdf = pool[(pool.Pos != "P") | (pool.Role == "SP")].copy()
        # projected value per player from the sims (mean blended with ceiling)
        tal = {}
        for nm in cdf["Name"].unique():
            a = score_k.get(normname(nm))
            if a is not None and len(a):
                tal[nm] = 0.5 * float(np.mean(a)) + 0.5 * float(np.percentile(a, 90))
        base = float(np.median(list(tal.values()))) if tal else 1.0

        def zmap(names):
            """z-score of talent within a player group (hitters or pitchers)."""
            vals = np.array([tal[n] for n in names if n in tal], float)
            if len(vals) == 0:
                return {}
            mu, sd = float(vals.mean()), float(vals.std()) + 1e-9
            return {n: (tal.get(n, mu) - mu) / sd for n in names}

        if talent_tilt > 0:
            # weight = exp(tilt · z); scale-invariant so `tilt` is a temperature.
            # z computed within hitters and within pitchers so each selection
            # context (intra-stack, one-off, pitcher) is calibrated on its own.
            zh = zmap(set(cdf[cdf["Pos"] != "P"]["Name"]))
            zp = zmap(set(cdf[cdf["Pos"] == "P"]["Name"]))
            cdf["Ownership"] = [
                float(np.exp(float(talent_tilt) *
                             (zp if r.Pos == "P" else zh).get(r.Name, 0.0)))
                for r in cdf.itertuples()]
        else:
            cdf["Ownership"] = 1.0   # projection-blind uniform players

        # stack-TEAM weights via a z-score softmax of team scoring power (sum of
        # hitters' talent): weight = exp(tilt · z). 0 (default) => uniform teams.
        team_weights = None
        if team_tilt > 0:
            hit = cdf[cdf["Pos"] != "P"]
            tteam = hit.groupby("Team")["Name"].apply(
                lambda s: sum(tal.get(n, base) for n in s)).to_dict()
            vals = np.array(list(tteam.values()), float)
            mu, sd = float(vals.mean()), float(vals.std()) + 1e-9
            team_weights = {t: float(np.exp(float(team_tilt) * (v - mu) / sd))
                            for t, v in tteam.items()}
        st.caption(f"Candidates: player talent tilt={talent_tilt:g}, "
                   f"stack-team tilt={team_tilt:g} "
                   f"({'teams favor better offenses' if team_tilt > 0 else 'teams uniform'}).")
        cb = Builder(Pool(cdf), params, seed=int(seed_cand), uniform=True,
                     team_weights=team_weights)
        cands, c_att = build_many(cb, int(num_candidates), "Candidates")
        if not cands:
            status.update(label="Could not build candidate lineups", state="error")
            st.error("Failed to construct any valid candidate lineup from this pool.")
            st.stop()
        cand_mat = score_matrix(cands, score_k, K)

        # ---- field for the chosen contest size ----
        st.write(f"Building an ownership-weighted field of {contest_size:,}…")
        beta = beta_for_size(contest_size, int(medium), float(chalk))
        fdf = adjust_ownership(normalize_to_slots(pool, 0.15), beta=beta)
        tilted = tilt_structures(
            [(tuple(s), w) for s, w in params["stack_structures"]],
            contest_size, int(medium), float(tilt))
        fp = dict(params)
        fp["stack_structures"] = [(list(s), w) for s, w in tilted]
        fb = Builder(Pool(fdf), fp, seed=int(seed_field), uniform=False)
        field, f_att = build_many(fb, contest_size, "Field")
        if len(field) < contest_size:
            st.warning(f"Built {len(field):,} of {contest_size:,} requested field "
                       "lineups (pool constrained); simulating against the field "
                       "that could be built.")
        field_mat = score_matrix(field, score_k, K)

        # ---- the contest (captures each candidate's finishing-place distro) ----
        st.write(f"Simulating the contest over {K:,} runs…")
        wins, t10, t100, avg, dist = run_contest_dist(
            field_mat, cand_mat, K, len(field))
        status.update(label=f"Done in {time.time()-t0:.1f}s", state="complete")

    # ---- per-lineup attributes for filtering/search ----
    own_map = {normname(rr.FullName): float(rr.Ownership) for rr in dk_df.itertuples()}
    cand_players = [frozenset(pl.Name for pl in lu["players"]) for lu in cands]
    prim_team, prim_size, own_sum = [], [], []
    for lu in cands:
        if lu["teams"]:
            pt, ps_ = max(lu["teams"].items(), key=lambda kv: kv[1])
        else:
            pt, ps_ = "", 0
        prim_team.append(pt); prim_size.append(int(ps_))
        own_sum.append(round(sum(own_map.get(normname(pl.Name), 0.0)
                                 for pl in lu["players"]), 1))

    # ---- results for the developed (candidate) lineups ----
    res = lineups_to_df(cands)
    res.insert(0, "Candidate", np.arange(1, len(cands) + 1))
    res["Wins"] = wins
    res["Win%"] = np.round(100 * wins / K, 3)
    res["Top10"] = t10
    res["Top10%"] = np.round(100 * t10 / K, 2)
    res["Top100"] = t100
    res["Top100%"] = np.round(100 * t100 / K, 2)
    res["AvgPlace"] = np.round(avg, 1)
    res["BestPlace"] = dist["best"]
    res["WorstPlace"] = dist["worst"]
    res["OwnSum"] = own_sum
    res["PrimaryTeam"] = prim_team
    res["PrimaryStack"] = prim_size
    res = res.sort_values(["Wins", "Top10", "Top100", "AvgPlace"],
                          ascending=[False, False, False, True]).reset_index(drop=True)

    # persist so the filter/export controls below can change without re-simulating
    st.session_state["sim"] = {
        "res": res, "cands": cands, "field_df": lineups_to_df(field),
        "K": K, "contest_size": contest_size, "field_n": len(field), "beta": beta,
        "dist": dist, "id_map": id_map,
        "cand_to_players": {i + 1: cand_players[i] for i in range(len(cands))},
        "pool_players": sorted({pl.Name for lu in cands for pl in lu["players"]}),
    }
    # fresh run -> clear prior marks / inspection state
    st.session_state["picked"] = set()
    st.session_state.pop("show_dist_for", None)


# --------------------------------------------------------------------------- #
# Results + DK upload  (rendered from session_state so widget tweaks below
# don't trigger a re-simulation)
# --------------------------------------------------------------------------- #
sim = st.session_state.get("sim")
if sim is None:
    st.info("Upload your slate file and make all three selections above, "
            "then press **Run simulation**.")
else:
    res = sim["res"]
    K = sim["K"]
    st.success(f"Simulated {len(sim['cands']):,} candidate lineups in a "
               f"{sim['field_n']:,}-entry field over {K:,} runs "
               f"(chalk β = {sim['beta']:.2f}).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Win%", f"{res['Win%'].max():.2f}%")
    m2.metric("Candidates that ever win", int((res["Wins"] > 0).sum()))
    m3.metric("Best Top100%", f"{res['Top100%'].max():.1f}%")
    m4.metric("Best AvgPlace", f"{res['AvgPlace'].min():,.0f}")

    res = res.copy()
    res["Rank"] = np.arange(1, len(res) + 1)

    st.subheader("Candidate lineups")

    # ---------------------- filter & search -------------------------------- #
    with st.expander("🔎 Filter & search lineups", expanded=False):
        f1, f2 = st.columns(2)
        players_sel = f1.multiselect("Must include player(s)", sim["pool_players"])
        match_mode = f1.radio("Player match", ["all", "any"], horizontal=True,
                              help="'all' = lineup contains every selected player; "
                                   "'any' = at least one.")
        shapes_sel = f2.multiselect("Stack shape (build style)",
                                    sorted(res["Stack"].unique()))
        teams_sel = f2.multiselect("Primary stack team",
                                   sorted(t for t in res["PrimaryTeam"].unique() if t))
        g1, g2, g3 = st.columns(3)
        psize_sel = g1.multiselect("Primary stack size",
                                   sorted(res["PrimaryStack"].unique(), reverse=True))

        def rng_slider(col, label, step, fmt):
            lo, hi = float(res[col].min()), float(res[col].max())
            if hi <= lo:
                return (lo, hi)
            return col_obj.slider(label, lo, hi, (lo, hi), step=step, format=fmt)

        col_obj = g2
        own_rng = rng_slider("OwnSum", "Combined ownership %", 0.5, "%.0f")
        col_obj = g3
        sal_rng = rng_slider("Salary", "Salary", 100.0, "$%d")
        h1, h2, h3 = st.columns(3)
        min_win = h1.number_input("Min Win%", 0.0, 100.0, 0.0, 0.1, format="%.2f")
        min_t10 = h2.number_input("Min Top10%", 0.0, 100.0, 0.0, 0.5)
        min_t100 = h3.number_input("Min Top100%", 0.0, 100.0, 0.0, 1.0)

    mask = pd.Series(True, index=res.index)
    if players_sel:
        want = set(players_sel)
        c2p = sim["cand_to_players"]
        mask &= res["Candidate"].map(
            lambda c: (want.issubset(c2p[int(c)]) if match_mode == "all"
                       else bool(want & c2p[int(c)])))
    if shapes_sel:
        mask &= res["Stack"].isin(shapes_sel)
    if teams_sel:
        mask &= res["PrimaryTeam"].isin(teams_sel)
    if psize_sel:
        mask &= res["PrimaryStack"].isin(psize_sel)
    mask &= res["OwnSum"].between(*own_rng)
    mask &= res["Salary"].between(*sal_rng)
    mask &= res["Win%"] >= min_win
    mask &= res["Top10%"] >= min_t10
    mask &= res["Top100%"] >= min_t100
    fres = res[mask].reset_index(drop=True)

    picked = st.session_state.setdefault("picked", set())
    c1, c2, c3 = st.columns([1.2, 1, 3])
    c1.caption(f"**{len(fres):,}** of {len(res):,} lineups match.")
    if c2.button("Mark all", help="Mark every lineup currently shown",
                 use_container_width=True):
        picked |= {int(x) for x in fres["Candidate"]}
    if c3.button("Clear marks", use_container_width=False):
        picked.clear()

    if len(fres) == 0:
        st.info("No lineups match these filters — loosen them.")
    else:
        # ---- player-focused lineups table (✓ to mark; metrics compact) ----
        slot_label = {"P1": "P", "P2": "P", "C": "C", "1B": "1B", "2B": "2B",
                      "3B": "3B", "SS": "SS", "OF1": "OF", "OF2": "OF", "OF3": "OF"}

        def _nm(v):
            return str(v).rsplit(" (", 1)[0]

        disp = pd.DataFrame({"✓": [int(x) in picked for x in fres["Candidate"]],
                             "Rank": fres["Rank"]})
        for c in COLS:
            disp[c] = fres[c].map(_nm)
        disp["Win%"] = fres["Win%"]
        disp["Top10%"] = fres["Top10%"]
        disp["Top100%"] = fres["Top100%"]
        disp["Salary"] = fres["Salary"]
        disp["Own%"] = fres["OwnSum"]
        disp["Stack"] = fres["Stack"]

        colcfg = {
            "✓": st.column_config.CheckboxColumn("✓", help="Mark for export",
                                                 width="small"),
            "Rank": st.column_config.NumberColumn(width="small"),
            "Win%": st.column_config.NumberColumn(format="%.2f%%", width="small"),
            "Top10%": st.column_config.NumberColumn(format="%.1f%%", width="small"),
            "Top100%": st.column_config.NumberColumn(format="%.1f%%", width="small"),
            "Salary": st.column_config.NumberColumn(format="$%d", width="small"),
            "Own%": st.column_config.NumberColumn(format="%.0f%%", width="small"),
            "Stack": st.column_config.TextColumn(width="small")}
        for c in COLS:
            colcfg[c] = st.column_config.TextColumn(slot_label[c])

        st.caption("Players are the focus — tick **✓** to mark lineups for "
                   "export. Win/Top-10/Top-100/own are compact on the right; "
                   "finishing-position detail is in the panel below.")
        edited = st.data_editor(
            disp, hide_index=True, height=460, use_container_width=True,
            disabled=[c for c in disp.columns if c != "✓"], column_config=colcfg,
            column_order=["✓", "Rank"] + COLS +
                         ["Win%", "Top10%", "Top100%", "Salary", "Own%", "Stack"])
        for cand_id, on in zip(fres["Candidate"], edited["✓"]):
            (picked.add if on else picked.discard)(int(cand_id))
        st.caption(f"☑️ **{len(picked):,}** lineup(s) marked for export.")

        d1, d2, d3 = st.columns(3)
        d1.download_button("Download filtered results (CSV)",
                           fres.to_csv(index=False).encode(),
                           file_name=f"candidate_results_{sim['contest_size']}.csv",
                           mime="text/csv", use_container_width=True)
        d2.download_button("Download all candidate lineups (CSV)",
                           lineups_to_df(sim["cands"]).to_csv(index=False).encode(),
                           file_name=f"candidates_{len(sim['cands'])}.csv",
                           mime="text/csv", use_container_width=True)
        d3.download_button("Download field (CSV)",
                           sim["field_df"].to_csv(index=False).encode(),
                           file_name=f"field_{sim['field_n']}.csv",
                           mime="text/csv", use_container_width=True)

        # ---- secondary: finishing-position detail (de-emphasized) ----
        with st.expander("📊 Finishing-position detail — pick a lineup",
                         expanded=False):
            labels = {int(c): f"Rank {int(rk)} · {stk} · Win {w:.2f}%"
                      for c, rk, stk, w in zip(fres["Candidate"], fres["Rank"],
                                               fres["Stack"], fres["Win%"])}
            chosen_cand = st.selectbox("Lineup", list(fres["Candidate"]),
                                       format_func=lambda c: labels[int(c)],
                                       label_visibility="collapsed")
            cand_idx = int(chosen_cand) - 1
            r = res[res["Candidate"] == chosen_cand].iloc[0]
            cc1, cc2 = st.columns([3, 2])
            with cc1:
                st.altair_chart(
                    place_distribution_chart(sim["dist"], cand_idx,
                                             sim["field_n"], K),
                    use_container_width=True)
            with cc2:
                st.caption(f"Rank #{int(r['Rank'])} · {r['Stack']} · "
                           f"${int(r['Salary']):,}")
                q1, q2, q3 = st.columns(3)
                q1.metric("Best", f"{int(r['BestPlace']):,}")
                q2.metric("Avg", f"{r['AvgPlace']:,.0f}")
                q3.metric("Worst", f"{int(r['WorstPlace']):,}")
                in_marks = int(chosen_cand) in picked
                if st.button(("☑️ Unmark" if in_marks else "⬜ Mark for export"),
                             use_container_width=True):
                    (picked.discard if in_marks else picked.add)(int(chosen_cand))
                    st.rerun()
            st.caption("Dashed lines mark 1st, Top-10, Top-100, and the lineup's "
                       "mean place. The x-axis covers valid places (1 … field).")

    # ----------------------------------------------------------------------- #
    # Step 3 — filled DraftKings upload file
    # ----------------------------------------------------------------------- #
    st.divider()
    st.subheader("3 · Build a DraftKings upload file")

    # IDs come from the slate file you already uploaded; a template is only a
    # fallback if that file carried no player IDs.
    dkid = dict(sim.get("id_map") or {})
    if dkid:
        st.caption(f"Using the {len(dkid)} player IDs from the slate file you "
                   "uploaded — no extra template needed.")
    else:
        st.caption("Your slate file had no player IDs, so upload a DKSalaries "
                   "template once to supply them (or re-upload a slate file that "
                   "includes IDs).")
        tmpl = st.file_uploader("DKSalaries template CSV", type=["csv"],
                                key="dk_template")
        if tmpl is not None:
            parsed = parse_dk_template(tmpl.getvalue().decode("utf-8", "replace"))
            if not parsed:
                st.error("Couldn't find the player table in that template — "
                         "expected a row whose 12th column is 'Position' with "
                         "Name/ID columns.")
            else:
                dkid = parsed

    mode = st.radio(
        "Which lineups to export?",
        [f"My marked selections ({len(picked)})", "Top N by ranking"],
        index=0 if picked else 1, horizontal=True)

    if not dkid:
        st.info("Player IDs are needed to build the upload file (see above).")
    elif mode.startswith("My marked"):
        if not picked:
            st.info("Mark some lineups above (tick the ✓ column), then export them here.")
        else:
            sel_df = res[res["Candidate"].isin(picked)]   # already in rank order
            csv_text, info = rows_to_upload_csv(sel_df, dkid)
            if info["chosen"] == 0:
                st.error("None of your marked lineups had a DK ID for every "
                         "player — check that the slate file's IDs cover these "
                         "players.")
            else:
                msg = f"Exporting **{info['chosen']}** marked lineup(s)."
                if info["skipped_unmapped"]:
                    msg += (f" ({info['skipped_unmapped']} skipped — a player had "
                            "no DK ID.)")
                st.success(msg)
                st.download_button(
                    "⬇ Download DraftKings upload CSV", csv_text.encode(),
                    file_name=f"DK_upload_marked_{info['chosen']}.csv",
                    mime="text/csv", type="primary", use_container_width=True)
    else:
        uc1, uc2 = st.columns(2)
        n_up = uc1.number_input("Number of lineups to export", min_value=1,
                                max_value=int(len(res)),
                                value=min(20, int(len(res))), step=1)
        sort_by = uc2.selectbox("Rank lineups by",
                                ["Win%", "Top10 Rate", "Top100 Rate"], index=0,
                                help="Win% favors tournament-winning ceiling; the "
                                     "Top10/Top100 rates favor consistent cashing.")
        with st.expander("Exposure caps (optional)"):
            pc1, pc2 = st.columns(2)
            player_cap = pc1.slider("Max player exposure", 0.05, 1.0, 1.0, 0.05,
                                    help="Cap the share of exported lineups any "
                                         "one player can appear in (1.0 = no cap).")
            team_cap = pc2.slider("Max primary-stack-team exposure", 0.05, 1.0,
                                  1.0, 0.05, help="Cap the share sharing the same "
                                                  "primary stack team.")
        csv_text, info = build_dk_upload(res, dkid, n_up, sort_by,
                                         player_cap, team_cap)
        if info["chosen"] == 0:
            st.error("No exportable lineups — players had no DK ID, or the caps "
                     "are too strict.")
        else:
            msg = (f"Selected **{info['chosen']} of {n_up}** lineups by "
                   f"**{sort_by}**. Max single-player exposure "
                   f"{info['max_player']}/{info['chosen']}, max stack-team "
                   f"{info['max_team']}/{info['chosen']}.")
            if info["skipped_unmapped"]:
                msg += (f" ({info['skipped_unmapped']} skipped: a player had no "
                        "DK ID.)")
            st.success(msg)
            st.download_button(
                "⬇ Download DraftKings upload CSV", csv_text.encode(),
                file_name=f"DK_upload_{info['chosen']}.csv",
                mime="text/csv", type="primary", use_container_width=True)

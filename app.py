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
from portfolio import select_portfolio, select_portfolio_ev, detect_value_groups
import portfolio_ev as pev
import dk_ids
from field_simulator import (normalize_to_slots, adjust_ownership,
                             beta_for_size, tilt_structures)
from stack_signal import team_stack_ownership, apply_stack_ownership_boost
import slate_ingest
import dk_slate_feed
import shared_store
import showdown_builder as sb
import showdown_contest as sc
import showdown_portfolio as sp
import showdown_upload as su

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

# Brand palette — navy / red
BRAND = "#F22E45"   # red accent
NAVY = "#002248"
INK = "#000D1A"     # near-black navy (page background)
PAPER = "#FFFFFF"


def _find_logo():
    # a user-provided raster logo wins over the bundled svg placeholder
    for ext in ("png", "jpg", "jpeg", "webp", "svg"):
        p = os.path.join(ASSETS, f"logo.{ext}")
        if os.path.exists(p):
            return p
    return None


LOGO = _find_logo()
LOCKUP = os.path.join(ASSETS, "rotowire_lockup.svg")   # RotoWire wordmark
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

# Persistent top-left RotoWire wordmark (Streamlit ≥1.35) + brand accents.
try:
    st.logo(LOCKUP if os.path.exists(LOCKUP) else LOGO)
except Exception:
    pass
st.markdown("""
<style>
  /* ---- RotoWire brand fonts (served from ./static) ---- */
  @font-face{font-family:'Integral CF';src:url('app/static/fonts/Integral-Heavy.otf') format('opentype'),url('app/static/fonts/Integral-Heavy.ttf') format('truetype');font-weight:700;font-display:swap;}
  @font-face{font-family:'Cosmica';src:url('app/static/fonts/Cosmica-Regular.otf') format('opentype');font-weight:400;font-display:swap;}
  @font-face{font-family:'Cosmica';src:url('app/static/fonts/Cosmica-Semibold.otf') format('opentype');font-weight:600;font-display:swap;}
  @font-face{font-family:'Cosmica';src:url('app/static/fonts/Cosmica-Heavy.otf') format('opentype');font-weight:700;font-display:swap;}
  @font-face{font-family:'Cosmica Mono';src:url('app/static/fonts/CosmicaMono-Semibold.otf') format('opentype');font-weight:600;font-display:swap;}

  :root{
    --rw-red:#f22e45; --rw-red-400:#f5566a; --rw-red-700:#c21e31;
    --rw-navy:#002248;
    --rw-ink:#000d1a; --rw-surface:#002248; --rw-raised:#083363; --rw-line:#1c4a7a;
    --rw-mut:#8ba0ba; --rw-turf:#00e657; --rw-ketchup:#ff4537;
    --font-body:'Cosmica',ui-sans-serif,system-ui,'Segoe UI',sans-serif;
    --font-display:'Integral CF','Impact',system-ui,sans-serif;
    --font-mono:'Cosmica Mono',ui-monospace,Menlo,Consolas,monospace;
  }

  html, body, .stApp, [data-testid="stAppViewContainer"],
  p, span, div, label, input, textarea, button, select, li, td, th {
    font-family: var(--font-body);
  }
  .stApp { background: var(--rw-ink); }

  /* Display headings — Integral CF, uppercase */
  h1, h2, h3, [data-testid="stHeading"] h1,
  [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
    font-family: var(--font-display) !important;
    text-transform: uppercase; letter-spacing: .02em; color: #fff;
  }

  /* Stat / metric cards */
  [data-testid="stMetric"]{
    background: var(--rw-surface); border:1px solid var(--rw-line);
    border-radius:12px; padding:14px 16px;
  }
  [data-testid="stMetricLabel"] p{
    font-family: var(--font-mono); text-transform:uppercase;
    letter-spacing:.06em; font-size:10px !important; color: var(--rw-mut);
  }
  [data-testid="stMetricValue"]{
    font-family: var(--font-display); color:#fff; font-size:30px;
  }

  /* Tabs — RotoWire purple active underline */
  .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--rw-line); }
  .stTabs [data-baseweb="tab"]{
    font-family: var(--font-display); text-transform:uppercase;
    letter-spacing:.04em; font-size:13px; color: var(--rw-mut); padding:6px 14px;
  }
  .stTabs [aria-selected="true"]{ color:#fff !important; }
  .stTabs [data-baseweb="tab-highlight"]{ background-color: var(--rw-red) !important; height:3px; }

  /* Buttons — primary purple, square-ish RW radius */
  .stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"]>button{
    font-family: var(--font-body); font-weight:600; border-radius:8px;
    border:1px solid var(--rw-line);
  }
  [data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"]>button{
    background: var(--rw-red) !important; border-color: var(--rw-red) !important; color:#fff !important;
  }
  [data-testid="stBaseButton-primary"]:hover, [data-testid="stFormSubmitButton"]>button:hover{
    background: var(--rw-red-400) !important; border-color: var(--rw-red-400) !important;
  }

  /* Surfaces: expanders, inputs, dataframes */
  [data-testid="stExpander"]{ background: var(--rw-surface); border:1px solid var(--rw-line); border-radius:12px; }
  [data-baseweb="input"], [data-baseweb="select"]>div, .stTextInput input, .stNumberInput input{
    background: var(--rw-raised) !important; border-radius:8px !important;
  }
  [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"]{ border:1px solid var(--rw-line); border-radius:10px; }
  .stProgress > div > div > div > div { background-color: var(--rw-red); }
  a, a:visited { color: var(--rw-red-400); }
  hr { border-top:1px solid var(--rw-line); }
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:#1c4a7a;border-radius:8px}
  ::-webkit-scrollbar-track{background:transparent}

  /* RotoWire branded header */
  .rw-header{display:flex;align-items:center;gap:16px;padding:14px 18px;margin:2px 0 6px;
    background:var(--rw-surface);border:1px solid var(--rw-line);border-radius:14px;}
  .rw-header .rw-logo{width:42px;height:42px;flex-shrink:0;color:var(--rw-red);}
  .rw-header .rw-logo svg{width:100%;height:auto;display:block;}
  .rw-header .rw-wordmark{height:30px;flex-shrink:0;display:flex;align-items:center;}
  .rw-header .rw-wordmark svg{height:30px;width:auto;display:block;}
  .rw-header .rw-divider{width:1px;height:34px;background:var(--rw-line);flex-shrink:0;}
  .rw-title{font-family:var(--font-display);text-transform:uppercase;letter-spacing:.02em;
    font-size:26px;line-height:1;color:#fff;}
  .rw-eyebrow{font-family:var(--font-mono);text-transform:uppercase;letter-spacing:.08em;
    font-size:10px;color:var(--rw-mut);margin-top:4px;}
  .rw-badge{margin-left:auto;font-family:var(--font-mono);font-weight:600;font-size:11px;
    text-transform:uppercase;letter-spacing:.04em;background:var(--rw-red);color:#fff;
    padding:6px 12px;border-radius:9999px;display:inline-flex;align-items:center;gap:7px;}
  .rw-badge .dot{width:7px;height:7px;border-radius:9999px;background:#fff;
    display:inline-block;animation:rwspin 2s linear infinite;}
  @keyframes rwspin{to{transform:rotate(360deg)}}
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


@st.cache_data(ttl=600, show_spinner=False)
def _load_slate_catalog():
    """Today's pickable DraftKings slates from the RotoWire feeds (salaries +
    ownership), cached for 10 min. Returns dk_slate_feed.build_catalog()."""
    return dk_slate_feed.build_catalog()


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
            # col 11 = Position, 13 = Name, 14 = ID, 16 = Salary, 18 = TeamAbbrev.
            # Keyed with team/pos/salary so same-named players stay distinct.
            dk_ids.add_id(dkid, r[13], r[18], r[14], pos=r[11], salary=r[16])
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
            dk_ids.add_id(idmap, nm, r[18], r[14], pos=r[11], salary=sal)
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

    has_team = "Team" in df.columns
    has_pos = "Position" in df.columns
    has_sal = "Salary" in df.columns

    def harvest(orig):
        m = {}
        for _, r in df.iterrows():
            cid = _clean_id(r[orig])
            if cid:
                # key by team/pos/salary so two same-named players stay distinct
                dk_ids.add_id(m, r["FullName"], r["Team"] if has_team else "", cid,
                              pos=r["Position"] if has_pos else "",
                              salary=r["Salary"] if has_sal else None)
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


def _split_cell(cell):
    """A result cell is ``"Name (TEAM)"``; return (name, team)."""
    s = str(cell)
    if s.endswith(")") and " (" in s:
        nm, tm = s.rsplit(" (", 1)
        return nm, tm[:-1]
    return s, ""


def _players_to_ids(players, dkid):
    """Resolve the DK id for each player OBJECT of a lineup (in slot order),
    matching on salary/team/position so two same-named players (e.g. the two Max
    Muncys) get their own id. Returns the id list, or None if any player has no
    id at all. Using the pool player objects (not the display cell) gives the
    exact salary, which is the reliable discriminator."""
    ids = []
    for pl in players:
        cid = dk_ids.lookup(dkid, pl.Name, getattr(pl, "Team", ""),
                            pos=getattr(pl, "Pos", ""),
                            salary=getattr(pl, "Salary", None))
        if cid is None:
            return None
        ids.append(cid)
    return ids


def _row_players(row, cands):
    """The player objects for a result row, via its Candidate index into `cands`
    (falls back to None when unavailable)."""
    try:
        return cands[int(row["Candidate"]) - 1]["players"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _lineup_ids(row, dkid, cands=None):
    """DK ids for a result row. Prefers the candidate's player objects (exact
    salary/team/pos); falls back to parsing name+team from the display cells."""
    players = _row_players(row, cands) if cands is not None else None
    if players is not None:
        return _players_to_ids(players, dkid)
    ids = []
    for c in COLS:
        nm, tm = _split_cell(row[c])
        cid = dk_ids.lookup(dkid, nm, tm)
        if cid is None:
            return None
        ids.append(cid)
    return ids


def build_dk_upload(res_df, dkid, n_select, sort_by, hitter_cap=1.0,
                    pitcher_cap=1.0, team_cap=1.0, pair_cap=1.0,
                    core_cap=1.0, max_overlap=1.0, group_of=None,
                    group_cap=1.0, player_caps=None, team_caps=None,
                    player_mins=None, team_mins=None, cands=None):
    """Pick n_select lineups from the ranked candidate results with the
    diversity-aware portfolio selector (per-player / stack-team / pairing /
    stack-core / value-group exposure caps + an overlap ceiling), map players to
    DK IDs, and emit a ready-to-upload CSV (P,P,C,1B,2B,3B,SS,OF,OF,OF + one ID
    row each). COLS[0:2] are the two pitcher slots; COLS[2:] are the 8 hitters."""
    keymap = {"Win%": ["Wins", "Top10", "Top100"],
              "Top10 Rate": ["Top10", "Top100", "Wins"],
              "Top100 Rate": ["Top100", "Top10", "Wins"]}[sort_by]

    def eligible(nms):
        return all(dk_ids.has_name(dkid, n) for n in nms)

    chosen, info = select_portfolio(
        res_df, n_select, keymap, cols=COLS, hitc=HITC, eligible=eligible,
        hitter_cap=hitter_cap, pitcher_cap=pitcher_cap, team_cap=team_cap,
        pair_cap=pair_cap, core_cap=core_cap, max_overlap=max_overlap,
        group_of=group_of, group_cap=group_cap,
        player_caps=player_caps, team_caps=team_caps,
        player_mins=player_mins, team_mins=team_mins)

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(SLOT)
    written = 0
    for row in chosen:
        ids = _lineup_ids(row, dkid, cands)   # salary/team/pos-aware id resolution
        if ids is None:
            continue
        w.writerow(ids)
        written += 1
    info["skipped_unmapped"] += info["chosen"] - written
    info["chosen"] = written
    return out.getvalue(), info


def build_dk_upload_ev(src, sim, dkid, n_select, *, entry_fee, pct_paid, rake,
                       top_heaviness, risk, shortlist, hitter_cap=1.0,
                       pitcher_cap=1.0, team_cap=1.0, pair_cap=1.0, core_cap=1.0,
                       max_overlap=1.0, group_of=None, group_cap=1.0,
                       player_caps=None, team_caps=None,
                       player_mins=None, team_mins=None, eval_sims=4000):
    """Payout-aware portfolio export. Instead of ranking each lineup by its
    standalone finish rate, this rebuilds the candidates' correlated per-sim
    scores, turns them into DOLLAR payouts against a parametric top-heavy curve,
    and greedily selects the set that maximizes the expected UTILITY of the whole
    portfolio's per-slate return (the `risk` posture picks the utility). All the
    same exposure / diversity caps still apply as hard constraints.

    Returns ``(csv_text, info, W, W_naive, extra)`` where `W` / `W_naive` are the
    per-sim portfolio returns for the EV set and a rank-selected set of the same
    size (for the coverage visualization), or ``None`` if the sim predates the
    stored placement ladder (caller falls back to ranked export)."""
    dist = sim.get("dist") or {}
    score_pool = sim.get("score_pool")
    if "field_cut_scores" not in dist or not score_pool:
        return None

    K = int(sim["K"])
    field_n = int(sim["field_n"])
    cands = sim["cands"]
    cut_scores = dist["field_cut_scores"]        # (K, n_cut)
    cut_places = dist["cut_places"]

    # shortlist: strongest candidates by the existing ranking (src is already
    # sorted) so the pay matrix stays small; the optimizer picks a decorrelated
    # subset from within it.
    short = src.head(int(shortlist)).reset_index(drop=True)
    M = len(short)
    if M == 0:
        return "", {"chosen": 0, "skipped_unmapped": 0}, None, None, {}

    # rebuild each shortlisted candidate's per-sim score from the pool arrays
    # (the Candidate column indexes back into the cands list).
    cand_scores = np.zeros((K, M), np.float32)
    for j, cid in enumerate(short["Candidate"].to_numpy()):
        lu = cands[int(cid) - 1]
        for pl in lu["players"]:
            arr = score_pool.get(normname(pl.Name))
            if arr is not None:
                cand_scores[:, j] += arr

    # payout curve uses the SIMULATED contest size, so a finishing place (relative
    # to that field) maps coherently onto the prize table.
    prize = pev.make_payout_curve(field_n, entry_fee, top_heaviness=top_heaviness,
                                  pct_paid=pct_paid, rake=rake)
    pay = pev.candidate_payout_matrix(cand_scores, cut_scores, cut_places, prize)

    def eligible(nms):
        return all(dk_ids.has_name(dkid, n) for n in nms)

    chosen, info, W = select_portfolio_ev(
        short, n_select, pay, pev.utility(risk), cols=COLS, hitc=HITC,
        eligible=eligible, hitter_cap=hitter_cap, pitcher_cap=pitcher_cap,
        team_cap=team_cap, pair_cap=pair_cap, core_cap=core_cap,
        max_overlap=max_overlap, group_of=group_of, group_cap=group_cap,
        player_caps=player_caps, team_caps=team_caps,
        player_mins=player_mins, team_mins=team_mins, eval_sims=eval_sims)

    # rank-selected baseline of the SAME size (eligible only) for the comparison
    naive_pos = []
    for i in range(M):
        nms = [str(short.iloc[i][c]).rsplit(" (", 1)[0] for c in COLS]
        if eligible(nms):
            naive_pos.append(i)
        if len(naive_pos) >= info["chosen"]:
            break
    W_naive = (pay[:, naive_pos].sum(axis=1) if naive_pos
               else np.zeros(K, np.float64))

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(SLOT)
    written = 0
    for row in chosen:
        ids = _lineup_ids(row, dkid, cands)   # salary/team/pos-aware id resolution
        if ids is None:
            continue
        w.writerow(ids)
        written += 1
    info["skipped_unmapped"] += info["chosen"] - written
    info["chosen"] = written

    extra = {"prize_summary": pev.payout_curve_summary(prize, entry_fee),
             "cost": info["chosen"] * float(entry_fee), "shortlist": M,
             "field_n": field_n}
    return out.getvalue(), info, W, W_naive, extra


def portfolio_return_chart(W, W_naive):
    """Overlaid histograms of per-slate portfolio $ return: the payout-aware set
    vs a rank-selected set of the same size. A ranked set piles its winning sims
    together (tall bar at $0, thin far-right tail); the EV set shifts mass off $0
    into the mid-range — that's the boom/bust being broken up."""
    hi = float(max(np.percentile(W, 99), np.percentile(W_naive, 99), 1.0))
    bins = np.linspace(0.0, hi, 31)

    def hist(w, label):
        c, _ = np.histogram(np.clip(w, 0, hi), bins=bins)
        return pd.DataFrame({"lo": bins[:-1], "hi": bins[1:],
                             "pct": 100.0 * c / max(1, len(w)), "which": label})

    df = pd.concat([hist(np.asarray(W), "Portfolio EV"),
                    hist(np.asarray(W_naive), "Ranked top-N")], ignore_index=True)
    return alt.Chart(df).mark_bar(opacity=0.55).encode(
        x=alt.X("lo:Q", title="Portfolio return ($ per simulated slate)",
                scale=alt.Scale(domain=[0, hi], nice=False, clamp=True)),
        x2="hi:Q",
        y=alt.Y("pct:Q", title="% of simulated slates",
                stack=None, scale=alt.Scale(zero=True)),
        color=alt.Color("which:N", title=None,
                        scale=alt.Scale(domain=["Portfolio EV", "Ranked top-N"],
                                        range=[BRAND, "#8a8a8a"])),
        tooltip=[alt.Tooltip("which:N", title="set"),
                 alt.Tooltip("lo:Q", title="$ from", format=",.0f"),
                 alt.Tooltip("hi:Q", title="$ to", format=",.0f"),
                 alt.Tooltip("pct:Q", title="% slates", format=".1f")]
    ).properties(height=240)


def rows_to_upload_csv(rows_df, dkid, cands=None):
    """Emit a DK upload CSV for an explicit, already-ordered set of lineup rows
    (used by the 'export my marked selections' path). Lineups whose players
    aren't all mapped to a DK ID are skipped. Each player's id is resolved with
    its salary/team/position so two same-named players never get swapped."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(SLOT)
    chosen = skipped = 0
    for _, row in rows_df.iterrows():
        ids = _lineup_ids(row, dkid, cands)
        if ids is None:
            skipped += 1
            continue
        w.writerow(ids)
        chosen += 1
    return out.getvalue(), {"chosen": chosen, "skipped_unmapped": skipped}


# --------------------------------------------------------------------------- #
# Showdown (Captain Mode): build the contest, render Results + Export.
# All of this runs only for showdown-format slates; the classic paths above and
# below are untouched. Backend lives in showdown_builder / showdown_contest /
# showdown_portfolio / showdown_upload.
# --------------------------------------------------------------------------- #
def run_showdown_sim(dk_df, score_k, K, contest_size, id_map, num_candidates,
                     seed_cand, seed_field, medium, chalk, cand_jitter, status):
    """Build a showdown contest (1 CPT + 5 UTIL) from the slate + sims and return
    the session-state dict the Results/Export tabs render from. Reuses the
    format-agnostic run_contest_dist (placement ladder + finishing distribution)
    so the payout-aware export works for showdown too."""
    pool = sc.build_pool(dk_df, score_k, normfn=normname)   # raises unless 2 teams
    teams = sorted(pool["Team"].unique())
    st.write(f"Showdown pool: **{len(pool)} players** — {teams[0]} vs {teams[1]}.")
    if len(pool) < sb.ROSTER_SIZE:
        raise RuntimeError(f"need at least {sb.ROSTER_SIZE} simmed players for a "
                           "showdown roster")

    st.write(f"Developing {int(num_candidates):,} candidate lineups…")
    cb = sb.Builder(sb.Pool(pool), seed=int(seed_cand), uniform=True,
                    jitter=float(cand_jitter))
    cands, _ = build_many(cb, int(num_candidates), "Candidates")
    if not cands:
        raise RuntimeError("could not build any showdown candidate lineup")

    beta = beta_for_size(int(contest_size), int(medium), float(chalk))
    st.write(f"Building an ownership-weighted field of {int(contest_size):,} "
             f"(chalk β={beta:.2f}, captain ceiling-tilted)…")
    fp = sc._field_pool(pool, score_k, normname, sc.DEFAULT_CPT_CEILING_TILT)
    fb = sb.Builder(sb.Pool(fp), {"cpt_chalk": beta, "util_chalk": beta},
                    seed=int(seed_field), uniform=False)
    field, _ = build_many(fb, int(contest_size), "Field")
    if not field:
        raise RuntimeError("could not build a showdown field")
    if len(field) < int(contest_size):
        st.warning(f"Built {len(field):,} of {int(contest_size):,} field lineups "
                   "(pool constrained); simulating against what could be built.")

    cand_mat = sb.score_matrix(cands, score_k, K, norm=normname)
    field_mat = sb.score_matrix(field, score_k, K, norm=normname)

    st.write(f"Simulating the contest over {K:,} runs…")
    cut_places = pev.field_place_cutpoints(len(field))
    wins, t10, t100, avg, dist = run_contest_dist(
        field_mat, cand_mat, K, len(field), cut_places=cut_places)

    own_map = {normname(r.FullName): float(r.Ownership) for r in dk_df.itertuples()}
    res = sb.lineups_to_df(cands)
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
    res["Captain"] = [lu["captain"].Name for lu in cands]
    res["CptTeam"] = [lu["captain"].Team for lu in cands]
    res["Split"] = ['-'.join(map(str, sorted(lu["teams"].values(), reverse=True)))
                    for lu in cands]
    res["OwnSum"] = [round(sum(own_map.get(normname(pl.Name), 0.0)
                               for pl in lu["players"]), 1) for lu in cands]
    res = res.sort_values(["Wins", "Top10", "Top100", "AvgPlace"],
                          ascending=[False, False, False, True]).reset_index(drop=True)

    pool_norm = {normname(n) for n in pool["Name"].unique()}
    score_pool = {k: np.asarray(v, np.float32)
                  for k, v in score_k.items() if k in pool_norm}

    tal, players_meta = {}, {}
    for nm in pool["Name"]:
        a = score_k.get(normname(nm))
        if a is not None and len(a):
            tal[nm] = 0.5 * float(np.mean(a)) + 0.5 * float(np.percentile(a, 90))
    for r in pool.itertuples():
        if r.Name not in players_meta:
            players_meta[r.Name] = {"pos": r.Pos, "salary": int(r.Salary),
                                    "team": r.Team, "proj": tal.get(r.Name)}

    return {
        "format": "showdown",
        "res": res, "cands": cands, "field": field,
        "field_df": sb.lineups_to_df(field),
        "K": K, "contest_size": int(contest_size), "field_n": len(field),
        "beta": beta, "dist": dist, "id_map": id_map, "score_pool": score_pool,
        "cand_to_players": {i + 1: frozenset(pl.Name for pl in lu["players"])
                            for i, lu in enumerate(cands)},
        "pool_players": sorted({pl.Name for lu in cands for pl in lu["players"]}),
        "players_meta": players_meta,
        "captains": sorted({lu["captain"].Name for lu in cands}),
        "teams": teams,
    }


def _sd_cand_scores(short, sim):
    """Rebuild each shortlisted showdown candidate's per-sim score (captain 1.5x)
    from the pool arrays — the payout-aware export path."""
    score_pool = sim["score_pool"]
    cands = sim["cands"]
    K = int(sim["K"])
    M = len(short)
    mat = np.zeros((K, M), np.float32)
    for j, cid in enumerate(short["Candidate"].to_numpy()):
        lu = cands[int(cid) - 1]
        for i, pl in enumerate(lu["players"]):
            arr = score_pool.get(normname(pl.Name))
            if arr is not None:
                mat[:, j] += (sb.CPT_MULT * arr) if i == 0 else arr
    return mat


def render_showdown_results(sim):
    """Results tab for showdown: metrics, captain/team-split filters, a CPT/UTIL
    lineup table with ✓-to-mark, downloads, and finishing-position detail."""
    res = sim["res"]
    K = sim["K"]
    st.success(f"Simulated {len(sim['cands']):,} showdown candidate lineups in a "
               f"{sim['field_n']:,}-entry field over {K:,} runs "
               f"(chalk β = {sim['beta']:.2f}). "
               f"Game: {sim['teams'][0]} vs {sim['teams'][1]}.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Win%", f"{res['Win%'].max():.2f}%")
    m2.metric("Candidates that ever win", int((res["Wins"] > 0).sum()))
    m3.metric("Best Top100%", f"{res['Top100%'].max():.1f}%")
    m4.metric("Distinct captains", res["Captain"].nunique())

    res = res.copy()
    res["Rank"] = np.arange(1, len(res) + 1)
    st.subheader("Candidate lineups")

    with st.expander("🔎 Filter & search lineups", expanded=False):
        f1, f2 = st.columns(2)
        players_sel = f1.multiselect("Must include player(s)", sim["pool_players"])
        match_mode = f1.radio("Player match", ["all", "any"], horizontal=True)
        exclude_sel = f1.multiselect("Exclude player(s)", sim["pool_players"])
        cpt_sel = f2.multiselect("Captain is", sim["captains"])
        split_sel = f2.multiselect("Team split", sorted(res["Split"].unique()))
        g1, g2 = st.columns(2)

        def rng_slider(col_obj, col, label, step, fmt):
            lo, hi = float(res[col].min()), float(res[col].max())
            if hi <= lo:
                return (lo, hi)
            return col_obj.slider(label, lo, hi, (lo, hi), step=step, format=fmt)

        own_rng = rng_slider(g1, "OwnSum", "Combined ownership %", 0.5, "%.0f")
        sal_rng = rng_slider(g2, "Salary", "Salary", 100.0, "$%d")
        h1, h2, h3 = st.columns(3)
        min_win = h1.number_input("Min Win%", 0.0, 100.0, 0.0, 0.1, format="%.2f")
        min_t10 = h2.number_input("Min Top10%", 0.0, 100.0, 0.0, 0.5)
        min_t100 = h3.number_input("Min Top100%", 0.0, 100.0, 0.0, 1.0)

    mask = pd.Series(True, index=res.index)
    c2p = sim["cand_to_players"]
    if players_sel:
        want = set(players_sel)
        mask &= res["Candidate"].map(
            lambda c: (want.issubset(c2p[int(c)]) if match_mode == "all"
                       else bool(want & c2p[int(c)])))
    if exclude_sel:
        avoid = set(exclude_sel)
        mask &= res["Candidate"].map(lambda c: not (avoid & c2p[int(c)]))
    if cpt_sel:
        mask &= res["Captain"].isin(cpt_sel)
    if split_sel:
        mask &= res["Split"].isin(split_sel)
    mask &= res["OwnSum"].between(*own_rng)
    mask &= res["Salary"].between(*sal_rng)
    mask &= res["Win%"] >= min_win
    mask &= res["Top10%"] >= min_t10
    mask &= res["Top100%"] >= min_t100
    fres = res[mask].reset_index(drop=True)
    st.session_state["_sd_fres_ids"] = list(fres["Candidate"])

    picked = st.session_state.setdefault("picked", set())
    c1, c2, c3 = st.columns([1.2, 1, 3])
    c1.caption(f"**{len(fres):,}** of {len(res):,} lineups match.")
    if c2.button("Mark all", width="stretch"):
        picked |= {int(x) for x in fres["Candidate"]}
    if c3.button("Clear marks", width="content"):
        picked.clear()

    if len(fres) == 0:
        st.info("No lineups match these filters — loosen them.")
        return

    def _nm(v):
        return str(v).rsplit(" (", 1)[0]

    disp = pd.DataFrame({"✓": [int(x) in picked for x in fres["Candidate"]],
                         "Rank": fres["Rank"]})
    for c in sb.SD_COLS:
        disp[c] = fres[c].map(_nm)
    disp["Win%"] = fres["Win%"]
    disp["Top10%"] = fres["Top10%"]
    disp["Top100%"] = fres["Top100%"]
    disp["Salary"] = fres["Salary"]
    disp["Own%"] = fres["OwnSum"]
    disp["Split"] = fres["Split"]

    colcfg = {
        "✓": st.column_config.CheckboxColumn("✓", help="Mark for export", width="small"),
        "Rank": st.column_config.NumberColumn(width="small"),
        "Win%": st.column_config.NumberColumn(format="%.2f%%", width="small"),
        "Top10%": st.column_config.NumberColumn(format="%.1f%%", width="small"),
        "Top100%": st.column_config.NumberColumn(format="%.1f%%", width="small"),
        "Salary": st.column_config.NumberColumn(format="$%d", width="small"),
        "Own%": st.column_config.NumberColumn(format="%.0f%%", width="small"),
        "Split": st.column_config.TextColumn(width="small")}
    colcfg["CPT"] = st.column_config.TextColumn("CPT (1.5×)")
    for c in sb.SD_COLS[1:]:
        colcfg[c] = st.column_config.TextColumn("UTIL")

    st.caption("The **CPT** column is your captain (1.5× points & salary). "
               "Tick **✓** to mark lineups for export.")
    edited = st.data_editor(
        disp, hide_index=True, height=460, width="stretch",
        disabled=[c for c in disp.columns if c != "✓"], column_config=colcfg,
        column_order=["✓", "Rank"] + sb.SD_COLS +
                     ["Win%", "Top10%", "Top100%", "Salary", "Own%", "Split"])
    for cand_id, on in zip(fres["Candidate"], edited["✓"]):
        (picked.add if on else picked.discard)(int(cand_id))
    st.caption(f"☑️ **{len(picked):,}** lineup(s) marked for export.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download filtered results (CSV)",
                       fres.to_csv(index=False).encode(),
                       file_name=f"showdown_results_{sim['contest_size']}.csv",
                       mime="text/csv", width="stretch")
    d2.download_button("Download all candidate lineups (CSV)",
                       sb.lineups_to_df(sim["cands"]).to_csv(index=False).encode(),
                       file_name=f"showdown_candidates_{len(sim['cands'])}.csv",
                       mime="text/csv", width="stretch")
    d3.download_button("Download field (CSV)",
                       sim["field_df"].to_csv(index=False).encode(),
                       file_name=f"showdown_field_{sim['field_n']}.csv",
                       mime="text/csv", width="stretch")

    with st.expander("📊 Finishing-position detail — click a lineup", expanded=False):
        pick_df = pd.DataFrame({
            "Rank": fres["Rank"], "Captain": fres["Captain"], "Split": fres["Split"],
            "Win%": fres["Win%"], "Top100%": fres["Top100%"], "Salary": fres["Salary"]})
        pick_evt = st.dataframe(
            pick_df, hide_index=True, width="stretch", height=200,
            on_select="rerun", selection_mode="single-row", key="sd_finish_pick",
            column_config={
                "Win%": st.column_config.NumberColumn(format="%.2f%%", width="small"),
                "Top100%": st.column_config.NumberColumn(format="%.1f%%", width="small"),
                "Salary": st.column_config.NumberColumn(format="$%d", width="small")})
        sel_rows = pick_evt.selection["rows"] if pick_evt.selection else []
        pos = sel_rows[0] if sel_rows else 0
        chosen_cand = int(fres["Candidate"].iloc[pos])
        r = res[res["Candidate"] == chosen_cand].iloc[0]
        cc1, cc2 = st.columns([3, 4])
        with cc1:
            st.altair_chart(
                place_distribution_chart(sim["dist"], chosen_cand - 1,
                                         sim["field_n"], K), width="stretch")
        with cc2:
            st.caption(f"Rank #{int(r['Rank'])} · captain {r['Captain']} · "
                       f"{r['Split']} · ${int(r['Salary']):,}")
            q1, q2, q3 = st.columns(3)
            q1.metric("Best", f"{int(r['BestPlace']):,}")
            q2.metric("Avg", f"{r['AvgPlace']:,.0f}")
            q3.metric("Worst", f"{int(r['WorstPlace']):,}")
            lu_rows = []
            for j, c in enumerate(sb.SD_COLS):
                v = str(r[c])
                nm = v.rsplit(" (", 1)[0]
                tm = v.rsplit(" (", 1)[1][:-1] if " (" in v else ""
                lu_rows.append({"Slot": "CPT" if j == 0 else "UTIL",
                                "Player": nm, "Team": tm})
            st.dataframe(pd.DataFrame(lu_rows), hide_index=True, width="stretch",
                         height=250)
            in_marks = int(chosen_cand) in picked
            if st.button(("☑️ Unmark" if in_marks else "⬜ Mark for export"),
                         width="stretch", key="sd_mark_btn"):
                (picked.discard if in_marks else picked.add)(int(chosen_cand))
                st.rerun()


def render_showdown_export(sim):
    """Export tab for showdown: requires a DKSalaries CSV (for CPT+UTIL ids),
    then exports marked or top-N lineups — ranked or payout-aware (Portfolio EV)
    — under per-player / per-captain / per-team exposure caps."""
    st.subheader("Build a DraftKings Showdown upload file")
    res = sim["res"]
    cands = sim["cands"]
    picked = st.session_state.setdefault("picked", set())

    st.caption("Showdown uploads need each player's **Captain-slot** and **UTIL-slot** "
               "DraftKings IDs. The RotoWire feed carries only the flex ID, so upload a "
               "DraftKings **DKSalaries.csv** for this slate (it lists both CPT and UTIL "
               "rows with their IDs). This is required to generate the upload file.")
    tmpl = st.file_uploader("DraftKings Showdown DKSalaries CSV", type=["csv"],
                            key="sd_template")
    tmap = None
    if tmpl is not None:
        tmap = su.parse_showdown_template(tmpl.getvalue().decode("utf-8", "replace"))
        if not tmap:
            st.error("Couldn't find CPT/UTIL rows in that CSV — expected DraftKings "
                     "columns Name, ID, and Roster Position (CPT/UTIL).")
        else:
            n_cpt = sum(1 for v in tmap.values() if 'CPT' in v)
            st.success(f"Loaded IDs for {len({k[0] for k in tmap})} players "
                       f"({n_cpt} CPT/UTIL rows).")

    fids = st.session_state.get("_sd_fres_ids") or list(res["Candidate"])
    fres = res[res["Candidate"].isin(fids)]

    mode = st.radio("Which lineups to export?",
                    [f"My marked selections ({len(picked)})", "Top N by ranking"],
                    index=0 if picked else 1, horizontal=True)

    if not tmap:
        st.info("Upload a DKSalaries CSV above to enable the export.")
        return

    def eligible(names):
        return su.eligible_names(tmap, names)

    if mode.startswith("My marked"):
        if not picked:
            st.info("Mark some lineups on the Results tab, then export them here.")
            return
        sel_df = res[res["Candidate"].isin(picked)]
        csv_text, uinfo = su.upload_csv([row for _, row in sel_df.iterrows()],
                                        tmap, cands)
        if uinfo["chosen"] == 0:
            st.error("None of your marked lineups had CPT+UTIL IDs for every player.")
            return
        msg = f"Exporting **{uinfo['chosen']}** marked lineup(s)."
        if uinfo["skipped_unmapped"]:
            msg += f" ({uinfo['skipped_unmapped']} skipped — missing an ID.)"
        st.success(msg)
        st.download_button("⬇ Download DraftKings upload CSV", csv_text.encode(),
                           file_name=f"DK_showdown_marked_{uinfo['chosen']}.csv",
                           mime="text/csv", type="primary", width="stretch")
        return

    # ---- Top-N ----
    from_filter = st.radio("Rank from", [f"Current filter ({len(fres):,})",
                                         f"All candidates ({len(res):,})"],
                           index=0, horizontal=True)
    src = fres if from_filter.startswith("Current") else res
    if len(src) == 0:
        st.info("No candidates to export.")
        return
    uc1, uc2 = st.columns(2)
    n_up = uc1.number_input("Number of lineups to export", 1, max(1, int(len(src))),
                            min(20, max(1, int(len(src)))), 1)
    sort_by = uc2.selectbox("Rank lineups by",
                            ["Win%", "Top10 Rate", "Top100 Rate"], index=0)
    keymap = {"Win%": ["Wins", "Top10", "Top100"],
              "Top10 Rate": ["Top10", "Top100", "Wins"],
              "Top100 Rate": ["Top100", "Top10", "Wins"]}[sort_by]

    _ev_ready = ("field_cut_scores" in (sim.get("dist") or {})
                 and bool(sim.get("score_pool")))
    sel_method = st.radio("Selection method",
                          ["Ranked (per-lineup rates)", "Portfolio EV (payout-aware)"],
                          index=0, horizontal=True, key="sd_sel_method")
    ev_mode = sel_method.startswith("Portfolio EV") and _ev_ready
    if sel_method.startswith("Portfolio EV") and not _ev_ready:
        st.info("Re-run the sim to enable payout-aware export. Using ranked for now.")

    with st.expander("Exposure caps (optional)", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        player_cap = cc1.slider("Max per player", 0.0, 1.0, 1.0, 0.05)
        captain_cap = cc2.slider("Max per captain", 0.0, 1.0, 1.0, 0.05)
        team_cap = cc3.slider("Max per team (majority side)", 0.0, 1.0, 1.0, 0.05)
        max_overlap = st.slider("Max lineup overlap", 0.5, 1.0, 1.0, 0.05,
                                help="Cap the share of players any two exported "
                                     "lineups may share (1.0 = no limit).")

    ev_entry_fee, ev_pct_paid, ev_rake = 20.0, 0.20, 0.15
    ev_top_heavy, ev_risk = 0.9, "Balanced"
    ev_shortlist = min(int(len(src)), 1000)
    if ev_mode:
        with st.expander("Payout structure & risk posture", expanded=True):
            st.caption(f"Modeling a **{int(sim['field_n']):,}-entry** contest.")
            ec1, ec2, ec3 = st.columns(3)
            ev_entry_fee = float(ec1.number_input("Entry fee ($)", 0.25, 10000.0,
                                                  20.0, 1.0))
            ev_pct_paid = ec2.slider("% of field paid", 0.05, 0.30, 0.20, 0.01)
            ev_rake = ec3.slider("Rake %", 0.0, 0.30, 0.15, 0.01)
            ev_top_heavy = st.slider("Top-heaviness", 0.3, 1.5, 0.9, 0.1)
            ev_risk = st.selectbox("How should the portfolio play out?",
                                   list(pev.UTILITIES.keys()), index=1)
            st.caption(pev.UTILITIES[ev_risk][1])
            ev_shortlist = int(st.number_input(
                "Candidate pool size", int(min(50, len(src))),
                int(min(4000, len(src))), int(min(1000, len(src))), 100))

    if ev_mode:
        short = (src.sort_values(keymap, ascending=False)
                 .head(int(ev_shortlist)).reset_index(drop=True))
        cand_scores = _sd_cand_scores(short, sim)
        prize = pev.make_payout_curve(int(sim["field_n"]), ev_entry_fee,
                                      top_heaviness=ev_top_heavy,
                                      pct_paid=ev_pct_paid, rake=ev_rake)
        pay = pev.candidate_payout_matrix(
            cand_scores, sim["dist"]["field_cut_scores"],
            sim["dist"]["cut_places"], prize)
        chosen, info, W = sp.select_showdown_portfolio_ev(
            short, int(n_up), pay, pev.utility(ev_risk), eligible=eligible,
            player_cap=player_cap, captain_cap=captain_cap, team_cap=team_cap,
            max_overlap=max_overlap, eval_sims=4000)
        csv_text, uinfo = su.upload_csv(chosen, tmap, cands)
        st.caption(f"Portfolio EV — exp return ${info['exp_return']:.0f}/slate · "
                   f"cash rate {100*info['cash_rate']:.0f}% · "
                   f"{info['distinct_captains']} distinct captains.")
    else:
        chosen, info = sp.select_showdown_portfolio(
            src, int(n_up), keymap, eligible=eligible, player_cap=player_cap,
            captain_cap=captain_cap, team_cap=team_cap, max_overlap=max_overlap)
        csv_text, uinfo = su.upload_csv(chosen, tmap, cands)

    if uinfo["chosen"] == 0:
        st.error("No exportable lineups — check the DKSalaries CSV covers these players.")
        return
    st.success(f"Selected **{uinfo['chosen']}** lineup(s) — max/player "
               f"{info['max_player']}, max/captain {info['max_captain']}, "
               f"max/team {info['max_team']} ({info['distinct_captains']} captains).")
    if uinfo["skipped_unmapped"]:
        st.caption(f"{uinfo['skipped_unmapped']} lineup(s) skipped — a player lacked "
                   "a CPT/UTIL ID in the DKSalaries CSV.")
    st.download_button("⬇ Download DraftKings upload CSV", csv_text.encode(),
                       file_name=f"DK_showdown_top{uinfo['chosen']}.csv",
                       mime="text/csv", type="primary", width="stretch")


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
    cares about: the game date, each team's batting order (lineups/players), each
    team's starting pitcher (matchups), and its Vegas implied total (which now
    drives offense scaling). A change in any of these means the correlated sims
    are stale. The total is rounded to 0.25 so genuine line moves trigger a
    rebuild but feed jitter doesn't thrash."""
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
        elif ta and tb and ta.get("order") != tb.get("order"):
            out.append(f"{t} lineup")
        elif ta and tb and ta.get("total") != tb.get("total"):
            out.append(f"{t} total {ta.get('total')}→{tb.get('total')}")
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


def slate_team_totals():
    """Per-team Vegas implied run totals for today's slate, from the live feed
    (preferred) or the stored slate.json. Returns [{Game, Team, Vegas total}].
    If the live feed returns all-identical totals (Vegas blocked → defaults) but
    a stored slate has differentiated totals, prefer the stored one."""
    def _rows(slate):
        out = []
        if slate:
            for g in slate.get("games", {}).values():
                for side in ("away", "home"):
                    t = g.get(side)
                    imp = (g.get("implied") or {}).get(side)
                    if t and imp is not None:
                        out.append({"Game": f"{g.get('away')} @ {g.get('home')}",
                                    "Team": t, "Vegas total": round(float(imp), 1)})
        return out

    def _varied(rows):
        return len({round(r["Vegas total"], 2) for r in rows}) > 1

    live = []
    try:
        live = _rows(slate_ingest.build_slate(write=False))
    except Exception:
        live = []
    if live and _varied(live):
        return live
    stored = _rows(load_stored_slate())
    if stored and _varied(stored):
        return stored
    return live or stored


def parse_team_totals_csv(raw_text, valid_teams=None):
    """Parse an uploaded CSV of team implied totals into a {TEAM: total} map.

    Accepts flexible headers — a team column (team/abbr/tm/club) and an
    implied-total column (implied/total/vegas/runs/itt/proj); if neither header
    is recognised it falls back to the first column for teams and the first
    numeric column for totals. Team codes are upper-cased and matched against
    `valid_teams` (the slate's teams) when provided so a stray/misspelled team
    is reported rather than silently applied. Returns
    (mapping, matched_teams, unmatched_labels)."""
    df = pd.read_csv(io.StringIO(raw_text))
    if df.empty or len(df.columns) < 2:
        raise ValueError("need at least a team column and an implied-total column")
    cols = list(df.columns)
    low = {c: str(c).strip().lower() for c in cols}

    def _find(keys, exclude=None):
        for c in cols:
            if c != exclude and any(k in low[c] for k in keys):
                return c
        return None

    team_col = _find(("team", "abbr", "tm", "club")) or cols[0]
    total_col = _find(("implied", "total", "vegas", "runs", "itt", "proj"),
                      exclude=team_col)
    if total_col is None:                       # first numeric non-team column
        for c in cols:
            if c != team_col and pd.to_numeric(df[c], errors="coerce").notna().any():
                total_col = c
                break
    if total_col is None:
        raise ValueError("couldn't find a numeric implied-total column")

    valid = {str(t).strip().upper(): t for t in (valid_teams or [])}
    mapping, matched, unmatched = {}, [], []
    for _, row in df.iterrows():
        raw_team = str(row[team_col]).strip()
        if not raw_team or raw_team.lower() == "nan":
            continue
        try:
            val = round(float(row[total_col]), 2)
        except (TypeError, ValueError):
            continue
        canon = valid.get(raw_team.upper(), None) if valid else raw_team
        if canon is None:
            unmatched.append(raw_team)
            continue
        mapping[canon] = val
        matched.append(canon)
    return mapping, matched, unmatched


TOTALS_OVERRIDE_PATH = os.path.join(HERE, "data", "team_totals_override.json")
SLATE_PLAYERS_PATH = os.path.join(HERE, "data", "slate_players.json")
SLATE_WINDOW_PATH = os.path.join(HERE, "data", "slate_window.json")


def run_vegas_diagnostic(date=None):
    """Live probe of the FantasyLabs Vegas feed → human-readable report lines.
    Runs on the host machine (needs egress to www.fantasylabs.com)."""
    import urllib.request
    import slate_config as C
    date = date or datetime.date.today().isoformat()
    url = C.FEED_VEGAS_TMPL.format(date=date)
    out = [f"URL: {url}"]
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "Chrome/124.0 Safari/537.36")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=30) as r:
            status = getattr(r, "status", r.getcode())
            ctype = r.headers.get("Content-Type")
            body = r.read().decode("utf-8", "replace")
        out.append(f"HTTP {status} | {ctype} | {len(body)} bytes")
        out.append(f"body head: {body[:240]}")
    except Exception as e:
        out.append(f"FETCH FAILED: {type(e).__name__}: {e}")
        return "\n".join(out)
    try:
        data = json.loads(body)
        rows = data if isinstance(data, list) else (
            data.get("Events") or data.get("data") or data.get("events") or [])
        out.append(f"JSON: {type(data).__name__}, {len(rows)} event rows")
        if rows:
            out.append(f"first-event keys: {sorted(rows[0].keys())}")
    except Exception as e:
        out.append(f"NOT JSON ({e}) — may need auth/cookies or returns HTML")
        return "\n".join(out)
    try:
        parsed = slate_ingest.fetch_vegas(date)
        out.append(f"fetch_vegas parsed {len(parsed)} teams; "
                   f"sample: {dict(list(parsed.items())[:6])}")
        slate = load_stored_slate() or {}
        teams = [g[s] for g in slate.get("games", {}).values()
                 for s in ("away", "home")]
        miss = [t for t in teams if C.canonical_team(t) not in parsed]
        if teams:
            out.append(f"slate team-matches: {len(teams)-len(miss)}/{len(teams)} "
                       f"matched; unmatched→default: {miss}")
    except Exception as e:
        out.append(f"parse/match step failed: {type(e).__name__}: {e}")
    return "\n".join(out)


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


def ensure_fresh(status, force=False, totals_path=None, slate_players=None,
                 slate_window=None):
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

    `slate_players` (optional): the FullNames on the chosen DK slate. Passed to
    the slate ingest so the off-slate game of a double-header is dropped (its
    pitcher/matchup would otherwise leak into today's rosterable set and the
    sims — see slate_ingest.filter_slate_doubleheaders).

    `slate_window` (optional): {"start","end"} with the chosen slate's
    SlateStart/SlateEnd. Passed through so games starting outside the slate
    window are dropped — the precise fix for a double-header, whose two games
    both appear in the slate's player list (see
    slate_ingest.filter_slate_by_window).

    Returns (notes, sims_changed, live_playable) where live_playable is the
    set of normnames eligible to be rostered today (lineup hitters + starting/
    opener/primary pitchers), or None if the live feed couldn't be read."""
    notes, sims_changed = [], False
    today = datetime.date.today().isoformat()
    # persist the slate's player list so the sim rebuild (a subprocess) can drop
    # the off-slate half of a double-header the same way the live read below does
    slate_players = list(slate_players) if slate_players is not None else None
    slate_players_path = None
    if slate_players:
        try:
            os.makedirs(os.path.dirname(SLATE_PLAYERS_PATH), exist_ok=True)
            json.dump(slate_players, open(SLATE_PLAYERS_PATH, "w"))
            slate_players_path = SLATE_PLAYERS_PATH
        except Exception:
            slate_players_path = None
    # persist the slate time window so the sim rebuild (a subprocess) drops the
    # same off-window (e.g. earlier double-header) games the live read below does
    slate_window_path = None
    if slate_window:
        try:
            os.makedirs(os.path.dirname(SLATE_WINDOW_PATH), exist_ok=True)
            json.dump(slate_window, open(SLATE_WINDOW_PATH, "w"))
            slate_window_path = SLATE_WINDOW_PATH
        except Exception:
            slate_window_path = None
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
        live = slate_ingest.build_slate(write=False, slate_players=slate_players,
                                        slate_window=slate_window)
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
                     or new_game_day or lineups_changed or totals_path is not None)
        if need_sims:
            if not sims_present() and live_sig is None and totals_path is None:
                st.error("No sims on disk and the live feed is unreachable — can't "
                         "build sims. Check your network and re-run.")
                return notes, False, live_playable
            if totals_path is not None:
                why = "edited team totals"
            elif not sims_present():
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
            cmd = ["run_slate.py"]
            if totals_path is not None:
                cmd += ["--team-totals", totals_path]
            if slate_players_path is not None:
                cmd += ["--slate-players", slate_players_path]
            if slate_window_path is not None:
                cmd += ["--slate-window", slate_window_path]
            ok, out = run_script(cmd, "Correlated sims (Stage C)", status)
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
def run_contest_dist(field_mat, cand_mat, n_sim, n_field, nbins=24,
                     cut_places=None):
    """Score each candidate against the field per sim and capture its
    finishing-place distribution as a compact ~`nbins`-bucket histogram (wider
    buckets read more clearly than per-position). Returns (wins, t10, t100, avg,
    dist) with exact best/mean/worst places.

    If `cut_places` is given (ascending place indices), the field's score at each
    of those places is also captured per sim into ``dist["field_cut_scores"]``
    (shape ``(n_sim, len(cut_places))``). This piggybacks on the per-sim sort we
    already do, giving the payout-aware export step the field placement ladder
    without a second pass."""
    N = cand_mat.shape[1]
    wins = np.zeros(N, np.int64); t10 = np.zeros(N, np.int64)
    t100 = np.zeros(N, np.int64); ps = np.zeros(N, np.int64)
    best = np.full(N, n_field + 1, np.int64); worst = np.zeros(N, np.int64)

    nb_target = max(6, min(int(nbins), int(n_field)))
    edges = np.unique(np.linspace(1, n_field + 1, nb_target + 1).astype(np.int64))
    nb = len(edges) - 1
    counts = np.zeros((N, nb), np.int32)
    idx = np.arange(N)

    cut_scores = None
    if cut_places is not None and len(cut_places):
        cut_places = np.asarray(cut_places, np.int64)
        cut_scores = np.empty((n_sim, len(cut_places)), np.float32)
        # ascending-sorted field: the score for place p is the p-th highest total
        take = n_field - cut_places                     # index into ascending fs

    for s in range(n_sim):
        fs = np.sort(field_mat[s]); cv = cand_mat[s]
        pl = (n_field - np.searchsorted(fs, cv, side="right")) + 1
        wins += (pl == 1); t10 += (pl <= 10); t100 += (pl <= 100); ps += pl
        best = np.minimum(best, pl); worst = np.maximum(worst, pl)
        b = np.clip(np.searchsorted(edges, pl, side="right") - 1, 0, nb - 1)
        np.add.at(counts, (idx, b), 1)
        if cut_scores is not None:
            cut_scores[s] = fs[take]
    dist = {"edges": edges, "counts": counts, "best": best, "worst": worst,
            "mean": ps / n_sim}
    if cut_scores is not None:
        dist["field_cut_scores"] = cut_scores
        dist["cut_places"] = cut_places
    return wins, t10, t100, ps / n_sim, dist


def place_distribution_chart(dist, i, n_field, n_sim):
    """Histogram of candidate i's finishing place — solid, full-height bars
    spanning each [lo, hi) place range, filled from the baseline up to the
    % of sims that landed in that bucket."""
    edges = dist["edges"].astype(float)
    counts = dist["counts"][i].astype(float)
    pct = 100 * counts / max(1, n_sim)
    df = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "pct": pct,
                       "zero": 0.0, "sims": counts.astype(np.int64)})
    xmax = max(2, int(n_field))
    xscale = alt.Scale(domain=[1, xmax], nice=False, zero=False, clamp=True)

    # x + x2 makes a ranged bar; pairing it with y + y2=0 fills each bar solidly
    # from the baseline (without y2 Vega-Lite only draws the thin top edge).
    bars = alt.Chart(df).mark_bar(color=BRAND, opacity=0.95).encode(
        x=alt.X("lo:Q", title="Finishing place  (1 = win)", scale=xscale,
                axis=alt.Axis(format="~s")),
        x2="hi:Q",
        y=alt.Y("pct:Q", title="% of sims", scale=alt.Scale(zero=True)),
        y2="zero:Q",
        tooltip=[alt.Tooltip("lo:Q", title="place ≥", format=",.0f"),
                 alt.Tooltip("hi:Q", title="< place", format=",.0f"),
                 alt.Tooltip("sims:Q", title="sims", format=","),
                 alt.Tooltip("pct:Q", title="% of sims", format=".2f")])

    rules = []
    for x, label, color in [(1, "1st", "#ffffff"), (10, "Top-10", "#f5566a"),
                            (100, "Top-100", "#c21e31"),
                            (float(dist["mean"][i]), "Mean", "#00e657")]:
        if 1 <= x <= xmax:
            rdf = pd.DataFrame({"x": [x], "label": [label]})
            rules.append(alt.Chart(rdf).mark_rule(
                color=color, strokeDash=[4, 3], size=1.5, opacity=0.9).encode(
                x=alt.X("x:Q", scale=xscale),
                tooltip=[alt.Tooltip("label:N", title="marker"),
                         alt.Tooltip("x:Q", title="place", format=",.0f")]))
    return alt.layer(bars, *rules).properties(height=230).configure_view(
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
# Header — RotoWire branded lockup
# --------------------------------------------------------------------------- #
def _lockup_svg():
    for p in (LOCKUP, LOGO):
        if p and str(p).endswith(".svg") and os.path.exists(p):
            try:
                return open(p, encoding="utf-8").read()
            except Exception:
                pass
    return ""


def _slate_label():
    d = (read_build_stamp().get("slate_date")
         or projections_built_date() or "")
    if d and len(d) == 10:
        try:
            dt = datetime.date.fromisoformat(d)
            return f"SLATE · {dt.strftime('%b').upper()} {dt.day}"
        except Exception:
            return "SLATE · " + d
    return "SLATE"


st.markdown(
    f"""<div class="rw-header">
      <div class="rw-wordmark">{_lockup_svg()}</div>
      <div class="rw-divider"></div>
      <div>
        <div class="rw-title">DFS Contest Sims</div>
        <div class="rw-eyebrow">MLB DFS contest simulator</div>
      </div>
      <span class="rw-badge"><span class="dot"></span>{_slate_label()}</span>
    </div>""",
    unsafe_allow_html=True)
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


tabs = st.tabs(["⚙️  Setup", "📊  Players", "🏆  Results", "⬇️  Export"])

with tabs[0]:
    st.subheader("Choose your slate")

    dk_df = None
    csv_ok = False
    id_map = {}
    id_col = None

    source = st.radio(
        "How do you want to load players?",
        ["Pick a slate (RotoWire feed)", "Upload your own file"],
        horizontal=True, label_visibility="collapsed", key="slate_source")

    if source == "Pick a slate (RotoWire feed)":
        top = st.columns([6, 1])
        if top[1].button("↻ Refresh", help="Re-fetch today's slates"):
            _load_slate_catalog.clear()
        catalog = None
        try:
            with st.spinner("Loading today's DraftKings slates…"):
                catalog = _load_slate_catalog()
        except Exception as e:
            st.error(f"Couldn't load the slate feed: {e}. Switch to "
                     "“Upload your own file” to proceed.")
        if catalog and catalog["slates"]:
            slates = catalog["slates"]
            labels = {s["slate_id"]: s["label"] for s in slates}
            sid = top[0].selectbox(
                "Slate", [s["slate_id"] for s in slates],
                format_func=lambda i: labels.get(i, i), key="slate_pick")
            slate = next(s for s in slates if s["slate_id"] == sid)
            dk_df, id_map = dk_slate_feed.to_dk_df(slate)
            st.session_state["_slate_fmt"] = slate.get("format", "classic")
            # the slate's time window pins which games are actually on the slate —
            # essential on a double-header day, where the player list carries both
            # games (see slate_ingest.filter_slate_by_window)
            st.session_state["_slate_window"] = (
                {"start": slate.get("start"), "end": slate.get("end")}
                if slate.get("start") and slate.get("end") else None)
            unowned = slate["n_players"] - slate["n_owned"]
            games_lbl = ("1 game (Showdown)" if slate.get("format") == "showdown"
                         else f"{slate['n_games']} games")
            st.caption(
                f"Slate **{sid}** ({catalog.get('date', '')}) — "
                f"{games_lbl}, {slate['n_players']} players, "
                f"{slate['n_owned']} with feed ownership"
                + (f" ({unowned} defaulted to 0%)." if unowned else "."))
        elif catalog is not None:
            st.warning("No slates are available from the feed right now — "
                       "switch to “Upload your own file”.")
    else:
        st.caption(
            "Upload either a **DraftKings salaries export** (the `DKSalaries.csv` with "
            "the player table — it already carries salary, position, team **and player "
            "IDs**, so it powers both the simulation and the upload export), or a "
            "**clean CSV** with columns " + ", ".join(f"`{c}`" for c in REQ_COLS) +
            " (add an `ID` column to enable the DK upload without a separate template). "
            "Ownership is the projected draft % (0–100).")
        st.session_state["_slate_fmt"] = "classic"   # uploads run the classic path
        st.session_state["_slate_window"] = None      # CSV uploads carry no window
        upload = st.file_uploader("Slate file", type=["csv"], label_visibility="collapsed")
        if upload is not None:
            try:
                raw = upload.getvalue().decode("latin-1", "replace")
                export = parse_dk_export(raw)
                if export is not None:
                    # raw DKSalaries export: has salary/pos/team/ID but no ownership
                    base_df, id_map = export
                    st.info(f"Detected a DraftKings salaries export — "
                            f"{len(base_df)} players, {dk_ids.count_ids(id_map)} IDs captured for the "
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
            except Exception as e:
                st.error(f"Could not read slate file: {e}")

    if dk_df is not None:
        simset = set(score)
        covered = int(dk_df["FullName"].map(lambda n: normname(n) in simset).sum())
        csv_ok = covered > 0
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Players in slate", len(dk_df))
        cc2.metric("Matched to sims", covered)
        cc3.metric("DK IDs available", dk_ids.count_ids(id_map) if id_map else 0)
        if covered == 0:
            st.error("None of the players in this slate matched the sim "
                     "universe — check names/teams. Nothing to simulate.")
        else:
            st.dataframe(dk_df.head(15), width="stretch")
            if id_map:
                if source.startswith("Pick"):
                    st.caption("✓ Player IDs from the RotoWire feed "
                               "(DraftKingsDraftableID) — the DK upload export "
                               "will use them automatically.")
                else:
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

    # --------------------------------------------------------------------------- #
    # Team totals (Vegas) — shown + editable. The loaded Vegas totals DRIVE the
    # sim by default (offense scaled by total vs a fixed 4.2-run league average);
    # editing is optional and just replaces a team's total before that scaling.
    # Only a flat/failed feed forces >=2 manual edits before a run can proceed.
    # --------------------------------------------------------------------------- #
    st.subheader("Team totals (Vegas) — adjust before running")
    if "_tt_fetched" not in st.session_state:
        with st.spinner("Reading today's team totals…"):
            st.session_state["_tt_fetched"] = slate_team_totals()
    fetched = st.session_state["_tt_fetched"]
    tt_override = {}
    n_tt_edits = 0
    feed_flat = False
    if not fetched:
        st.info("Couldn't read today's team totals (live feed unavailable and no "
                "slate on disk). Sims use the pipeline's Vegas totals as-is.")
    else:
        fetched_map = {r["Team"]: r["Vegas total"] for r in fetched}
        vals = list(fetched_map.values())
        if len(set(round(v, 2) for v in vals)) <= 1:
            feed_flat = True
            st.warning(f"⚠️ The live Vegas feed looks unavailable — every team "
                       f"defaulted to **{vals[0]:.1f}** runs, so Vegas can't drive "
                       "the sim. Edit at least 2 totals manually (or use the "
                       "diagnostic below) before running.")
        else:
            st.caption("Today's Vegas implied run totals **drive the sim by "
                       "default** — each team's offense is scaled by its total vs a "
                       "4.2-run league average (higher total → higher mean **and** "
                       "fatter ceiling, and a tougher night for the pitcher it "
                       "faces). Editing is **optional**: any change replaces that "
                       "team's total before the same scaling, rebuilds the sims, and "
                       "reshapes its player projections.")

        with st.expander("🔬 Diagnose the live Vegas feed"):
            st.caption("Probes www.fantasylabs.com live (from this host) and "
                       "shows the status, JSON shape, and whether the totals "
                       "match the slate. Use it to see why totals look flat.")
            if st.button("Run Vegas feed diagnostic", key="vdiag"):
                with st.spinner("Probing the Vegas feed…"):
                    st.code(run_vegas_diagnostic(), language="text")

        # --- Upload team totals to pre-fill the overrides ------------------- #
        # A CSV of teams + implied totals seeds the editor below; matching teams
        # are pre-filled and any change vs the live Vegas total becomes an
        # override on Run. Unmatched teams are reported, not silently dropped.
        with st.expander("⬆ Upload team totals (CSV)"):
            st.caption("CSV with a **team** column (DK abbreviations, e.g. NYY) "
                       "and an **implied total** column. Matching teams are "
                       "pre-filled below; edit further or remove the file to "
                       "revert to the live Vegas totals.")
            st.download_button(
                "⬇ Download template (today's teams)",
                pd.DataFrame(fetched)[["Team", "Vegas total"]]
                    .rename(columns={"Vegas total": "Implied total"})
                    .to_csv(index=False).encode(),
                file_name="team_totals_template.csv", mime="text/csv",
                key="tt_csv_tmpl")
            tt_csv = st.file_uploader(
                "Team totals CSV", type=["csv"], key="tt_csv_uploader",
                label_visibility="collapsed")
            if tt_csv is not None and \
                    st.session_state.get("_tt_csv_fid") != tt_csv.file_id:
                st.session_state["_tt_csv_fid"] = tt_csv.file_id
                try:
                    raw_tt = tt_csv.getvalue().decode("latin-1", "replace")
                    cmap, matched, unmatched = parse_team_totals_csv(
                        raw_tt, set(fetched_map))
                except Exception as e:
                    st.session_state["_tt_csv_map"] = {}
                    st.session_state["_tt_csv_report"] = None
                    st.error(f"Couldn't read that CSV: {e}")
                else:
                    st.session_state["_tt_csv_map"] = cmap
                    st.session_state["_tt_csv_report"] = {
                        "matched": matched, "unmatched": unmatched}
                    st.session_state["_tt_csv_ver"] = \
                        st.session_state.get("_tt_csv_ver", 0) + 1
            elif tt_csv is None and st.session_state.get("_tt_csv_map"):
                # File cleared → drop the overlay and re-seed the editor.
                st.session_state["_tt_csv_map"] = {}
                st.session_state["_tt_csv_fid"] = None
                st.session_state["_tt_csv_report"] = None
                st.session_state["_tt_csv_ver"] = \
                    st.session_state.get("_tt_csv_ver", 0) + 1

            _tt_report = st.session_state.get("_tt_csv_report") or {}
            _tt_map = st.session_state.get("_tt_csv_map") or {}
            if _tt_map:
                st.success(f"Loaded {len(_tt_map)} team total(s) from CSV.")
            if _tt_report.get("unmatched"):
                st.caption("⚠️ Not matched to any slate team (ignored): "
                           + ", ".join(_tt_report["unmatched"]))

        # Seed the editor with any uploaded totals (matching teams only).
        _base_tt = pd.DataFrame(fetched)
        _csv_map = st.session_state.get("_tt_csv_map") or {}
        if _csv_map:
            _base_tt["Vegas total"] = [
                _csv_map.get(t, v)
                for t, v in zip(_base_tt["Team"], _base_tt["Vegas total"])]

        _tt_edit = st.data_editor(
            _base_tt, hide_index=True, width="stretch",
            height=min(460, 60 + 34 * len(fetched)),
            key=f"tt_editor_{st.session_state.get('_tt_csv_ver', 0)}",
            disabled=["Game", "Team"],
            column_config={
                "Game": st.column_config.TextColumn(width="medium"),
                "Team": st.column_config.TextColumn(width="small"),
                "Vegas total": st.column_config.NumberColumn(
                    "Implied total", min_value=0.0, max_value=18.0, step=0.1,
                    format="%.1f")})
        for _, rr in _tt_edit.iterrows():
            try:
                val = round(float(rr["Vegas total"]), 2)
            except (TypeError, ValueError):
                continue
            if abs(val - float(fetched_map.get(rr["Team"], val))) > 1e-6:
                tt_override[rr["Team"]] = val      # only edited teams override
                n_tt_edits += 1
        if feed_flat:
            st.caption(f"✏️ {n_tt_edits} team total(s) changed"
                       + (" — run enabled." if n_tt_edits >= 2 else
                          f" · change **{2 - n_tt_edits} more** to enable the run "
                          "(the live feed is flat)."))
        elif n_tt_edits:
            st.caption(f"✏️ {n_tt_edits} team total(s) overridden; the rest use the "
                       "live Vegas totals.")

    # Vegas drives the sim by default; only a flat/failed feed needs manual edits.
    tt_ready = (n_tt_edits >= 2) if feed_flat else True

    # --------------------------------------------------------------------------- #
    # Step 2 — forced decisions, gated behind a Run button
    # --------------------------------------------------------------------------- #
    st.subheader("Make your selections, then run")

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
            cand_jitter = st.slider(
                "Candidate diversity jitter", min_value=0.0, max_value=1.5,
                value=0.0, step=0.1,
                help="Adds a per-pick random shock to every weighted selection "
                     "(stack team, stack members, one-off bats, pitchers) when "
                     "developing candidates. 0 (default) = deterministic weighting. "
                     "Higher = near-equally-projected options trade places between "
                     "lineups, so two similar players at the same price both get "
                     "used, a team's stack rotates its members, and the same "
                     "primary team pairs with different secondaries. Diversifies "
                     "the candidate POOL at the source (complements the export-time "
                     "exposure caps).")
            stack_boost = st.slider(
                "Stack-ownership ceiling boost", min_value=0.0, max_value=0.25,
                value=0.05, step=0.01,
                help="Uses projected STACK ownership (sum of a team's hitter "
                     "ownership) as a small upside signal: in each team's high-end "
                     "sims (its top ~20% of games), that team's hitters' DK points "
                     "are scaled up by a factor that grows with the team's stack "
                     "ownership — the lowest-owned stack gets no bump, the chalkiest "
                     "gets the full value here. The same boosted sims score BOTH the "
                     "field and your candidates, so popular stacks hit their ceiling "
                     "a touch more often and the top projected stacks surface a bit "
                     "higher. 0 = off (pure projection); 0.05 (default) = a gentle "
                     "nudge, never a driver. Correlation is preserved (a stack still "
                     "booms together).")

        force_refresh = st.checkbox(
            "Force full refresh (rebuild projections + sims now)", value=False,
            help="Rebuild projections (Stage B) and correlated sims (Stage C) "
                 "regardless of staleness — bypasses the once-a-day retry guard. "
                 "Use after fixing a data/connection issue.")
        submitted = st.form_submit_button(
            "▶ Run simulation", type="primary", width="stretch",
            disabled=(not tt_ready),
            help=None if tt_ready else
            "The live Vegas feed is flat — adjust at least 2 team totals above to "
            "enable the run.")

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

        # write the user's team-total overrides so Stage C rebuilds with them
        tt_path = None
        if tt_override:
            try:
                os.makedirs(os.path.dirname(TOTALS_OVERRIDE_PATH), exist_ok=True)
                json.dump(tt_override, open(TOTALS_OVERRIDE_PATH, "w"))
                tt_path = TOTALS_OVERRIDE_PATH
            except Exception as e:
                st.warning(f"Couldn't save team-total overrides ({e}); using Vegas.")

        t0 = time.time()
        with st.status("Running contest simulation…", expanded=True) as status:
            if tt_path:
                st.write(f"Applying {len(tt_override)} team-total override(s): "
                         + ", ".join(f"{t} {v:g}" for t, v in tt_override.items()))
            # ---- 0) freshness: rebuild projections / sims when stale ----
            # the chosen slate's players tell the ingest which half of a
            # double-header is actually on the slate (only the pitchers differ)
            slate_players = (dk_df["FullName"].tolist()
                             if dk_df is not None and "FullName" in dk_df else None)
            slate_window = st.session_state.get("_slate_window")
            notes, sims_changed, live_playable = ensure_fresh(
                status, force=force_refresh, totals_path=tt_path,
                slate_players=slate_players, slate_window=slate_window)
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

            # ---- Showdown slates take a dedicated path (1 CPT + 5 UTIL, single
            # game). Everything above (freshness, sim load, name-match guard) is
            # format-agnostic; everything below is the classic build, so we branch
            # here and st.rerun() to render the tabs from session state. ----
            if st.session_state.get("_slate_fmt", "classic") == "showdown":
                try:
                    sim_state = run_showdown_sim(
                        dk_df, score_k, K, int(contest_size), id_map,
                        int(num_candidates), int(seed_cand), int(seed_field),
                        int(medium), float(chalk), float(cand_jitter), status)
                except Exception as e:
                    status.update(label="Showdown build failed", state="error")
                    st.error(f"Could not build the showdown contest: {e}")
                    st.stop()
                st.session_state["sim"] = sim_state
                st.session_state["picked"] = set()
                st.session_state.pop("show_dist_for", None)
                st.session_state.pop("_sd_fres_ids", None)
                st.session_state["_goto_players"] = True
                status.update(label=f"Done in {time.time()-t0:.1f}s", state="complete")
                st.rerun()

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

            # ---- guard: restrict the pool to today's PLAYABLE players, but only
            # if the live feed is complete enough to leave a viable roster.
            # An incomplete/mismatched live feed (lineups not posted yet, name/
            # team differences) must NOT strip out valid simmed players. ----
            matched = int(dk_df["FullName"].map(
                lambda n: normname(n) in simnames).sum())
            applied_live = False
            if live_playable:
                keep = pool["Name"].map(lambda n: normname(n) in live_playable)
                kept = pool[keep]
                kh = kept[kept.Pos != "P"].Name.nunique()
                kp = kept[kept.Pos == "P"].Name.nunique()
                if kp >= 2 and kh >= 8:
                    drop_df = pool[~keep]
                    dh = drop_df[drop_df.Pos != "P"].Name.nunique()
                    dp = drop_df[drop_df.Pos == "P"].Name.nunique()
                    pool = kept.reset_index(drop=True)
                    applied_live = True
                    if dh or dp:
                        st.write(f"⛔ Restricted to today's projected/confirmed "
                                 f"lineups + starters — excluded **{dh} hitter(s)** "
                                 f"and **{dp} pitcher(s)**.")
                else:
                    st.warning(f"Live lineups look incomplete — only {kh} hitters / "
                               f"{kp} starters in your pool match today's live feed "
                               "(not a full roster). Keeping all simmed players for "
                               "this run rather than over-filtering; re-run once "
                               "lineups are posted to tighten it.")

            nh = pool[pool.Pos != "P"].Name.nunique()
            npi = pool[pool.Pos == "P"].Name.nunique()
            nt = pool.Team.nunique()
            st.write(f"Pool: **{nh} hitters + {npi} starters** across **{nt} teams**"
                     + (" — restricted to today's posted lineups + starters."
                        if applied_live else " — from your slate ∩ sims."))
            dropped = len(dk_df) - matched
            if dropped:
                st.caption(f"{dropped} of {len(dk_df)} slate players had no sim and "
                           "were excluded (they can't be scored).")
            if npi < 2 or nh < 8:
                status.update(label="Pool too small to build lineups", state="error")
                st.error(
                    f"Pool too small ({npi} starters, {nh} hitters; need ≥2 and ≥8). "
                    f"Your slate matched {matched} of {len(dk_df)} players to the "
                    "sims. If your slate is normal and lineups are posted, the sims "
                    "are likely **stale for today's slate** — tick **Force full "
                    "refresh** and run again to rebuild them; otherwise check that "
                    "the slate file's player names/teams match the slate.")
                st.stop()

            # ---- stack-ownership upside signal: give popular stacks a small bump
            #      to their high-end (ceiling) outcomes, tied to projected stack
            #      ownership. The SAME boosted sims score both the field and the
            #      candidates below, so it's a coherent re-weighting (not a thumb on
            #      the candidate scale). Raw `score_k` still drives candidate talent
            #      tilt — construction stays projection-honest. ----
            score_b = score_k
            if float(stack_boost) > 0:
                hit_pool = pool[pool.Pos != "P"]
                names_by_team, own_by_name = {}, {}
                for r in hit_pool.itertuples():
                    nn = normname(r.Name)
                    names_by_team.setdefault(r.Team, set()).add(nn)
                    own_by_name[nn] = float(r.Ownership)
                names_by_team = {t: sorted(ns) for t, ns in names_by_team.items()}
                stack_own = team_stack_ownership(names_by_team, own_by_name)
                score_b = apply_stack_ownership_boost(
                    score_k, names_by_team, stack_own, K,
                    strength=float(stack_boost), quantile=0.80)
                top_stacks = sorted(stack_own.items(), key=lambda kv: kv[1],
                                    reverse=True)[:3]
                st.caption(
                    f"Stack-ownership ceiling boost = {float(stack_boost):g}: "
                    "popular stacks' high-end games nudged up "
                    + "(top projected: "
                    + ", ".join(f"{t} {o:.0f}%" for t, o in top_stacks) + ").")

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
                         team_weights=team_weights, jitter=float(cand_jitter))
            cands, c_att = build_many(cb, int(num_candidates), "Candidates")
            if not cands:
                status.update(label="Could not build candidate lineups", state="error")
                st.error("Failed to construct any valid candidate lineup from this pool.")
                st.stop()
            cand_mat = score_matrix(cands, score_b, K)

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
            field_mat = score_matrix(field, score_b, K)

            # ---- the contest (captures each candidate's finishing-place distro,
            #      plus the field's per-sim placement ladder for payout-aware
            #      portfolio export) ----
            st.write(f"Simulating the contest over {K:,} runs…")
            cut_places = pev.field_place_cutpoints(len(field))
            wins, t10, t100, avg, dist = run_contest_dist(
                field_mat, cand_mat, K, len(field), cut_places=cut_places)
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

        # per-player metadata (position / salary / projected value) so the export
        # step can auto-detect near-twin value groups without re-simulating.
        players_meta = {}
        for r in pool.itertuples():
            nm = r.Name
            if nm in players_meta:
                continue
            players_meta[nm] = {
                "pos": "P" if r.Pos == "P" else r.Pos,
                "salary": int(r.Salary), "team": r.Team,
                "proj": float(tal[nm]) if nm in tal else None,
            }

        # per-player (boosted) sim arrays for just the pool players, so the
        # payout-aware export can rebuild any candidate's per-sim score cheaply
        # without persisting the full (n_sim x n_candidate) matrix.
        pool_norm = {normname(n) for n in pool["Name"].unique()}
        score_pool = {k: np.asarray(v, np.float32)
                      for k, v in score_b.items() if k in pool_norm}

        # persist so the filter/export controls below can change without re-simulating
        st.session_state["sim"] = {
            "res": res, "cands": cands, "field_df": lineups_to_df(field),
            "K": K, "contest_size": contest_size, "field_n": len(field), "beta": beta,
            "dist": dist, "id_map": id_map, "score_pool": score_pool,
            "cand_to_players": {i + 1: cand_players[i] for i in range(len(cands))},
            "pool_players": sorted({pl.Name for lu in cands for pl in lu["players"]}),
            "players_meta": players_meta,
        }
        # fresh run -> clear prior marks / inspection state
        st.session_state["picked"] = set()
        st.session_state.pop("show_dist_for", None)
        # signal the UI to jump to the Players tab as a "done" indicator
        st.session_state["_goto_players"] = True


# --------------------------------------------------------------------------- #
# Players tab — per-player projected ranges & thresholds.
# The Results and Export tabs below render from st.session_state["sim"], so
# tweaking their widgets doesn't trigger a re-simulation.
# --------------------------------------------------------------------------- #
with tabs[1]:
    with st.expander("📊 Players — projected ranges & thresholds", expanded=True):
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
            view, width="stretch", height=420, hide_index=True,
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
                st.altair_chart(player_score_chart(arr), width="stretch")
                st.caption(f"{psel}: min {r['Min']:.1f} · max {r['Max']:.1f} · "
                           f"std {r['Std']:.1f} · bust {r['Bust% (≤0)']:.0f}% · "
                           f"30+ {r['30+%']:.0f}%. Dashed line = mean.")


sim = st.session_state.get("sim")
with tabs[2]:
    if sim is None:
        st.info("On the **Setup** tab, upload your slate file and make all three "
                "selections, then press **Run simulation**.")
    elif sim.get("format") == "showdown":
        render_showdown_results(sim)
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
            exclude_sel = f1.multiselect(
                "Exclude player(s)", sim["pool_players"],
                help="Hide any lineup that contains any of these players.")
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
        c2p = sim["cand_to_players"]
        if players_sel:
            want = set(players_sel)
            mask &= res["Candidate"].map(
                lambda c: (want.issubset(c2p[int(c)]) if match_mode == "all"
                           else bool(want & c2p[int(c)])))
        if exclude_sel:
            avoid = set(exclude_sel)
            mask &= res["Candidate"].map(lambda c: not (avoid & c2p[int(c)]))
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
                     width="stretch"):
            picked |= {int(x) for x in fres["Candidate"]}
        if c3.button("Clear marks", width="content"):
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
                disp, hide_index=True, height=460, width="stretch",
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
                               mime="text/csv", width="stretch")
            d2.download_button("Download all candidate lineups (CSV)",
                               lineups_to_df(sim["cands"]).to_csv(index=False).encode(),
                               file_name=f"candidates_{len(sim['cands'])}.csv",
                               mime="text/csv", width="stretch")
            d3.download_button("Download field (CSV)",
                               sim["field_df"].to_csv(index=False).encode(),
                               file_name=f"field_{sim['field_n']}.csv",
                               mime="text/csv", width="stretch")

            # ---- secondary: finishing-position detail (de-emphasized) ----
            with st.expander("📊 Finishing-position detail — click a lineup",
                             expanded=False):
                st.caption("Click a lineup row below to see its finishing-place "
                           "distribution and the players it's built on.")

                # Click-to-select picker: a single-row selection drives the detail.
                pick_df = pd.DataFrame({
                    "Rank": fres["Rank"], "Stack": fres["Stack"],
                    "Win%": fres["Win%"], "Top10%": fres["Top10%"],
                    "Top100%": fres["Top100%"], "Salary": fres["Salary"]})
                pick_evt = st.dataframe(
                    pick_df, hide_index=True, width="stretch", height=200,
                    on_select="rerun", selection_mode="single-row",
                    key="finish_pick",
                    column_config={
                        "Rank": st.column_config.NumberColumn(width="small"),
                        "Win%": st.column_config.NumberColumn(format="%.2f%%",
                                                              width="small"),
                        "Top10%": st.column_config.NumberColumn(format="%.1f%%",
                                                                width="small"),
                        "Top100%": st.column_config.NumberColumn(format="%.1f%%",
                                                                 width="small"),
                        "Salary": st.column_config.NumberColumn(format="$%d",
                                                                width="small")})
                sel_rows = pick_evt.selection["rows"] if pick_evt.selection else []
                pos = sel_rows[0] if sel_rows else 0
                chosen_cand = int(fres["Candidate"].iloc[pos])
                cand_idx = chosen_cand - 1
                r = res[res["Candidate"] == chosen_cand].iloc[0]

                # Narrower chart on the left; the freed space shows the lineup.
                cc1, cc2 = st.columns([3, 4])
                with cc1:
                    st.altair_chart(
                        place_distribution_chart(sim["dist"], cand_idx,
                                                 sim["field_n"], K),
                        width="stretch")
                    st.caption("Dashed lines mark 1st, Top-10, Top-100, and the "
                               "lineup's mean place. The x-axis covers valid "
                               "places (1 … field).")
                with cc2:
                    st.caption(f"Rank #{int(r['Rank'])} · {r['Stack']} · "
                               f"${int(r['Salary']):,}")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Best", f"{int(r['BestPlace']):,}")
                    q2.metric("Avg", f"{r['AvgPlace']:,.0f}")
                    q3.metric("Worst", f"{int(r['WorstPlace']):,}")

                    lu_rows = []
                    for slot, c in zip(SLOT, COLS):
                        v = str(r[c])
                        nm = v.rsplit(" (", 1)[0]
                        tm = v.rsplit(" (", 1)[1][:-1] if " (" in v else ""
                        lu_rows.append({"Slot": slot, "Player": nm, "Team": tm})
                    st.dataframe(
                        pd.DataFrame(lu_rows), hide_index=True,
                        width="stretch", height=388,
                        column_config={
                            "Slot": st.column_config.TextColumn(width="small"),
                            "Team": st.column_config.TextColumn(width="small")})

                    in_marks = int(chosen_cand) in picked
                    if st.button(("☑️ Unmark" if in_marks else "⬜ Mark for export"),
                                 width="stretch"):
                        (picked.discard if in_marks else picked.add)(int(chosen_cand))
                        st.rerun()

with tabs[3]:
    if sim is None:
        st.info("Run a simulation first (Setup tab) to build a DraftKings upload.")
    elif sim.get("format") == "showdown":
        render_showdown_export(sim)
    else:
            st.subheader("Build a DraftKings upload file")

            # IDs come from the slate file you already uploaded; a template is only a
            # fallback if that file carried no player IDs.
            dkid = dict(sim.get("id_map") or {})
            if dkid:
                st.caption(f"Using the {dk_ids.count_ids(dkid)} player IDs from the slate you "
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
                    csv_text, info = rows_to_upload_csv(sel_df, dkid, sim.get("cands"))
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
                            mime="text/csv", type="primary", width="stretch")
            else:
                from_filter = st.radio(
                    "Rank from", [f"Current filter ({len(fres):,})",
                                  f"All candidates ({len(res):,})"],
                    index=0, horizontal=True,
                    help="Top-N is taken from your active filters by default, so e.g. "
                         "filter to 5 primary-stack teams and export the top 20 by Win%.")
                src = fres if from_filter.startswith("Current") else res
                uc1, uc2 = st.columns(2)
                n_up = uc1.number_input("Number of lineups to export", min_value=1,
                                        max_value=max(1, int(len(src))),
                                        value=min(20, max(1, int(len(src)))), step=1)
                sort_by = uc2.selectbox("Rank lineups by",
                                        ["Win%", "Top10 Rate", "Top100 Rate"], index=0,
                                        help="Win% favors tournament-winning ceiling; the "
                                             "Top10/Top100 rates favor consistent cashing.")

                _ev_ready = ("field_cut_scores" in (sim.get("dist") or {})
                             and bool(sim.get("score_pool")))
                sel_method = st.radio(
                    "Selection method",
                    ["Ranked (per-lineup rates)", "Portfolio EV (payout-aware)"],
                    index=0, horizontal=True, key="sel_method",
                    help="Ranked takes the top lineups by the metric above, then "
                         "fills under your caps — each lineup judged on its own. "
                         "Portfolio EV instead picks the SET that maximizes the "
                         "expected dollar return of the whole portfolio, so the "
                         "lineups cover different slate outcomes instead of "
                         "booming/busting together.")
                ev_mode = sel_method.startswith("Portfolio EV")
                # payout-aware defaults (overridden by the expander when in EV mode)
                ev_entry_fee = 20.0; ev_pct_paid = 0.20; ev_rake = 0.15
                ev_top_heavy = 0.9; ev_risk = "Balanced"
                ev_shortlist = min(int(len(src)), 1000)
                if ev_mode and not _ev_ready:
                    st.info("This simulation predates the payout-aware export — "
                            "re-run the sim (Setup tab) to enable it. Falling back "
                            "to ranked selection for now.")
                    ev_mode = False
                elif ev_mode:
                    with st.expander("Payout structure & risk posture", expanded=True):
                        st.caption(
                            f"Modeling a **{int(sim['field_n']):,}-entry** contest "
                            "(your simulated field). Dollar payouts come from a "
                            "parametric top-heavy GPP curve.")
                        ec1, ec2, ec3 = st.columns(3)
                        ev_entry_fee = float(ec1.number_input(
                            "Entry fee ($)", min_value=0.25, max_value=10000.0,
                            value=20.0, step=1.0))
                        ev_pct_paid = ec2.slider(
                            "% of field paid", 0.05, 0.30, 0.20, 0.01,
                            help="Share of the field that cashes (GPPs ~20%).")
                        ev_rake = ec3.slider(
                            "Rake %", 0.0, 0.30, 0.15, 0.01,
                            help="Operator's cut; sets the prize pool = "
                                 "entries x fee x (1 - rake).")
                        ev_top_heavy = st.slider(
                            "Top-heaviness", 0.3, 1.5, 0.9, 0.1,
                            help="How concentrated the prizes are at the top. "
                                 "~0.3 = flat (double-up-like), ~0.9 = typical GPP, "
                                 "~1.5 = winner-take-most.")
                        ev_risk = st.selectbox(
                            "How should the portfolio play out?",
                            list(pev.UTILITIES.keys()), index=1,
                            help="Sets the risk posture. "
                                 + " ".join(f"**{k.split(' (')[0]}** — {v[1]}"
                                            for k, v in pev.UTILITIES.items()))
                        st.caption(pev.UTILITIES[ev_risk][1])
                        ev_shortlist = int(st.number_input(
                            "Candidate pool size", min_value=int(min(50, len(src))),
                            max_value=int(min(4000, len(src))),
                            value=int(min(1000, len(src))), step=100,
                            help="The optimizer picks from this many top-ranked "
                                 "candidates. Larger = more freedom to diversify, "
                                 "slower."))

                # global caps default to no effect; per-entity overrides start empty
                hitter_cap = pitcher_cap = team_cap = 1.0
                player_caps: dict[str, float] = {}
                team_caps: dict[str, float] = {}
                player_mins: dict[str, float] = {}
                team_mins: dict[str, float] = {}
                with st.expander("Exposure caps (optional)"):
                    _cap_mode = st.radio(
                        "Exposure cap mode",
                        ["Global caps", "Per-player / per-team caps"],
                        index=0, horizontal=True, key="cap_mode",
                        help="Global applies one cap to every hitter / pitcher / "
                             "stack team. Per-player / per-team lets you set a "
                             "different max exposure for specific players or "
                             "teams.")

                    if _cap_mode == "Global caps":
                        pc1, pc2, pc3 = st.columns(3)
                        hitter_cap = pc1.slider(
                            "Max hitter exposure", 0.05, 1.0, 1.0, 0.05,
                            help="Cap the share of exported lineups any one HITTER "
                                 "can appear in (1.0 = no cap).")
                        pitcher_cap = pc2.slider(
                            "Max pitcher exposure", 0.05, 1.0, 1.0, 0.05,
                            help="Cap the share of exported lineups any one PITCHER "
                                 "can appear in (1.0 = no cap).")
                        team_cap = pc3.slider(
                            "Max stack-team exposure", 0.05, 1.0, 1.0, 0.05,
                            help="Cap the share sharing the same primary stack team.")
                    else:
                        st.caption(
                            "Set a **Min %** and/or **Max %** for the players and/or "
                            "teams you want to control. **Max %** caps exposure "
                            "(100 = no cap, 0 = exclude); **Min %** forces a floor — "
                            "the entity is seeded into at least that share of lineups "
                            "before the rest fill by rank (0 = no floor). Anything "
                            "left at Min 0 / Max 100 is unconstrained.")
                        _meta = sim.get("players_meta") or {}
                        _pool = sim.get("pool_players") or sorted(_meta.keys())
                        _min_conflicts = []

                        def _collect(_row, name_key, caps, mins):
                            """Read a Min %/Max % editor row into the caps/mins maps;
                            flag a min that exceeds its own max (engine clamps it)."""
                            nm = str(_row[name_key])
                            try:
                                mx = float(_row["Max %"])
                            except (TypeError, ValueError):
                                mx = 100.0
                            try:
                                mn = float(_row["Min %"])
                            except (TypeError, ValueError):
                                mn = 0.0
                            if mx < 100:
                                caps[nm] = mx / 100.0
                            if mn > 0:
                                mins[nm] = mn / 100.0
                                if mx < 100 and mn > mx:
                                    _min_conflicts.append(
                                        f"{nm} (min {mn:g}% > max {mx:g}%)")

                        # ---- per-player caps / mins ----
                        _prows = []
                        for _nm in _pool:
                            _m = _meta.get(_nm, {})
                            _prows.append({
                                "Player": _nm,
                                "Pos": _m.get("pos", ""),
                                "Team": _m.get("team", ""),
                                "Proj": (round(float(_m["proj"]), 1)
                                         if _m.get("proj") is not None else None),
                                "Min %": 0,
                                "Max %": 100,
                            })
                        # most-used (highest proj) players first so caps are handy
                        _pdf = pd.DataFrame(_prows)
                        if not _pdf.empty and _pdf["Proj"].notna().any():
                            _pdf = _pdf.sort_values(
                                "Proj", ascending=False, na_position="last"
                            ).reset_index(drop=True)
                        st.markdown("**Per-player exposure (min / max)**")
                        _pedit = st.data_editor(
                            _pdf, hide_index=True, width="stretch",
                            key="player_caps_editor",
                            column_config={
                                "Player": st.column_config.TextColumn(disabled=True),
                                "Pos": st.column_config.TextColumn(disabled=True),
                                "Team": st.column_config.TextColumn(disabled=True),
                                "Proj": st.column_config.NumberColumn(disabled=True),
                                "Min %": st.column_config.NumberColumn(
                                    min_value=0, max_value=100, step=5,
                                    help="Min share of exported lineups this player "
                                         "must appear in (best-effort)."),
                                "Max %": st.column_config.NumberColumn(
                                    min_value=0, max_value=100, step=5,
                                    help="Max share of exported lineups this player "
                                         "may appear in."),
                            })
                        for _, _r in _pedit.iterrows():
                            _collect(_r, "Player", player_caps, player_mins)

                        # ---- per-team caps / mins ----
                        _teams = sorted({_m.get("team", "") for _m in _meta.values()
                                         if _m.get("team")})
                        if _teams:
                            st.markdown("**Per-team (primary stack) exposure (min / max)**")
                            _tdf = pd.DataFrame(
                                [{"Team": _t, "Min %": 0, "Max %": 100}
                                 for _t in _teams])
                            _tedit = st.data_editor(
                                _tdf, hide_index=True, width="content",
                                key="team_caps_editor",
                                column_config={
                                    "Team": st.column_config.TextColumn(disabled=True),
                                    "Min %": st.column_config.NumberColumn(
                                        min_value=0, max_value=100, step=5,
                                        help="Min share of exported lineups whose "
                                             "primary stack is this team "
                                             "(best-effort)."),
                                    "Max %": st.column_config.NumberColumn(
                                        min_value=0, max_value=100, step=5,
                                        help="Max share of exported lineups whose "
                                             "primary stack is this team."),
                                })
                            for _, _r in _tedit.iterrows():
                                _collect(_r, "Team", team_caps, team_mins)

                        if _min_conflicts:
                            st.warning(
                                "Min above max — the floor is clamped to the cap for: "
                                + ", ".join(_min_conflicts))
                        _n_set = (len(player_caps) + len(team_caps)
                                  + len(player_mins) + len(team_mins))
                        if _n_set:
                            st.caption(
                                f"{len(player_caps)} player cap(s), "
                                f"{len(team_caps)} team cap(s), "
                                f"{len(player_mins)} player min(s) and "
                                f"{len(team_mins)} team min(s) active.")

                with st.expander("Portfolio diversity (optional)"):
                    st.caption("Spread the exported set so the lineups work "
                               "together instead of piling onto the single "
                               "best build. All default to no effect.")
                    dc1, dc2, dc3 = st.columns(3)
                    pair_cap = dc1.slider(
                        "Max stack-pairing exposure", 0.05, 1.0, 1.0, 0.05,
                        help="Cap the share of lineups sharing the SAME "
                             "(primary, secondary) team pair — so e.g. a Cleveland "
                             "primary gets paired with several different secondary "
                             "stacks instead of always Kansas City. 1.0 = no cap.")
                    core_cap = dc2.slider(
                        "Max stack-core exposure", 0.05, 1.0, 1.0, 0.05,
                        help="Cap the share of lineups using the exact SAME set of "
                             "hitters for the primary stack — forces a team's stack "
                             "to rotate its teammates across the portfolio. "
                             "1.0 = no cap.")
                    max_overlap = dc3.slider(
                        "Max lineup similarity", 0.30, 1.0, 1.0, 0.05,
                        help="Reject a lineup if it shares more than this fraction "
                             "of players with one already exported (Jaccard "
                             "overlap). 1.0 = allow near-duplicates; ~0.8 keeps "
                             "every exported lineup meaningfully distinct.")

                    meta = sim.get("players_meta") or {}
                    group_of, groups = ({}, [])
                    group_cap = 1.0
                    if meta:
                        gc1, gc2 = st.columns(2)
                        sal_tol = gc1.number_input(
                            "Value-group salary tolerance ($)", min_value=0,
                            max_value=2000, value=300, step=100,
                            help="Players at the same position within this salary "
                                 "gap and a close projection are treated as "
                                 "near-twins.")
                        proj_tol = gc2.number_input(
                            "Value-group projection tolerance (pts)",
                            min_value=0.0, max_value=10.0, value=1.5, step=0.5,
                            help="Max projected-point gap for two same-position, "
                                 "similar-salary players to count as near-twins.")
                        group_of, groups = detect_value_groups(
                            meta, salary_tol=int(sal_tol), proj_tol=float(proj_tol))
                        if groups:
                            group_cap = st.slider(
                                "Max value-group exposure", 0.05, 1.0, 1.0, 0.05,
                                help="Cap the combined share of lineups using ANY "
                                     "member of a near-twin group, so a slightly "
                                     "higher projection doesn't let one player eat "
                                     "the whole group's exposure. 1.0 = no cap.")
                            st.caption(f"{len(groups)} near-twin value group(s) "
                                       "detected (capped together):")
                            gdf = pd.DataFrame([{
                                "Pos": g["pos"],
                                "Players": ", ".join(g["players"]),
                                "Salary": (f"${g['salary_lo']:,}" if
                                           g['salary_lo'] == g['salary_hi'] else
                                           f"${g['salary_lo']:,}–${g['salary_hi']:,}"),
                                "Proj": f"{g['proj_lo']:.1f}–{g['proj_hi']:.1f}",
                            } for g in groups])
                            st.dataframe(gdf, width="stretch",
                                         hide_index=True)
                        else:
                            st.caption("No near-twin value groups at these "
                                       "tolerances.")

                W = W_naive = ev_extra = None
                if ev_mode:
                    _ev = build_dk_upload_ev(
                        src, sim, dkid, n_up, entry_fee=ev_entry_fee,
                        pct_paid=ev_pct_paid, rake=ev_rake,
                        top_heaviness=ev_top_heavy, risk=ev_risk,
                        shortlist=ev_shortlist, hitter_cap=hitter_cap,
                        pitcher_cap=pitcher_cap, team_cap=team_cap,
                        pair_cap=pair_cap, core_cap=core_cap, max_overlap=max_overlap,
                        group_of=group_of, group_cap=group_cap,
                        player_caps=player_caps, team_caps=team_caps,
                        player_mins=player_mins, team_mins=team_mins)
                    if _ev is None:
                        ev_mode = False
                    else:
                        csv_text, info, W, W_naive, ev_extra = _ev
                if not ev_mode:
                    csv_text, info = build_dk_upload(
                        src, dkid, n_up, sort_by, hitter_cap, pitcher_cap, team_cap,
                        pair_cap=pair_cap, core_cap=core_cap, max_overlap=max_overlap,
                        group_of=group_of, group_cap=group_cap,
                        player_caps=player_caps, team_caps=team_caps,
                        player_mins=player_mins, team_mins=team_mins,
                        cands=sim.get("cands"))

                if info["chosen"] == 0:
                    st.error("No exportable lineups — players had no DK ID, or the caps "
                             "are too strict.")
                else:
                    if ev_mode:
                        _by = f"portfolio EV ({ev_risk.split(' (')[0]})"
                    else:
                        _by = f"**{sort_by}**"
                    msg = (f"Selected **{info['chosen']} of {n_up}** lineups by "
                           f"{_by}. Max hitter exposure "
                           f"{info['max_hitter']}/{info['chosen']}, max pitcher "
                           f"{info['max_pitcher']}/{info['chosen']}, max stack-team "
                           f"{info['max_team']}/{info['chosen']}.")
                    if info["chosen"] < n_up:
                        msg += (f" Caps/pool limited the set to "
                                f"{info['chosen']} — loosen them for more.")
                    msg += (f" Distinct stack pairings: {info['distinct_pairs']}; "
                            f"distinct stack cores: {info['distinct_cores']}.")
                    if info["skipped_unmapped"]:
                        msg += (f" ({info['skipped_unmapped']} skipped: a player had no "
                                "DK ID.)")
                    st.success(msg)
                    _unmet = info.get("unmet_mins") or []
                    if _unmet:
                        st.warning(
                            "Couldn't fully meet these minimum exposures (not enough "
                            "distinct lineups or caps left no room): "
                            + ", ".join(
                                f"{u['name']} {u['have']}/{u['need']}" for u in _unmet))
                    st.download_button(
                        "⬇ Download DraftKings upload CSV", csv_text.encode(),
                        file_name=f"DK_upload_{info['chosen']}.csv",
                        mime="text/csv", type="primary", width="stretch")

                    # ---- exposure breakdown (player & team) ----
                    _pexpo = info.get("player_expo") or {}
                    _texpo = info.get("team_expo") or {}
                    _n_lu = info["chosen"] or 1
                    if _pexpo or _texpo:
                        _meta = sim.get("players_meta") or {}
                        _pset = set(info.get("pitchers") or [])
                        with st.expander(
                                "Exposure breakdown (player & team)",
                                expanded=False):
                            st.caption(
                                f"How the {info['chosen']} exported lineups spread "
                                "across players and stack teams. **Exposure** is the "
                                "share of exported lineups the player (or primary "
                                "stack team) appears in.")
                            ec1, ec2 = st.columns([3, 2])

                            with ec1:
                                st.markdown("**Player exposure**")
                                _prows = []
                                for _nm, _ct in _pexpo.items():
                                    _m = _meta.get(_nm, {})
                                    _prows.append({
                                        "Player": _nm,
                                        "Pos": ("P" if _nm in _pset
                                                else _m.get("pos", "")),
                                        "Team": _m.get("team", ""),
                                        "Lineups": int(_ct),
                                        "Exposure": int(_ct) / _n_lu,
                                    })
                                _pdf = pd.DataFrame(_prows).sort_values(
                                    ["Lineups", "Player"],
                                    ascending=[False, True]).reset_index(drop=True)
                                st.dataframe(
                                    _pdf, hide_index=True, width="stretch",
                                    column_config={
                                        "Lineups": st.column_config.NumberColumn(
                                            help=f"of {info['chosen']} lineups"),
                                        "Exposure": st.column_config.ProgressColumn(
                                            format="percent", min_value=0.0,
                                            max_value=1.0),
                                    })
                                st.download_button(
                                    "⬇ Player exposure CSV",
                                    _pdf.assign(Exposure=(_pdf["Exposure"] * 100)
                                                .round(1)).to_csv(index=False)
                                    .encode(),
                                    file_name=f"player_exposure_{info['chosen']}.csv",
                                    mime="text/csv", width="stretch")

                            with ec2:
                                st.markdown("**Stack-team exposure (primary)**")
                                _trows = [{
                                    "Team": _tm,
                                    "Lineups": int(_ct),
                                    "Exposure": int(_ct) / _n_lu,
                                } for _tm, _ct in _texpo.items()]
                                _tdf = pd.DataFrame(_trows).sort_values(
                                    ["Lineups", "Team"],
                                    ascending=[False, True]).reset_index(drop=True)
                                st.dataframe(
                                    _tdf, hide_index=True, width="stretch",
                                    column_config={
                                        "Lineups": st.column_config.NumberColumn(
                                            help=f"of {info['chosen']} lineups"),
                                        "Exposure": st.column_config.ProgressColumn(
                                            format="percent", min_value=0.0,
                                            max_value=1.0),
                                    })
                                st.download_button(
                                    "⬇ Team exposure CSV",
                                    _tdf.assign(Exposure=(_tdf["Exposure"] * 100)
                                                .round(1)).to_csv(index=False)
                                    .encode(),
                                    file_name=f"team_exposure_{info['chosen']}.csv",
                                    mime="text/csv", width="stretch")

                    # ---- payout-aware coverage visualization ----
                    if ev_mode and W is not None and ev_extra:
                        ps = ev_extra["prize_summary"]
                        cost = ev_extra["cost"] or 1.0
                        st.markdown("##### Portfolio outcome across simulated slates")
                        st.caption(
                            f"Curve: **${ps['first_place']:,.0f}** to 1st, "
                            f"**${ps['min_cash']:,.0f}** min-cash, "
                            f"{ps['places_paid']:,} paid of "
                            f"{ev_extra['field_n']:,} "
                            f"(pool ${ps['prize_pool']:,.0f}). "
                            f"Entry cost {info['chosen']}×${ev_entry_fee:,.0f} "
                            f"= ${cost:,.0f}.")

                        # EV set vs a rank-selected set of the same size
                        naive_ret = float(np.mean(W_naive))
                        naive_cash = float(np.mean(W_naive > 0))
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Expected return", f"${info['exp_return']:,.0f}",
                                   f"{info['exp_return'] - naive_ret:+,.0f} vs ranked")
                        mc2.metric("ROI", f"{100*(info['exp_return']/cost - 1):+.1f}%")
                        mc3.metric("Cash rate (≥1 lineup)",
                                   f"{100*info['cash_rate']:.1f}%",
                                   f"{100*(info['cash_rate'] - naive_cash):+.1f} pts vs ranked")
                        mc4.metric("Floor (p10) / Ceiling (p90)",
                                   f"${info['floor_p10']:,.0f} / ${info['ceiling_p90']:,.0f}")
                        st.altair_chart(portfolio_return_chart(W, W_naive),
                                        use_container_width=True)
                        st.caption(
                            "“Cash rate” is the share of simulated slates where at "
                            "least one exported lineup finishes in the money — the "
                            "portfolio-level win/cash metric. Compared against a "
                            "top-N-by-rank set of the same size, drawn from the same "
                            "candidate pool.")


# --------------------------------------------------------------------------- #
# After a completed run, jump to the Players tab as a "done" indicator.
# (Streamlit has no API to set the active tab, so click it from a one-shot
#  same-origin script.)
# --------------------------------------------------------------------------- #
if st.session_state.pop("_goto_players", False):
    import streamlit.components.v1 as _components
    _components.html(
        """
        <script>
          const doc = window.parent.document;
          let tries = 0;
          const jump = () => {
            const tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs && tabs.length > 1) { tabs[1].click(); }
            else if (tries++ < 20) { setTimeout(jump, 100); }
          };
          setTimeout(jump, 150);
        </script>
        """,
        height=0,
    )

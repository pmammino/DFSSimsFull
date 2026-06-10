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

Nothing runs until ALL four decisions are made and the user clicks Run, then the
app builds an ownership-weighted field + a uniform candidate pool and reports the
simulated contest outcomes (Win% / Top10% / Top100% / AvgPlace) for every
candidate lineup.

Launch:
    streamlit run app.py
"""
import csv, io, json, os, re, tempfile, time
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st

from stage_d import (load_sims, build_pool, lineups_to_df, score_matrix,
                     run_contest, norm as normname, COLS, HITC, SLOT)
from mlb_lineup_builder import Pool, Builder
from field_simulator import (normalize_to_slots, adjust_ownership,
                             beta_for_size, tilt_structures)

HERE = os.path.dirname(os.path.abspath(__file__))
DELIV = os.path.join(HERE, "deliverables")
PARAMS_PATH = os.path.join(HERE, "field_params.json")
REQ_COLS = ["FullName", "Team", "Position", "Salary", "Ownership"]
SIZE_PRESETS = [150, 1000, 6000, 20000, 50000, 150000]

st.set_page_config(page_title="DFS Contest Simulator", page_icon="⚾", layout="wide")


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


def ids_from_clean(df):
    """Pull a name -> DK ID map from a clean CSV if it carries IDs, via either
    an 'ID' column or a DK-style 'Name + ID' column ('Player Name (1234567)')."""
    idmap = {}
    if "ID" in df.columns:
        for _, r in df.iterrows():
            v = str(r["ID"]).strip()
            if v and v.lower() != "nan":
                idmap[normname(r["FullName"])] = v.split(".")[0]
    elif "Name + ID" in df.columns:
        for _, r in df.iterrows():
            m = re.search(r"\((\d+)\)\s*$", str(r["Name + ID"]))
            if m:
                idmap[normname(r["FullName"])] = m.group(1)
    return idmap


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


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("⚾ DFS Contest Simulator")
st.caption(
    "Simulate DraftKings MLB contest outcomes for machine-developed candidate "
    "lineups, against an ownership-weighted field, using the day's correlated "
    "player sims. You provide the expected ownership; you choose the contest "
    "size, the number of sim runs, and how many candidate lineups to develop."
)

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

with st.expander("Which players are in the sim universe? (your CSV must cover these)"):
    universe = pd.DataFrame(
        {"Player": sorted(list(H.keys())) + sorted(list(P.keys())),
         "Type": ["Hitter"] * n_hit + ["Pitcher"] * n_pit}
    )
    st.dataframe(universe, use_container_width=True, height=240)
    st.download_button("Download player list (CSV)",
                       universe.to_csv(index=False).encode(),
                       file_name="sim_player_universe.csv", mime="text/csv")

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
            # clean CSV
            dk_df = pd.read_csv(io.StringIO(raw))
            dk_df.columns = [c.strip() for c in dk_df.columns]
            missing = [c for c in REQ_COLS if c not in dk_df.columns]
            if missing:
                st.error("CSV is missing required column(s): " + ", ".join(missing))
                dk_df = None
            else:
                id_map = ids_from_clean(dk_df)

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
                if not id_map:
                    st.caption("No player IDs in this file — you'll be able to "
                               "supply a DKSalaries template at export time, or "
                               "add an `ID` column here.")
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

    K = int(sim_runs)
    if contest_size > 100_000 or num_candidates > 50_000:
        st.info("Large request — field/candidate construction and scoring may "
                "take a while and use significant memory.")

    t0 = time.time()
    # subsample the sims to the requested number of runs (sim index is aligned
    # across all players, so a prefix slice preserves the correlation structure)
    score_k = {k: v[:K] for k, v in score.items()}

    # ---- build the pool (write CSV to a temp path for build_pool) ----
    with st.status("Running contest simulation…", expanded=True) as status:
        st.write("Building player pool from your CSV + sims…")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="") as tf:
            dk_df.to_csv(tf.name, index=False)
            tmp_csv = tf.name
        try:
            pool = build_pool(tmp_csv, H, P, score_k)
        finally:
            os.unlink(tmp_csv)

        nh = pool[pool.Pos != "P"].Name.nunique()
        npi = pool[pool.Pos == "P"].Name.nunique()
        nt = pool.Team.nunique()
        st.write(f"Pool: **{nh} hitters + {npi} starters** across **{nt} teams** "
                 "— drawn only from your uploaded ownership (players with sims). "
                 "Both the field and the candidate lineups use only these players.")
        simnames = set(score_k)
        dropped = len(dk_df) - int(dk_df["FullName"].map(
            lambda n: normname(n) in simnames).sum())
        if dropped:
            st.caption(f"{dropped} player(s) in your CSV had no sim and were "
                       "excluded (they can't be scored).")
        if npi < 2 or nh < 8:
            status.update(label="Pool too small to build lineups", state="error")
            st.error("Need at least 2 starting pitchers and 8 hitters in the "
                     "matched pool to fill a roster. Check your CSV.")
            st.stop()

        # ---- candidate lineups (uniform, ownership-blind, starters only) ----
        st.write(f"Developing {int(num_candidates):,} candidate lineups…")
        cdf = pool[(pool.Pos != "P") | (pool.Role == "SP")].copy()
        cdf["Ownership"] = 1.0
        cb = Builder(Pool(cdf), params, seed=int(seed_cand), uniform=True)
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

        # ---- the contest ----
        st.write(f"Simulating the contest over {K:,} runs…")
        wins, t10, t100, avg = run_contest(field_mat, cand_mat, K, len(field))
        status.update(label=f"Done in {time.time()-t0:.1f}s", state="complete")

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
    res = res.sort_values(["Wins", "Top10", "Top100", "AvgPlace"],
                          ascending=[False, False, False, True]).reset_index(drop=True)

    # persist so the DK-upload controls below can change without re-simulating
    st.session_state["sim"] = {
        "res": res, "cands": cands, "field_df": lineups_to_df(field),
        "K": K, "contest_size": contest_size, "field_n": len(field), "beta": beta,
        "best_lineup": cands[res.loc[0, "Candidate"] - 1],
        "id_map": id_map,  # IDs captured from the uploaded slate file (may be empty)
    }


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

    st.subheader("Simulated outcomes — candidate lineups")
    st.caption("Sorted best-first by Wins → Top10 → Top100 → AvgPlace.")
    st.dataframe(res, use_container_width=True, height=420)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download candidate results (CSV)",
                       res.to_csv(index=False).encode(),
                       file_name=f"candidate_results_{sim['contest_size']}.csv",
                       mime="text/csv", use_container_width=True)
    d2.download_button("Download candidate lineups (CSV)",
                       lineups_to_df(sim["cands"]).to_csv(index=False).encode(),
                       file_name=f"candidates_{len(sim['cands'])}.csv",
                       mime="text/csv", use_container_width=True)
    d3.download_button("Download field (CSV)",
                       sim["field_df"].to_csv(index=False).encode(),
                       file_name=f"field_{sim['field_n']}.csv",
                       mime="text/csv", use_container_width=True)

    with st.expander("Top candidate lineup detail"):
        best = sim["best_lineup"]
        rows = [{"Slot": COLS[i], "Player": pl.Name, "Team": pl.Team,
                 "Pos": pl.Pos, "Salary": pl.Salary}
                for i, pl in enumerate(best["players"])]
        st.table(pd.DataFrame(rows))
        st.caption(f"Total salary ${best['salary']:,} · stack "
                   f"{'-'.join(map(str, sorted(best['teams'].values(), reverse=True)))}")

    # ----------------------------------------------------------------------- #
    # Step 3 — filled DraftKings upload file
    # ----------------------------------------------------------------------- #
    st.divider()
    st.subheader("3 · Download a filled DraftKings upload file")

    # IDs come from the slate file you already uploaded; a template is only
    # needed as a fallback if that file carried no player IDs.
    dkid = dict(sim.get("id_map") or {})
    if dkid:
        st.caption(f"Using the {len(dkid)} player IDs from the slate file you "
                   "uploaded — no extra template needed. Choose how many lineups "
                   "and how to rank them, then download.")
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

    uc1, uc2 = st.columns(2)
    n_up = uc1.number_input("Number of lineups to export", min_value=1,
                            max_value=int(len(res)),
                            value=min(20, int(len(res))), step=1)
    sort_by = uc2.selectbox("Rank lineups by",
                            ["Win%", "Top10 Rate", "Top100 Rate"], index=0,
                            help="Win% favors tournament-winning ceiling; the "
                                 "Top10/Top100 rates favor consistent cashing "
                                 "near the top.")
    with st.expander("Exposure caps (optional)"):
        pc1, pc2 = st.columns(2)
        player_cap = pc1.slider("Max player exposure", 0.05, 1.0, 1.0, 0.05,
                                help="Cap the share of exported lineups any one "
                                     "player can appear in (1.0 = no cap).")
        team_cap = pc2.slider("Max primary-stack-team exposure", 0.05, 1.0, 1.0,
                              0.05, help="Cap the share of exported lineups that "
                                         "share the same primary stack team.")

    if not dkid:
        st.info("Player IDs are needed to build the upload file (see above).")
    else:
        cand_players = {pl.Name for lu in sim["cands"] for pl in lu["players"]}
        mapped = sum(1 for nm in cand_players if normname(nm) in dkid)
        if mapped < len(cand_players):
            st.caption(f"{mapped} of {len(cand_players)} candidate-pool players "
                       f"have a DK ID ({len(dkid)} IDs available).")
        csv_text, info = build_dk_upload(res, dkid, n_up, sort_by,
                                         player_cap, team_cap)
        if info["chosen"] == 0:
            st.error(
                "No exportable lineups — none of the candidate lineups had a DK "
                f"ID for every player (only {mapped}/{len(cand_players)} candidate "
                "players matched), or the exposure caps are too strict. Make sure "
                "the slate file's player IDs cover the same players as your "
                "ownership.")
        else:
            msg = (f"Selected **{info['chosen']} of {n_up}** lineups by "
                   f"**{sort_by}**. Max single-player exposure "
                   f"{info['max_player']}/{info['chosen']}, max stack-team "
                   f"{info['max_team']}/{info['chosen']}.")
            if info["chosen"] < n_up:
                msg += (" Fewer than requested — loosen the exposure caps or "
                        "develop more candidate lineups.")
            if info["skipped_unmapped"]:
                msg += (f" ({info['skipped_unmapped']} candidate lineups skipped: "
                        "a player had no DK ID.)")
            st.success(msg)
            st.download_button(
                "⬇ Download DraftKings upload CSV", csv_text.encode(),
                file_name=f"DK_upload_{info['chosen']}.csv",
                mime="text/csv", type="primary", use_container_width=True)

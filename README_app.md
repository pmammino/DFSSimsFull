# DFS Contest Simulator — web interface (`app.py`)

A Streamlit front end over **Stage D**. It uses the correlated player DK sims
already built in `deliverables/` (Stage C output) and asks you for the only
inputs the pipeline doesn't produce itself, then simulates contest outcomes for
machine-developed candidate lineups.

```
deliverables/{hitter,pitcher}_dk_sims.npy   (Stage C, already built)
        +  your ownership CSV  ─────────────►  app.py  ─►  candidate lineups
                                                            + simulated outcomes
```

## Run

```bash
pip install -r requirements.txt      # includes streamlit
streamlit run app.py
```

Then open the URL it prints (default http://localhost:8501).

## What you provide (the app forces every choice)

1. **Slate file** — either of:
   - a **DraftKings salaries export** (`DKSalaries.csv`, the one with the player
     table). It already carries salary, position, team **and player IDs**, so the
     *same* file powers both the simulation and the upload export — no separate
     template needed. Because it has no ownership, the app then asks for a small
     ownership CSV (`FullName, Ownership`) to merge in; or
   - a **clean CSV** with columns `FullName, Team, Position, Salary, Ownership`
     (add an optional `ID` column to enable the DK upload without a template).

   Only players that match the sim universe are used; the app shows how many
   matched and how many DK IDs are available. `sample_ownership.csv` is a
   synthetic clean-CSV example (fabricated salaries/ownership — **not** real DK
   data).
2. **Contest size** — number of entries in the simulated field.
3. **Number of sim runs** — how many of the available correlated sims to score
   the contest over (capped at the number built, currently 10,000). More runs =
   smoother estimates, slower.
4. **Number of candidate lineups** — how many lineups to develop and evaluate.

Nothing runs until all four are set and you press **Run simulation**.

> **The ownership CSV defines the entire player universe.** Both the field and
> the candidate lineups are built only from players in your uploaded CSV (that
> also have sims). Nothing outside your ownership file can appear in either.

## Freshness check on every Run (auto-refresh)

When you press **Run**, before anything else the app makes sure the underlying
data is current:

1. **Projections** — if the projection outputs in `out/` weren't built *today*,
   it rebuilds projections **and** the correlated sims (Stage A–C via
   `refresh_and_run.py`).
2. **Confirmed lineups** — if projections are already current, it fetches the
   live lineup feed and compares it to the slate the sims were built from; if a
   confirmed lineup changed, it reruns the correlated sims (Stage C via
   `run_slate.py`).

This runs in a live progress panel and then continues straight into the contest
sim. It needs network access (Statcast/statsapi and the lineup feed); if a
refresh can't run or fails, the app warns and falls back to the existing sims
rather than stopping. A full projection rebuild can take several minutes.

## Download a filled DraftKings upload file (step 3)

After a run, you can export a ready-to-upload DK file:

1. **Player IDs come from the slate file you already uploaded** — if it was a
   DraftKings export (or a clean CSV with an `ID` column), no further upload is
   needed. Only if your slate file had no IDs does the app ask for a DKSalaries
   template (once).
2. Choose **how many lineups** to export and how to **rank** them —
   **Win%**, **Top10 Rate**, or **Top100 Rate**.
3. Optionally set **exposure caps** (max share of exported lineups any one
   player, or any one primary-stack team, may appear in).
4. Download `DK_upload_<N>.csv` — header `P,P,C,1B,2B,3B,SS,OF,OF,OF` followed by
   one row of DraftKings player IDs per lineup.

Tweaking the count / sort / caps re-selects instantly without re-simulating.

> A lineup is exportable only if every one of its players has a DK ID. The app
> reports how many candidate players had IDs, so if your ownership and your DK
> IDs cover different players you'll see it immediately.

## What it does

1. Joins your CSV with the sim universe → a scorable player pool (multi-position
   expanded, `SP/RP → P`, opponents inferred from the sims).
2. Builds a uniform, ownership-blind **candidate** pool (the lineups you're
   evaluating).
3. Builds an ownership-weighted **field** at your contest size, using the
   contest-size chalk model (chalk sharpens in small fields, flattens in large).
4. Inserts each candidate into the field per sim and ranks it →
   **Win% / Top10% / Top100% / AvgPlace** for every candidate.

## Output & inspecting lineups

- Headline metrics + a ranked **candidate-lineups** table (Win% / Top10% /
  Top100% / Avg-Best-Worst place / salary / stack).
- **Select any row** to inspect that lineup: a clean, player-focused table
  (slot, player, team, position, salary) with its key rates and best/avg/worst
  finishing place.
- **📊 Show finishing-position distribution** — a histogram of where that lineup
  finished across *all* sim runs, with dashed markers at 1st, Top-10, Top-100
  and the lineup's mean place.
- Downloads: candidate results, candidate lineups, and the field — each as CSV.

## Advanced (optional)

The "Advanced field model" expander exposes the field knobs — medium baseline
size, chalk sensitivity, stack-shape tilt, and the field/candidate RNG seeds —
all with sensible defaults. These shape the field, not the requirement to choose
the four inputs above.

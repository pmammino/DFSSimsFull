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
     (column-name variants like `Name`, `TeamAbbrev`, `Pos`, `Own` are accepted).
     Include a player-ID column — named `ID`, `Id`, `Player ID`, `DK ID`,
     `player_id`, or a DK `Name + ID` column — to enable the DK upload without a
     separate template. The app shows which column it read IDs from.

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

## Freshness check on every Run (auto-refresh, persistent)

When you press **Run**, the app reads the **live** lineup/matchup feed and
compares it against a build stamp (`out/.build_stamp.json`) that records the
game day, every team's batting order, and every team's starting pitcher as of
the last build. The projection rebuild and the sim rebuild are **decoupled**:

1. **Projections (Stage B, best-effort).** If projections aren't from today it
   tries to rebuild them (`run_pipeline.py`). If that fails, it **keeps going**
   on the existing projections (they change little day to day) and shows the
   error; it won't re-attempt the same day.
2. **Correlated sims (Stage C, the daily essential).** It rebuilds the sims from
   **today's live slate** — lineups, starting-pitcher matchups, and Vegas game
   totals (`run_slate.py`, using the existing projections) — whenever the slate
   moved, a new game day started, projections were rebuilt, or no sims exist.
   This is what pulls the new lineups/matchups/totals into the sims.
3. **Nothing changed** → no rebuild; it uses the sims already on disk.

A **Force full refresh** checkbox (in step 2) rebuilds both projections and
sims regardless of staleness and bypasses the once-a-day projection-retry guard
— use it right after fixing a data/connection issue.

Because Stage B and Stage C run independently, a projection-build failure no
longer blocks the sim rebuild. The progress panel shows the live slate it read
and exactly what changed (e.g. `ATL SP Sale→Lopez`, `NYM lineup`, `game day …`),
and surfaces the real error if a stage fails. State is written to the stamp on
every successful build, so it **persists across sessions/restarts**.

It needs network access (Statcast/statsapi and the lineup feed). If the feed
can't be reached, the app says so and falls back to the existing sims rather
than silently using stale data.

**Stage B (projections) requirements.** The projection rebuild additionally
needs `scikit-learn`, `xgboost`, `pybaseball`, `pyarrow` and reachable
`statsapi.mlb.com`. Before attempting it the app preflights those packages and,
if any aren't importable, skips the rebuild with a precise message instead of a
long failure (a common case on **Python 3.14**, where `scikit-learn`/`xgboost`
may not have wheels yet — run the app on Python 3.11–3.12 for the full
projection rebuild). If statsapi is unreachable, the pipeline now fails with a
clear "statsapi returned no rate data" message rather than a cryptic error. In
all these cases the **sims still rebuild** from today's slate on the existing
projections.

**Starter guard.** Independently of the sims, the pool is filtered to only the
pitchers confirmed as today's **starters on the live slate**. So a pitcher who
isn't starting (e.g. threw yesterday) can never appear in a lineup even if the
sims are a build behind — the app reports any pitchers it excluded.

> Persistence is file-based (`deliverables/`, `out/`, and the stamp). On a local
> or persistent server these survive restarts. In an **ephemeral/cloud** session
> that re-clones the repo each time, point those at a persistent volume (or
> commit them) so they carry over.

## Download a filled DraftKings upload file (step 3)

After a run, you can export a ready-to-upload DK file:

1. **Player IDs come from the slate file you already uploaded** — if it was a
   DraftKings export (or a clean CSV with an `ID` column), no further upload is
   needed. Only if your slate file had no IDs does the app ask for a DKSalaries
   template (once).
2. Choose **which lineups to export**:
   - **My marked selections** — exactly the lineups you ticked above, in rank
     order; or
   - **Top N by ranking** — choose how many and rank by **Win%**, **Top10
     Rate**, or **Top100 Rate**, with optional per-player / stack-team exposure
     caps.
3. Download `DK_upload_<N>.csv` — header `P,P,C,1B,2B,3B,SS,OF,OF,OF` followed by
   one row of DraftKings player IDs per lineup.

Tweaking the selection / count / sort / caps re-selects instantly without
re-simulating.

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

## Output, filtering & inspecting lineups

- Headline metrics + a **candidate-lineups** table you can mark up and filter.
- **🔎 Filter & search** by: player(s) the lineup must include (all/any), stack
  shape (build style), primary stack team, primary stack size, combined
  ownership %, salary range, and minimum Win% / Top10% / Top100%.
- **Mark off lineups** — tick the ✓ column on any rows (or **Mark all** the
  filtered set); marks persist across filtering and feed the export below.
- **Inspect a lineup** — pick one to see a clean, player-focused table (slot,
  player, team, position, salary) plus its rates and best/avg/worst place, and a
  one-click mark/unmark.
- **📊 Show finishing-position distribution** — a histogram of where that lineup
  finished across *all* sim runs, with dashed markers at 1st, Top-10, Top-100
  and the lineup's mean place.
- Downloads: filtered results, all candidate lineups, and the field — each CSV.

## Advanced (optional)

The "Advanced field model" expander exposes the field knobs — medium baseline
size, chalk sensitivity, stack-shape tilt, and the field/candidate RNG seeds —
all with sensible defaults. These shape the field, not the requirement to choose
the four inputs above.

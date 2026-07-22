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

## Design — RotoWire (full dark) + Tabbed Workspace

The UI implements the **RotoWire product design system** (from the Claude Design
handoff): full-dark **navy/red** theme — **Navy #002248** panels on a **Dark
#000D1A** background, **Red #F22E45** accent, and **White #FFFFFF** text — the
licensed brand fonts (**Integral CF** uppercase display, **Cosmica** body,
**Cosmica Mono** labels, served from `static/fonts/` via Streamlit static
serving), a branded header lockup with an animated **SLATE** date badge, and
RotoWire-styled stat cards, pill badges, tabs (red active underline), buttons,
and tables.

The flow is organized as a **Tabbed Workspace** (design Option C):

- **⚙️ Setup** — pick the slate from the **RotoWire feed**; review the day's
  **Vegas team totals**.
  The loaded totals **drive the sim by default**: each team's offense is scaled
  by its implied total vs a fixed 4.2-run league average, so a high-total team
  gets a higher **mean and a fatter ceiling** (its lineup booms together more on
  its big days)
  and **tanks the opposing starter** (more hits/HR/runs allowed), while a
  low-total team is suppressed — no editing required. Editing is **optional** and
  just replaces a team's total before that same scaling (it rebuilds the sims and
  visibly reshapes that team's player projections). Only a flat/failed feed
  forces ≥2 manual edits before Run enables. Then configure
  contest size / sim runs / candidates / tilts, and Run.
- **📊 Players** — per-player projected ranges & thresholds. (The app jumps here
  automatically when a run finishes, as a "done" indicator.)
- **🏆 Results** — candidate lineups: metrics, filter/search, quick export.
- **⬇️ Export** — build the DraftKings upload file.

### Classic vs. Showdown slates

The app supports both DraftKings contest formats and routes automatically on the
slate you pick from the RotoWire feed (each slate is tagged Classic or Showdown):

- **Classic** — the standard 2×P + C/1B/2B/3B/SS/3×OF roster with hitter stacks
  (everything described elsewhere in this doc).
- **Showdown / Captain Mode** — a single game, **1 CPT + 5 UTIL**. The captain
  scores **1.5× points** and costs **1.5× salary**; any player (hitter or
  pitcher) may fill any slot; rosters must include players from **both teams**.
  The same correlated sims drive it — only the roster, the captain multiplier,
  and the opponent-field model differ. The field is **ownership-driven with a
  captain ceiling-tilt** (studs get captained more than flex ownership implies);
  it is a heuristic, not yet calibrated to real showdown standings. Results shows
  a **CPT/UTIL** table with captain and team-split filters; Export offers the
  same ranked / Portfolio-EV selection under **per-player / per-captain /
  per-team** exposure caps.

  **Showdown upload:** the RotoWire feed carries only each player's flex ID, so
  the Export tab **requires you to upload a DraftKings `DKSalaries.csv`** for the
  slate — it supplies the distinct **Captain-slot and UTIL-slot** IDs needed for
  a valid upload (the captain is written under its CPT id).

Theme tokens live in `.streamlit/config.toml`. Drop your logo at
**`assets/logo.svg`** (or `.png`/`.jpg`/`.webp`) — it's used in the header
lockup and as the favicon (favicon needs a raster file).

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
     separate template. If both a **contest ID** (`PlayerContestID`, `Contest ID`,
     `Draftable ID`) and a generic player ID are present, the export uses the
     **contest ID** (what DraftKings uploads require). The app shows which column
     it read IDs from.

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

1. **Projections (Stage B, best-effort — the SLOW step).** If projections aren't
   from today it tries to rebuild them (`run_pipeline.py`). If that fails, it
   **keeps going** on the existing projections (they change little day to day)
   and shows the error; it won't re-attempt the same day. **This is the
   expensive stage**, so it's best done ahead of time by the scheduled morning
   job (see *Scheduled morning rebuild* below) — when that has run, projections
   are already dated today and this step is skipped on every interactive Run.
2. **Correlated sims (Stage C, the daily essential).** It rebuilds the sims from
   **today's live slate** — lineups, starting-pitcher matchups, and Vegas game
   totals (`run_slate.py`, using the existing projections) — whenever the slate
   moved, a new game day started, projections were rebuilt, or no sims exist.
   This is what pulls the new lineups/matchups/totals into the sims.
3. **Nothing changed** → no rebuild; it uses the sims already on disk.

A **Force full refresh** checkbox (in step 2) rebuilds both projections and
sims regardless of staleness and bypasses the once-a-day projection-retry guard
— use it right after fixing a data/connection issue.

### Scheduled morning rebuild (so Stage B never blocks a Run)

The projection rebuild (Stage B) is by far the slowest part of a cold Run. To
keep it off the interactive path, a scheduled job rebuilds the projections
**every morning** and publishes them; the app then finds them already dated
today and skips Stage B, leaving only the fast Stage C re-sim when lineups move.

- **Deployed (R2/S3 + Actions):** `.github/workflows/refresh.yml` runs daily at
  13:00 UTC (~9am ET). It runs `refresh_and_run.py --skip-bip` (Stage B + C,
  reusing the committed BIP inputs), stamps the build with
  `scripts/stamp_build.py --projections`, and pushes to the object store. The
  **full** dispatch mode additionally runs the Statcast BIP scrape (Stage A);
  run it occasionally to refresh the underlying data. See DEPLOYMENT.md, Step 5.
- **Single server (no object store):** schedule the same two commands with cron
  so they write to the app's `deliverables/`+`out/` directly:

  ```cron
  0 7 * * *  cd /path/to/DFSSimsFull && python refresh_and_run.py --skip-bip \
               && python scripts/stamp_build.py --projections >> refresh.log 2>&1
  ```

`scripts/stamp_build.py` writes the same `out/.build_stamp.json` the app writes
itself (`projections_date`, `slate_date`, `slate_sig`), which is exactly what
the freshness check above reads to decide the rebuild can be skipped.

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

**Playable-pool guard.** Independently of the sims, the pool is restricted to
players in today's **projected/confirmed batting orders** plus each game's
**starting / opener / primary pitcher** (per the live slate). Anyone not in
today's lineups — benched hitters, non-starting pitchers — is dropped before the
field and candidates are built (even if the sims lag), and projected ownership
is then **renormalized over the remaining pool**. The app reports how many
hitters and pitchers it excluded.

## Sharing across users on a live deployment (S3)

On Streamlit Community Cloud the filesystem is **ephemeral and per-instance**, so
local files don't persist across restarts or replicas. To share one refreshed
build across all users, configure an **S3 (or S3-compatible) bucket** in secrets:

```toml
# .streamlit/secrets.toml  (or the app's Settings → Secrets on Streamlit Cloud)
[shared_store]
bucket = "your-dfs-bucket"
prefix = "dfs"
region = "us-east-1"
# endpoint_url = "https://<acct>.r2.cloudflarestorage.com"   # R2/MinIO/B2
access_key_id = "AKIA..."
secret_access_key = "..."
```

When configured, the app:
- **pulls** the latest shared sims/projections/stamp on load (re-syncing once per
  day per session, so a tab left open across someone else's refresh still picks up
  the newer build) and again before each Run — downloading only when the shared
  build is newer than local. The Setup tab's **↻ Refresh** button also pulls on
  demand, so a second user can grab a freshly published build without running a
  rebuild themselves,
- **rebuilds under a lock** — an in-process lock plus an S3 lock object — so two
  users can't trigger the heavy rebuild at once; a user who arrives mid-rebuild
  just gets the shared result,
- **pushes** the regenerated artifacts back to S3 after a successful refresh.

A **freshness banner** at the top of the app states whether today's sims already
exist (built by anyone earlier) and are shared — so if a teammate has already run
the refresh for the day, the next user just makes their selections and runs the
scoring on those same sims, no rebuild needed.

So one daily refresh is shared by everyone and survives restarts. With no
`[shared_store]` config the app runs purely on the local filesystem, where all
sessions on that one instance still share the same on-disk sims (via the build
stamp + mtime-keyed cache); the S3 store is what extends that sharing across
restarts and replicas.
Requires `boto3` (in `requirements.txt`). See `.streamlit/secrets.toml.example`.

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
     Rate**, or **Top100 Rate**, with optional separate **hitter / pitcher /
     stack-team** exposure caps. Ranks from the **current filter** by default
     (or all candidates).
   - Within *Top N by ranking* you can also pick a **Selection method**:
     *Ranked (per-lineup rates)* (the classic behaviour above) or **Portfolio EV
     (payout-aware)** — see below.
3. Download `DK_upload_<N>.csv` — header `P,P,C,1B,2B,3B,SS,OF,OF,OF` followed by
   one row of DraftKings player IDs per lineup.

Tweaking the selection / count / sort / caps re-selects instantly without
re-simulating.

### Portfolio diversity

A ranked portfolio tends to pile onto the single best build — the same teammates
fill every stack and the same primary team is always paired with the same
secondary. Two places now let you spread it (all default to *no effect*, so
behaviour is unchanged until you turn them on):

- **Candidate diversity jitter** (Setup → *Advanced field model*) — adds a
  per-pick random shock to every weighted selection while *developing*
  candidates. Higher values let two near-equally-projected players at the same
  price both get used, rotate a team's stack members, and pair a primary team
  with different secondaries. This diversifies the candidate **pool at the
  source**.
- **Portfolio diversity** (Export → *Top N by ranking*) — diversifies the
  **exported set** after ranking:
  - **Max stack-pairing exposure** — caps the share of lineups sharing the same
    *(primary, secondary)* team pair (e.g. spread a Cleveland primary across
    several secondaries instead of always Kansas City).
  - **Max stack-core exposure** — caps the share using the exact same set of
    primary-stack hitters, forcing the stack to rotate teammates.
  - **Max lineup similarity** — rejects a lineup that overlaps an
    already-exported one by more than the chosen fraction of players (Jaccard).
  - **Value groups** — the app auto-detects near-twin players (same position,
    similar salary and projection) and a **Max value-group exposure** cap shares
    the load across them, so a hair-higher projection doesn't let one player eat
    the whole group's exposure.

> A lineup is exportable only if every one of its players has a DK ID. The app
> reports how many candidate players had IDs, so if your ownership and your DK
> IDs cover different players you'll see it immediately.
>
> When two players share a name on the same slate (e.g. **Max Muncy** on the
> Dodgers *and* the Athletics), the upload ID is resolved by the player's
> **team**, so the lineup's actual player — not whichever one happened to be
> listed last — is the one written to the DK file.

### Portfolio EV (payout-aware selection)

Ranked selection judges every lineup **in a vacuum** — it takes the highest
Win% / Top100% builds. The catch is those builds almost all win in the *same*
simulations (the slates where the same chalk stack booms), so an exported set
that looks diverse on paper is concentrated in **outcome-space** and tends to
succeed or fail together on a given slate.

**Portfolio EV** optimizes the set as a whole. It replays the correlated
simulations as **dollar outcomes** and greedily picks the lineups that maximize
the expected *utility* of the whole portfolio's per-slate return — so each added
lineup is chosen for the slate outcomes it covers that the set doesn't already
win. All the exposure/diversity caps above still apply as hard constraints.

Controls (Export → *Top N by ranking* → Selection method → **Portfolio EV**):

- **Payout structure** — a parametric top-heavy GPP prize curve built from the
  **entry fee**, **% of field paid**, **rake**, and **top-heaviness** (flat
  double-up ↔ winner-take-most). The contest size is your simulated field, so a
  finishing place maps coherently onto the prize table.
- **How should the portfolio play out?** — the risk posture (the utility knob):
  - **Aggressive (max ceiling)** — near-linear utility; chases raw expected
    dollars, barely diversifies (best for large-field GPP ceiling).
  - **Balanced** — square-root utility; spreads winning sims across slate
    outcomes without giving up much ceiling (default).
  - **Conservative (consistent cashing)** — log / Kelly-style utility; strong
    boom/bust aversion, prioritizes cashing across as many slate states as
    possible.
- **Candidate pool size** — the optimizer picks from this many top-ranked
  candidates; larger = more freedom to diversify, slower.

After selecting, a **coverage panel** shows the portfolio's outcome across every
simulated slate — expected return, ROI, **cash rate** (share of slates where at
least one exported lineup finishes in the money — the portfolio-level cash
metric), floor/ceiling, and a return distribution — all compared against a
top-N-by-rank set of the same size drawn from the same pool, so you can see the
boom/bust being broken up.

> Requires a fresh simulation: the payout-aware path reuses the field placement
> ladder and per-player sim arrays captured during the run. If you loaded an
> older session, re-run the sim to enable it (the app falls back to ranked
> selection and tells you).

## What it does

1. Joins your CSV with the sim universe → a scorable player pool (multi-position
   expanded, `SP/RP → P`, opponents inferred from the sims).
2. Builds a uniform, ownership-blind **candidate** pool (the lineups you're
   evaluating).
3. Builds an ownership-weighted **field** at your contest size, using the
   contest-size chalk model (chalk sharpens in small fields, flattens in large).
4. Inserts each candidate into the field per sim and ranks it →
   **Win% / Top10% / Top100% / AvgPlace** for every candidate.

## Players — projected ranges & thresholds

A **📊 Players** section (expander above the workflow) shows, for every simmed
player, the DK-point distribution from the current sims: projected mean, floor
(p10) / median / ceiling (p90) / p99, min/max, std, and bust (≤0) / 2× / 30+
rates. Filter by hitter/pitcher or name, download the table, and pick any player
to see its outcome histogram. It reflects the latest refreshed sims.

## Output, filtering & inspecting lineups

- Headline metrics + a **candidate-lineups** table you can mark up and filter.
- **🔎 Filter & search** by: player(s) the lineup must include (all/any),
  player(s) to **exclude**, stack shape (build style), primary stack team,
  primary stack size, combined ownership %, salary range, and minimum
  Win% / Top10% / Top100%.
- **Mark off lineups** — tick the ✓ column on any rows (or **Mark all** the
  filtered set); marks persist across filtering and feed the export below.
- **Inspect a lineup** — pick one to see a clean, player-focused table (slot,
  player, team, position, salary) plus its rates and best/avg/worst place, and a
  one-click mark/unmark.
- **📊 Finishing-position detail** — click a lineup row in the picker to see a
  solid, full-height histogram of where it finished across *all* sim runs (with
  dashed markers at 1st, Top-10, Top-100 and the lineup's mean place) alongside
  the actual lineup it's built on (slot, player, team) and its best/avg/worst
  place, with a one-click mark/unmark.
- Downloads: filtered results, all candidate lineups, and the field — each CSV.

## Advanced (optional)

The "Advanced field model" expander exposes:

- **Field knobs** — medium baseline size, chalk sensitivity, stack-shape tilt,
  and the field/candidate RNG seeds.
- **Candidate talent tilt (players)** — how strongly candidate lineups favor
  higher-projected **players** (incl. one-off bats on any team) when filling
  stacks, one-off slots, and pitchers. Applied as `exp(tilt·z)` on each player's
  projected value, so it's a temperature: 0 = projection-blind uniform; ~0.7
  (default) ≈3.5× more for the top players; higher sharpens further. **Stack
  teams stay diverse** at any setting.
- **Candidate stack-team tilt (Vegas/talent)** — *separate* lever for how
  strongly candidates stack the better **offenses** (team scoring power from the
  sims, applied as a z-score softmax). **0 = default** (every team equally likely
  to be stacked); raise it to concentrate stacks on the top offenses (e.g. ~0.8
  puts ~half the primary stacks on the top-5 offenses).
- **Stack-ownership ceiling boost** — uses projected **stack ownership** (sum of
  a team's hitter ownership) as a small *upside* signal. In each team's high-end
  sims (its top ~20% of games) that team's hitters' DK points are scaled up by a
  factor that grows with the team's stack ownership — the lowest-owned stack gets
  no bump, the chalkiest gets the full slider value. The same boosted sims score
  **both** the field and your candidates, so popular stacks hit their ceiling a
  touch more often and the **top projected stacks surface a bit higher** in the
  results, without the optimizer being driven entirely onto contrarian, low-owned
  stacks. **0.05 = default** (a gentle nudge); 0 turns it off (pure projection).
  Correlation is preserved — a stack still booms together, just slightly bigger.

All default to sensible values and don't change the four required inputs.

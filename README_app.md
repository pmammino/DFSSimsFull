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

1. **Ownership CSV** — a DraftKings salary/ownership export with columns:
   `FullName, Team, Position, Salary, Ownership`
   (`Ownership` = projected draft %, 0–100). Only players that match the sim
   universe are used; the app shows how many matched. A synthetic example you
   can load to try it out is `sample_ownership.csv` (fabricated salaries/
   ownership for the current slate — **not** real DK data).
2. **Contest size** — number of entries in the simulated field.
3. **Number of sim runs** — how many of the available correlated sims to score
   the contest over (capped at the number built, currently 10,000). More runs =
   smoother estimates, slower.
4. **Number of candidate lineups** — how many lineups to develop and evaluate.

Nothing runs until all four are set and you press **Run simulation**.

## What it does

1. Joins your CSV with the sim universe → a scorable player pool (multi-position
   expanded, `SP/RP → P`, opponents inferred from the sims).
2. Builds a uniform, ownership-blind **candidate** pool (the lineups you're
   evaluating).
3. Builds an ownership-weighted **field** at your contest size, using the
   contest-size chalk model (chalk sharpens in small fields, flattens in large).
4. Inserts each candidate into the field per sim and ranks it →
   **Win% / Top10% / Top100% / AvgPlace** for every candidate.

## Output

- On-page sortable results table (best-first) plus headline metrics.
- Downloads: candidate results, candidate lineups, and the field — each as CSV.
- The top candidate lineup is broken out player-by-player.

## Advanced (optional)

The "Advanced field model" expander exposes the field knobs — medium baseline
size, chalk sensitivity, stack-shape tilt, and the field/candidate RNG seeds —
all with sensible defaults. These shape the field, not the requirement to choose
the four inputs above.

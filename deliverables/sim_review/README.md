# Sim-engine performance review — reproducible analysis

`SIM_ENGINE_PERFORMANCE_REVIEW.md` is the write-up. To reproduce the numbers:

1. Unzip the sim history so the `.npy` files live in a `History/` dir, then point `HIST_DIR` at it:
   ```
   HIST_DIR=/path/to/History python3 grade.py     # bias / MAE / RMSE / PIT / coverage
   HIST_DIR=/path/to/History python3 grade2.py    # per-day, boom coverage, biggest misses, tiers
   HIST_DIR=/path/to/History python3 grade3.py    # predicted-vs-observed tail probabilities
   ```
2. `actuals/2026-07-2X.json` hold the actual box-score lines (see *Data provenance* in the report:
   pitcher lines near-complete/high-confidence, hitter lines partial/best-effort).

Actual DK points are computed with the engine's own `dk_hitter` / `dk_pitcher` formulas so scoring
matches the sim exactly.

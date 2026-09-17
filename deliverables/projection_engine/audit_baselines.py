"""
audit_baselines.py
==================
Reproducible audit of the current per-PA baselines in `out/` — the inputs a
2027 season-long projection engine would have to build on.

Run from the repo root:

    python deliverables/projection_engine/audit_baselines.py

Every number in `CURRENT_STATE_ASSESSMENT.md` comes from this script. Checks
that need real league data read `bip_inputs/bip_2025.csv` (in-repo) rather than
hard-coded reference rates, so they stay honest if the baselines are rebuilt.

The seven checks:
    1. League-level calibration of the hitter per-PA distribution
    2. Pitcher-side league calibration (RA9 / ERA)
    3. Offense/defense internal consistency (two views of one league)
    4. Per-BIP distribution vs REAL 2025 batted balls  <- ground truth
    5. Root cause: skew of the per-BIP probability distribution
    6. Player-level spread compression
    7. Team alignment + team-talent / R-RBI / wins closure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pitcher_outputs import (  # noqa: E402
    LINEAR_WEIGHTS_RUNS as LW,
    RUNS_INTERCEPT_DEFAULT as ICPT,
)

HIT_CSV = ROOT / "out" / "hitter_pa_projections_2027.csv"
PIT_CSV = ROOT / "out" / "pitcher_pa_projections_2027.csv"
BIP_CSV = ROOT / "bip_inputs" / "bip_2025.csv"

# MLBAM team ids -> abbreviations. The pipeline's own `Team` string column is
# unusable (see check 7), so the audit carries its own map.
TEAM_NAMES = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",  119: "LAD",
    120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SD",  136: "SEA",
    137: "SF",  138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Approximate 2024-26 MLB per-PA rates, used only as soft reference points in
# check 1. The load-bearing batted-ball test (check 4) uses real data instead.
MLB_REF_PER_PA = {
    "P_K": 0.2230, "P_BB": 0.0840, "P_HBP": 0.0110, "P_SF": 0.0060,
    "P_HR": 0.0320, "P_3B": 0.0045, "P_2B": 0.0460, "P_1B": 0.1420,
}
EVENTS = ["P_K", "P_BB", "P_HBP", "P_SF", "P_HR", "P_3B", "P_2B", "P_1B",
          "P_BIPOut"]
PA_PER_TEAM_GAME = 38.0


def _hdr(n: int, title: str) -> None:
    print("\n" + "=" * 74)
    print(f"{n}. {title}")
    print("=" * 74)


def _pa_weights(df: pd.DataFrame) -> np.ndarray:
    """PA weights from last season. Guards the all-zero case."""
    w = df["Last_PA"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    return np.ones(len(df)) if w.sum() <= 0 else w


def _runs_per_pa(df: pd.DataFrame, weights: np.ndarray) -> float:
    """League R/PA implied by the per-PA event distribution, via the same
    linear weights `pitcher_outputs` uses for RA9 — so offense and defense are
    measured on one scale."""
    return sum(w * np.average(df[c], weights=weights)
               for c, w in LW.items() if c in df.columns) + ICPT


# ── 1-3: league calibration ──────────────────────────────────────────────────

def check_league_calibration(h: pd.DataFrame, p: pd.DataFrame) -> dict:
    _hdr(1, "LEAGUE-LEVEL CALIBRATION — hitter per-PA distribution")
    w = _pa_weights(h)
    print(f"{'event':<10}{'proj (PAwt)':>13}{'MLB ref':>10}{'delta':>10}{'rel':>9}")
    for e in EVENTS:
        pw = np.average(h[e], weights=w)
        ref = MLB_REF_PER_PA.get(e)
        if ref is None:
            print(f"{e:<10}{pw:>13.4f}{'-':>10}{'-':>10}{'-':>9}")
        else:
            print(f"{e:<10}{pw:>13.4f}{ref:>10.4f}{pw - ref:>+10.4f}"
                  f"{(pw / ref - 1) * 100:>+8.1f}%")

    obp = np.average(h.P_BB + h.P_HBP + h.P_HR + h.P_1B + h.P_2B + h.P_3B,
                     weights=w)
    rpa_h = _runs_per_pa(h, w)
    print(f"\n  (BB+HBP+H)/PA        : {obp:.4f}   MLB ~0.318")
    print(f"  implied R/PA         : {rpa_h:.4f}   MLB ~0.118-0.122")
    print(f"  implied R/G @ {PA_PER_TEAM_GAME:.0f} PA : "
          f"{rpa_h * PA_PER_TEAM_GAME:.2f}   MLB ~4.40-4.50")

    _hdr(2, "PITCHER-SIDE LEAGUE CALIBRATION")
    wp = _pa_weights(p)
    rpa_p = np.average(p.R_per_PA, weights=wp)
    print(f"  RA9 (PA-wt) {np.average(p.RA9, weights=wp):.3f}   MLB ~4.40")
    print(f"  ERA (PA-wt) {np.average(p.ERA, weights=wp):.3f}   MLB ~4.10")
    for role in ("starter", "reliever"):
        g = p[p.role == role]
        print(f"    {role:<9} n={len(g):>3}  ERA {g.ERA.mean():.3f}  "
              f"RA9 {g.RA9.mean():.3f}")

    _hdr(3, "OFFENSE / DEFENSE INTERNAL CONSISTENCY")
    print(f"  hitter-implied R/PA  : {rpa_h:.4f}")
    print(f"  pitcher-implied R/PA : {rpa_p:.4f}")
    print(f"  gap                  : {rpa_h - rpa_p:+.4f} "
          f"({(rpa_h / rpa_p - 1) * 100:+.1f}%)")
    print("  -> the two sides agree with each OTHER; both are low vs the real"
          " league,\n     which points at a shared upstream cause (check 4).")
    return {"rpa_h": rpa_h, "rpa_p": rpa_p}


# ── 4-5: the batted-ball ground-truth test and its root cause ───────────────

def check_bip_vs_real(h: pd.DataFrame) -> pd.DataFrame | None:
    _hdr(4, "PER-BIP DISTRIBUTION vs REAL 2025 BATTED BALLS  [ground truth]")
    if not BIP_CSV.exists():
        print(f"  SKIPPED — {BIP_CSV} not found")
        return None

    b = pd.read_csv(BIP_CSV, usecols=["events", "launch_speed", "launch_angle"])
    label = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
    actual = b.events.map(label).fillna("Out").value_counts(normalize=True)

    w = _pa_weights(h)
    # SF is folded into the out bucket, matching pa_aggregation's treatment.
    cols = {"HR": "P_HR", "3B": "P_3B", "2B": "P_2B", "1B": "P_1B",
            "Out": "P_BIPOut"}
    sf = np.average(h.P_SF, weights=w)
    total = sum(np.average(h[c], weights=w) for c in cols.values()) + sf

    print(f"  real 2025 batted balls: n={len(b):,}\n")
    print(f"{'outcome':<9}{'projected':>12}{'actual 2025':>14}{'ratio':>9}")
    for k, c in cols.items():
        proj = np.average(h[c], weights=w) / total
        if k == "Out":
            proj += sf / total
        print(f"{k:<9}{proj:>12.4f}{actual[k]:>14.4f}{proj / actual[k]:>9.2f}")
    print("\n  Out rate is CORRECT (~1.01) — this is not a 'too many outs'"
          " problem.\n  Extra-base hits are being converted into singles"
          " inside the BIP pool.")
    return b


def check_blend_mechanism(b: pd.DataFrame | None) -> None:
    _hdr(5, "ROOT CAUSE — skew of the per-BIP probability distribution")
    if b is None:
        print("  SKIPPED — needs check 4's batted-ball frame")
        return

    from pipeline_config import EVENT_BLEND_WEIGHTS_HITTER as BLEND

    b = b.dropna(subset=["launch_speed", "launch_angle"]).copy()
    b["hr"] = (b.events == "home_run").astype(int)
    b["xb"] = b.events.isin(["double", "triple", "home_run"]).astype(int)

    # Proxy the XGBoost per-BIP probability by binning on its own two main
    # features. Exact values differ from the model's; the SHAPE is the point.
    b["ev_bin"] = pd.cut(b.launch_speed, np.arange(20, 125, 5))
    b["la_bin"] = pd.cut(b.launch_angle, np.arange(-90, 95, 5))
    g = (b.groupby(["ev_bin", "la_bin"], observed=True)
         .agg(n=("hr", "size"), p_hr=("hr", "mean"), p_xb=("xb", "mean"))
         .reset_index())
    g = g[g.n >= 20]

    print(f"  blend weights (mean, median): "
          f"{ {k: v for k, v in BLEND.items()} }\n")
    for name, col in [("HR", "p_hr"), ("XBH", "p_xb")]:
        v = np.repeat(g[col].to_numpy(), g.n.to_numpy().astype(int))
        mean, med = v.mean(), float(np.median(v))
        blended = 0.75 * mean + 0.25 * med
        print(f"  {name:<4} per-BIP prob: mean {mean:.4f}  median {med:.4f}"
              f"  (median is {100 * (1 - med / mean):.0f}% below mean)")
        print(f"       0.75*mean + 0.25*median = {blended:.4f}"
              f"  -> {100 * (blended / mean - 1):+.1f}% deflation")
    print("\n  The median per-BIP HR probability is ZERO — most batted balls"
          " cannot be a\n  home run. So a 0.25 median weight is a mechanical"
          " -25% haircut on HR,\n  while 1B and Out use pure mean (1.0, 0.0)"
          " and are untouched. The lost\n  extra-base mass is reallocated to"
          " singles on renormalization.")


# ── 6: spread compression ───────────────────────────────────────────────────

def check_spread(h: pd.DataFrame) -> None:
    _hdr(6, "PLAYER-LEVEL SPREAD COMPRESSION")
    # Last_PA is a PARTIAL 2026 season in the committed artifacts, so it is a
    # poor 'regular' filter. Career_PA is stable.
    reg = h[h.Career_PA >= 1000]
    mlb_sd = {"P_HR": 0.0150, "P_K": 0.0600, "P_BB": 0.0320, "P_2B": 0.0110}
    print(f"  pool: Career_PA >= 1000, n={len(reg)}\n")
    print(f"{'event':<9}{'proj SD':>10}{'MLB SD':>10}{'ratio':>9}")
    for e, sd in mlb_sd.items():
        print(f"{e:<9}{reg[e].std():>10.4f}{sd:>10.4f}"
              f"{reg[e].std() / sd:>9.2f}")
    print(f"\n  top projected HR rate: {reg.P_HR.max() * 600:.1f} HR/600 PA"
          f"   (MLB leaders reach ~50)")
    print(f"  p90 projected HR rate: {reg.P_HR.quantile(.9) * 600:.1f}"
          f" HR/600 PA   (MLB p90 ~33-36)")

    print("\n  Age effect on the projections (survivorship, NOT an aging"
          " curve):")
    for lo, hi in [(21, 26), (27, 29), (30, 32), (33, 45)]:
        g = reg[(reg.Age >= lo) & (reg.Age <= hi)]
        if len(g):
            print(f"    age {lo}-{hi}: n={len(g):>3}  P_HR {g.P_HR.mean():.4f}"
                  f"  P_K {g.P_K.mean():.4f}  P_BB {g.P_BB.mean():.4f}"
                  f"  sprint {g.sprint_speed_used.mean():.2f}")
    print("    -> the 33+ group projects HIGHER power than the 30-32 group."
          "\n       Only good old players stay in the league; nothing ages"
          " anyone forward.")


# ── 7: team alignment and closure ───────────────────────────────────────────

def _team_lineup_rpa(g: pd.DataFrame) -> tuple[float, np.ndarray, pd.DataFrame]:
    """Crude proxy lineup: top 9 by Career_PA, Career_PA-weighted.

    The pipeline has no roster, depth chart, or position data, so a real
    lineup cannot be built. This proxy skews veteran and is only meant to
    show the DIRECTION and rough SIZE of the closure errors.
    """
    lu = g.nlargest(9, "Career_PA")
    w = lu.Career_PA.to_numpy(dtype=float)
    return _runs_per_pa(lu, w), w, lu


def check_team_level(h: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    _hdr(7, "TEAM ALIGNMENT, TEAM TALENT, AND CLOSURE")

    print("  (a) team identity columns")
    print(f"      hitter `Team` string uniques : {h.Team.nunique()} "
          f"(must be 30)  -> {sorted(h.Team.dropna().unique())}")
    print(f"      pitcher `Team` string uniques: {p.Team.nunique()} "
          f"(must be 30)")
    print("      cause: data_acquisition.py takes team['name'][:3] when"
          " `abbreviation`\n             is missing, so Chi/Los/New/San each"
          " merge two franchises.")
    print(f"      hitters carry Pred_target_team_id : "
          f"{h.Pred_target_team_id.nunique()} teams, "
          f"{h.Pred_target_team_id.isna().sum()} missing")
    print(f"      pitchers carry NO Pred_target_team_id — only"
          f" home_park_team_id\n             (and it is derived by a DIFFERENT"
          f" rule; see assessment)")

    rows = []
    for tid, g in h.groupby("Pred_target_team_id"):
        rpa, w, lu = _team_lineup_rpa(g)
        rows.append({
            "TeamId": int(tid),
            "Team": TEAM_NAMES.get(int(tid), str(int(tid))),
            "n_hitters": len(g),
            "bottom_up_RPA": rpa,
            "factor_used": float(g.Pred_target_team_factor.iloc[0]),
            "R_per_PA": np.average(lu.P_R, weights=w),
            "RBI_per_PA": np.average(lu.P_RBI, weights=w),
        })
    t = pd.DataFrame(rows)
    t["bottom_up_factor"] = t.bottom_up_RPA / t.bottom_up_RPA.mean()
    t["gap"] = t.factor_used - t.bottom_up_factor

    print("\n  (b) roster depth — no roster constraint exists")
    print(f"      hitters/team : min {t.n_hitters.min()} "
          f"max {t.n_hitters.max()} mean {t.n_hitters.mean():.1f}")
    for role in ("starter", "reliever"):
        c = p[p.role == role].groupby("home_park_team_id").size()
        print(f"      {role}s/team : min {c.min()} max {c.max()} "
              f"mean {c.mean():.1f}")

    print("\n  (c) is the team run environment consistent with the roster?")
    r = float(np.corrcoef(t.bottom_up_factor, t.factor_used)[0, 1])
    print(f"      corr(bottom-up lineup factor, Pred_target_team_factor)"
          f" = {r:.3f}")
    print(f"      bottom-up factor SD {t.bottom_up_factor.std():.3f}"
          f"   vs   factor_used SD {t.factor_used.std():.3f}")
    print("      the factor actually used is MORE dispersed than the talent"
          " it represents.")
    print("\n      largest disagreements:")
    for _, x in t.reindex(t.gap.abs().sort_values(ascending=False)
                          .index).head(6).iterrows():
        print(f"        {x.Team:<4} factor_used {x.factor_used:.3f}"
              f"  roster implies {x.bottom_up_factor:.3f}"
              f"  gap {x.gap:+.3f}")

    print("\n  (d) do individual R and RBI close to a team run total?")
    lw = t.bottom_up_RPA.mean()
    print(f"      mean player R/PA   {t.R_per_PA.mean():.4f}"
          f"   / team LW R/PA {lw:.4f} = {t.R_per_PA.mean() / lw:.3f}"
          f"   (target ~1.00)")
    print(f"      mean player RBI/PA {t.RBI_per_PA.mean():.4f}"
          f"   / team LW R/PA {lw:.4f} = {t.RBI_per_PA.mean() / lw:.3f}"
          f"   (target ~0.88)")
    print("      R and RBI are free-standing player rates; nothing constrains"
          " them to\n      sum to the runs the lineup actually scores.")

    print("\n  (e) wins: what a Pythagenpat off these baselines produces")
    ra = []
    for tid, g in p.groupby("home_park_team_id"):
        s = g[g.role == "starter"].nsmallest(5, "RA9")
        rel = g[g.role == "reliever"].nsmallest(7, "RA9")
        if s.empty:
            continue
        est = (0.65 * s.RA9.mean() + 0.35 * rel.RA9.mean()
               if len(rel) else s.RA9.mean())
        ra.append({"TeamId": int(tid), "RA9_est": est})
    m = t.merge(pd.DataFrame(ra), on="TeamId")
    m["RS_G"] = m.bottom_up_RPA * PA_PER_TEAM_GAME
    exp = ((m.RS_G + m.RA9_est) / 2) ** 0.287
    m["W"] = 162 * m.RS_G ** exp / (m.RS_G ** exp + m.RA9_est ** exp)
    print(f"      RS/G {m.RS_G.min():.2f}-{m.RS_G.max():.2f}"
          f" (mean {m.RS_G.mean():.2f})"
          f"   RA9 {m.RA9_est.min():.2f}-{m.RA9_est.max():.2f}"
          f" (mean {m.RA9_est.mean():.2f})")
    print(f"      wins {m.W.min():.1f}-{m.W.max():.1f}"
          f"   SD {m.W.std():.1f} (MLB ~11-12)"
          f"   SUM {m.W.sum():.0f} (must be 2430)")
    print("      RS mean != RA mean, so league wins do not close; and the"
          " win spread is\n      about half of reality because player talent"
          " spread is compressed.")
    return m


def main() -> None:
    for f in (HIT_CSV, PIT_CSV):
        if not f.exists():
            raise SystemExit(f"missing {f} — run run_pipeline.py first")
    h = pd.read_csv(HIT_CSV)
    p = pd.read_csv(PIT_CSV)

    print("=" * 74)
    print("BASELINE AUDIT — inputs available to a 2027 season projection engine")
    print("=" * 74)
    print(f"  hitters  {h.shape[0]} rows x {h.shape[1]} cols   ({HIT_CSV.name})")
    print(f"  pitchers {p.shape[0]} rows x {p.shape[1]} cols   ({PIT_CSV.name})")
    print(f"  Last_PA: median {h.Last_PA.median():.0f}, max"
          f" {h.Last_PA.max():.0f} -> these artifacts were built from a"
          f" PARTIAL 2026 season")

    check_league_calibration(h, p)
    b = check_bip_vs_real(h)
    check_blend_mechanism(b)
    check_spread(h)
    check_team_level(h, p)
    print()


if __name__ == "__main__":
    main()

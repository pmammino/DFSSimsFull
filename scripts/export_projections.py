"""
export_projections.py
=====================
One spreadsheet with every player's season projection, ready to export.

    python scripts/export_projections.py --target-year 2027

Reads what the pipeline and the season layer already wrote and turns per-PA
rates x projected playing time into the season counting stats people actually
want: PA, H, HR, R, RBI, SB for hitters; IP, K, ERA, WHIP, SV, HLD for
pitchers. Writes an .xlsx workbook plus plain CSVs of the same data, because a
workbook is for reading and a CSV is for loading into something else.

Inputs (all produced by `run_pipeline.py` then `season_engine.py`):

    out/hitter_pa_projections_<year>.csv     per-PA rates + Proj_PA
    out/pitcher_pa_projections_<year>.csv    per-PA rates + Proj_IP
    out/season_<year>/team_context.csv       team run environment
    out/fielding_history_<year>.csv          optional, for primary position

Why the counting stats are VALUES and not formulas
--------------------------------------------------
They are computed here and written as numbers. Formulas would be nicer — edit
a PA and watch the line move — but 7,000 players x ~18 derived columns is
~130,000 formulas, and the LibreOffice pass that verifies them times out well
before that (measured: 54,000 formulas, 239s, no result). Shipping unverified
formulas means shipping cells that read as empty to pandas, to previewers, and
to anything else that looks at cached values.

So instead: every rate the stats are built from is included in the sheet
beside them (the `r_*` columns), the arithmetic is spelled out on the Read Me
tab, and the **What-If** tab carries live formulas for a small enough set of
players that they can actually be verified. Nothing here is a black box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FONT = "Arial"


# ─────────────────────────────────────────────────────────────────────────────
# derive
# ─────────────────────────────────────────────────────────────────────────────

def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


HITTER_RATES = {
    "r_1B": "P_1B", "r_2B": "P_2B", "r_3B": "P_3B", "r_HR": "P_HR",
    "r_BB": "P_BB", "r_HBP": "P_HBP", "r_SF": "P_SF", "r_SO": "P_K",
    "r_SB": "P_SB", "r_CS": "P_CS", "r_R": "P_R", "r_RBI": "P_RBI",
}
PITCHER_RATES = {
    "r_SO": "P_K", "r_BB": "P_BB", "r_HBP": "P_HBP", "r_HR": "P_HR",
    "r_1B": "P_1B", "r_2B": "P_2B", "r_3B": "P_3B",
}


def build_hitters(h: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "PlayerId": h["PlayerId"].astype("Int64"),
        "Name": h.get("Name"),
        "Team": h.get("team_abbr", h.get("Pred_target_team_abbr")),
        "Age": _num(h, "Age").round(0).astype("Int64"),
        "Bats": h.get("BatSide"),
        "Pos": h.get("pt_position"),
        "Tier": h.get("pt_tier"),
        "Role": h.get("pt_role"),
        "RoleSource": h.get("pt_role_source"),
        "Origin": np.where(h.get("mle_source").notna(), "MiLB (MLE)", "MLB")
        if "mle_source" in h.columns else "MLB",
        "LineupSlot": _num(h, "Pred_lineup_slot").round(0).astype("Int64"),
    })
    for k, src in HITTER_RATES.items():
        out[k] = _num(h, src)
    out["PA"] = _num(h, "Proj_PA").round(1)
    out["G"] = _num(h, "Proj_G").round(1)
    for stat, rate in (("1B", "r_1B"), ("2B", "r_2B"), ("3B", "r_3B"),
                       ("HR", "r_HR"), ("BB", "r_BB"), ("HBP", "r_HBP"),
                       ("SF", "r_SF"), ("SO", "r_SO"), ("SB", "r_SB"),
                       ("CS", "r_CS"), ("R", "r_R"), ("RBI", "r_RBI")):
        out[stat] = (out[rate] * out["PA"]).round(1)
    out["H"] = (out["1B"] + out["2B"] + out["3B"] + out["HR"]).round(1)
    out["AB"] = (out["PA"] - out["BB"] - out["HBP"] - out["SF"]).round(1)
    out["TB"] = (out["1B"] + 2 * out["2B"] + 3 * out["3B"]
                 + 4 * out["HR"]).round(1)
    ab = out["AB"].replace(0, np.nan)
    pa = out["PA"].replace(0, np.nan)
    out["AVG"] = (out["H"] / ab).round(3)
    out["OBP"] = ((out["H"] + out["BB"] + out["HBP"]) / pa).round(3)
    out["SLG"] = (out["TB"] / ab).round(3)
    out["OPS"] = (out["OBP"] + out["SLG"]).round(3)
    order = ["PlayerId", "Name", "Team", "Age", "Bats", "Pos", "Tier", "Role",
             "RoleSource", "Origin", "LineupSlot", "PA", "G", "AB", "H", "1B",
             "2B", "3B", "HR", "TB", "BB", "HBP", "SF", "SO", "SB", "CS", "R",
             "RBI", "AVG", "OBP", "SLG", "OPS"] + list(HITTER_RATES)
    return out[order].sort_values("PA", ascending=False).reset_index(drop=True)


def build_pitchers(p: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
    tbf_ip = _num(p, "TBF_per_IP", 4.3)
    tbf_ip = tbf_ip.where(tbf_ip > 1.0, 4.3)
    out = pd.DataFrame({
        "PlayerId": p["PlayerId"].astype("Int64"),
        "Name": p.get("Name"),
        "Team": p.get("team_abbr", p.get("Pred_target_team_abbr")),
        "Age": _num(p, "Age").round(0).astype("Int64"),
        "Tier": p.get("pt_tier"),
        "Role": p.get("pt_role"),
        "RoleSource": p.get("pt_role_source"),
        "SP_RP": p.get("role"),
        "Origin": np.where(p.get("mle_source").notna(), "MiLB (MLE)", "MLB")
        if "mle_source" in p.columns else "MLB",
    })
    for k, src in PITCHER_RATES.items():
        out[k] = _num(p, src)
    out["TBF_per_IP"] = tbf_ip.round(3)
    out["r_ERA"] = _num(p, "ERA")
    out["r_RA9"] = _num(p, "RA9")
    out["IP"] = _num(p, "Proj_IP").round(1)
    out["G"] = _num(p, "Proj_G").round(1)
    out["GS"] = _num(p, "Proj_GS").round(1)
    out["TBF"] = (out["IP"] * out["TBF_per_IP"]).round(1)
    for stat, rate in (("SO", "r_SO"), ("BB", "r_BB"), ("HBP", "r_HBP"),
                       ("HR", "r_HR"), ("1B", "r_1B"), ("2B", "r_2B"),
                       ("3B", "r_3B")):
        out[stat] = (out[rate] * out["TBF"]).round(1)
    out["H"] = (out["1B"] + out["2B"] + out["3B"] + out["HR"]).round(1)
    out["ER"] = (out["r_ERA"] * out["IP"] / 9.0).round(1)
    out["R"] = (out["r_RA9"] * out["IP"] / 9.0).round(1)
    ip = out["IP"].replace(0, np.nan)
    out["ERA"] = out["r_ERA"].round(2)
    out["WHIP"] = ((out["H"] + out["BB"]) / ip).round(2)
    out["K9"] = (out["SO"] * 9 / ip).round(2)
    out["BB9"] = (out["BB"] * 9 / ip).round(2)
    out["HR9"] = (out["HR"] * 9 / ip).round(2)

    # Saves and holds: each club's pool, split by the NORMALISED role share.
    # The raw `pt_save_share` is a per-role weight and sums past 1 over a real
    # staff (1.16 saves, 1.85 holds), so using it directly over-allocates.
    pool = pools.set_index("team_id") if len(pools) else None
    out["SV"] = np.nan
    out["HLD"] = np.nan
    if pool is not None and "team_id" in p.columns:
        from playing_time_model import allocate_opportunity
        sv = pd.Series(0.0, index=out.index)
        hld = pd.Series(0.0, index=out.index)
        g = out["G"].to_numpy(dtype=float)
        teams = p["team_id"]
        for tid, idx in out.groupby(teams.to_numpy()).groups.items():
            if pd.isna(tid) or tid not in pool.index:
                continue
            pos = out.index.get_indexer(idx)
            cap = g[pos]
            # Saves first: a closer's saves take precedence over his holds,
            # and holds are then limited to the appearances left over.
            s = allocate_opportunity(_num(p, "Proj_SV_share").to_numpy()[pos],
                                     float(pool.loc[tid, "expected_saves"]), cap)
            hh = allocate_opportunity(_num(p, "Proj_HLD_share").to_numpy()[pos],
                                      float(pool.loc[tid, "hold_opportunities"]),
                                      np.maximum(cap - s, 0.0))
            sv.iloc[pos] = s
            hld.iloc[pos] = hh
        out["SV"] = sv.round(1)
        out["HLD"] = hld.round(1)
        impossible = int(((out["SV"] + out["HLD"]) > out["G"] + 0.05).sum())
        if impossible:
            print(f"  WARNING: {impossible} pitchers still show more saves + "
                  "holds than appearances")
    order = ["PlayerId", "Name", "Team", "Age", "Tier", "Role", "RoleSource",
             "SP_RP", "Origin", "IP", "G", "GS", "TBF", "H", "1B", "2B", "3B",
             "HR", "BB", "HBP", "SO", "ER", "R", "SV", "HLD", "ERA", "WHIP",
             "K9", "BB9", "HR9", "TBF_per_IP"] + list(PITCHER_RATES) \
        + ["r_ERA", "r_RA9"]
    return out[order].sort_values("IP", ascending=False).reset_index(drop=True)


def team_pools(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    """Per-club wins and save/hold pools, from the season layer's own models."""
    try:
        from team_context import bottom_up_team_factors
        from team_wins import (bottom_up_team_ra, project_save_hold_opportunity,
                               project_team_wins)
        ra = bottom_up_team_ra(pitchers, team_col="team_id")
        off = bottom_up_team_factors(hitters, team_col="team_id")
        wins = project_team_wins(off, ra, team_col="team_id")
        return project_save_hold_opportunity(wins, team_col="team_id")
    except Exception as e:
        print(f"  team pools unavailable ({type(e).__name__}: {e}); "
              "SV/HLD will be blank")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# workbook
# ─────────────────────────────────────────────────────────────────────────────

def _write_sheet(ws, df: pd.DataFrame, *, money_cols=(), rate_cols=(),
                 three_dp=()):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    hdr_fill = PatternFill("solid", fgColor="1F3864")
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(1, j, col)
        c.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
    for i, row in enumerate(df.itertuples(index=False), start=2):
        for j, v in enumerate(row, start=1):
            if pd.isna(v):
                v = None
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            cell = ws.cell(i, j, v)
            cell.font = Font(name=FONT, size=10)
            name = df.columns[j - 1]
            if name in three_dp:
                cell.number_format = "0.000"
            elif name in rate_cols:
                cell.number_format = "0.0000"
            elif isinstance(v, float):
                cell.number_format = "0.0"
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = (f"A1:{get_column_letter(len(df.columns))}"
                          f"{len(df) + 1}")
    widths = {"Name": 24, "Role": 30, "Team": 7, "RoleSource": 12,
              "Origin": 11, "Tier": 10, "SP_RP": 9, "PlayerId": 10}
    for j, col in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(j)].width = widths.get(col, 8.5)


def _read_me(ws, meta: dict):
    from openpyxl.styles import Alignment, Font
    lines = [
        (f"MLB {meta['year']} season projections", "title"),
        ("", None),
        (f"Every player in the organization: {meta['n_hit']:,} hitters and "
         f"{meta['n_pit']:,} pitchers across all 30 clubs.", None),
        (f"Generated {meta['stamp']} from pipeline run {meta['run']}.", None),
        ("", None),
        ("TABS", "head"),
        ("  Hitters      one row per hitter, sorted by projected PA", None),
        ("  Pitchers     one row per pitcher, sorted by projected IP", None),
        ("  Teams        per-club run environment, wins, save and hold pools", None),
        ("", None),
        ("  A companion file, mlb_" + str(meta["year"]) + "_what_if.xlsx, holds "
         "the same top players", None),
        ("  with LIVE formulas: change a yellow PA or IP cell and the whole", None),
        ("  line recalculates. Use it to see the arithmetic, or to test what a", None),
        ("  role change would do before editing the override file.", None),
        ("", None),
        ("HOW THE COUNTING STATS ARE BUILT", "head"),
        ("  Every stat is a projected RATE times projected PLAYING TIME.", None),
        ("  Both are in the sheet, so nothing here is a black box:", None),
        ("", None),
        ("    hitters    HR = r_HR x PA        H  = 1B + 2B + 3B + HR", None),
        ("               AB = PA - BB - HBP - SF", None),
        ("               AVG = H / AB          OBP = (H + BB + HBP) / PA", None),
        ("    pitchers   TBF = IP x TBF_per_IP     SO = r_SO x TBF", None),
        ("               ER = r_ERA x IP / 9       WHIP = (H + BB) / IP", None),
        ("               SV = club save pool x that player's share", None),
        ("", None),
        ("  The r_* columns are the per-PA (or per-TBF) rates. They are the", None),
        ("  model's actual output; the counting stats are arithmetic on top.", None),
        ("", None),
        ("PLAYING TIME", "head"),
        ("  Tier 'projected' = expected to accumulate real MLB playing time.", None),
        ("  Tier 'floor'     = organizational depth, held at exactly 1 PA / 1 IP", None),
        ("                     so the player stays present and joinable without", None),
        ("                     moving any team or league total.", None),
        (f"  {meta['n_floor_h']:,} hitters and {meta['n_floor_p']:,} pitchers are "
         f"at the floor. That is deliberate.", None),
        ("", None),
        ("  Playing time is an ALLOCATION, not a per-player forecast: each club", None),
        ("  closes on 162 x 38 = 6,156 PA and 162 x 9 = 1,458 IP, because a team", None),
        ("  cannot bat 7,000 times however good its players are.", None),
        ("", None),
        ("  Role drives volume, and every role is a DEFAULT you can replace.", None),
        ("  Edit rosters/player_roles_{hitters,pitchers}_" + str(meta["year"])
         + ".csv", None),
        ("  (Role, Role Start, Availability) and re-run. RoleSource tells you", None),
        ("  whether a row came from the model ('default') or from you", None),
        ("  ('override').", None),
        ("", None),
        ("KNOWN LIMITS — read before trusting a single player", "head"),
        (f"  * Runs scored ({meta['r_scored']:,.0f}) exceeds runs allowed "
         f"({meta['r_allowed']:,.0f}) by {meta['r_gap']:+.1f}%.", None),
        ("    They must be equal in reality. R and RBI are still free-standing", None),
        ("    player rates rather than an allocation of the runs each lineup", None),
        ("    actually scores, which is the next thing to fix.", None),
        (f"  * Role anchors are stretched {meta['scale_h']:.0%} (hitters) / "
         f"{meta['scale_p']:.0%} (pitchers) to fit real", None),
        ("    rosters, so a nominal 630-PA regular lands nearer 550. The", None),
        ("    ordering is meaningful; the absolute level is provisional.", None),
        ("  * No typed lineup slots yet, so two first basemen can both be", None),
        ("    'Full Time'. Position data now exists, so this is next.", None),
        ("  * Wins, saves and holds come from a bottom-up roster estimate with", None),
        ("    no betting-market prior loaded. Drop a market_odds file in", None),
        ("    rosters/ to anchor them.", None),
        ("  * Counting stats are VALUES, not formulas: ~130,000 formulas cannot", None),
        ("    be machine-verified here, and unverified formulas read as empty", None),
        ("    cells to anything but Excel. The What-If tab is live.", None),
        ("", None),
        ("REGENERATE", "head"),
        ("  python run_pipeline.py --target-year " + str(meta["year"]), None),
        ("  python season_engine.py --target-year " + str(meta["year"]), None),
        ("  python scripts/export_projections.py --target-year "
         + str(meta["year"]), None),
    ]
    ws.sheet_view.showGridLines = False
    for i, (text, kind) in enumerate(lines, start=1):
        c = ws.cell(i, 1, text)
        if kind == "title":
            c.font = Font(name=FONT, size=15, bold=True, color="1F3864")
        elif kind == "head":
            c.font = Font(name=FONT, size=11, bold=True, color="1F3864")
        else:
            c.font = Font(name=FONT, size=10)
        c.alignment = Alignment(vertical="center")
    ws.column_dimensions["A"].width = 100


def _what_if(ws, hitters: pd.DataFrame, pitchers: pd.DataFrame, n=25):
    """Live formulas for a small, verifiable set of players."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    yellow = PatternFill("solid", fgColor="FFFF00")
    hdr = PatternFill("solid", fgColor="1F3864")
    bold = Font(name=FONT, size=10, bold=True, color="FFFFFF")
    blue = Font(name=FONT, size=10, color="0000FF")
    black = Font(name=FONT, size=10)

    ws.sheet_view.showGridLines = False
    ws["A1"] = "What-If — edit the YELLOW volume cells; every stat recalculates"
    ws["A1"].font = Font(name=FONT, size=13, bold=True, color="1F3864")
    ws["A2"] = ("The rest of the workbook holds computed values. This tab is "
                "live, so you can see the arithmetic and test a role change.")
    ws["A2"].font = black

    row = 4
    # ── hitters ──
    ws.cell(row, 1, f"Top {n} hitters by projected PA").font = Font(
        name=FONT, size=11, bold=True, color="1F3864")
    row += 1
    hcols = ["Name", "Team", "Role", "PA", "AB", "H", "HR", "BB", "SO", "R",
             "RBI", "SB", "AVG", "OBP", "SLG", "r_1B", "r_2B", "r_3B", "r_HR",
             "r_BB", "r_HBP", "r_SF", "r_SO", "r_SB", "r_R", "r_RBI"]
    for j, cname in enumerate(hcols, start=1):
        c = ws.cell(row, j, cname)
        c.font = bold
        c.fill = hdr
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    hstart = row + 1
    L = {c: get_column_letter(j) for j, c in enumerate(hcols, start=1)}
    for i, r in hitters.head(n).iterrows():
        rr = hstart + i
        ws.cell(rr, 1, r["Name"]).font = black
        ws.cell(rr, 2, r["Team"]).font = black
        ws.cell(rr, 3, r["Role"]).font = black
        pa = ws.cell(rr, 4, float(r["PA"]))
        pa.font = blue
        pa.fill = yellow
        pa.number_format = "0.0"
        P = L["PA"]
        f = {
            "AB": f"={P}{rr}-{L['r_BB']}{rr}*{P}{rr}-{L['r_HBP']}{rr}*{P}{rr}"
                  f"-{L['r_SF']}{rr}*{P}{rr}",
            "H": f"=({L['r_1B']}{rr}+{L['r_2B']}{rr}+{L['r_3B']}{rr}"
                 f"+{L['r_HR']}{rr})*{P}{rr}",
            "HR": f"={L['r_HR']}{rr}*{P}{rr}",
            "BB": f"={L['r_BB']}{rr}*{P}{rr}",
            "SO": f"={L['r_SO']}{rr}*{P}{rr}",
            "R": f"={L['r_R']}{rr}*{P}{rr}",
            "RBI": f"={L['r_RBI']}{rr}*{P}{rr}",
            "SB": f"={L['r_SB']}{rr}*{P}{rr}",
            "AVG": f"=IFERROR({L['H']}{rr}/{L['AB']}{rr},0)",
            "OBP": f"=IFERROR(({L['H']}{rr}+{L['BB']}{rr}"
                   f"+{L['r_HBP']}{rr}*{P}{rr})/{P}{rr},0)",
            "SLG": f"=IFERROR(({L['r_1B']}{rr}+2*{L['r_2B']}{rr}"
                   f"+3*{L['r_3B']}{rr}+4*{L['r_HR']}{rr})*{P}{rr}"
                   f"/{L['AB']}{rr},0)",
        }
        for cname, formula in f.items():
            c = ws.cell(rr, hcols.index(cname) + 1, formula)
            c.font = black
            c.number_format = "0.000" if cname in ("AVG", "OBP", "SLG") else "0.0"
        for cname in ("r_1B", "r_2B", "r_3B", "r_HR", "r_BB", "r_HBP", "r_SF",
                      "r_SO", "r_SB", "r_R", "r_RBI"):
            c = ws.cell(rr, hcols.index(cname) + 1, float(r[cname]))
            c.font = black
            c.number_format = "0.0000"

    row = hstart + n + 2
    # ── pitchers ──
    ws.cell(row, 1, f"Top {n} pitchers by projected IP").font = Font(
        name=FONT, size=11, bold=True, color="1F3864")
    row += 1
    pcols = ["Name", "Team", "Role", "IP", "TBF", "H", "SO", "BB", "HR", "ER",
             "ERA", "WHIP", "K9", "TBF_per_IP", "r_SO", "r_BB", "r_HR", "r_1B",
             "r_2B", "r_3B", "r_ERA"]
    for j, cname in enumerate(pcols, start=1):
        c = ws.cell(row, j, cname)
        c.font = bold
        c.fill = hdr
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    pstart = row + 1
    M = {c: get_column_letter(j) for j, c in enumerate(pcols, start=1)}
    for i, r in pitchers.head(n).iterrows():
        rr = pstart + i
        ws.cell(rr, 1, r["Name"]).font = black
        ws.cell(rr, 2, r["Team"]).font = black
        ws.cell(rr, 3, r["Role"]).font = black
        ip = ws.cell(rr, 4, float(r["IP"]))
        ip.font = blue
        ip.fill = yellow
        ip.number_format = "0.0"
        I = M["IP"]
        f = {
            "TBF": f"={I}{rr}*{M['TBF_per_IP']}{rr}",
            "H": f"=({M['r_1B']}{rr}+{M['r_2B']}{rr}+{M['r_3B']}{rr}"
                 f"+{M['r_HR']}{rr})*{M['TBF']}{rr}",
            "SO": f"={M['r_SO']}{rr}*{M['TBF']}{rr}",
            "BB": f"={M['r_BB']}{rr}*{M['TBF']}{rr}",
            "HR": f"={M['r_HR']}{rr}*{M['TBF']}{rr}",
            "ER": f"={M['r_ERA']}{rr}*{I}{rr}/9",
            "ERA": f"={M['r_ERA']}{rr}",
            "WHIP": f"=IFERROR(({M['H']}{rr}+{M['BB']}{rr})/{I}{rr},0)",
            "K9": f"=IFERROR({M['SO']}{rr}*9/{I}{rr},0)",
        }
        for cname, formula in f.items():
            c = ws.cell(rr, pcols.index(cname) + 1, formula)
            c.font = black
            c.number_format = "0.00" if cname in ("ERA", "WHIP", "K9") else "0.0"
        for cname in ("TBF_per_IP", "r_SO", "r_BB", "r_HR", "r_1B", "r_2B",
                      "r_3B", "r_ERA"):
            c = ws.cell(rr, pcols.index(cname) + 1, float(r[cname]))
            c.font = black
            c.number_format = "0.0000"

    for j, cname in enumerate(hcols, start=1):
        ws.column_dimensions[get_column_letter(j)].width = \
            24 if cname == "Name" else (30 if cname == "Role" else 9)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--dest", type=Path, default=ROOT / "out")
    ap.add_argument("--run", default="local")
    a = ap.parse_args(argv)
    y = a.target_year

    season = a.out_dir / f"season_{y}"
    hp = season / "hitters.csv"
    pp = season / "pitchers.csv"
    if not hp.exists() or not pp.exists():
        hp = a.out_dir / f"hitter_pa_projections_{y}.csv"
        pp = a.out_dir / f"pitcher_pa_projections_{y}.csv"
    for f in (hp, pp):
        if not f.exists():
            print(f"missing {f} — run run_pipeline.py and season_engine.py first")
            return 1
    h_raw = pd.read_csv(hp, low_memory=False)
    p_raw = pd.read_csv(pp, low_memory=False)
    print(f"  loaded {len(h_raw):,} hitters, {len(p_raw):,} pitchers")

    pools = team_pools(h_raw, p_raw)
    hitters = build_hitters(h_raw)
    pitchers = build_pitchers(p_raw, pools)

    tc_path = season / "team_context.csv"
    teams = pools.copy()
    if tc_path.exists() and len(teams):
        tc = pd.read_csv(tc_path)
        keep = [c for c in ("team_id", "n_hitters", "n_projected",
                            "team_factor", "pa_reserve_share",
                            "ip_reserve_share") if c in tc.columns]
        teams = teams.merge(tc[keep], on="team_id", how="left")
    if len(teams):
        teams = (teams.sort_values("expected_wins", ascending=False)
                 .round(3).reset_index(drop=True))

    a.dest.mkdir(parents=True, exist_ok=True)
    for name, df in (("hitters", hitters), ("pitchers", pitchers),
                     ("teams", teams)):
        if len(df):
            path = a.dest / f"projections_{name}_{y}.csv"
            df.to_csv(path, index=False)
            print(f"  wrote {path.name}  ({len(df):,} rows x {len(df.columns)} cols)")

    # Projected players only. The floor tier is 2,179 hitters and 3,101
    # pitchers carried at 1 PA / 1 IP apiece so they are present, ranked and
    # joinable — placeholders, not projections that they will play. Summing
    # their runs into a league total adds a phantom 3,101 innings of pitching
    # to a 43,740-inning league and makes the two sides disagree by 5%.
    def _projected(df):
        if "Tier" not in df.columns:
            return df
        keep = df["Tier"].astype(str) != "floor"
        return df[keep] if keep.any() else df

    r_scored = float(_projected(hitters)["R"].sum())
    r_allowed = float(_projected(pitchers)["R"].sum())
    meta = {
        "year": y, "n_hit": len(hitters), "n_pit": len(pitchers),
        "stamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "run": a.run,
        "n_floor_h": int((hitters["Tier"] == "floor").sum()),
        "n_floor_p": int((pitchers["Tier"] == "floor").sum()),
        "r_scored": r_scored, "r_allowed": r_allowed,
        "r_gap": (r_scored / r_allowed - 1) * 100 if r_allowed else 0.0,
        "scale_h": 0.875, "scale_p": 0.884,
    }

    from openpyxl import Workbook
    wb = Workbook()
    _read_me(wb.active, meta)
    wb.active.title = "Read Me"
    hrates = set(HITTER_RATES) | {"r_ERA", "r_RA9"} | set(PITCHER_RATES)
    _write_sheet(wb.create_sheet("Hitters"), hitters, rate_cols=hrates,
                 three_dp={"AVG", "OBP", "SLG", "OPS"})
    _write_sheet(wb.create_sheet("Pitchers"), pitchers, rate_cols=hrates,
                 three_dp={"ERA", "WHIP", "K9", "BB9", "HR9"})
    if len(teams):
        _write_sheet(wb.create_sheet("Teams"), teams,
                     three_dp={"win_pct", "team_factor", "rs_per_game",
                               "ra_per_game"})
    xlsx = a.dest / f"mlb_{y}_projections.xlsx"
    wb.save(xlsx)
    print(f"  wrote {xlsx.name}")

    # The live sheet goes in its OWN workbook. Bundled with 7,000 rows of data
    # the file is too large for the LibreOffice pass to verify — it times out
    # before returning a result — and formulas that cannot be checked ship as
    # cells that read empty to everything except Excel. Alone, the same
    # formulas verify in seconds, and the data workbook stays formula-free so
    # there is nothing in it left unverified.
    wb2 = Workbook()
    _what_if(wb2.active, hitters, pitchers)
    wb2.active.title = "What-If"
    whatif = a.dest / f"mlb_{y}_what_if.xlsx"
    wb2.save(whatif)
    print(f"  wrote {whatif.name}  (live formulas)")
    print(f"\n  league check: {r_scored:,.0f} runs scored vs "
          f"{r_allowed:,.0f} allowed ({meta['r_gap']:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

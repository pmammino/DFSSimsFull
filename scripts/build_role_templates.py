"""
build_role_templates.py
=======================
Generate the role-assignment workbooks the playing-time model will read.

    python scripts/build_role_templates.py --target-year 2027

Writes `rosters/player_roles_hitters_<year>.xlsx` and
`rosters/player_roles_pitchers_<year>.xlsx`, pre-populated with every player in
`out/*_pa_projections_<year>.csv` so the sheets are ready to fill in rather
than empty.

Design (see "Designing the role taxonomy" in README_projection_engine.md):

  * **Roles are columns holding probabilities**, not a single label. A player
    in a job battle is 50% full-time / 30% platoon / 20% bench, and one label
    would produce the mean of a bimodal distribution.
  * **Job and timing are separate.** `Role Start` supplies the fraction of the
    season the player holds the job, so "mid-season callup who becomes a
    full-timer" and "...who becomes a platoon bat" are expressible — a single
    "Mid Season Callup" role cannot tell them apart. It also unifies callups
    with return-from-injury: both are "when does this role start".
  * **Availability is orthogonal to role.** A full-time player who misses April
    is Full Time x 0.85 availability, which no role label can express.

The workbooks are live: `Proj PA` / `Proj IP` are SUMPRODUCT formulas over the
role probabilities and the anchors, so editing a probability updates the
projection in the sheet. The pipeline reads the same columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from team_context import abbr_for_team_id  # noqa: E402

FONT = "Arial"
HDR_FILL = PatternFill("solid", fgColor="1F3864")
ANCHOR_FILL = PatternFill("solid", fgColor="FFF2CC")   # assumptions
INPUT_FILL = PatternFill("solid", fgColor="FFFFCC")    # cells to fill in
EXAMPLE_FILL = PatternFill("solid", fgColor="E2EFDA")
BLUE = Font(name=FONT, size=10, color="0000FF")        # hardcoded input
BLACK = Font(name=FONT, size=10)
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# ─────────────────────────────────────────────────────────────────────────────
# Role definitions
# The role vocabulary, its anchors, the timing table, the default-role
# heuristics and the full-time reference all live in role_taxonomy.py. They
# used to live HERE, which meant playing_time_model.py would have needed its
# own copy — and two copies of a table nobody diffs drift, with a renamed role
# failing as a silent lookup miss rather than an error.
from role_taxonomy import (  # noqa: E402
    HITTER_ROLES, PITCHER_ROLES, TIMING, full_time_reference,
    suggest_hitter_role as _suggest_hitter_role,
    suggest_pitcher_role as _suggest_pitcher_role,
)


def _style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border = BOX
    ws.row_dimensions[row].height = 46


def _legend(wb, role_defs, kind):
    ws = wb.create_sheet("Legend", 0)
    ws.sheet_view.showGridLines = False
    vol = "PA" if kind == "hitter" else "IP"
    lines = [
        (f"Role assignments — {kind}s", True),
        ("", False),
        ("WHAT TO EDIT — the yellow cells on the Assignments sheet:", True),
        (f"  * One probability per role column. They should sum to 1.00 for "
         f"each player; the 'Role Prob Sum' column checks this.", False),
        ("  * Role Start — when the player takes the job (see the Timing "
         "table on the Roles sheet).", False),
        ("  * Availability — share of the season he is healthy and on an MLB "
         "roster. 1.00 = full year.", False),
        ("  * Pos / Bats (or Throws) — needed for the typed-slot allocation "
         "and the platoon splits.", False),
        ("", False),
        ("WHY PROBABILITIES AND NOT ONE ROLE:", True),
        ("  A player in a job battle is ~50% full-time / 30% platoon / 20% "
         "bench. Picking one label", False),
        (f"  yields a plausible-looking {vol} figure that is the mean of a "
         f"bimodal distribution — wrong in", False),
        ("  both worlds. Probabilities also give the model the variance, "
         "which is what season-long", False),
        ("  ranking and DFS leverage both want.", False),
        ("", False),
        ("WHY ROLE START IS SEPARATE FROM THE ROLE:", True),
        ("  'Mid-season callup' conflates two things. A callup who becomes a "
         "full-timer and one who", False),
        ("  becomes a platoon bat have very different rate lines. Role x "
         "timing expresses both; one", False),
        ("  combined label cannot. It also means a mid-season callup and a "
         "player back from the IL in", False),
        ("  July are the same parameter.", False),
        ("", False),
        ("WHY AVAILABILITY IS SEPARATE TOO:", True),
        ("  'Injured' is not a role. A full-time player who misses April is "
         "Full Time x 0.85, which no", False),
        ("  role label can say. A season-ending injury gives availability ~0, "
         f"so the role collapses to", False),
        (f"  the 1 {vol} floor — the existing rule, reached through the model "
         f"rather than bolted on.", False),
        ("", False),
        ("COLOUR KEY:", True),
        ("  Yellow = fill this in.   Blue text = a hardcoded anchor you may "
         "tune.   Black = formula.", False),
        ("  Green row = the example; delete it before use.", False),
        ("", False),
        ("FREE AGENTS AND UNDER-PROJECTED TEAMS:", True),
        ("  Set Team to 'FA' for an unsigned player. He keeps a full rate line "
         "and can be given a role", False),
        ("  and playing time, and is excluded from every team aggregate. Then "
         "use the Teams sheet to", False),
        ("  reserve part of a club's budget for the signing you expect it to "
         "make, so that club is", False),
        ("  deliberately UNDER-projected instead of spreading those "
         f"{vol} across the players on hand.", False),
        ("", False),
        ("THE PROJECTION COLUMNS:", True),
        (f"  Proj {vol} and the columns beside it are FORMULAS, so they update "
         f"as you edit probabilities.", False),
        ("  They show blank until you open this file in Excel or LibreOffice, "
         "which computes them on", False),
        ("  open — openpyxl cannot write cached values alongside a formula. "
         "That is cosmetic: the", False),
        ("  pipeline reads the INPUT columns (role probabilities, Role Start, "
         "Availability) and recomputes", False),
        ("  the volumes itself, so a blank formula cell never reaches a "
         "projection.", False),
        ("", False),
        ("CALIBRATION STATUS:", True),
        (f"  Every anchor on the Roles sheet is a documented starting point, "
         f"not a fitted value. Fit them", False),
        ("  from history (group real player-seasons by observed role) before "
         "trusting the levels. The", False),
        ("  cross-role ORDERING is more reliable than the absolute numbers.", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, size=11, bold=bold)
    ws.column_dimensions["A"].width = 105
    return ws


def _roles_sheet(wb, role_defs, kind):
    ws = wb.create_sheet("Roles")
    if kind == "hitter":
        cols = [("Role", 32), ("PA Anchor", 11), ("vL Share (LHB)", 13),
                ("vL Share (RHB)", 13), ("Definition / notes", 78)]
    else:
        cols = [("Role", 32), ("IP Anchor", 11), ("GS Anchor", 11),
                ("G Anchor", 10), ("Save Share Wt", 13), ("Hold Share Wt", 13),
                ("Definition / notes", 78)]
    for j, (name, width) in enumerate(cols, start=1):
        ws.cell(row=1, column=j, value=name)
        ws.column_dimensions[get_column_letter(j)].width = width
    _style_header(ws, 1, len(cols))

    for i, r in enumerate(role_defs, start=2):
        if kind == "hitter":
            vals = [r["role"], r["pa"], r["vl_lhb"], r["vl_rhb"], r["note"]]
        else:
            vals = [r["role"], r["ip"], r["gs"], r["g"], r["sv"], r["hld"],
                    r["note"]]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = BLACK if j in (1, len(vals)) else BLUE
            c.border = BOX
            if j == len(vals):
                c.alignment = Alignment(wrap_text=True, vertical="top")
            if isinstance(v, float):
                c.number_format = "0.00"
        ws.row_dimensions[i].height = 30

    # Timing table
    base = len(role_defs) + 4
    ws.cell(row=base - 1, column=1, value="Role Start -> Role Share").font = \
        Font(name=FONT, size=11, bold=True)
    for j, name in enumerate(["Role Start", "Role Share", "Meaning"], start=1):
        ws.cell(row=base, column=j, value=name)
    _style_header(ws, base, 3)
    for i, (label, share, note) in enumerate(TIMING, start=base + 1):
        ws.cell(row=i, column=1, value=label).font = BLACK
        c = ws.cell(row=i, column=2, value=share)
        c.font = BLUE
        c.number_format = "0.00"
        ws.cell(row=i, column=3, value=note).font = BLACK
        for j in range(1, 4):
            ws.cell(row=i, column=j).border = BOX

    note_row = base + len(TIMING) + 2
    ws.cell(row=note_row, column=1,
            value="Share weights are RELATIVE within a team — the allocator "
                  "normalizes them to that club's save/hold pool from "
                  "team_wins.py, so they need not sum to 1. They are scaled "
                  "by Role Share x Availability first, so a closer who misses "
                  "half the season claims half as much of the pool."
            if kind == "pitcher" else
            "Anchors are per 162 team games at full availability with the role "
            "held all season. Multiply by Role Share and Availability.").font = \
        Font(name=FONT, size=10, italic=True)
    return ws, base


def _assignments(wb, players, role_defs, kind, roles_ws_name="Roles"):
    ws = wb.create_sheet("Assignments", 1)
    role_names = [r["role"] for r in role_defs]
    vol = "PA" if kind == "hitter" else "IP"
    hand_col = "Bats" if kind == "hitter" else "Throws"

    last_col = "Last PA" if kind == "hitter" else "Last TBF"
    meta = ["PlayerId", "Name", "Team", "Age", hand_col, "Pos",
            last_col, "Career", "Suggested Role", "Role Start", "Role Share",
            "Availability"]
    derived = [f"Proj {vol}", "Role Prob Sum"]
    if kind == "hitter":
        derived += ["Proj vL Share"]
    else:
        derived += ["Proj GS", "Proj G", "Save Wt", "Hold Wt"]
    headers = meta + role_names + derived + ["Notes"]

    for j, h in enumerate(headers, start=1):
        ws.cell(row=1, column=j, value=h)
    _style_header(ws, 1, len(headers))

    r0 = len(meta) + 1                       # first role column
    r1 = len(meta) + len(role_names)         # last role column
    RC0, RC1 = get_column_letter(r0), get_column_letter(r1)

    # Row 2 — anchors, positioned directly above the role columns they scale,
    # so SUMPRODUCT needs no TRANSPOSE (an array function LibreOffice would
    # only partially evaluate in an openpyxl-written file).
    ws.cell(row=2, column=1, value="ANCHORS ->").font = \
        Font(name=FONT, size=10, bold=True, italic=True)
    key = "pa" if kind == "hitter" else "ip"
    for j, r in enumerate(role_defs, start=r0):
        c = ws.cell(row=2, column=j, value=r[key])
        c.font = BLUE
        c.fill = ANCHOR_FILL
        c.border = BOX
    ws.cell(row=2, column=r1 + 1,
            value=f"<- {vol} per 162 G at full availability").font = \
        Font(name=FONT, size=9, italic=True)
    # Hidden anchor rows for the secondary quantities.
    extra_rows = {}
    if kind == "hitter":
        extra_rows = {"vl_lhb": 3, "vl_rhb": 4}
    else:
        extra_rows = {"gs": 3, "g": 4, "sv": 5, "hld": 6}
    for field, row in extra_rows.items():
        ws.cell(row=row, column=1, value=f"anchor:{field}").font = \
            Font(name=FONT, size=9, italic=True, color="808080")
        for j, r in enumerate(role_defs, start=r0):
            c = ws.cell(row=row, column=j, value=r[field])
            c.font = Font(name=FONT, size=9, color="808080")
            c.number_format = "0.00"
        ws.row_dimensions[row].outlineLevel = 1
        ws.row_dimensions[row].hidden = True

    first_data = max(extra_rows.values()) + 1

    def write_row(i, rec, is_example=False):
        vals = [rec.get(h) for h in meta]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = BLACK
            c.border = BOX
            if headers[j - 1] in (hand_col, "Pos", "Role Start",
                                  "Availability"):
                c.fill = EXAMPLE_FILL if is_example else INPUT_FILL
            if headers[j - 1] == "Availability":
                c.number_format = "0.00"
        # Role Share is looked up from the timing table, so the two stay in sync.
        tim_first = len(role_defs) + 5
        tim_last = tim_first + len(TIMING) - 1
        ws.cell(row=i, column=11).value = (
            f'=IFERROR(INDEX({roles_ws_name}!$B${tim_first}:$B${tim_last},'
            f'MATCH(J{i},{roles_ws_name}!$A${tim_first}:$A${tim_last},0)),1)')
        ws.cell(row=i, column=11).number_format = "0.00"

        for j, name in enumerate(role_names, start=r0):
            c = ws.cell(row=i, column=j)
            c.value = rec.get("_probs", {}).get(name)
            c.fill = EXAMPLE_FILL if is_example else INPUT_FILL
            c.font = BLACK
            c.border = BOX
            c.number_format = "0.00"

        col = r1
        col += 1
        ws.cell(row=i, column=col,
                value=f"=SUMPRODUCT({RC0}{i}:{RC1}{i},"
                      f"${RC0}$2:${RC1}$2)*$K{i}*$L{i}").number_format = "0.0"
        col += 1
        ws.cell(row=i, column=col,
                value=f"=SUM({RC0}{i}:{RC1}{i})").number_format = "0.00"
        if kind == "hitter":
            col += 1
            lr = extra_rows["vl_lhb"]
            rr = extra_rows["vl_rhb"]
            pa_col = get_column_letter(r1 + 1)      # the Proj PA column
            # Anchor-weighted, so the blended share reflects where the plate
            # appearances actually come from. The denominator reuses Proj PA
            # (backing out Role Share and Availability) rather than repeating
            # a third SUMPRODUCT — three per row over 600+ rows was enough to
            # time LibreOffice out during recalculation.
            ws.cell(row=i, column=col, value=(
                f'=IFERROR(IF(UPPER($E{i})="L",'
                f'SUMPRODUCT({RC0}{i}:{RC1}{i},${RC0}${lr}:${RC1}${lr},'
                f'${RC0}$2:${RC1}$2),'
                f'SUMPRODUCT({RC0}{i}:{RC1}{i},${RC0}${rr}:${RC1}${rr},'
                f'${RC0}$2:${RC1}$2))'
                f'/({pa_col}{i}/($K{i}*$L{i})),"")'
            )).number_format = "0.00"
        else:
            # Every derived quantity is scaled by Role Share x Availability,
            # save and hold weights included: a closer who misses half the
            # season should claim half as much of his team's save pool, and
            # the allocator normalizes the claims within the club afterwards.
            # Leaving the weights unscaled would hand a hurt closer a full
            # claim and quietly take saves away from the healthy arm behind
            # him.
            for field, fmt in (("gs", "0.0"), ("g", "0.0"),
                               ("sv", "0.000"), ("hld", "0.000")):
                col += 1
                ar = extra_rows[field]
                ws.cell(row=i, column=col, value=(
                    f"=SUMPRODUCT({RC0}{i}:{RC1}{i},"
                    f"${RC0}${ar}:${RC1}${ar})*$K{i}*$L{i}"
                )).number_format = fmt
        col += 1
        c = ws.cell(row=i, column=col, value=rec.get("Notes"))
        c.fill = EXAMPLE_FILL if is_example else INPUT_FILL
        c.font = BLACK

    # Example row, per the workbook convention: realistic values showing the
    # expected format, clearly marked for deletion.
    if kind == "hitter":
        example = {
            "PlayerId": 999999, "Name": "EXAMPLE — delete this row",
            "Team": "FA", "Age": 28, hand_col: "R", "Pos": "2B",
            "Suggested Role": "Full Time",
            "Role Start": "Early Season (~May)", "Availability": 0.85,
            "Notes": "Unsigned; job battle. 60% full-time / 30% strong-side "
                     "platoon / 10% utility. Missed April.",
            "_probs": {"Full Time": 0.60, "Strong Side Platoon": 0.30,
                       "Utility IF": 0.10},
        }
    else:
        example = {
            "PlayerId": 999999, "Name": "EXAMPLE — delete this row",
            "Team": "FA", "Age": 30, hand_col: "R", "Pos": "RP",
            "Suggested Role": "Late Inning RP (Setup)",
            "Role Start": "Opening Day", "Availability": 0.90,
            "Notes": "70% closer / 30% setup — job not settled.",
            "_probs": {"Closer": 0.70, "Late Inning RP (Setup)": 0.30},
        }
    write_row(first_data, example, is_example=True)

    for i, rec in enumerate(players, start=first_data + 1):
        write_row(i, rec)

    widths = {"Name": 26, "Suggested Role": 28, "Role Start": 20, "Notes": 46}
    for j, h in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(j)].width = widths.get(
            h, 15 if j > len(meta) else 11)
    ws.freeze_panes = ws.cell(row=first_data, column=3)
    return ws


def _teams_sheet(wb, kind):
    ws = wb.create_sheet("Teams")
    share = "pa_share" if kind == "hitter" else "ip_share"
    cols = [("Team", 10), (share, 12), ("Expected signing / note", 70)]
    for j, (n, w) in enumerate(cols, start=1):
        ws.cell(row=1, column=j, value=n)
        ws.column_dimensions[get_column_letter(j)].width = w
    _style_header(ws, 1, len(cols))

    from team_context import TEAM_ABBR_BY_ID
    for i, abbr in enumerate(sorted(TEAM_ABBR_BY_ID.values()), start=2):
        ws.cell(row=i, column=1, value=abbr).font = BLACK
        c = ws.cell(row=i, column=2, value=0.0)
        c.font = BLUE
        c.fill = INPUT_FILL
        c.number_format = "0.00"
        ws.cell(row=i, column=3).fill = INPUT_FILL
        for j in range(1, 4):
            ws.cell(row=i, column=j).border = BOX

    n = len(TEAM_ABBR_BY_ID) + 3
    for k, text in enumerate([
        f"Fraction of each club's {'PA' if kind == 'hitter' else 'IP'} budget "
        f"held back for a signing it has not made yet. 0 to 0.5.",
        "Without a reserve the allocator spreads the full budget across the "
        "players currently on hand, so every incumbent absorbs the playing "
        "time that will actually go to the free agent.",
        "A reserve says 'this club's playing time is not all accounted for'. "
        "It does NOT say the club is better than its roster looks — the "
        "quality of an unsigned player is unknowable, so reserves never touch "
        "the rate projections.",
        "Copy these into the \"reserves\" block of "
        "rosters/team_assignments_<year>.json, which is what the pipeline "
        "reads.",
    ]):
        c = ws.cell(row=n + k, column=1, value=text)
        c.font = Font(name=FONT, size=10, italic=True)
    return ws


def _load(target_year: int, out_dir: Path, kind: str) -> list[dict]:
    side = "hitter" if kind == "hitter" else "pitcher"
    path = out_dir / f"{side}_pa_projections_{target_year}.csv"
    if not path.exists():
        print(f"  {path} not found — emitting an empty template")
        return []
    df = pd.read_csv(path)
    team_col = next((c for c in ("Pred_target_team_id", "Pred_home_team_id",
                                 "home_park_team_id") if c in df.columns), None)
    reference = full_time_reference(df)
    # Bullpen roles are a within-team standing, so rank relievers by RA9 on
    # their own staff rather than league-wide.
    staff_rank = {}
    if kind == "pitcher" and {"role", "RA9"} <= set(df.columns) and team_col:
        rp = df[df["role"].astype(str).str.lower() == "reliever"]
        for _, g in rp.groupby(team_col):
            for n, (idx, _r) in enumerate(
                    g.sort_values("RA9").iterrows(), start=1):
                staff_rank[int(_r["PlayerId"])] = n
    recs = []
    for _, r in df.iterrows():
        abbr = abbr_for_team_id(r[team_col]) if team_col else None
        recs.append({
            "PlayerId": int(r["PlayerId"]),
            "Name": r.get("Name"),
            "Team": abbr or "",
            "Age": (int(r["Age"]) if pd.notna(r.get("Age")) else None),
            "Bats": r.get("BatSide"),
            "Throws": r.get("PitchHand"),
            "Pos": r.get("pt_position") if pd.notna(r.get("pt_position")) else None,
            "Last PA": (float(r["Last_PA"]) if pd.notna(r.get("Last_PA")) else None),
            "Last TBF": (float(r["Last_PA"]) if pd.notna(r.get("Last_PA")) else None),
            "Career": (float(r["Career_PA"]) if pd.notna(r.get("Career_PA")) else None),
            # The role the PLAYING-TIME MODEL assigned, where it ran. That is
            # the assignment the projections were actually built on, and this
            # sheet exists to be a view of it that a person can edit.
            #
            # It used to re-derive a suggestion here instead, from
            # `suggest_*_role` called WITHOUT the position or the platoon
            # share — the weaker signature from before those existed. The two
            # disagreed badly: the workbook showed four hitter roles with no
            # catchers, no designated hitters and no platoon bats, and a
            # pitching staff with no closer on any club, while the
            # projections beside it had all twelve. Anyone opening the sheet
            # to adjust a role was editing a different set of labels from the
            # ones the numbers came from.
            #
            # The fallback is the old call, for a frame written before the
            # playing-time step existed.
            "Suggested Role": (
                r["pt_role"] if pd.notna(r.get("pt_role"))
                else (_suggest_hitter_role(r, reference) if kind == "hitter"
                      else _suggest_pitcher_role(
                          r, reference, staff_rank.get(int(r["PlayerId"]))))),
            "Role Start": (r.get("pt_role_start")
                           if pd.notna(r.get("pt_role_start"))
                           else "Opening Day"),
            "Availability": (float(r["pt_availability"])
                             if pd.notna(r.get("pt_availability")) else 1.00),
            "Notes": None,
            # Pre-fill the assignment as a probability of 1 on its own role,
            # so the sheet round-trips: export it unchanged and the model
            # reads back exactly what it assigned. A person expressing a job
            # battle edits these cells into a split — 0.6 / 0.3 / 0.1 — and
            # the model blends the anchors rather than picking one.
            "_probs": ({_role: 1.0} if (_role := (
                r["pt_role"] if pd.notna(r.get("pt_role")) else None)) else {}),
        })
    return recs


def build(kind: str, target_year: int, out_dir: Path, dest: Path) -> Path:
    role_defs = HITTER_ROLES if kind == "hitter" else PITCHER_ROLES
    players = _load(target_year, out_dir, kind)

    wb = Workbook()
    wb.remove(wb.active)
    _legend(wb, role_defs, kind)
    _assignments(wb, players, role_defs, kind)
    _roles_sheet(wb, role_defs, kind)
    _teams_sheet(wb, kind)
    wb._sheets = [wb["Legend"], wb["Assignments"], wb["Roles"], wb["Teams"]]

    dest.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dest)
    print(f"  wrote {dest} ({len(players)} {kind}s, {len(role_defs)} roles)")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--dest-dir", type=Path, default=ROOT / "rosters")
    a = ap.parse_args(argv)

    for kind in ("hitter", "pitcher"):
        build(kind, a.target_year, a.out_dir,
              a.dest_dir / f"player_roles_{kind}s_{a.target_year}.xlsx")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
roles_xlsx_to_csv.py
====================
Turn the human-maintained role workbook into the CSV the pipeline reads.

    python scripts/roles_xlsx_to_csv.py --target-year 2027

`build_role_templates.py` writes `rosters/player_roles_{hitters,pitchers}_<year>
.xlsx`, but `playing_time_model.load_role_overrides` reads only the matching
`.csv` — so an edited workbook changes nothing until it is exported. This
reads the workbook's Assignments sheet and writes the CSV beside it: PlayerId,
Name, Role Start, Availability, and one probability column per role. The
workbook's own formula columns (Role Share, Proj PA, ...) are dropped; the
pipeline recomputes them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from role_taxonomy import role_names  # noqa: E402


def convert(kind: str, target_year: int, rosters: Path) -> Path | None:
    plural = f"{kind}s"
    src = rosters / f"player_roles_{plural}_{target_year}.xlsx"
    if not src.exists():
        print(f"  {src} not found; skipped")
        return None
    df = pd.read_excel(src, sheet_name="Assignments")
    df["PlayerId"] = pd.to_numeric(df["PlayerId"], errors="coerce")
    df = df.dropna(subset=["PlayerId"])
    df = df[df["PlayerId"] != 999999]                # the worked example row
    df["PlayerId"] = df["PlayerId"].astype(int)
    keep = ["PlayerId", "Name", "Role Start", "Availability"] + [
        r for r in role_names(kind) if r in df.columns]
    out = df[keep].copy()
    if "Notes" in df.columns:
        out["Note"] = df["Notes"]
    dest = src.with_suffix(".csv")
    out.to_csv(dest, index=False)
    print(f"  wrote {dest.name} ({len(out):,} {kind}s)")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--rosters", type=Path, default=ROOT / "rosters")
    a = ap.parse_args(argv)
    for kind in ("hitter", "pitcher"):
        convert(kind, a.target_year, a.rosters)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""What the pipeline writes is a whitelist, and a whitelist can silently lie.

`run_pipeline._format_output` emits only the columns named in its column
groups. A column a model produces and the list omits is dropped between the
model and the CSV, and nothing downstream can tell that apart from the model
never having produced it.

That happened to the entire role taxonomy. `PLAYING_TIME_COLS` named only
`pt_tier`, `pt_source`, `Proj_PA` and `Proj_IP`, so `pt_role` and
`pt_position` never once reached a pipeline-built export across every
refresh run. Role and Pos were blank for all 6,941 players, and since the
export derives starts, appearances, saves and holds from the role, GS, G, SV
and HLD summed to ZERO — the closer-saves work and the hitter-role work both
shipped into a file that could not show them.

It stayed hidden because the artifacts being checked had been rebuilt
locally by calling `project_playing_time` directly, which does not go
through `_format_output` at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import playing_time_model as PT  # noqa: E402
import run_pipeline as RP  # noqa: E402

# Columns the export reads by name. Each one is a visible column in the
# delivered spreadsheet, so losing it is not a diagnostic inconvenience.
EXPORT_NEEDS = {
    "pt_tier":        "Tier",
    "pt_role":        "Role, and every save / hold / start derived from it",
    "pt_role_source": "RoleSource",
    "pt_position":    "Pos",
    "Proj_PA":        "PA",
    "Proj_IP":        "IP",
    "Proj_G":         "G",
    "Proj_GS":        "GS",
}


def _frame(n=40, kind="hitter"):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "PlayerId": range(1, n + 1),
        "Name": [f"P{i}" for i in range(n)],
        "Team": "NYY",
        "Pred_target_team_id": 147,
        "Last_PA": rng.integers(50, 600, n),
        "Career_PA": rng.integers(100, 4000, n),
        "P_K": 0.22, "P_BB": 0.085, "P_HBP": 0.011, "P_SF": 0.006,
        "P_HR": 0.030, "P_3B": 0.004, "P_2B": 0.042, "P_1B": 0.142,
        "P_BIPOut": 0.460,
        "N_BIP": rng.integers(50, 500, n),
    })
    for c in RP.SD_COLS:
        df[c] = 0.01
    if kind == "pitcher":
        df["role"] = "starter"
        df["TBF"] = rng.integers(100, 700, n)
    return df


@pytest.mark.parametrize("kind", ["hitter", "pitcher"])
def test_every_column_the_model_produces_survives_to_the_csv(kind):
    """The guard that would have caught it."""
    src = _frame(kind=kind)
    out, _, _ = PT.project_playing_time(src, kind, target_year=2027)
    produced = [c for c in out.columns if c not in src.columns]
    assert produced, "fixture produced no playing-time columns"

    written = set(RP._format_output(out).columns)
    missing = sorted(c for c in produced if c not in written)
    assert not missing, (
        f"the playing-time model produces {missing} and _format_output drops "
        f"them — add them to run_pipeline.PLAYING_TIME_COLS")


@pytest.mark.parametrize("kind", ["hitter", "pitcher"])
def test_the_columns_the_export_reads_by_name_are_written(kind):
    src = _frame(kind=kind)
    out, _, _ = PT.project_playing_time(src, kind, target_year=2027)
    written = set(RP._format_output(out).columns)
    for col, what in EXPORT_NEEDS.items():
        if col not in out.columns:
            continue          # not produced for this side, e.g. Proj_GS
        assert col in written, f"{col} is dropped; the export shows {what}"


def test_a_role_actually_arrives_rather_than_an_empty_column():
    """Present-but-empty is the failure mode that looked like success."""
    src = _frame(kind="hitter")
    out, _, _ = PT.project_playing_time(src, "hitter", target_year=2027)
    written = RP._format_output(out)
    assert written["pt_role"].notna().any()
    assert (written["pt_role"].astype(str).str.len() > 0).any()


def test_the_whitelist_has_no_names_the_model_does_not_produce():
    """A stale entry is dead weight that reads as coverage."""
    hit, _, _ = PT.project_playing_time(_frame(), "hitter", target_year=2027)
    pit, _, _ = PT.project_playing_time(_frame(kind="pitcher"), "pitcher",
                                        target_year=2027)
    known = set(hit.columns) | set(pit.columns)
    # evidence_* are set upstream of the model, in the tier step.
    allowed = known | {"evidence_volume", "evidence_season"}
    stale = [c for c in RP.PLAYING_TIME_COLS if c not in allowed]
    assert not stale, f"PLAYING_TIME_COLS names columns nobody produces: {stale}"

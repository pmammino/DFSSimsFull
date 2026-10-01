"""A player in a job battle is not 100% anything.

The role workbook has carried one probability column per role, and a
`Role Prob Sum` check beside them, since it was written. Its own legend says
so in as many words — "WHY PROBABILITIES AND NOT ONE ROLE ... the pipeline
reads the INPUT columns (role probabilities, Role Start, Availability)".

It did not. `load_role_overrides` read a single `Role` name and nothing else,
so a 60/30/10 split typed into the sheet was silently discarded and the
player kept whatever one label the default had given him. These tests cover
the mixture actually reaching the model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import playing_time_model as M  # noqa: E402
import role_taxonomy as RT  # noqa: E402

MIX = "Full Time:0.6|Strong Side Platoon:0.3|Utility IF:0.1"


# ─────────────────────────────────────────────────────────────────────────────
# Parsing and formatting
# ─────────────────────────────────────────────────────────────────────────────

def test_a_mixture_parses_and_normalizes():
    m = RT.parse_role_mix(MIX, "hitter")
    assert set(m) == {"Full Time", "Strong Side Platoon", "Utility IF"}
    assert sum(m.values()) == pytest.approx(1.0)
    assert m["Full Time"] == pytest.approx(0.6)


def test_weights_that_do_not_sum_to_one_are_normalized():
    m = RT.parse_role_mix("Full Time:3|Bench Bat:1", "hitter")
    assert m["Full Time"] == pytest.approx(0.75)
    assert m["Bench Bat"] == pytest.approx(0.25)


def test_an_unknown_role_is_dropped_rather_than_raising():
    """A typo in a hand-edited file is the likeliest failure here."""
    m = RT.parse_role_mix("Clozer:0.5|Closer:0.5", "pitcher")
    assert m == {"Closer": 1.0}


def test_a_mixture_of_nothing_but_typos_comes_back_none():
    assert RT.parse_role_mix("Clozer:1.0", "pitcher") is None
    assert RT.parse_role_mix("", "hitter") is None
    assert RT.parse_role_mix(np.nan, "hitter") is None


def test_the_string_form_round_trips():
    m = RT.parse_role_mix(MIX, "hitter")
    assert RT.parse_role_mix(RT.format_role_mix(m), "hitter") == pytest.approx(m)


def test_the_modal_role_is_the_heaviest():
    assert RT.modal_role(RT.parse_role_mix(MIX, "hitter")) == "Full Time"
    # Ties break by name, so the same input always gives the same answer.
    tie = RT.parse_role_mix("Bench Bat:0.5|Full Time:0.5", "hitter")
    assert RT.modal_role(tie) == RT.modal_role(tie) == "Bench Bat"


# ─────────────────────────────────────────────────────────────────────────────
# Blending
# ─────────────────────────────────────────────────────────────────────────────

def test_volume_blends_as_an_expectation():
    a = RT.blend_anchors(RT.parse_role_mix(MIX, "hitter"), "hitter")
    assert a["pa"] == pytest.approx(0.6 * 630 + 0.3 * 480 + 0.1 * 290)


def test_save_shares_blend_as_an_expectation():
    """Half a closer's job is half a closer's saves."""
    a = RT.blend_anchors(
        RT.parse_role_mix("Closer:0.5|Late Inning RP (Setup):0.5", "pitcher"),
        "pitcher")
    assert a["sv"] == pytest.approx(0.5 * 0.80 + 0.5 * 0.06)
    assert a["ip"] == pytest.approx(0.5 * 62 + 0.5 * 65)


def test_platoon_exposure_weights_by_plate_appearances_not_probability():
    """vL is a share OF his plate appearances, so the role that gives him
    more trips to the plate has more say in who he faces.

    Half a season of 630 PA at .29 and half of 150 PA at .36 is .30
    exposure, not the .325 a straight probability average would give.
    """
    m = RT.parse_role_mix("Full Time:0.5|Bench Bat:0.5", "hitter")
    a = RT.blend_anchors(m, "hitter")
    expected = (0.5 * 630 * 0.29 + 0.5 * 150 * 0.36) / (0.5 * 630 + 0.5 * 150)
    assert a["vl_rhb"] == pytest.approx(expected)
    assert a["vl_rhb"] != pytest.approx(0.5 * 0.29 + 0.5 * 0.36)


def test_a_single_role_mixture_is_that_role():
    a = RT.blend_anchors(RT.parse_role_mix("Full Time:1", "hitter"), "hitter")
    assert a["pa"] == pytest.approx(630)
    assert a["vl_rhb"] == pytest.approx(0.29)


def test_the_spread_is_the_uncertainty_that_was_expressed():
    assert RT.mix_volume_sd(RT.parse_role_mix("Full Time:1", "hitter"),
                            "hitter") == pytest.approx(0.0)
    sd = RT.mix_volume_sd(RT.parse_role_mix(MIX, "hitter"), "hitter")
    assert sd > 50, "a 630/480/290 split is not a settled player"


# ─────────────────────────────────────────────────────────────────────────────
# Reaching the model
# ─────────────────────────────────────────────────────────────────────────────

def _frame(n=30):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "PlayerId": range(1, n + 1),
        "Name": [f"p{i}" for i in range(n)],
        "Pred_target_team_id": 147,
        "Last_PA": rng.integers(200, 600, n),
        "Career_PA": rng.integers(500, 4000, n),
        "P_K": 0.22, "P_BB": 0.085, "P_HBP": 0.011, "P_SF": 0.006,
        "P_HR": 0.030, "P_3B": 0.004, "P_2B": 0.042, "P_1B": 0.142,
        "P_BIPOut": 0.460,
    })


def _run(tmp_path, overrides: pd.DataFrame | None, kind="hitter"):
    roster = tmp_path / "rosters"
    roster.mkdir(exist_ok=True)
    if overrides is not None:
        plural = "hitters" if kind == "hitter" else "pitchers"
        overrides.to_csv(roster / f"player_roles_{plural}_2027.csv", index=False)
    out, _, stats = M.project_playing_time(
        _frame(), kind, target_year=2027, roster_path=str(roster))
    return out, stats


def test_the_workbooks_own_column_shape_is_read(tmp_path):
    """One column per role, which is what exporting the sheet produces."""
    base, _ = _run(tmp_path, None)
    pid = int(base.nlargest(1, "Proj_PA")["PlayerId"].iloc[0])

    out, stats = _run(tmp_path, pd.DataFrame([{
        "PlayerId": pid, "Full Time": 0.6,
        "Strong Side Platoon": 0.3, "Utility IF": 0.1}]))
    row = out[out["PlayerId"] == pid].iloc[0]
    assert stats["overrides"]["mix"] == 1
    assert row["pt_anchor"] == pytest.approx(0.6 * 630 + 0.3 * 480 + 0.1 * 290)
    assert RT.parse_role_mix(row["pt_role_mix"], "hitter")["Full Time"] \
        == pytest.approx(0.6)


def test_a_ready_made_mix_string_is_read_too(tmp_path):
    base, _ = _run(tmp_path, None)
    pid = int(base.nlargest(1, "Proj_PA")["PlayerId"].iloc[0])
    out, stats = _run(tmp_path, pd.DataFrame([{"PlayerId": pid,
                                               "Role Mix": MIX}]))
    assert stats["overrides"]["mix"] == 1
    row = out[out["PlayerId"] == pid].iloc[0]
    assert row["pt_anchor"] == pytest.approx(0.6 * 630 + 0.3 * 480 + 0.1 * 290)


def test_the_modal_role_is_what_the_roster_sees(tmp_path):
    """A 60/40 split still has to occupy one roster spot."""
    base, _ = _run(tmp_path, None)
    pid = int(base.nlargest(1, "Proj_PA")["PlayerId"].iloc[0])
    out, _ = _run(tmp_path, pd.DataFrame([{"PlayerId": pid,
                                           "Role Mix": "Bench Bat:0.4|Full Time:0.6"}]))
    row = out[out["PlayerId"] == pid].iloc[0]
    assert row["pt_role"] == "Full Time"
    assert row["pt_family"] == RT.role_family("Full Time")


def test_naming_one_role_outright_beats_a_mixture_in_the_same_file(tmp_path):
    """The more specific statement wins."""
    base, _ = _run(tmp_path, None)
    pid = int(base.nlargest(1, "Proj_PA")["PlayerId"].iloc[0])
    out, _ = _run(tmp_path, pd.DataFrame([{
        "PlayerId": pid, "Role": "Catcher - Backup", "Role Mix": MIX}]))
    assert out[out["PlayerId"] == pid].iloc[0]["pt_role"] == "Catcher - Backup"


def test_a_settled_player_carries_an_empty_mix_and_no_spread(tmp_path):
    out, _ = _run(tmp_path, None)
    assert "pt_role_mix" in out.columns
    assert (out["pt_role_mix"].astype(str) == "").all()
    assert (out["pt_role_sd"] == 0).all()


def test_the_team_still_closes_with_mixtures_in_play(tmp_path):
    """A mixture changes a player's SHARE of the budget, never its size."""
    base, _ = _run(tmp_path, None)
    ids = base.nlargest(5, "Proj_PA")["PlayerId"].astype(int)
    out, _ = _run(tmp_path, pd.DataFrame(
        [{"PlayerId": int(p), "Role Mix": MIX} for p in ids]))
    for frame in (base, out):
        projected = frame[frame["pt_tier"].astype(str) != "floor"]
        total = projected.groupby("Pred_target_team_id")["Proj_PA"].sum()
        assert total.to_numpy() == pytest.approx(M.TEAM_PA_BUDGET)


def test_a_mixture_moves_platoon_exposure(tmp_path):
    """The reason this matters beyond playing time: the daily sim reads it."""
    base, _ = _run(tmp_path, None)
    pid = int(base.nlargest(1, "Proj_PA")["PlayerId"].iloc[0])
    before = base[base["PlayerId"] == pid].iloc[0]["Proj_vL_share"]
    out, _ = _run(tmp_path, pd.DataFrame([{
        "PlayerId": pid, "Role Mix": "Full Time:0.5|Weak Side Platoon:0.5"}]))
    after = out[out["PlayerId"] == pid].iloc[0]["Proj_vL_share"]
    assert after > before + 0.05, "half a weak-side job should raise vL exposure"


def test_probabilities_that_do_not_sum_to_one_are_reported(tmp_path, capsys):
    base, _ = _run(tmp_path, None)
    pid = int(base.nlargest(1, "Proj_PA")["PlayerId"].iloc[0])
    _run(tmp_path, pd.DataFrame([{"PlayerId": pid, "Full Time": 0.6,
                                  "Bench Bat": 0.1}]))
    assert "sum to 0.70" in capsys.readouterr().out


def test_pitcher_mixtures_blend_the_bullpen_pools(tmp_path):
    out, stats = _run(tmp_path, pd.DataFrame([{
        "PlayerId": 1, "Role Mix": "Closer:0.5|Late Inning RP (Setup):0.5"}]),
        kind="pitcher")
    row = out[out["PlayerId"] == 1].iloc[0]
    assert stats["overrides"]["mix"] == 1
    assert row["pt_save_share"] == pytest.approx(0.5 * 0.80 + 0.5 * 0.06)
    assert row["pt_hold_share"] == pytest.approx(0.5 * 0.02 + 0.5 * 0.29)

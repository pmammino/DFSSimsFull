"""
Tests for the refresh gate's own checks.

The gate decides whether a rebuild ships, so a check that reports the wrong
thing is worse than no check. Both failure directions have now happened for
real and both are pinned here:

  FALSE POSITIVE — `Team.nunique() == 30` failed a CORRECT refresh. Once the
    minor-league feed worked, the output legitimately held 150 labels (30 clubs
    + 120 affiliates), and the check blamed `data_acquisition._team_code`,
    which was fine. Two good rows were buried under it.

  FALSE NEGATIVE — the offense/defense check applied `.fillna(0)` to the VALUE,
    so a pitcher with no `R_per_PA` counted as allowing ZERO runs per PA at
    full weight. And nothing at all checked per-player plausibility, so a
    0.283 per-BIP home-run rate and a NEGATIVE ERA passed every row except one
    aggregate that named the symptom rather than the cause.

Run with:  python -m pytest tests/test_verify_refresh_checks.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from team_context import TEAM_ABBR_BY_ID
from verify_refresh import (  # noqa: E402
    FAIL, PASS, WARN, Checks, check_physically_possible,
    check_pool_composition, check_team_identity, volume_weights,
)

MLB = sorted(TEAM_ABBR_BY_ID.values())
AFFILIATES = ["Round Rock Express", "Toledo Mud Hens", "Syracuse Mets",
              "Tulsa Drillers", "Peoria Chiefs"]


def _rows(checks: Checks) -> dict:
    """{check name: status} for assertions."""
    return {name: status for status, name, _ in checks.rows}


def _detail(checks: Checks, name: str) -> str:
    return next(d for _, n, d in checks.rows if n == name)


def _team_frame(labels, n_missing_id=0):
    """A frame carrying `labels` as Team, cycled to cover them all."""
    n = max(len(labels), 30)
    team = [labels[i % len(labels)] for i in range(n)]
    tid = [1.0] * n
    for i in range(min(n_missing_id, n)):
        tid[i] = np.nan
    return pd.DataFrame({"Team": team, "Pred_target_team_id": tid,
                         "Name": [f"p{i}" for i in range(n)]})


# ─────────────────────────────────────────────────────────────────────────────
# team identity
# ─────────────────────────────────────────────────────────────────────────────

def test_thirty_clubs_alone_passes():
    c = Checks()
    check_team_identity(c, _team_frame(MLB), _team_frame(MLB))
    r = _rows(c)
    assert r["MLB team labels (hitters)"] == PASS
    assert r["MLB team labels (pitchers)"] == PASS


def test_affiliates_alongside_the_clubs_do_not_fail_the_gate():
    """THE false positive: a correct refresh with a live MiLB feed."""
    c = Checks()
    labels = MLB + AFFILIATES
    check_team_identity(c, _team_frame(labels), _team_frame(labels))
    r = _rows(c)
    assert r["MLB team labels (hitters)"] == PASS
    assert r["non-MLB labels (hitters)"] == WARN, (
        "affiliates must be reported, not failed on")


def test_collapsed_club_labels_still_fail():
    """The ORIGINAL defect must still be caught: 30 clubs -> 26 labels."""
    collapsed = [l for l in MLB if l not in ("CHC", "LAA", "NYM", "SF")]
    c = Checks()
    check_team_identity(c, _team_frame(collapsed), _team_frame(collapsed))
    assert _rows(c)["MLB team labels (hitters)"] == FAIL
    d = _detail(c, "MLB team labels (hitters)")
    assert "26/30" in d
    for missing in ("CHC", "LAA", "NYM", "SF"):
        assert missing in d, "the message must name what went missing"


def test_old_name_prefix_codes_fail():
    """`name[:3]` output ("Ari", "Atl") is not an abbreviation."""
    prefixes = ["Ari", "Atl", "Bos", "Chi", "Los", "New", "San"]
    c = Checks()
    check_team_identity(c, _team_frame(prefixes), _team_frame(prefixes))
    assert _rows(c)["MLB team labels (hitters)"] == FAIL
    assert "0/30" in _detail(c, "MLB team labels (hitters)")


def test_affiliate_warn_is_suppressed_when_the_clubs_are_missing():
    """Don't count one defect twice — the FAIL already says it."""
    c = Checks()
    check_team_identity(c, _team_frame(AFFILIATES), _team_frame(AFFILIATES))
    assert "non-MLB labels (hitters)" not in _rows(c)


def test_present_but_mostly_empty_team_ids_fail():
    """A column that exists and is 90% NaN is not a pass."""
    c = Checks()
    df = _team_frame(MLB, n_missing_id=27)
    check_team_identity(c, df, df)
    assert _rows(c)["team ids (hitters)"] == FAIL
    assert "no park factor" in _detail(c, "team ids (hitters)")


def test_a_few_missing_team_ids_still_pass():
    """Free agents and unsigned players are legitimately without a club."""
    c = Checks()
    df = _team_frame(MLB, n_missing_id=3)
    check_team_identity(c, df, df)
    assert _rows(c)["team ids (hitters)"] == PASS


def test_absent_team_id_column_fails():
    c = Checks()
    df = _team_frame(MLB).drop(columns=["Pred_target_team_id"])
    check_team_identity(c, df, df)
    assert _rows(c)["team ids (hitters)"] == FAIL
    assert "ABSENT" in _detail(c, "team ids (hitters)")


# ─────────────────────────────────────────────────────────────────────────────
# physical plausibility
# ─────────────────────────────────────────────────────────────────────────────

def _sane_pitchers(n=30):
    return pd.DataFrame({"RA9": [4.3] * n, "ERA": [4.0] * n,
                         "R_per_PA": [0.115] * n, "Career_PA": [500.0] * n})


def _sane_hitters(n=30):
    return pd.DataFrame({"P_HR": [0.035] * n, "Name": [f"h{i}" for i in range(n)]})


def test_sane_projections_pass_every_bound():
    c = Checks()
    check_physically_possible(c, _sane_hitters(), _sane_pitchers())
    assert set(_rows(c).values()) == {PASS}, _rows(c)


def test_the_shipped_home_run_rate_is_caught():
    """0.283/PA — the value that passed the old gate."""
    h = _sane_hitters()
    h.loc[0, "P_HR"] = 0.283
    h.loc[0, "Name"] = "Carter Garate"
    c = Checks()
    check_physically_possible(c, h, _sane_pitchers())
    assert _rows(c)["P_HR plausible"] == FAIL
    d = _detail(c, "P_HR plausible")
    assert "Carter Garate" in d, "the message must name the player"


def test_a_record_breaking_but_possible_rate_passes():
    """The bound is physical, not editorial: it can't flag a great season."""
    h = _sane_hitters()
    h.loc[0, "P_HR"] = 0.105          # ~Bonds 2001
    c = Checks()
    check_physically_possible(c, h, _sane_pitchers())
    assert _rows(c)["P_HR plausible"] == PASS


@pytest.mark.parametrize("col", ["RA9", "ERA", "R_per_PA"])
def test_negative_pitcher_values_are_caught(col):
    """RA9 -0.815 and ERA -0.761 both shipped."""
    p = _sane_pitchers()
    p.loc[0, col] = -0.815
    c = Checks()
    check_physically_possible(c, _sane_hitters(), p)
    assert _rows(c)[f"{col} non-negative"] == FAIL
    assert "NEGATIVE" in _detail(c, f"{col} non-negative")


def test_the_shipped_league_ra9_is_caught():
    """Mean RA9 2.15 against a real ~4.40 is a 50% error in every run total."""
    p = _sane_pitchers()
    p["RA9"] = 2.15
    c = Checks()
    check_physically_possible(c, _sane_hitters(), p)
    assert _rows(c)["league RA9"] == FAIL
    assert "2.15" in _detail(c, "league RA9")


def test_league_ra9_accepts_the_real_range():
    for val in (3.9, 4.2, 4.4, 4.6, 5.0):
        p = _sane_pitchers()
        p["RA9"] = val
        c = Checks()
        check_physically_possible(c, _sane_hitters(), p)
        assert _rows(c)["league RA9"] == PASS, f"rejected a real {val}"


def test_missing_columns_are_skipped_not_failed():
    """A frame without these columns must not fabricate failures."""
    c = Checks()
    check_physically_possible(c, pd.DataFrame({"Name": ["x"]}),
                              pd.DataFrame({"Career_PA": [1.0]}))
    assert FAIL not in _rows(c).values()


# ─────────────────────────────────────────────────────────────────────────────
# the gate must not crash — a dead verifier reads as a broken run
# ─────────────────────────────────────────────────────────────────────────────

DEGENERATE = {
    # Career_PA absent was a live AttributeError: pd.to_numeric(df.get(col))
    # returns a SCALAR nan, and .fillna(0) on a numpy float raises.
    "no Career_PA": (pd.DataFrame({"P_HR": [0.03], "Name": ["a"]}),
                     pd.DataFrame({"RA9": [4.3], "ERA": [4.0],
                                   "R_per_PA": [0.115]})),
    "all-NaN P_HR": (pd.DataFrame({"P_HR": [np.nan], "Name": ["a"]}),
                     pd.DataFrame({"RA9": [4.3], "Career_PA": [10.0]})),
    "no Name":      (pd.DataFrame({"P_HR": [0.30]}),
                     pd.DataFrame({"RA9": [4.3], "Career_PA": [10.0]})),
    "all-NaN RA9":  (pd.DataFrame({"P_HR": [0.03], "Name": ["a"]}),
                     pd.DataFrame({"RA9": [np.nan], "Career_PA": [10.0]})),
    "empty":        (pd.DataFrame(), pd.DataFrame()),
}


@pytest.mark.parametrize("tag", sorted(DEGENERATE))
def test_checks_survive_degenerate_frames(tag):
    from verify_refresh import check_league_calibration
    h, p = DEGENERATE[tag]
    for fn in (check_physically_possible, check_league_calibration):
        fn(Checks(), h, p)      # must not raise


def test_a_flagged_row_without_a_name_still_fails():
    """Naming the worst player is a convenience, not a precondition."""
    c = Checks()
    check_physically_possible(c, pd.DataFrame({"P_HR": [0.30]}),
                              pd.DataFrame({"RA9": [4.3]}))
    assert _rows(c)["P_HR plausible"] == FAIL


def test_absent_ra9_data_warns_rather_than_fails():
    """No data is not a calibration failure."""
    c = Checks()
    check_physically_possible(c, pd.DataFrame({"P_HR": [0.03], "Name": ["a"]}),
                              pd.DataFrame({"RA9": [np.nan]}))
    assert _rows(c)["league RA9"] == WARN


# ─────────────────────────────────────────────────────────────────────────────
# league-aggregate weighting: the floor tier must not steer it
# ─────────────────────────────────────────────────────────────────────────────

def _mixed_pool():
    """700 active regulars, 20 retired stars, 280 fringe, 1,875 MLE."""
    rows = [{"Career_PA": 3000.0, "pt_tier": "projected"} for _ in range(700)]
    rows += [{"Career_PA": 9000.0, "pt_tier": "floor"} for _ in range(20)]
    rows += [{"Career_PA": 400.0, "pt_tier": "floor"} for _ in range(280)]
    rows += [{"Career_PA": 0.0, "pt_tier": "floor"} for _ in range(1875)]
    return pd.DataFrame(rows)


def test_floor_tier_carries_no_aggregate_weight():
    """Miguel Cabrera's 11,000 career PA must not vote on league R/PA."""
    df = _mixed_pool()
    w = volume_weights(df)
    floor = df.pt_tier == "floor"
    assert w[floor].sum() == 0.0
    assert w[~floor].sum() > 0


def test_the_bias_being_removed_is_real():
    """Guard: the fixture must exercise the problem, or this proves nothing."""
    df = _mixed_pool()
    career = df.Career_PA
    floor = (df.pt_tier == "floor").to_numpy()
    rate = np.where(floor, 0.095, 0.1150)      # stale rates vs real
    biased = np.average(rate, weights=career)
    fixed = np.average(rate, weights=volume_weights(df))
    assert biased < 0.1150 * 0.99, "fixture no longer shows the bias"
    assert fixed == pytest.approx(0.1150)


def test_projected_players_keep_their_career_weight():
    """This is not a switch to Proj_PA — that is NaN until the PT model lands,
    and 1.0 only for the floor tier, so using it would invert the bias."""
    df = pd.DataFrame({"Career_PA": [1000.0, 4000.0],
                       "pt_tier": ["projected", "projected"]})
    assert list(volume_weights(df)) == [1000.0, 4000.0]


def test_weights_degrade_without_pt_tier():
    df = pd.DataFrame({"Career_PA": [10.0, 20.0]})
    assert list(volume_weights(df)) == [10.0, 20.0]


def test_weights_degrade_without_career_pa():
    assert list(volume_weights(pd.DataFrame({"x": [1, 2, 3]}))) == [1.0, 1.0, 1.0]


def test_an_all_floor_pool_does_not_produce_zero_weights():
    """Every weight zero would make np.average raise, killing the gate."""
    df = pd.DataFrame({"Career_PA": [500.0, 900.0],
                       "pt_tier": ["floor", "floor"]})
    w = volume_weights(df)
    assert w.sum() > 0


def test_pool_composition_reports_the_share_without_failing():
    df = _mixed_pool()
    c = Checks()
    check_pool_composition(c, df, df)
    r = _rows(c)
    assert r["floor-tier weight (hitters)"] == PASS
    assert "12.2%" in _detail(c, "floor-tier weight (hitters)")


def test_pool_composition_is_skipped_without_tiers():
    c = Checks()
    check_pool_composition(c, pd.DataFrame({"Career_PA": [1.0]}),
                           pd.DataFrame({"Career_PA": [1.0]}))
    assert not c.rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

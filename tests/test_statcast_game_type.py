"""Batted balls from games that are not the regular season.

The savant search endpoint takes a date range and returns every game played
in it, with no game_type filter of its own. Both BIP scrapes run March to
November, which brackets spring training at one end and the whole postseason
at the other, so without an explicit filter the batted-ball pool carries:

    "S"  spring training   -- rosters full of arms who never reach the majors
    "E"  exhibition
    "F"/"D"/"L"/"W"        -- the postseason, the opposite selection

Those rows then get counted against regular-season PA denominators all the
way downstream, which inflates the implied share of plate appearances that
end in a ball in play — and every per-PA batted-ball rate is measured
against that share.
"""

import pandas as pd
import pytest

from data_acquisition import regular_season_only
from pipeline_config import STATCAST_REGULAR_SEASON


def _frame(types):
    return pd.DataFrame({
        "game_type": list(types),
        "description": ["hit_into_play"] * len(types),
        "launch_speed": [95.0] * len(types),
        "batter": range(len(types)),
    })


def test_keeps_only_regular_season_rows():
    out = regular_season_only(_frame(["R", "S", "R", "F", "D", "L", "W", "E"]))
    assert len(out) == 2
    assert set(out["game_type"]) == {"R"}


def test_spring_training_is_dropped():
    assert len(regular_season_only(_frame(["S", "S", "S"]))) == 0


def test_postseason_is_dropped():
    """Postseason batted balls are real, but they are not a 162-game season."""
    assert len(regular_season_only(_frame(["F", "D", "L", "W"]))) == 0


def test_a_frame_without_game_type_passes_through_untouched():
    """Some cached pulls predate the column, and must not be emptied."""
    df = pd.DataFrame({"batter": [1, 2, 3], "launch_speed": [90.0, 95.0, 100.0]})
    out = regular_season_only(df)
    assert out is df or out.equals(df)
    assert len(out) == 3


def test_none_is_passed_through():
    assert regular_season_only(None) is None


def test_non_string_game_type_does_not_raise():
    df = _frame(["R", "S"])
    df["game_type"] = [None, "R"]
    out = regular_season_only(df)
    assert len(out) == 1


def test_the_constant_is_the_regular_season_code():
    assert STATCAST_REGULAR_SEASON == "R"


def test_both_scrapers_apply_the_filter():
    """A filter only one of the two scrapes applies is a filter that leaks."""
    import inspect
    import data_acquisition
    src = inspect.getsource(data_acquisition.fetch_statcast_season)
    assert "regular_season_only" in src

    prefetch = (pytest.importorskip("pathlib").Path(__file__).parent.parent
                / "prefetch_bip.py").read_text()
    assert "regular_season_only" in prefetch

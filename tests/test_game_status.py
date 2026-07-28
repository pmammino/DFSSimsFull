"""Tests for game_status — classifying StatsAPI schedule statuses and folding
game matchups onto canonical team codes.

Run: python -m pytest tests/test_game_status.py  (or run this file directly).
All cases inject a pre-fetched schedule dict, so nothing touches the network.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import game_status as GS


def _sched(*games):
    """games: (away_name, home_name, detailedState, codedGameState, reason)."""
    return {"dates": [{"games": [
        {"gamePk": 100 + i,
         "status": {"detailedState": ds, "codedGameState": cd, "reason": rsn},
         "teams": {"away": {"team": {"name": a}},
                   "home": {"team": {"name": h}}}}
        for i, (a, h, ds, cd, rsn) in enumerate(games)]}]}


def test_classify_detailed_states():
    assert GS._classify("Postponed", "D") == "postponed"
    assert GS._classify("Cancelled", "C") == "cancelled"
    assert GS._classify("Canceled", "C") == "cancelled"     # US spelling
    assert GS._classify("Suspended", "U") == "suspended"
    assert GS._classify("Scheduled", "S") is None
    assert GS._classify("In Progress", "I") is None
    assert GS._classify("Final", "F") is None


def test_classify_falls_back_to_coded_state():
    # detailedState unrecognised but the coded state still flags it.
    assert GS._classify("Delayed Start: Rain", "D") == "postponed"
    assert GS._classify("", "C") == "cancelled"
    assert GS._classify(None, None) is None


def test_game_statuses_maps_full_names_to_canonical_codes():
    rows = GS.game_statuses(schedule_json=_sched(
        ("Athletics", "New York Yankees", "Scheduled", "S", ""),
        ("Arizona Diamondbacks", "St. Louis Cardinals", "Postponed", "D", "Rain"),
    ))
    by_pk = {r["game_pk"]: r for r in rows}
    assert by_pk[100]["away"] == "OAK" and by_pk[100]["home"] == "NYY"
    assert by_pk[100]["not_played"] is False
    assert by_pk[101]["away"] == "ARI" and by_pk[101]["home"] == "STL"
    assert by_pk[101]["not_played"] is True
    assert by_pk[101]["category"] == "postponed"
    assert by_pk[101]["matchup"] == frozenset({"ARI", "STL"})


def test_postponed_matchups_keys_by_canonical_frozenset():
    pm = GS.postponed_matchups(schedule_json=_sched(
        ("New York Mets", "Philadelphia Phillies", "Cancelled", "C", ""),
        ("Boston Red Sox", "Tampa Bay Rays", "Scheduled", "S", ""),
    ))
    assert set(pm) == {frozenset({"NYM", "PHI"})}
    info = pm[frozenset({"NYM", "PHI"})]
    assert info["category"] == "cancelled"
    assert info["away"] == "NYM" and info["home"] == "PHI"


def test_postponed_list_is_display_friendly():
    lst = GS.postponed_list(schedule_json=_sched(
        ("New York Mets", "Philadelphia Phillies", "Postponed", "D", "Rain"),
    ))
    assert lst == [{"away": "NYM", "home": "PHI", "status": "Postponed",
                    "reason": "Rain", "category": "postponed"}]


def test_string_schedule_json_is_parsed():
    lst = GS.postponed_list(schedule_json="{}")
    assert lst == []


def test_fetch_failure_is_best_effort_empty():
    """A raising fetch (network/HTTP down) must be swallowed by the best-effort
    wrappers and reported as 'no postponements' — never over-drop a slate."""
    import unittest.mock as mock
    with mock.patch.object(GS, "fetch_schedule",
                           side_effect=RuntimeError("boom")):
        assert GS.postponed_matchups(date="2026-07-17") == {}
        assert GS.postponed_list(date="2026-07-17") == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")

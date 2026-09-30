"""
Tests for the statsapi minor-league fetch → MLE feed mapping.

This path cannot be validated from the dev sandbox (statsapi is unreachable
through the proxy), and it is the path that silently cost 4,435 players their
organization: the /stats response names only the AFFILIATE, and the code read a
`currentTeam` field that only the older RotoWire feed supplied.

So the response shape is pinned here against mocked payloads instead. The
mocks describe what statsapi documents; if the live response differs, these
tests still guarantee the mapping does the right thing with whatever fields ARE
present, including none of them.

Run with:  python -m pytest tests/test_minors_fetch_mapping.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_acquisition as DA


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch(monkeypatch, payload):
    monkeypatch.setattr(DA.requests, "get",
                        lambda *a, **k: _Resp(payload))


# ─────────────────────────────────────────────────────────────────────────────
# the affiliate -> parent map
# ─────────────────────────────────────────────────────────────────────────────

def test_parent_map_reads_id_and_name(monkeypatch):
    _patch(monkeypatch, {"teams": [
        {"id": 102, "name": "Round Rock Express",
         "parentOrgId": 140, "parentOrgName": "Texas Rangers"},
        {"id": 103, "name": "Toledo Mud Hens",
         "parentOrgId": 116, "parentOrgName": "Detroit Tigers"},
    ]})
    m = DA._fetch_affiliate_parent_map(11)
    assert m[102] == (140, "Texas Rangers")
    assert m[103] == (116, "Detroit Tigers")


def test_parent_map_keeps_a_row_with_only_a_name(monkeypatch):
    """team_id_for_abbr resolves full club names, so a name alone is usable."""
    _patch(monkeypatch, {"teams": [
        {"id": 102, "parentOrgName": "Texas Rangers"},
    ]})
    assert DA._fetch_affiliate_parent_map(11)[102] == (None, "Texas Rangers")


def test_parent_map_keeps_a_row_with_only_an_id(monkeypatch):
    _patch(monkeypatch, {"teams": [{"id": 102, "parentOrgId": 140}]})
    assert DA._fetch_affiliate_parent_map(11)[102] == (140, None)


def test_parent_map_skips_rows_with_no_parent_at_all(monkeypatch):
    _patch(monkeypatch, {"teams": [
        {"id": 102, "name": "Round Rock Express"},
        {"id": 103, "parentOrgId": 116},
    ]})
    m = DA._fetch_affiliate_parent_map(11)
    assert 102 not in m
    assert 103 in m


def test_parent_map_coerces_a_stringified_id(monkeypatch):
    _patch(monkeypatch, {"teams": [
        {"id": "102", "parentOrgId": "140", "parentOrgName": "Texas Rangers"},
    ]})
    assert DA._fetch_affiliate_parent_map(11)[102] == (140, "Texas Rangers")


def test_parent_map_survives_a_missing_field_entirely(monkeypatch):
    """The documented field may not exist — degrade, never raise."""
    _patch(monkeypatch, {"teams": [{"id": 102, "name": "Round Rock"}]})
    assert DA._fetch_affiliate_parent_map(11) == {}


def test_parent_map_survives_a_request_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network")
    monkeypatch.setattr(DA.requests, "get", boom)
    assert DA._fetch_affiliate_parent_map(11) == {}


def test_parent_map_survives_a_junk_payload(monkeypatch):
    for payload in ({}, {"teams": None}, {"teams": []}):
        _patch(monkeypatch, payload)
        assert DA._fetch_affiliate_parent_map(11) == {}


# ─────────────────────────────────────────────────────────────────────────────
# the record mapping
# ─────────────────────────────────────────────────────────────────────────────

def _stats_payload(group="hitting", team=None):
    stat = ({"gamesPlayed": 120, "atBats": 450, "baseOnBalls": 40,
             "strikeOuts": 95, "hits": 130, "doubles": 28, "triples": 3,
             "homeRuns": 18, "stolenBases": 9, "caughtStealing": 3,
             "runs": 70, "rbi": 75}
            if group == "hitting" else
            {"gamesPlayed": 25, "inningsPitched": "140.2", "baseOnBalls": 45,
             "strikeOuts": 150, "hits": 120, "homeRuns": 14, "earnedRuns": 55,
             "battersFaced": 580, "gamesStarted": 25})
    return {"stats": [{"splits": [{
        "player": {"id": 691234, "fullName": "A Prospect",
                   "primaryPosition": {"abbreviation": "3B"}},
        "team": team if team is not None else {"id": 102,
                                              "name": "Round Rock Express"},
        "stat": stat,
    }]}]}


def test_record_takes_the_parent_from_the_map(monkeypatch):
    _patch(monkeypatch, _stats_payload())
    rec = DA._fetch_minors_statsapi_one(
        2026, 11, "hitting", {102: (140, "Texas Rangers")})[0]
    assert rec["parent_org_id"] == 140
    assert rec["currentTeam"] == "Texas Rangers"
    # The affiliate is still recorded, just no longer mistaken for the org.
    assert rec["team"] == "Round Rock Express"
    assert rec["affiliate_id"] == 102
    assert rec["mlbam_id"] == 691234


def test_record_prefers_an_inline_parent_over_the_map(monkeypatch):
    """If statsapi inlines it, the /teams call becomes unnecessary."""
    _patch(monkeypatch, _stats_payload(team={
        "id": 102, "name": "Round Rock Express",
        "parentOrgId": 140, "parentOrgName": "Texas Rangers"}))
    rec = DA._fetch_minors_statsapi_one(2026, 11, "hitting", {})[0]
    assert rec["parent_org_id"] == 140
    assert rec["currentTeam"] == "Texas Rangers"


def test_record_degrades_with_no_map_and_no_inline_parent(monkeypatch):
    """No guessing: unresolved is unresolved, and must not raise."""
    _patch(monkeypatch, _stats_payload())
    rec = DA._fetch_minors_statsapi_one(2026, 11, "hitting", None)[0]
    assert rec["parent_org_id"] is None
    assert rec["currentTeam"] is None
    assert rec["team"] == "Round Rock Express"


def test_record_joins_a_stringified_split_team_id(monkeypatch):
    _patch(monkeypatch, _stats_payload(team={"id": "102", "name": "RRE"}))
    rec = DA._fetch_minors_statsapi_one(
        2026, 11, "hitting", {102: (140, "Texas Rangers")})[0]
    assert rec["parent_org_id"] == 140


def test_record_survives_a_missing_team_object(monkeypatch):
    _patch(monkeypatch, _stats_payload(team={}))
    rec = DA._fetch_minors_statsapi_one(
        2026, 11, "hitting", {102: (140, "Texas Rangers")})[0]
    assert rec["parent_org_id"] is None


def test_pitcher_record_carries_the_parent_too(monkeypatch):
    """Hitters and pitchers disagreeing about the team column was a past bug."""
    _patch(monkeypatch, _stats_payload("pitching"))
    rec = DA._fetch_minors_statsapi_one(
        2026, 11, "pitching", {102: (140, "Texas Rangers")})[0]
    assert rec["parent_org_id"] == 140
    assert rec["bf"] == 580
    assert rec["ip"] == "140.2"


def test_parent_org_resolves_end_to_end(monkeypatch):
    """Fetch -> _parent_org -> a real MLB team id, the whole point of this."""
    from mle_translations import _parent_org
    _patch(monkeypatch, _stats_payload())
    rec = DA._fetch_minors_statsapi_one(
        2026, 11, "hitting", {102: (140, "Texas Rangers")})[0]
    assert _parent_org(rec) == ("TEX", 140.0)


def test_parent_org_resolves_end_to_end_from_the_name_alone(monkeypatch):
    """The redundant path: no id in the response, name still gets there."""
    from mle_translations import _parent_org
    _patch(monkeypatch, _stats_payload())
    rec = DA._fetch_minors_statsapi_one(
        2026, 11, "hitting", {102: (None, "Texas Rangers")})[0]
    abbr, tid = _parent_org(rec)
    assert (abbr, tid) == ("TEX", 140.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

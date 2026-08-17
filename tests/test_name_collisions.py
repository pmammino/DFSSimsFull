"""Same-name collision handling across the sim -> pool -> upload chain.

Two different real players can share a name on one slate (the Dodgers' Max Muncy
and the Athletics' Max Muncy; Héctor Rodríguez, etc.). Before this fix every
layer keyed players by name, so the two collapsed onto ONE sim array — the
cheaper/worse player then inherited the star's projection and flooded lineups.
These tests pin the disambiguation: each real player keeps its own projection,
a duplicate that can't be resolved fails safe (dropped, never mis-scored), and
the DK upload id still resolves.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matchup as M
import dk_ids
from stage_d import build_pool, norm, _sim_key_for


def _hproj(rows):
    df = pd.DataFrame(rows)
    df["name_key"] = df["Name"].map(M._norm)
    return df


def _slate(lineups):
    """lineups: {gid: {'away':(team,[names]), 'home':(team,[names])}}"""
    games = {}
    for gid, sides in lineups.items():
        g = {}
        for side, (team, names) in sides.items():
            g[side] = team
            g.setdefault("lineups", {})[side] = [
                {"name": n, "slot": i + 1, "pos": "3B"} for i, n in enumerate(names)]
        games[gid] = g
    return {"date": "2026-08-17", "games": games}


# --------------------------------------------------------------------------- #
# resolve_collisions
# --------------------------------------------------------------------------- #
def test_two_on_slate_resolved_by_team_and_elimination():
    # Dodgers Muncy team "LA"->LAD (projection team "Los" does NOT canonicalise),
    # Athletics Muncy "ATH"->OAK (projection "Ath"->OAK resolves). The A's row is
    # matched by team; the Dodgers row falls out by elimination.
    hproj = _hproj([
        {"Name": "Max Muncy", "Team": "Los"},     # Dodgers (unresolvable code)
        {"Name": "Max Muncy", "Team": "Ath"},     # Athletics (-> OAK)
        {"Name": "Mookie Betts", "Team": "Los"},
    ])
    slate = _slate({"g1": {"away": ("ATH", ["Max Muncy"]), "home": ("SEA", ["Mookie Betts"])},
                    "g2": {"away": ("LA", ["Max Muncy"]), "home": ("SF", ["Some Guy"])}})
    assign, unresolved, collided = M.resolve_collisions(slate, hproj)
    assert M._norm("Max Muncy") in collided
    assert not unresolved
    # A's Muncy -> the "Ath" row; Dodgers Muncy -> the "Los" row (by elimination)
    oak_row = assign[(M._norm("Max Muncy"), "OAK")]
    lad_row = assign[(M._norm("Max Muncy"), "LAD")]
    assert hproj.loc[oak_row, "Team"] == "Ath"
    assert hproj.loc[lad_row, "Team"] == "Los"


def test_unresolvable_collision_fails_safe():
    # two players share a name but there is only ONE projection row -> cannot
    # separate -> both flagged unresolved (dropped, never mis-scored).
    hproj = _hproj([{"Name": "John Smith", "Team": "Bos"}])
    slate = _slate({"g1": {"away": ("BOS", ["John Smith"]), "home": ("NYY", ["John Smith"])}})
    assign, unresolved, collided = M.resolve_collisions(slate, hproj)
    assert M._norm("John Smith") in collided
    assert len(unresolved) == 2 and not assign


def test_single_occurrence_with_twin_is_still_disambiguated():
    # Only the A's Muncy is in a posted lineup, but a Dodgers Muncy exists in the
    # projection set (and the DK feed lists both). The single occurrence must
    # STILL be disambiguated to its own team's row, so its plain-named twin can't
    # later share its sim in build_pool.
    hproj = _hproj([{"Name": "Max Muncy", "Team": "Los", "BatSide": "L"},
                    {"Name": "Max Muncy", "Team": "Ath", "BatSide": "R"}])
    slate = _slate({"g2": {"away": ("ATH", ["Max Muncy"]), "home": ("SF", ["X"])}})
    assign, unresolved, collided = M.resolve_collisions(slate, hproj)
    assert M._norm("Max Muncy") in collided               # ambiguous in projections
    assert not unresolved
    # the on-slate A's Muncy -> the "Ath" (OAK) row, keyed "Max Muncy (OAK)"
    row = hproj.loc[assign[(M._norm("Max Muncy"), "OAK")]]
    assert row["Team"] == "Ath"


def test_lineup_muncy_resolved_when_only_it_is_on_slate():
    # symmetric: only the Dodgers Muncy is posted. Its team ("LA"->LAD) can't be
    # canonicalised from the projection's "Los", but the A's row is confidently
    # OAK (not on this slate) and is excluded, leaving "Los" by elimination.
    hproj = _hproj([{"Name": "Max Muncy", "Team": "Los"},
                    {"Name": "Max Muncy", "Team": "Ath"}])
    slate = _slate({"g1": {"away": ("LA", ["Max Muncy"]), "home": ("SEA", ["Z"])}})
    assign, unresolved, _ = M.resolve_collisions(slate, hproj)
    assert not unresolved
    assert hproj.loc[assign[(M._norm("Max Muncy"), "LAD")]]["Team"] == "Los"


# --------------------------------------------------------------------------- #
# build_pool: two distinct rows, each with its own sim; mismatch dropped
# --------------------------------------------------------------------------- #
def _write_dk(tmp_path, rows):
    p = tmp_path / "dk.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_build_pool_separates_both_muncys(tmp_path):
    # sims carry team-qualified keys for the collision
    score = {norm("Max Muncy (LAD)"): np.full(10, 12.0),
             norm("Max Muncy (OAK)"): np.full(10, 6.0),
             norm("Mike Trout"): np.full(10, 11.0)}
    dk = _write_dk(tmp_path, [
        {"FullName": "Max Muncy", "Team": "LAD", "Position": "3B",
         "Salary": 5000, "Ownership": 10},
        {"FullName": "Max Muncy", "Team": "OAK", "Position": "SS",
         "Salary": 3500, "Ownership": 8},
        {"FullName": "Mike Trout", "Team": "LAA", "Position": "OF",
         "Salary": 6000, "Ownership": 20},
    ])
    pool = build_pool(dk, {}, {}, score)
    names = set(pool["Name"])
    assert "Max Muncy (LAD)" in names and "Max Muncy (OAK)" in names
    # each row maps to its OWN sim (distinct projections downstream)
    assert norm(pool[pool.Name == "Max Muncy (LAD)"].iloc[0]["Name"]) in score
    assert norm(pool[pool.Name == "Max Muncy (OAK)"].iloc[0]["Name"]) in score


def test_build_pool_drops_collision_with_no_matching_sim(tmp_path):
    # only the Dodgers Muncy was simmed; the A's Muncy (team matches no sim key)
    # is dropped rather than inheriting the Dodgers array.
    score = {norm("Max Muncy (LAD)"): np.full(10, 12.0)}
    dk = _write_dk(tmp_path, [
        {"FullName": "Max Muncy", "Team": "LAD", "Position": "3B",
         "Salary": 5000, "Ownership": 10},
        {"FullName": "Max Muncy", "Team": "OAK", "Position": "SS",
         "Salary": 3500, "Ownership": 8},
    ])
    pool = build_pool(dk, {}, {}, score)
    assert list(pool["Name"]) == ["Max Muncy (LAD)"]      # A's Muncy dropped


def test_stale_plain_sim_never_shared_across_collision(tmp_path):
    # sims predate the fix: a single plain "max muncy" key. Two DK Muncys must
    # NOT both map onto it (the original bug) — with no team-qualified key for
    # either, both are dropped (fail safe) rather than one wearing the other's
    # projection.
    score = {norm("Max Muncy"): np.full(10, 12.0), norm("Mike Trout"): np.full(10, 11.0)}
    dk = _write_dk(tmp_path, [
        {"FullName": "Max Muncy", "Team": "LAD", "Position": "3B",
         "Salary": 5000, "Ownership": 10},
        {"FullName": "Max Muncy", "Team": "OAK", "Position": "SS",
         "Salary": 3500, "Ownership": 8},
        {"FullName": "Mike Trout", "Team": "LAA", "Position": "OF",
         "Salary": 6000, "Ownership": 20},
    ])
    pool = build_pool(dk, {}, {}, score)
    assert "Max Muncy" not in set(pool["Name"])        # the shared plain key is refused
    assert not any(n.startswith("Max Muncy") for n in pool["Name"])
    assert "Mike Trout" in set(pool["Name"])           # non-collision unaffected


def test_non_colliding_same_name_single_player_keeps_plain(tmp_path):
    # a name that is NOT a collision (one DK team) still uses the plain sim key.
    score = {norm("Max Muncy"): np.full(10, 12.0)}
    dk = _write_dk(tmp_path, [{"FullName": "Max Muncy", "Team": "LAD",
                               "Position": "3B", "Salary": 5000, "Ownership": 10}])
    pool = build_pool(dk, {}, {}, score)
    assert list(pool["Name"]) == ["Max Muncy"]


def test_non_colliding_player_keeps_plain_name(tmp_path):
    score = {norm("Shohei Ohtani"): np.full(10, 15.0)}
    dk = _write_dk(tmp_path, [{"FullName": "Shohei Ohtani", "Team": "LAD",
                               "Position": "OF", "Salary": 6000, "Ownership": 30}])
    pool = build_pool(dk, {}, {}, score)
    assert list(pool["Name"]) == ["Shohei Ohtani"]        # untouched


def test_sim_key_for_prefers_team_qualified():
    simset = {norm("Max Muncy (LAD)"), norm("Max Muncy (OAK)")}
    disp, key = _sim_key_for("Max Muncy", "LAD", simset)
    assert disp == "Max Muncy (LAD)" and key == norm("Max Muncy (LAD)")
    disp, key = _sim_key_for("Nobody Here", "LAD", simset)
    assert disp is None and key is None


# --------------------------------------------------------------------------- #
# dk_ids: the disambiguated pool name still resolves to the right upload id
# --------------------------------------------------------------------------- #
def test_upload_id_resolves_for_disambiguated_name():
    idmap = {}
    dk_ids.add_id(idmap, "Max Muncy", "LAD", "111", pos="3B", salary=5000)
    dk_ids.add_id(idmap, "Max Muncy", "OAK", "222", pos="SS", salary=3500)
    # the build hands dk_ids the disambiguated name; the suffix + salary/team pick
    # the correct id, not "last write wins".
    assert dk_ids.lookup(idmap, "Max Muncy (LAD)", team="LAD", salary=5000) == "111"
    assert dk_ids.lookup(idmap, "Max Muncy (OAK)", team="OAK", salary=3500) == "222"
    # even with no explicit team, the "(TEAM)" suffix disambiguates
    assert dk_ids.lookup(idmap, "Max Muncy (OAK)") == "222"
    assert dk_ids.has_name(idmap, "Max Muncy (LAD)")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

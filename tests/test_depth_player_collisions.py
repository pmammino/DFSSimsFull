"""
test_depth_player_collisions.py
===============================
Guards the DAILY path against the organizational-depth players the season
engine added to the projection set.

The risk is specific and has bitten this repo twice already (the "Muncy-style
same-name drop"). `matchup.resolve_collisions` treats any name with 2+ rows in
the projection set as needing disambiguation:

    proj_counts = hproj['name_key'].value_counts()
    ambiguous = set(proj_counts[proj_counts >= 2].index)

The projection set now spans whole organizations, down to Single-A, so a real
MLB player very often has a namesake on a farm team. Counting those rows would
mark him ambiguous, push him through a disambiguation he cannot win (the
depth row's team may be the same org!), and land him in `unresolved` — which
the caller fails safe on by DROPPING him from the slate.

The fix is that depth rows never participate in ambiguity or resolution, while
still being available for a direct match (so a just-promoted player keeps his
own baseline). These tests pin both halves.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchup import _mlb_rows, _norm, _row_for, resolve_collisions  # noqa: E402


def _proj(rows):
    """rows: list of (Name, Team, pt_tier)."""
    df = pd.DataFrame(rows, columns=["Name", "Team", "pt_tier"])
    df["name_key"] = df["Name"].map(_norm)
    return df


def _slate(*games):
    """games: ((away_team, [away names]), (home_team, [home names])) pairs."""
    out = {"games": {}}
    for i, ((at, an), (ht, hn)) in enumerate(games):
        out["games"][f"g{i}"] = {
            "away": at, "home": ht,
            "lineups": {"away": [{"name": n} for n in an],
                        "home": [{"name": n} for n in hn]},
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# _mlb_rows
# ─────────────────────────────────────────────────────────────────────────────

def test_mlb_rows_drops_floor_tier():
    df = _proj([("A B", "NYY", "projected"), ("C D", "LAD", "floor")])
    assert list(_mlb_rows(df)["Name"]) == ["A B"]


def test_mlb_rows_passes_through_a_legacy_frame():
    """A projection file predating playing-time tiers must behave as before."""
    df = _proj([("A B", "NYY", "projected")]).drop(columns=["pt_tier"])
    assert len(_mlb_rows(df)) == 1


def test_mlb_rows_never_empties_a_frame():
    """A misconfigured tier column must not blank out a slate."""
    df = _proj([("A B", "NYY", "floor"), ("C D", "LAD", "floor")])
    assert len(_mlb_rows(df)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# The regression: a farmhand namesake must not drop a real player
# ─────────────────────────────────────────────────────────────────────────────

def test_a_farmhand_namesake_does_not_drop_the_mlb_player():
    """THE case this protects. One MLB regular, one Single-A namesake in the
    same organization — the depth row cannot be separated by team, so before
    the fix the real player went unresolved and was dropped."""
    proj = _proj([
        ("Mike Smith", "NYY", "projected"),
        ("Mike Smith", "NYY", "floor"),      # A-ball, same org
    ])
    slate = _slate((("BOS", ["Other Guy"]), ("NYY", ["Mike Smith"])))
    assign, unresolved, collided = resolve_collisions(slate, proj)
    nk = _norm("Mike Smith")

    assert nk not in unresolved and not unresolved, (
        f"real player was dropped: {unresolved}"
    )
    # Better than merely resolving: the name is no longer treated as colliding
    # at all, so it skips the collision machinery and keeps its plain sim key —
    # zero blast radius, exactly as for any unique name.
    assert nk not in collided
    assert nk not in [k[0] for k in assign]
    # It is then resolved normally, to the MLB row.
    assert _row_for(proj, "Mike Smith", team="NYY")["pt_tier"] == "projected"


def test_two_real_mlb_twins_still_resolve_by_team():
    """The genuine collision case must keep working — depth rows present or not."""
    proj = _proj([
        ("Max Muncy", "LAD", "projected"),
        ("Max Muncy", "OAK", "projected"),
        ("Max Muncy", "LAD", "floor"),        # a third, irrelevant row
    ])
    slate = _slate((("LAD", ["Max Muncy"]), ("OAK", ["Max Muncy"])))
    assign, unresolved, collided = resolve_collisions(slate, proj)

    assert not unresolved
    nk = _norm("Max Muncy")
    assert proj.loc[assign[(nk, "LAD")], "Team"] == "LAD"
    assert proj.loc[assign[(nk, "OAK")], "Team"] == "OAK"
    assert nk in collided


def test_a_depth_only_name_is_not_treated_as_ambiguous():
    """Two farmhands sharing a name must not make an unrelated slate name
    require disambiguation."""
    proj = _proj([
        ("Real Player", "NYY", "projected"),
        ("Farm Guy", "NYY", "floor"),
        ("Farm Guy", "LAD", "floor"),
    ])
    slate = _slate((("BOS", ["Other Guy"]), ("NYY", ["Real Player"])))
    assign, unresolved, collided = resolve_collisions(slate, proj)
    assert not unresolved
    assert _norm("Real Player") not in collided


def test_many_depth_namesakes_do_not_degrade_resolution():
    """Scale check: a common name with a dozen farmhands attached."""
    rows = [("John Lopez", "NYY", "projected")]
    rows += [("John Lopez", t, "floor")
             for t in ("NYY", "NYY", "LAD", "BOS", "SF", "COL")]
    proj = _proj(rows)
    slate = _slate((("BOS", ["Someone Else"]), ("NYY", ["John Lopez"])))
    assign, unresolved, collided = resolve_collisions(slate, proj)
    assert not unresolved
    assert _norm("John Lopez") not in collided, (
        "six farmhand namesakes should not make one MLB player collide"
    )
    assert _row_for(proj, "John Lopez", team="NYY")["pt_tier"] == "projected"


# ─────────────────────────────────────────────────────────────────────────────
# ...but a depth row is still usable when it is the only match
# ─────────────────────────────────────────────────────────────────────────────

def test_a_just_promoted_depth_player_is_still_matchable():
    """Tiers are set from last season's evidence, so a player called up in
    March is `floor` while genuinely in a lineup. He should get his own
    baseline rather than no projection — an improvement on the old behavior,
    where he was absent from the file entirely."""
    proj = _proj([("Rookie Callup", "NYY", "floor")])
    row = _row_for(proj, "Rookie Callup", team="NYY")
    assert row is not None
    assert row["Name"] == "Rookie Callup"


def test_row_for_prefers_the_mlb_row_over_a_depth_namesake():
    proj = _proj([
        ("Mike Smith", "NYY", "floor"),
        ("Mike Smith", "NYY", "projected"),
    ])
    assert _row_for(proj, "Mike Smith", team="NYY")["pt_tier"] == "projected"


def test_row_for_still_disambiguates_real_twins_by_team():
    proj = _proj([
        ("Max Muncy", "LAD", "projected"),
        ("Max Muncy", "OAK", "projected"),
    ])
    assert _row_for(proj, "Max Muncy", team="OAK")["Team"] == "OAK"
    assert _row_for(proj, "Max Muncy", team="LAD")["Team"] == "LAD"


def test_resolution_falls_back_to_depth_rows_when_there_is_no_mlb_row():
    """Two promoted prospects with the same name on two slate teams: the only
    rows available are floor-tier, and they must still be separable."""
    proj = _proj([
        ("Twin Prospect", "NYY", "floor"),
        ("Twin Prospect", "LAD", "floor"),
    ])
    slate = _slate((("LAD", ["Twin Prospect"]), ("NYY", ["Twin Prospect"])))
    assign, unresolved, _ = resolve_collisions(slate, proj)
    assert not unresolved
    nk = _norm("Twin Prospect")
    assert proj.loc[assign[(nk, "NYY")], "Team"] == "NYY"
    assert proj.loc[assign[(nk, "LAD")], "Team"] == "LAD"


def test_legacy_projection_frame_behaves_exactly_as_before():
    """No pt_tier column anywhere — the tier filter must be a no-op."""
    proj = _proj([
        ("Max Muncy", "LAD", "projected"),
        ("Max Muncy", "OAK", "projected"),
    ]).drop(columns=["pt_tier"])
    slate = _slate((("LAD", ["Max Muncy"]), ("OAK", ["Max Muncy"])))
    assign, unresolved, collided = resolve_collisions(slate, proj)
    assert not unresolved
    assert len(assign) == 2

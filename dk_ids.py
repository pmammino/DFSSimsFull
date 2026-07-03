#!/usr/bin/env python3
"""
dk_ids.py
=========
DraftKings player-ID maps that disambiguate same-named players.

Two different players can share a name on a single slate — e.g. **Max Muncy**
(Dodgers) and **Max Muncy** (Athletics). A plain ``name -> id`` map collapses
them (last write wins), so the upload file can embed the WRONG player's DK id
even though the lineup itself is correct (right salary, right team). That's a
pure export bug: the lineup means one Muncy, the CSV ships the other.

These helpers key ids by ``(normalized name, team)`` and resolve using the team
already carried in each lineup cell (``"Name (TEAM)"``), so the lineup decides
which Muncy is uploaded. When a name is unique the team is ignored, so nothing
changes for the ~99% of players with no name clash.

Map shape: ``{ norm(name): { TEAMKEY: "dk_id", ... }, ... }``.
"""
from stage_d import norm


def team_key(t):
    """Normalize a team code for keying/lookup (case- and space-insensitive)."""
    return str(t or "").strip().upper()


def add_id(idmap, name, team, cid):
    """Record ``cid`` for (name, team). Safe to call repeatedly; a later id for
    the same (name, team) overwrites, different teams coexist."""
    if not cid:
        return
    idmap.setdefault(norm(name), {})[team_key(team)] = str(cid).strip()


def count_ids(idmap):
    """Total distinct player ids in the map (counts both Muncys, not one name)."""
    return sum(len(v) if isinstance(v, dict) else 1 for v in idmap.values())


def has_name(idmap, name):
    """True if any id exists for this name (team-agnostic membership test)."""
    return norm(name) in idmap


def lookup(idmap, name, team=""):
    """Resolve the DK id for a lineup player.

    Unique name -> that id (team ignored). Clashing name -> the id whose team
    matches `team`; returns ``None`` if the team can't disambiguate it. Tolerates
    a legacy flat ``name -> id`` map (returns the id) so old sessions don't crash.
    """
    d = idmap.get(norm(name))
    if not d:
        return None
    if isinstance(d, str):                 # legacy flat map
        return d
    if len(d) == 1:
        return next(iter(d.values()))
    return d.get(team_key(team))


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    m = {}
    add_id(m, "Max Muncy", "LAD", "111")
    add_id(m, "Max Muncy", "OAK", "222")     # same name, different team
    add_id(m, "Shohei Ohtani", "LAD", "333")

    # clashing name resolves by the lineup's team
    assert lookup(m, "Max Muncy", "LAD") == "111", m
    assert lookup(m, "Max Muncy", "OAK") == "222", m
    # unique name resolves regardless of team
    assert lookup(m, "Shohei Ohtani", "") == "333"
    assert lookup(m, "Shohei Ohtani", "NYY") == "333"
    # clashing name with an unknown team can't be disambiguated
    assert lookup(m, "Max Muncy", "NYY") is None
    assert lookup(m, "Max Muncy", "") is None
    # membership + counts
    assert has_name(m, "max muncy") and not has_name(m, "nobody")
    assert count_ids(m) == 3, count_ids(m)
    # legacy flat map tolerated
    assert lookup({"max muncy": "999"}, "Max Muncy", "LAD") == "999"

    print("dk_ids.py self-test passed:", m)

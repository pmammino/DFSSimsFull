#!/usr/bin/env python3
"""
dk_ids.py
=========
DraftKings player-ID maps that disambiguate same-named players.

Two DIFFERENT players can share a name on one slate — e.g. **Max Muncy** on the
Dodgers (3B, $3548) and **Max Muncy** on the Athletics (SS, $3558). The sim layer
now keeps them apart: a colliding player is keyed by ``"<name> (<CANON_TEAM>)"``
end to end (see ``matchup.resolve_collisions`` / ``matchup.sim_name`` and
``stage_d.build_pool``), so each real player carries its own projection and a
distinct pool row — the field can no longer roster one wearing the other's
numbers. This module handles the UPLOAD side of the same clash: the pool row
arrives with that disambiguated name, so :func:`_records` strips the ``(TEAM)``
suffix back to the plain name the id map is keyed by, and :func:`lookup` resolves
the correct id from the suffix's team plus salary/position (a plain
``name -> id`` map would collapse the two ids, last write wins, and ship the same
— e.g. cheaper — Muncy every time).

Keying by team alone is not enough: DraftKings' team codes don't always match the
projection feed's (DK "OAK"/"LAD" vs a feed's "ATH"/"LA"), and if the id source
lists only one of the two, a team match silently returns the wrong id. So we keep
EVERY id record per name and resolve a lineup player against them using the most
reliable signal first:

    salary (exact int, identical across any DK-derived file)
      > team (normalized)
      > position (the slot the player fills)

Salary is the clincher — two same-named players almost always differ in salary,
and DK salaries are the same integer in the salary file and the upload template.

Map shape: ``{ norm(name): [ {"id","team","pos":set(),"salary":int|None}, ... ] }``.
"""
import re

from stage_d import norm

# A colliding player is carried through the build as ``"<name> (<CANON_TEAM>)"``
# (see matchup.sim_name / stage_d.build_pool). Strip that suffix before keying
# into the id map — which is keyed by the plain name — and reuse the suffix's
# team to disambiguate the two ids when the caller didn't pass one.
_DISAMBIG_RE = re.compile(r"\s*\(([A-Z]{2,4})\)\s*$")


def _base_name(name):
    return _DISAMBIG_RE.sub("", str(name or ""))


def _suffix_team(name):
    m = _DISAMBIG_RE.search(str(name or ""))
    return m.group(1) if m else ""


def team_key(t):
    """Normalize a team/position code for comparison (case/space-insensitive)."""
    return str(t or "").strip().upper()


def _pos_set(pos):
    """DK positions can be multi ('3B/SS'); return the set of normalized slots."""
    return {team_key(p) for p in str(pos or "").replace(",", "/").split("/") if p.strip()}


def add_id(idmap, name, team, cid, pos="", salary=None):
    """Record one DK id for a player. Multiple players with the same name are all
    kept (as separate records); a repeated id for the same name is ignored."""
    if not cid:
        return
    cid = str(cid).strip()
    try:
        sal = int(float(salary)) if salary not in (None, "") else None
    except (TypeError, ValueError):
        sal = None
    recs = idmap.setdefault(norm(name), [])
    for r in recs:
        if r["id"] == cid:                 # same id already present: enrich, don't dup
            if sal is not None and r.get("salary") is None:
                r["salary"] = sal
            r["pos"] |= _pos_set(pos)
            if team and not r.get("team"):
                r["team"] = team_key(team)
            return
    recs.append({"id": cid, "team": team_key(team),
                 "pos": _pos_set(pos), "salary": sal})


def _records(idmap, name):
    """Normalized list of id records for a name, tolerating legacy map shapes
    (flat ``name->id`` and the earlier ``name->{team:id}``)."""
    v = idmap.get(norm(_base_name(name)))
    if not v:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):                                  # legacy flat map
        return [{"id": v, "team": "", "pos": set(), "salary": None}]
    if isinstance(v, dict):                                 # legacy name->{team:id}
        return [{"id": i, "team": team_key(t), "pos": set(), "salary": None}
                for t, i in v.items()]
    return []


def count_ids(idmap):
    """Total distinct player ids in the map (both Muncys count, not one name)."""
    return sum(len({r["id"] for r in _records(idmap, k)}) for k in idmap)


def has_name(idmap, name):
    """True if any id exists for this name (used for export eligibility)."""
    return bool(_records(idmap, name))


def lookup(idmap, name, team="", pos="", salary=None):
    """Resolve the DK id for a lineup player.

    Scores every id record for the name by how well it matches the player's
    salary (weight 100), team (10) and position (1), and returns the best. With a
    unique name any record wins; with a clash the exact salary decides. Returns
    ``None`` only when the name has no ids at all.
    """
    recs = _records(idmap, name)
    if not recs:
        return None
    if len(recs) == 1:
        return recs[0]["id"]

    try:
        sal = int(float(salary)) if salary not in (None, "") else None
    except (TypeError, ValueError):
        sal = None
    tk = team_key(team or _suffix_team(name))   # fall back to the "(TEAM)" suffix
    pk = team_key(pos)

    best, best_score = None, -1
    for r in recs:
        s = 0
        if sal is not None and r.get("salary") is not None and sal == r["salary"]:
            s += 100
        if tk and r.get("team") and tk == r["team"]:
            s += 10
        if pk and r.get("pos") and pk in r["pos"]:
            s += 1
        if s > best_score:
            best, best_score = r, s
    return best["id"] if best else None


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    m = {}
    # the real collision: two Max Muncys, close salaries, different pos/team
    add_id(m, "Max Muncy", "LA", "3548id", pos="3B", salary=3548)
    add_id(m, "Max Muncy", "ATH", "3558id", pos="SS", salary=3558)
    add_id(m, "Shohei Ohtani", "LA", "999", pos="OF", salary=6000)

    # exact salary resolves regardless of team-code convention
    assert lookup(m, "Max Muncy", salary=3548) == "3548id"
    assert lookup(m, "Max Muncy", salary=3558) == "3558id"
    # team-code MISMATCH across files (DK 'LAD'/'OAK' vs feed 'LA'/'ATH'):
    # salary still nails it
    assert lookup(m, "Max Muncy", team="LAD", salary=3548) == "3548id"
    assert lookup(m, "Max Muncy", team="OAK", salary=3558) == "3558id"
    # no salary -> fall back to team, then position
    assert lookup(m, "Max Muncy", team="LA") == "3548id"
    assert lookup(m, "Max Muncy", pos="SS") == "3558id"
    # unique name resolves with nothing
    assert lookup(m, "Shohei Ohtani") == "999"
    # membership + count
    assert has_name(m, "max muncy") and not has_name(m, "nobody")
    assert count_ids(m) == 3, count_ids(m)

    # legacy shapes tolerated
    assert lookup({"max muncy": "flat"}, "Max Muncy", salary=1) == "flat"
    assert lookup({"max muncy": {"LA": "a", "ATH": "b"}},
                  "Max Muncy", team="ATH") == "b"

    print("dk_ids.py self-test passed:", {k: len(v) for k, v in m.items()})

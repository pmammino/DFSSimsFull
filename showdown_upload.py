#!/usr/bin/env python3
"""
showdown_upload.py
==================
Turn selected DraftKings **Showdown** lineups into an uploadable CSV.

Showdown differs from classic at upload time: every player has TWO DraftKings
draftable ids on a slate — one for the **CPT** (captain) roster slot and one for
the **UTIL** slot — and the captain must be entered under its CPT id. The
RotoWire salaries feed carries only the flex/UTIL id, so a valid upload needs a
DraftKings ``DKSalaries.csv`` (or the wide contest-entry template), which lists
both roster positions with their distinct ids. This module parses that file and
emits the upload CSV.

The parser is column-NAME driven (not fixed offsets), so it handles both the
9-column salaries download (Position, Name+ID, Name, ID, Roster Position, Salary,
Game Info, TeamAbbrev, AvgPointsPerGame) and the wider entry template whose
player table starts partway across the row. A player is keyed by
(normalized name, team) with a name-only fallback, so two same-named players on
the two teams keep distinct ids when the team is known.
"""
import csv
import io

from stage_d import norm

CPT_HDR = ['CPT', 'UTIL', 'UTIL', 'UTIL', 'UTIL', 'UTIL']


def _roster_pos(val):
    """Normalize a roster-position cell to 'CPT' or 'UTIL' (None if neither)."""
    u = str(val or '').upper()
    if 'CPT' in u or 'CAPT' in u:
        return 'CPT'
    if 'UTIL' in u or 'FLEX' in u:
        return 'UTIL'
    return None


def parse_showdown_template(text):
    """Parse a DK showdown salaries/template CSV into
    ``{(norm(name), TEAM): {'CPT': id, 'UTIL': id}}`` (plus a (norm(name), '')
    name-only fallback). Returns None if no usable player table is found."""
    rows = list(csv.reader(io.StringIO(text)))
    hdr, col = None, {}
    for i, r in enumerate(rows):
        low = [c.strip().lower() for c in r]
        if 'name' not in low or 'id' not in low:
            continue

        def find(cands, low=low):
            for c in cands:
                if c in low:
                    return low.index(c)
            return None

        rp = find(['roster position'])
        pos = find(['position'])
        if rp is None and pos is None:
            continue
        col = {'name': find(['name']), 'id': find(['id']),
               'rp': rp if rp is not None else pos,
               'salary': find(['salary']),
               'team': find(['teamabbrev', 'team'])}
        hdr = i
        break
    if hdr is None or col['name'] is None or col['id'] is None or col['rp'] is None:
        return None

    tmap = {}

    def put(key, rp, cid):
        tmap.setdefault(key, {}).setdefault(rp, cid)

    need = max(col['name'], col['id'], col['rp'])
    for r in rows[hdr + 1:]:
        if len(r) <= need:
            continue
        nm = r[col['name']].strip()
        cid = r[col['id']].strip()
        rp = _roster_pos(r[col['rp']])
        if not nm or not cid or rp is None:
            continue
        team = ''
        if col['team'] is not None and len(r) > col['team']:
            team = r[col['team']].strip().upper()
        put((norm(nm), team), rp, cid)
        put((norm(nm), ''), rp, cid)          # name-only fallback
    return tmap or None


def lookup_id(tmap, name, team, roster_pos):
    """CPT/UTIL id for a player, trying (name, team) then name-only."""
    for key in ((norm(name), str(team or '').upper()), (norm(name), '')):
        d = tmap.get(key)
        if d and roster_pos in d:
            return d[roster_pos]
    return None


def eligible_names(tmap, names):
    """Name-only export eligibility for the portfolio selector: names[0] is the
    captain (needs a CPT id), the rest need a UTIL id."""
    if not names:
        return False
    if lookup_id(tmap, names[0], '', 'CPT') is None:
        return False
    return all(lookup_id(tmap, n, '', 'UTIL') is not None for n in names[1:])


def lineup_ids(tmap, players):
    """Resolve ids for a lineup's player OBJECTS (index 0 = captain -> CPT id,
    the rest -> UTIL id). Returns the 6 ids in slot order, or None if any is
    missing."""
    ids = []
    for i, pl in enumerate(players):
        rp = 'CPT' if i == 0 else 'UTIL'
        cid = lookup_id(tmap, pl.Name, getattr(pl, 'Team', ''), rp)
        if cid is None:
            return None
        ids.append(cid)
    return ids


def upload_csv(chosen_rows, tmap, cands):
    """Emit the DK showdown upload CSV (header ``CPT,UTIL,UTIL,UTIL,UTIL,UTIL``)
    for the chosen result rows, mapping each lineup's players to CPT/UTIL ids via
    its Candidate index into `cands`. Returns (csv_text, info)."""
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(CPT_HDR)
    written, skipped = 0, 0
    for row in chosen_rows:
        try:
            players = cands[int(row['Candidate']) - 1]['players']
        except (KeyError, IndexError, TypeError, ValueError):
            players = None
        ids = lineup_ids(tmap, players) if players is not None else None
        if ids is None:
            skipped += 1
            continue
        w.writerow(ids)
        written += 1
    return out.getvalue(), {'chosen': written, 'skipped_unmapped': skipped}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    # 9-column DKSalaries download: each player twice (CPT then UTIL)
    dl = (
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
        "CPT,Zack Wheeler (111),Zack Wheeler,111,CPT,15900,NYM@PHI,PHI,20\n"
        "UTIL,Zack Wheeler (112),Zack Wheeler,112,UTIL,10600,NYM@PHI,PHI,20\n"
        "CPT,Francisco Alvarez (211),Francisco Alvarez,211,CPT,9600,NYM@PHI,NYM,12\n"
        "UTIL,Francisco Alvarez (212),Francisco Alvarez,212,UTIL,6400,NYM@PHI,NYM,12\n"
    )
    tmap = parse_showdown_template(dl)
    assert tmap is not None
    assert lookup_id(tmap, 'Zack Wheeler', 'PHI', 'CPT') == '111'
    assert lookup_id(tmap, 'Zack Wheeler', 'PHI', 'UTIL') == '112'
    assert lookup_id(tmap, 'Francisco Alvarez', 'NYM', 'CPT') == '211'
    assert lookup_id(tmap, 'Francisco Alvarez', '', 'UTIL') == '212'   # name-only

    # wide entry-template layout: player table offset by entry columns
    wide_hdr = "Entry ID,Contest Name,Contest ID,Entry Fee,,,,,,,,Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame"
    wide = (wide_hdr + "\n"
            + ",,,,,,,,,,,CPT,Zack Wheeler (111),Zack Wheeler,111,CPT,15900,NYM@PHI,PHI,20\n"
            + ",,,,,,,,,,,UTIL,Zack Wheeler (112),Zack Wheeler,112,UTIL,10600,NYM@PHI,PHI,20\n")
    tw = parse_showdown_template(wide)
    assert tw and lookup_id(tw, 'Zack Wheeler', 'PHI', 'CPT') == '111', tw

    # eligibility + upload
    class P:
        def __init__(s, n, t):
            s.Name = n; s.Team = t
    cands = [{'players': [P('Zack Wheeler', 'PHI'), P('Francisco Alvarez', 'NYM'),
                          P('Zack Wheeler', 'PHI'), P('Francisco Alvarez', 'NYM'),
                          P('Francisco Alvarez', 'NYM'), P('Zack Wheeler', 'PHI')]}]
    assert eligible_names(tmap, ['Zack Wheeler', 'Francisco Alvarez'])
    assert not eligible_names(tmap, ['Unknown Guy', 'Zack Wheeler'])
    csv_text, info = upload_csv([{'Candidate': 1}], tmap, cands)
    assert info['chosen'] == 1, info
    body = csv_text.strip().splitlines()
    assert body[0] == 'CPT,UTIL,UTIL,UTIL,UTIL,UTIL'
    assert body[1].split(',')[0] == '111'      # captain uses the CPT id
    assert body[1].split(',')[1] == '212'      # util uses the UTIL id
    print("showdown_upload.py self-test passed")

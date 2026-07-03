"""
dk_slate_feed.py — build a pickable DraftKings slate catalog from the RotoWire
feeds, so the app no longer requires a manual slate-file upload.

Two feeds (same proxy host the lineup ingest already uses):

  * salaries-dk : every DK player for the day, each tagged with its SlateID,
    GameType (Classic/Showdown), salary, position(s), team, DraftKingsDraftableID
    (the slate-specific contest ID the upload export needs) and RotoID.
  * MLBOwnership: projected ownership per slate, players keyed by RotoWire
    player Id (== salaries-dk RotoID). Carries no team/salary, so it is joined
    onto the salaries feed.

build_catalog() returns a plain (JSON-serialisable) dict of Classic slates with
their players + ownership already joined, ready for a dropdown. to_dk_df() turns
a chosen slate into the (dk_df, id_map) pair the rest of the app expects —
identical in shape to what the CSV upload path produces:

    dk_df : FullName, Team, Position, Salary, Ownership, PlayerContestID
    id_map: norm(name) -> {TEAM -> DraftKingsDraftableID} (see dk_ids.py)

Slates are matched between the two feeds on SlateID == SlateId; players within a
slate are matched on RotoID, falling back to normalized name when an id is
missing. Only Classic slates are surfaced (the sim pipeline doesn't build
Showdown).
"""
import re
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

from stage_d import norm
import dk_ids

FEED_SALARIES  = ("https://rotowire-secrets-ebgmaeh8ecc4huhf.canadaeast-01."
                  "azurewebsites.net/api/proxy?feed=salaries-dk")
FEED_OWNERSHIP = ("https://rotowire-secrets-ebgmaeh8ecc4huhf.canadaeast-01."
                  "azurewebsites.net/api/proxy?feed=MLBOwnership")


def _http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (pipeline)'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def _root(xml_text, tag):
    """Parse xml_text, tolerating the proxy occasionally concatenating docs by
    extracting the first <tag>…</tag> block."""
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        m = re.search(rf'<{tag}\b.*?</{tag}>', xml_text, re.DOTALL)
        if not m:
            raise
        return ET.fromstring(m.group(0))


def _txt(el, tag, default=''):
    v = el.findtext(tag)
    return (v or default).strip()


def parse_salaries(xml_text):
    """Return (date, {slate_id: slate}). Classic slates only.
    slate = {slate_id, game_type, slate_type, start, end, games(set),
             players:[{name, team, position, salary, draftable_id, roto_id}]}."""
    root = _root(xml_text, 'Salaries')
    date = (root.get('Date') or '').strip()
    slates = {}
    for p in root.findall('.//Player'):
        if _txt(p, 'GameType').lower() != 'classic':
            continue
        sid = _txt(p, 'SlateID')
        if not sid:
            continue
        s = slates.setdefault(sid, {
            'slate_id': sid, 'game_type': _txt(p, 'GameType'),
            'slate_type': _txt(p, 'SlateType'),
            'start': _txt(p, 'SlateStart'), 'end': _txt(p, 'SlateEnd'),
            'games': set(), 'players': []})
        name = f"{_txt(p, 'FirstName')} {_txt(p, 'LastName')}".strip()
        pos, pos2 = _txt(p, 'Position'), _txt(p, 'Position2')
        position = f"{pos}/{pos2}" if pos2 else pos
        ga = _txt(p, 'GameAbbreviation')
        if ga:
            s['games'].add(ga)
        try:
            sal = int(float(_txt(p, 'Salary') or 0))
        except ValueError:
            sal = 0
        s['players'].append({
            'name': name, 'team': _txt(p, 'Team'), 'position': position,
            'salary': sal, 'draftable_id': _txt(p, 'DraftKingsDraftableID'),
            'roto_id': _txt(p, 'RotoID')})
    return date, slates


def parse_ownership(xml_text):
    """Return (date, {slate_id: {game_type, slate_type, start, end,
    own_by_roto, own_by_name}}). Ownership is the projected draft % (0–100)."""
    root = _root(xml_text, 'Ownership')
    date = _txt(root, 'Date')
    slates = {}
    for sl in root.findall('.//Slate'):
        sid = _txt(sl, 'SlateId')
        if not sid:
            continue
        own_by_roto, own_by_name = {}, {}
        for p in sl.findall('Players/Player'):
            try:
                ov = float(_txt(p, 'Ownership') or 0)
            except ValueError:
                ov = 0.0
            rid = (p.get('Id') or '').strip()
            nm = norm(f"{_txt(p, 'FirstName')} {_txt(p, 'LastName')}")
            if rid:
                own_by_roto[rid] = ov
            if nm:
                own_by_name[nm] = ov
        slates[sid] = {
            'slate_id': sid, 'game_type': _txt(sl, 'GameType'),
            'slate_type': _txt(sl, 'SlateType'),
            'start': _txt(sl, 'SlateStart'), 'end': _txt(sl, 'SlateEnd'),
            'own_by_roto': own_by_roto, 'own_by_name': own_by_name}
    return date, slates


def _fmt_time(start):
    """'06/18/2026 6:40 PM' or ISO '2026-06-18T13:35:00-07:00' -> a short time."""
    if not start:
        return ''
    if 'T' in start:                      # ISO from the ownership feed
        t = start.split('T', 1)[1]
        return t[:5] if len(t) >= 5 else t
    parts = start.split(' ', 1)           # '06/18/2026 6:40 PM'
    return parts[1] if len(parts) > 1 else start


def _label(slate, n_games, n_players):
    gt = (slate['game_type'] or 'Classic').replace('_', ' ').title()
    t = _fmt_time(slate['start'])
    bits = [gt, f"{n_games} games"]
    if t:
        bits.append(t)
    bits.append(f"{n_players} players")
    return " · ".join(bits)


def build_catalog(salaries_xml=None, ownership_xml=None):
    """Fetch + join both feeds into a catalog of Classic slates that have both
    salary and ownership data. Returns {'date', 'slates':[slate, …]} where each
    slate is JSON-serialisable and carries its joined players."""
    salaries_xml = salaries_xml or _http_get(FEED_SALARIES)
    ownership_xml = ownership_xml or _http_get(FEED_OWNERSHIP)
    date, sal_slates = parse_salaries(salaries_xml)
    _, own_slates = parse_ownership(ownership_xml)

    out = []
    for sid, s in sal_slates.items():
        own = own_slates.get(sid)
        if not own:                       # this flow needs ownership
            continue
        obr, obn = own['own_by_roto'], own['own_by_name']
        players, n_owned = [], 0
        for pl in s['players']:
            ov = obr.get(pl['roto_id'])
            if ov is None:
                ov = obn.get(norm(pl['name']))
            if ov is None:
                ov = 0.0
            else:
                n_owned += 1
            players.append({**pl, 'ownership': float(ov)})
        n_games = len(s['games'])
        # the ownership feed's GameType (EARLY/AFTERNOON/…) is the descriptive
        # time-window label; the salaries feed's is just "Classic".
        meta = dict(s)
        meta['game_type'] = own.get('game_type') or s['game_type']
        out.append({
            'slate_id': sid, 'date': date,
            'game_type': meta['game_type'],
            'slate_type': own.get('slate_type') or s['slate_type'],
            'start': s['start'], 'end': s['end'],
            'n_games': n_games, 'n_players': len(players), 'n_owned': n_owned,
            'games': sorted(s['games']),
            'label': _label(meta, n_games, len(players)),
            'players': players})
    out.sort(key=lambda x: (x['start'], -x['n_games']))
    return {'date': date, 'slates': out}


def to_dk_df(slate):
    """Turn a catalog slate into (dk_df, id_map), matching the CSV upload path.
    dk_df: FullName, Team, Position, Salary, Ownership, PlayerContestID."""
    recs, id_map = [], {}
    for pl in slate['players']:
        recs.append({
            'FullName': pl['name'], 'Team': pl['team'],
            'Position': pl['position'], 'Salary': pl['salary'],
            'Ownership': pl['ownership'], 'PlayerContestID': pl['draftable_id']})
        if pl['draftable_id']:
            # key by team/pos/salary so two same-named players (e.g. Max Muncy on
            # two teams) keep distinct upload ids
            dk_ids.add_id(id_map, pl['name'], pl['team'], pl['draftable_id'],
                          pos=pl['position'], salary=pl['salary'])
    return pd.DataFrame(recs), id_map


if __name__ == '__main__':
    cat = build_catalog()
    print(f"Slate catalog {cat['date']}: {len(cat['slates'])} Classic slate(s)")
    for s in cat['slates']:
        print(f"  [{s['slate_id']}] {s['label']} "
              f"({s['n_owned']}/{s['n_players']} with ownership)")

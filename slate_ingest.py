"""
ingest.py — fetch and normalize the slate.

Sources, in priority order:
  1. CONFIRMED feed (feed=lineups): authoritative for pitchers
     (StartingPitcher / OpenerPitcher / PrimaryPitcher) and for any batting
     order that has already been confirmed.
  2. EXPECTED feed (feed=explineups): fallback batting orders (LineupStatus
     X = expected, C = confirmed) for teams whose order isn't in the
     confirmed feed yet.
  3. FantasyLabs Vegas feed: HomeVegasRuns / VisitorVegasRuns implied totals.

Produces a single normalized slate dict and writes it to data/slate.json.

Network note: this environment can fetch the rotowire proxy and statsapi from
bash, but the FantasyLabs host may be blocked. fetch_vegas() therefore accepts
an optional pre-fetched JSON payload (vegas_json) so the caller can paste the
feed when direct fetch fails. If neither works, implied totals fall back to
DEFAULT_TEAM_RUNS and the pipeline still runs.
"""
import json, os, re, sys
import xml.etree.ElementTree as ET
import urllib.request

import slate_config as C


def _http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (pipeline)'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def _pname(el):
    if el is None:
        return None
    fn, ln = el.findtext('FirstName'), el.findtext('LastName')
    if fn is None and ln is None:
        return None
    nm = f"{fn or ''} {ln or ''}".strip()
    return nm or None


def parse_confirmed(xml_text):
    """Return {gid: {date, datetime, away, home, away_pitchers, home_pitchers, lineups}}.
    *_pitchers = {'starter':name,'opener':name|None,'primary':name|None}
    lineups[side] = [ {name, slot, pos, pid} ] (may be empty if not yet posted)
    """
    # The proxy sometimes concatenates docs; take the <Lineups> section.
    m = re.search(r'<Lineups>.*?</Lineups>', xml_text, re.DOTALL)
    root = ET.fromstring(m.group(0) if m else xml_text)
    date = root.findtext('Date')
    games = {}
    for g in root.findall('.//Game'):
        gid = g.get('Id')
        rec = {'date': date, 'datetime': g.findtext('DateTime', ''),
               'pitchers': {}, 'lineups': {'away': [], 'home': []}}
        for t in g.findall('Teams/Team'):
            side = 'home' if t.get('IsHome') == '1' else 'away'
            rec[side] = t.get('Code')
            rec['pitchers'][side] = {
                'starter': _pname(t.find('StartingPitcher')),
                'opener':  _pname(t.find('OpenerPitcher')),
                'primary': _pname(t.find('PrimaryPitcher')),
            }
            order = []
            for p in t.findall('Players/Player'):
                nm = f"{p.findtext('FirstName','')} {p.findtext('LastName','')}".strip()
                order.append({'name': nm, 'slot': int(p.findtext('BattingSpot', 0)),
                              'pos': p.findtext('Position', ''), 'pid': p.get('Id')})
            order.sort(key=lambda x: x['slot'])
            rec['lineups'][side] = order
        games[gid] = rec
    return games


def parse_expected(xml_text):
    """Return {gid: {away, home, status:{side:X/C}, lineups:{side:[...]}}}.
    Used only as a fallback for batting orders."""
    m = re.search(r'<ExpectedLineups>.*?</ExpectedLineups>', xml_text, re.DOTALL)
    root = ET.fromstring(m.group(0) if m else xml_text)
    games = {}
    for g in root.findall('.//Game'):
        gid = g.get('Id')
        rec = {'status': {}, 'lineups': {'away': [], 'home': []}}
        for t in g.findall('Teams/Team'):
            side = 'home' if t.get('IsHome') == '1' else 'away'
            rec[side] = t.get('Code')
            rec['status'][side] = t.get('LineupStatus', '')
            order = []
            for p in t.findall('Players/Player'):
                nm = f"{p.findtext('FirstName','')} {p.findtext('LastName','')}".strip()
                order.append({'name': nm, 'slot': int(p.findtext('BattingSpot', 0)),
                              'pos': p.findtext('Position', ''), 'pid': p.get('Id')})
            order.sort(key=lambda x: x['slot'])
            rec['lineups'][side] = order
        games[gid] = rec
    return games


def fetch_vegas(date, vegas_json=None):
    """Return {gid_or_matchup: {away,home,away_runs,home_runs,total}}.
    `date` is YYYY-MM-DD. If vegas_json (raw text or dict) is provided it is
    parsed directly; otherwise we try the live feed and fall back to {} on any
    failure."""
    payload = None
    if vegas_json is not None:
        payload = vegas_json if isinstance(vegas_json, (dict, list)) else json.loads(vegas_json)
    else:
        try:
            payload = json.loads(_http_get(C.FEED_VEGAS_TMPL.format(date=date)))
        except Exception as e:
            print(f"  [vegas] live fetch failed ({e}); implied totals will fall back to default", file=sys.stderr)
            return {}

    # FantasyLabs returns a list of sportevent dicts. Field names observed:
    # HomeTeamShort / VisitorTeamShort / HomeVegasRuns / VisitorVegasRuns / OverUnder
    out = {}
    rows = payload if isinstance(payload, list) else payload.get('Events', payload.get('data', []))
    for ev in rows:
        home = ev.get('HomeTeamShort') or ev.get('HomeTeam') or ev.get('HomeTeamAbbrev')
        away = ev.get('VisitorTeamShort') or ev.get('VisitorTeam') or ev.get('VisitorTeamAbbrev')
        hr = ev.get('HomeVegasRuns'); ar = ev.get('VisitorVegasRuns')
        ou = ev.get('OverUnder') or ev.get('Total')
        if home and away and hr is not None and ar is not None:
            key = f"{away}@{home}"
            out[key] = {'away': away, 'home': home,
                        'away_runs': float(ar), 'home_runs': float(hr),
                        'total': float(ou) if ou is not None else float(ar) + float(hr)}
    return out


def build_slate(confirmed_xml=None, expected_xml=None, vegas_json=None, write=True):
    """Merge everything into one normalized slate.

    Returns slate = {
        'date': 'YYYY-MM-DD',
        'games': { gid: {
            'away','home','datetime',
            'pitchers': {'away':{starter,opener,primary}, 'home':{...}},
            'lineups':  {'away':[...], 'home':[...]},
            'lineup_source': {'away':'confirmed|expected|none', 'home':...},
            'implied': {'away':float,'home':float,'total':float},
        }}
    }
    """
    confirmed_xml = confirmed_xml or _http_get(C.FEED_CONFIRMED)
    expected_xml  = expected_xml  or _http_get(C.FEED_EXPECTED)

    conf = parse_confirmed(confirmed_xml)
    exp  = parse_expected(expected_xml)
    date = next(iter(conf.values()))['date'] if conf else None

    # Build an index of expected lineups by (team_code) for fallback matching,
    # since expected gids won't match confirmed gids on different dates.
    exp_by_team = {}
    for gid, rec in exp.items():
        for side in ('away', 'home'):
            exp_by_team[rec[side]] = rec['lineups'][side]

    vegas = fetch_vegas(date, vegas_json=vegas_json) if date else {}

    games = {}
    for gid, rec in conf.items():
        away, home = rec['away'], rec['home']
        out = {'away': away, 'home': home, 'datetime': rec['datetime'],
               'pitchers': rec['pitchers'], 'lineups': {}, 'lineup_source': {}, 'implied': {}}

        for side in ('away', 'home'):
            order = rec['lineups'][side]
            if order:
                out['lineups'][side] = order
                out['lineup_source'][side] = 'confirmed'
            elif exp_by_team.get(rec[side]):
                out['lineups'][side] = exp_by_team[rec[side]]
                out['lineup_source'][side] = 'expected'
            else:
                out['lineups'][side] = []
                out['lineup_source'][side] = 'none'

        # implied totals: match on standard codes
        astd, hstd = C.std_code(away), C.std_code(home)
        key = f"{astd}@{hstd}"
        v = vegas.get(key)
        if v:
            out['implied'] = {'away': v['away_runs'], 'home': v['home_runs'], 'total': v['total']}
        else:
            out['implied'] = {'away': C.DEFAULT_TEAM_RUNS, 'home': C.DEFAULT_TEAM_RUNS,
                              'total': 2 * C.DEFAULT_TEAM_RUNS}
        games[gid] = out

    slate = {'date': date, 'games': games}
    if write:
        with open(os.path.join(C.DATA_DIR, 'slate.json'), 'w') as f:
            json.dump(slate, f, indent=2)
    return slate


if __name__ == '__main__':
    s = build_slate()
    n_conf = sum(1 for g in s['games'].values() for side in ('away', 'home')
                 if g['lineup_source'][side] == 'confirmed')
    n_exp = sum(1 for g in s['games'].values() for side in ('away', 'home')
                if g['lineup_source'][side] == 'expected')
    print(f"Slate {s['date']}: {len(s['games'])} games | "
          f"{n_conf} confirmed lineups, {n_exp} expected fallbacks")
    for gid, g in s['games'].items():
        ps = g['pitchers']
        def fmt(side):
            p = ps[side]
            return p['opener'] and f"{p['opener']}(O)->{p['primary']}(P)" or p['starter']
        print(f"  {C.std_code(g['away'])}@{C.std_code(g['home'])}: "
              f"{fmt('away')} vs {fmt('home')} | "
              f"impl {g['implied']['away']}/{g['implied']['home']} "
              f"[{g['lineup_source']['away']}/{g['lineup_source']['home']}]")

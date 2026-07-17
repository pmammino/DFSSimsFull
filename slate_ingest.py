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
import unicodedata
import xml.etree.ElementTree as ET
import urllib.request

import slate_config as C


def _norm_name(n):
    """Normalize a player name for matching. Mirrors stage_d.norm but without the
    numpy-heavy import, so the slate ingest stays dependency-light."""
    n = unicodedata.normalize('NFKD', str(n)).encode('ascii', 'ignore').decode()
    n = n.lower().replace('.', '').replace(',', '').replace("'", "")
    for s in (' jr', ' sr', ' ii', ' iii', ' iv'):
        if n.endswith(s):
            n = n[:-len(s)]
    return n.strip()


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

    # FantasyLabs returns a list of sportevent dicts. The per-team implied total
    # is ProjHomeScore/ProjVisitorScore in the current feed, or HomeVegasRuns/
    # VisitorVegasRuns in older exports; fall back to deriving from OU + spread.
    # Returns {canonical_team_code: implied_runs} so matching is per-team and
    # robust to matchup ordering / abbreviation variants.
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    out = {}
    rows = payload if isinstance(payload, list) else payload.get('Events', payload.get('data', payload.get('events', [])))
    for ev in rows:
        if not isinstance(ev, dict):
            continue
        h = C.canonical_team(ev.get('HomeTeam') or ev.get('HomeTeamShort')
                             or ev.get('HomeTeamAbbrev'))
        a = C.canonical_team(ev.get('VisitorTeam') or ev.get('VisitorTeamShort')
                             or ev.get('VisitorTeamAbbrev'))
        h_imp = _num(ev.get('ProjHomeScore'))
        if h_imp is None:
            h_imp = _num(ev.get('HomeVegasRuns'))
        a_imp = _num(ev.get('ProjVisitorScore'))
        if a_imp is None:
            a_imp = _num(ev.get('VisitorVegasRuns'))
        if h_imp is None or a_imp is None:   # derive from OU + home spread
            ou = _num(ev.get('OU') or ev.get('OverUnder') or ev.get('Total'))
            sp = ev.get('SpreadHome')
            sp = _num(sp if sp is not None else ev.get('Spread'))
            if ou is not None and sp is not None:
                h_imp = ou / 2.0 - sp / 2.0
                a_imp = ou / 2.0 + sp / 2.0
        if h and h_imp is not None:
            out[h] = round(h_imp, 2)
        if a and a_imp is not None:
            out[a] = round(a_imp, 2)
    return out


def _with_date(url, date):
    """Append a &date=YYYY-MM-DD parameter to a feed URL (for historical slates).
    The rotowire proxy returns the slate for that date instead of today's."""
    if not date:
        return url
    sep = '&' if '?' in url else '?'
    return f"{url}{sep}date={date}"


def _canon_matchup(rec):
    """Order-independent key identifying the two teams playing in a game."""
    return frozenset({C.canonical_team(rec.get('away')),
                      C.canonical_team(rec.get('home'))})


def filter_slate_doubleheaders(games, slate_players):
    """Drop off-slate games of a double-header.

    On a double-header day the confirmed-lineup feed returns BOTH games of a
    matchup (same two teams, two game ids), but usually only one of them is on
    the DFS slate. Keeping both makes the pipeline treat the OTHER game's
    (confirmed) pitcher as if it applied to the slate, and lets the wrong
    matchup drive the sims.

    The two games of a double-header share the same two lineups, so hitters
    can't tell them apart — only the PITCHERS differ. `slate_players` is the set
    of players actually on the slate (the DK slate / uploaded file, which lists
    each game's probable pitcher by name), so the on-slate game is the one whose
    starter/opener/primary is in that set. When no game's pitcher matches (e.g.
    neither is confirmed yet and the names don't line up), the earliest game by
    start time is kept as a best guess.

    `games` is {gid: rec}; `slate_players` an iterable of raw player names.
    Returns a new {gid: rec} with the off-slate double-header games removed.
    Matchups that appear only once are always kept unchanged, so ordinary
    (non-double-header) slates are unaffected; a falsy `slate_players` returns
    the games untouched.
    """
    if not slate_players:
        return games
    _norm = _norm_name
    sp = {_norm(n) for n in slate_players if n}

    groups = {}
    for gid, rec in games.items():
        groups.setdefault(_canon_matchup(rec), []).append(gid)

    def _pitcher_hits(gid):
        rec = games[gid]
        hits = 0
        for side in ('away', 'home'):
            roles = rec.get('pitchers', {}).get(side) or {}
            for role in ('starter', 'opener', 'primary'):
                nm = roles.get(role)
                if nm and _norm(nm) in sp:
                    hits += 1
        return hits

    keep = set()
    for gids in groups.values():
        if len(gids) == 1:                       # not a double-header
            keep.update(gids)
            continue
        scored = {gid: _pitcher_hits(gid) for gid in gids}
        best = max(scored.values())
        if best > 0:                             # keep the game whose SP is on the slate
            keep.update(gid for gid, s in scored.items() if s == best)
        else:                                    # can't disambiguate — keep the earliest
            keep.add(min(gids, key=lambda g: (games[g].get('datetime') or '')))

    dropped = [gid for gid in games if gid not in keep]
    if dropped:
        print(f"  [slate] double-header: kept {len(keep)} game(s), dropped "
              f"{len(dropped)} off-slate game(s) {dropped}", file=sys.stderr)
    return {gid: rec for gid, rec in games.items() if gid in keep}


def build_slate(confirmed_xml=None, expected_xml=None, vegas_json=None, write=True,
                date=None, slate_players=None):
    """Merge everything into one normalized slate.

    `date` (YYYY-MM-DD, optional): rebuild a *historical* slate. When given and
    the lineup feeds aren't passed in directly, the confirmed/expected feeds are
    fetched with a &date= parameter and Vegas totals are fetched for that date,
    so the pipeline reproduces a past day's slate (lineups + matchups + Vegas).

    `slate_players` (optional): raw player names that are on the DFS slate (the
    DK slate / uploaded file). Used to drop the off-slate game of a
    double-header so its pitcher isn't treated as a slate starter — see
    filter_slate_doubleheaders(). Ignored when the day has no double-header.

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
    confirmed_xml = confirmed_xml or _http_get(_with_date(C.FEED_CONFIRMED, date))
    expected_xml  = expected_xml  or _http_get(_with_date(C.FEED_EXPECTED, date))

    conf = parse_confirmed(confirmed_xml)
    exp  = parse_expected(expected_xml)
    # Prefer the feed's own date; fall back to the requested historical date.
    date = (next(iter(conf.values()))['date'] if conf else None) or date

    # Index expected lineups two ways for the batting-order fallback:
    #  - by game id (then team code): the authoritative match on a normal
    #    same-day slate. This is essential on double-header days, where a team
    #    plays the same opponent twice: a team-code-only lookup collides between
    #    the two games and can pull the OFF-slate game's expected lineup into the
    #    on-slate game, injecting hitters who aren't in the slate game at all.
    #  - by team code alone: fallback for historical rebuilds where the expected
    #    feed's gids may not line up with the confirmed feed's.
    exp_by_gid = {}
    exp_by_team = {}
    for gid, rec in exp.items():
        exp_by_gid[gid] = {rec['away']: rec['lineups']['away'],
                           rec['home']: rec['lineups']['home']}
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
                continue
            # Prefer the expected lineup for THIS exact game (gid + team code),
            # so a double-header can't cross-contaminate the two games' hitters;
            # fall back to the team-code index for historical rebuilds where
            # gids may not match.
            exp_order = exp_by_gid.get(gid, {}).get(rec[side]) \
                or exp_by_team.get(rec[side])
            if exp_order:
                out['lineups'][side] = exp_order
                out['lineup_source'][side] = 'expected'
            else:
                out['lineups'][side] = []
                out['lineup_source'][side] = 'none'

        # implied totals: match per-team on canonical codes (robust to feed
        # abbreviation variants and matchup ordering)
        a_imp = vegas.get(C.canonical_team(away))
        h_imp = vegas.get(C.canonical_team(home))
        a_imp = a_imp if a_imp is not None else C.DEFAULT_TEAM_RUNS
        h_imp = h_imp if h_imp is not None else C.DEFAULT_TEAM_RUNS
        out['implied'] = {'away': a_imp, 'home': h_imp, 'total': a_imp + h_imp}
        games[gid] = out

    games = filter_slate_doubleheaders(games, slate_players)

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

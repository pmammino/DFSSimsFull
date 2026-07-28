"""
game_status.py — detect games that MLB will NOT play as scheduled.

The problem this solves
-----------------------
The RotoWire salaries / ownership / lineup feeds keep a game's players on the
slate even after MLB postpones or cancels it (a rainout, doubleheader split,
etc.). Left unchecked, the pipeline simulates and rosters players who won't
take the field, so lineups get built around a game that isn't happening.

The fix is to ask the *authoritative* source — the MLB StatsAPI schedule
endpoint — which games for a slate date are postponed / cancelled / suspended,
and to key that answer by canonical team code so it reconciles with the slate
exactly the way the Vegas totals already do (see slate_config.canonical_team).

    GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD

Each game carries a ``status`` object; the human-readable ``detailedState``
("Scheduled", "Postponed", "Cancelled", "Suspended", "In Progress", "Final", …)
is the primary signal, with ``codedGameState`` ('D' postponed, 'C' cancelled)
as a backstop.

Everything here is best-effort: any network / parse failure is swallowed and
reported as "no postponements", so a StatsAPI outage can never wipe a slate —
the pipeline just falls back to its prior behaviour of keeping every game.
"""
import sys

import requests

import slate_config as C

STATSAPI_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"

# statsapi returns 403 to the default python-requests UA on some networks;
# present a browser-like UA (mirrors data_acquisition.HEADERS).
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36"}

# detailedState values (matched case-insensitively as substrings) that mean the
# game will NOT be played as scheduled and its players must not be rostered.
_NOT_PLAYED_STATES = ("postponed", "cancelled", "canceled", "suspended")
# codedGameState backstop: 'D' = postponed, 'C' = cancelled.
_NOT_PLAYED_CODES = {"D", "C"}


def _classify(detailed_state, coded_state):
    """Return a normalized category ('postponed'|'cancelled'|'suspended') if the
    game won't be played as scheduled, else None."""
    ds = (detailed_state or "").strip().lower()
    for key in _NOT_PLAYED_STATES:
        if key in ds:
            return "cancelled" if key in ("cancelled", "canceled") else key
    if (coded_state or "").strip().upper() in _NOT_PLAYED_CODES:
        return "postponed" if (coded_state or "").upper() == "D" else "cancelled"
    return None


def fetch_schedule(date=None, timeout=15, session=None):
    """Fetch the raw StatsAPI schedule JSON for `date` (YYYY-MM-DD, or None for
    today). Raises on network / HTTP error — callers that want best-effort
    behaviour should go through game_statuses()/postponed_matchups()."""
    params = {"sportId": 1}
    if date:
        params["date"] = date
    getter = session.get if session is not None else requests.get
    r = getter(STATSAPI_SCHEDULE, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def game_statuses(date=None, schedule_json=None, timeout=15):
    """Return one status record per game StatsAPI lists for `date`:

        {away, home, matchup(frozenset of codes), status(detailedState),
         reason, coded, category, not_played(bool), game_pk}

    `away`/`home` are canonical team codes (slate_config.canonical_team), so
    they line up with the slate's team codes. `schedule_json` (a pre-fetched
    dict, e.g. in tests) bypasses the network fetch. Raises on fetch failure
    when `schedule_json` is not supplied."""
    data = schedule_json if schedule_json is not None else fetch_schedule(date, timeout)
    if isinstance(data, str):
        import json
        data = json.loads(data or "{}")
    out = []
    for dd in (data.get("dates") or []):
        for g in (dd.get("games") or []):
            teams = g.get("teams") or {}
            a_team = (teams.get("away") or {}).get("team") or {}
            h_team = (teams.get("home") or {}).get("team") or {}
            away = C.canonical_team(a_team.get("abbreviation") or a_team.get("name"))
            home = C.canonical_team(h_team.get("abbreviation") or h_team.get("name"))
            status = g.get("status") or {}
            ds = status.get("detailedState")
            cat = _classify(ds, status.get("codedGameState"))
            out.append({
                "away": away, "home": home,
                "matchup": frozenset({away, home}) - {None},
                "status": ds, "reason": (status.get("reason") or "").strip(),
                "coded": status.get("codedGameState"),
                "category": cat, "not_played": cat is not None,
                "game_pk": g.get("gamePk"),
            })
    return out


def postponed_matchups(date=None, schedule_json=None, timeout=15):
    """Best-effort map of games that will NOT be played on `date`:

        { frozenset({away_code, home_code}): {
              away, home, status, reason, category } }

    Only two-team matchups are keyed (a TBD/placeholder game is skipped). On any
    fetch/parse failure returns {} and logs a note — so a StatsAPI outage never
    causes games to be dropped."""
    try:
        rows = game_statuses(date, schedule_json=schedule_json, timeout=timeout)
    except Exception as e:  # network, HTTP, JSON — all non-fatal
        print(f"  [game_status] schedule fetch failed ({type(e).__name__}: {e}); "
              "assuming no postponements", file=sys.stderr)
        return {}
    out = {}
    for r in rows:
        if r["not_played"] and len(r["matchup"]) == 2:
            out[r["matchup"]] = {
                "away": r["away"], "home": r["home"], "status": r["status"],
                "reason": r["reason"], "category": r["category"]}
    return out


def postponed_list(date=None, schedule_json=None, timeout=15):
    """Best-effort JSON/cache-friendly list of not-played games for `date`:

        [{away, home, status, reason, category}, …]

    Same data as postponed_matchups() without the frozenset keys — convenient
    for display (the app's slate warning) and for Streamlit's cache."""
    return [dict(v) for v in postponed_matchups(
        date, schedule_json=schedule_json, timeout=timeout).values()]


if __name__ == "__main__":
    import datetime
    d = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
    rows = game_statuses(d)
    print(f"Schedule {d}: {len(rows)} game(s)")
    for r in rows:
        flag = f"  ⛔ {r['category'].upper()}" + (f" ({r['reason']})" if r['reason'] else "") \
            if r["not_played"] else ""
        print(f"  {r['away']}@{r['home']}: {r['status']}{flag}")
    pp = postponed_list(d)
    print(f"{len(pp)} not-played game(s): "
          + ", ".join(f"{p['away']}@{p['home']} [{p['category']}]" for p in pp))

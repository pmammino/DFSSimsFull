#!/usr/bin/env python3
"""
vegas_diag.py — diagnose the live FantasyLabs Vegas team-totals feed.

Run this ON THE MACHINE that hosts the app (where the host is reachable):

    python vegas_diag.py            # today's date
    python vegas_diag.py 2026-06-16 # a specific slate date

It prints, in order:
  1. the exact URL + HTTP status / content-type / size + a body sample
  2. the JSON shape and the field names of the first event
  3. how many matchups fetch_vegas() actually parses (and a sample)
  4. whether each slate game's team-code key matches a parsed Vegas key
     (unmatched games fall back to the default 4.4 — the "all teams same" bug)

Paste the output back and the parser / team-code map can be fixed precisely.
"""
import sys
import json
import datetime
import urllib.request

import slate_config as C
import slate_ingest as SI

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
    url = C.FEED_VEGAS_TMPL.format(date=date)
    print(f"[1] URL: {url}")

    body = None
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status = getattr(r, "status", r.getcode())
            ctype = r.headers.get("Content-Type")
            body = r.read().decode("utf-8", "replace")
        print(f"    HTTP {status} | Content-Type: {ctype} | {len(body)} bytes")
        print(f"    body head: {body[:300]!r}")
    except urllib.error.HTTPError as e:
        print(f"    HTTPError {e.code}: {e.read()[:200]!r}")
        return
    except Exception as e:
        print(f"    FETCH FAILED: {type(e).__name__}: {e}")
        return

    print("\n[2] JSON shape")
    try:
        data = json.loads(body)
    except Exception as e:
        print(f"    NOT JSON ({e}) — the feed may require auth/cookies or returns HTML.")
        return
    if isinstance(data, list):
        print(f"    list of {len(data)} items")
        rows = data
    elif isinstance(data, dict):
        print(f"    dict; top-level keys: {sorted(data.keys())[:30]}")
        rows = data.get("Events") or data.get("data") or data.get("events") or []
        print(f"    rows under Events/data/events: {len(rows)}")
    else:
        print(f"    unexpected type: {type(data)}")
        rows = []
    if rows:
        print(f"    FIRST EVENT KEYS: {sorted(rows[0].keys())}")
        print(f"    FIRST EVENT: {json.dumps(rows[0])[:600]}")

    print("\n[3] fetch_vegas() parse result (per-team implied totals)")
    parsed = SI.fetch_vegas(date)
    print(f"    parsed {len(parsed)} teams")
    for k, v in list(parsed.items())[:8]:
        print(f"      {k}: {v}")

    print("\n[4] slate team-match (does each team find its Vegas total?)")
    try:
        slate = SI.build_slate(write=False)
        print(f"    slate {slate.get('date')}: {len(slate.get('games', {}))} games")
        miss = 0
        for g in slate["games"].values():
            for side in ("away", "home"):
                canon = C.canonical_team(g[side])
                hit = canon in parsed
                miss += (0 if hit else 1)
                print(f"      {g[side]:6} → canon {canon:5} matched={hit}  "
                      f"implied={g['implied'][side]}")
        print(f"\n    UNMATCHED teams: {miss} "
              f"(these default to {C.DEFAULT_TEAM_RUNS} → 'all teams same')")
    except Exception as e:
        print(f"    slate build failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

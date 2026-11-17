#!/usr/bin/env python3
"""
fetch_role_feeds.py — download the four RotoWire usage feeds into feeds/.

    python scripts/fetch_role_feeds.py            # fetch all four
    python scripts/fetch_role_feeds.py --only depth closers
    python scripts/fetch_role_feeds.py --dry-run  # fetch + validate, write nothing

The feeds decide player roles (role_feeds.py) and, for depth.xml, each
player's club (team context). Until now they were saved by hand; this replaces
that.

URLs come from ENVIRONMENT VARIABLES, never from the repo — two of them carry
an API key:

    RW_DEPTH_URL      required   .../Baseball/MLB/DepthChart.php?key=...
    RW_CLOSERS_URL    required   .../Baseball/MLB/Closers.php?key=...
    RW_ORDERS_URL     optional   batting-orders proxy (defaults to the same
                                 proxy host the slate feeds use)
    RW_PROSPECTS_URL  optional   rotowire.com top-prospects table (public)

    bash:        export RW_DEPTH_URL='https://api.rotowire.com/...'
    PowerShell:  $env:RW_DEPTH_URL = 'https://api.rotowire.com/...'
    Windows:     setx RW_DEPTH_URL "https://api.rotowire.com/..."   (new shells)

SAFETY. A feed is only written over the last good copy if it PARSES and looks
like the real thing (right root element, enough clubs/players). A failed or
garbled download leaves the existing file alone and the script exits non-zero,
so `refresh_and_run.py` stops rather than silently projecting on a stale or
truncated feed. A rejected payload is saved to feeds/.rejected/ (git-ignored)
so you can see what the server actually sent. Pass --allow-stale to carry on
with the existing files after a failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEED_DIR = ROOT / "feeds"
REJECT_DIR = FEED_DIR / ".rejected"
LOG_PATH = FEED_DIR / "fetch_log.json"

DEFAULT_PROSPECTS_URL = ("https://www.rotowire.com/baseball/tables/"
                         "top-prospects.php?pos=All&league=3&eta=All"
                         "&signed=All&fypd=All")

# name -> (env var, file name, required?)
FEEDS = {
    "depth":     ("RW_DEPTH_URL",     "depth.xml",     True),
    "orders":    ("RW_ORDERS_URL",    "orders.xml",    False),
    "closers":   ("RW_CLOSERS_URL",   "closers.xml",   True),
    "prospects": ("RW_PROSPECTS_URL", "prospects.xml", False),
}


def default_url(name: str) -> str | None:
    """Defaults for the feeds that carry no secret."""
    if name == "prospects":
        return DEFAULT_PROSPECTS_URL
    if name == "orders":
        # Same proxy host the daily slate feeds already use (slate_config).
        try:
            sys.path.insert(0, str(ROOT))
            from slate_config import FEED_CONFIRMED
            base = FEED_CONFIRMED.split("?")[0]
            return f"{base}?feed=batting-orders"
        except Exception:
            return None
    return None


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (dfs-pipeline feed refresh)",
        "Accept": "application/xml,text/xml,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_xml(raw: bytes) -> ET.Element:
    """Parse, tolerating a banner or BOM before the first tag."""
    text = raw.decode("utf-8", "replace")
    start = text.find("<")
    if start < 0:
        raise ValueError("response contains no XML")
    return ET.fromstring(text[start:])


def _count(root: ET.Element, xpath: str) -> int:
    return len(root.findall(xpath))


def validate(name: str, root: ET.Element) -> dict:
    """Raise ValueError unless `root` looks like the real feed. Returns counts.

    The floors are deliberately loose — they exist to catch an error page, an
    empty body or a truncated download, not to police roster size.
    """
    if name == "depth":
        teams = _count(root, ".//Team")
        players = _count(root, ".//Team/Players/Player")
        ok = root.tag == "DepthChart" and teams >= 28 and players >= 1000
    elif name == "closers":
        teams = _count(root, ".//Team")
        players = _count(root, ".//Team//Player")
        ok = teams >= 28 and players >= 150
    elif name == "orders":
        teams = len({t.get("Code") for t in root.iter("Team")
                     if t.find(".//BattingOrder") is not None})
        players = _count(root, ".//BattingSpot")
        ok = teams >= 15 and players >= 200
    elif name == "prospects":
        teams = len({t.get("Code") for t in root.iter("Team")})
        players = _count(root, ".//Player")
        ok = root.tag == "Prospects" and players >= 200
    else:  # pragma: no cover
        raise KeyError(name)
    counts = {"root": root.tag, "teams": teams, "players": players}
    if not ok:
        raise ValueError(f"does not look like the {name} feed: {counts}")
    return counts


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)            # atomic on POSIX and Windows


def fetch_one(name: str, *, dry_run: bool = False, getter=http_get,
              feed_dir: Path = FEED_DIR) -> dict:
    """Fetch + validate + write one feed. Never raises; returns a status dict."""
    env, fname, required = FEEDS[name]
    url = os.environ.get(env) or default_url(name)
    status = {"feed": name, "file": fname, "ok": False}
    if not url:
        status["error"] = (f"{env} is not set" +
                           ("" if required else " and has no default"))
        return status
    try:
        raw = getter(url)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        status["error"] = f"download failed: {type(e).__name__}: {e}"
        return status
    try:
        counts = validate(name, parse_xml(raw))
    except (ET.ParseError, ValueError) as e:
        status["error"] = f"rejected ({type(e).__name__}: {e})"
        try:
            REJECT_DIR.mkdir(parents=True, exist_ok=True)
            (REJECT_DIR / fname).write_bytes(raw)
            status["rejected_copy"] = str(REJECT_DIR / fname)
        except OSError:
            pass
        return status
    status.update(ok=True, bytes=len(raw), **counts)
    if not dry_run:
        feed_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(feed_dir / fname, raw)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FEEDS),
                    help="fetch just these feeds")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and validate but write nothing")
    ap.add_argument("--allow-stale", action="store_true",
                    help="exit 0 even if a feed failed (keeps the old file)")
    args = ap.parse_args(argv)

    names = args.only or list(FEEDS)
    results = [fetch_one(n, dry_run=args.dry_run) for n in names]

    print("RotoWire role feeds" + (" (dry run)" if args.dry_run else ""))
    for r in results:
        if r["ok"]:
            print(f"  OK    {r['feed']:<10} {r['teams']:>3} clubs "
                  f"{r['players']:>6} rows  {r['bytes']/1024:>7.0f} KB")
        else:
            print(f"  FAIL  {r['feed']:<10} {r['error']}")
            if r.get("rejected_copy"):
                print(f"        payload saved to {r['rejected_copy']}")

    if not args.dry_run and any(r["ok"] for r in results):
        log = {}
        if LOG_PATH.exists():
            try:
                log = json.loads(LOG_PATH.read_text())
            except ValueError:
                log = {}
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        for r in results:
            if r["ok"]:
                log[r["feed"]] = {"fetched_utc": stamp, "teams": r["teams"],
                                  "players": r["players"]}
        LOG_PATH.write_text(json.dumps(log, indent=2) + "\n")

    failed = [r["feed"] for r in results if not r["ok"]]
    if failed:
        msg = f"{len(failed)} feed(s) not refreshed: {', '.join(failed)}"
        if args.allow_stale:
            print(f"  WARNING: {msg}; continuing on the existing files")
            return 0
        print(f"  ERROR: {msg}; existing files left untouched")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

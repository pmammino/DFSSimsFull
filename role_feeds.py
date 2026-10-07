"""
role_feeds.py — turn RotoWire usage feeds into ROLE MIXTURES and availability.

The playing-time model has always been able to express "this player is 60% an
everyday regular and 40% a bench bat" (`pt_role_mix`, `role_taxonomy.
parse_role_mix`), and until now nothing filled it in. Every player arrived at
the allocator as 100% of one role at 100% availability, which is a statement
nobody has enough information to make: it says a club's eighth-best outfielder
and its left fielder are each a settled fact.

Four feeds say otherwise, and between them they say it in the only way worth
having — by DISAGREEING. That is the design idea here and it is worth stating
before the mechanics:

    A mixture's WIDTH is not an opinion. It is how much the feeds disagree
    about a player.

A left fielder batting third against both hands who is also the depth chart's
rank-1 left fielder is not uncertain, and he comes out ~0.90 on his role. A
player the depth chart calls a starter who appears in neither batting order is
genuinely uncertain, and he comes out split ~0.55/0.45 between what each feed
implies. A closer the feed itself labels "Very Low" stability is uncertain
because RotoWire says so, and he comes out 0.50 rather than 0.92. Nothing here
invents a spread to look humble: every number traces to a feed saying
something, two feeds saying different things, or one feed saying nothing.

THE FEEDS
---------

`depth.xml`      All 30 clubs, ~1,690 players, ranked WITHIN a position group:
                 1B/2B/3B/SS/C/LF/CF/RF/DH for the lineup, P for the rotation,
                 BP for the bullpen, CL for the closer, PROS for prospects.
                 This is the only feed that covers everybody.

`orders.xml`     A batting order vs LHP and vs RHP. The two orders together are
                 the platoon signal: in both = regular, in the vs-R order only
                 = strong-side platoon, in the vs-L order only = weak side. The
                 batting SPOT is the volume signal — a leadoff hitter takes
                 about 4.65 plate appearances a game and a ninth-place hitter
                 about 4.05.

`closers.xml`    Bullpen pecking order, 1..12, with RotoWire's own role label
                 (Closer / Committee / Setup Man / Middle Reliever /
                 Multi-Inning / Closer of Future) and, on the top arm only, a
                 Stability rating. Stability is the single most useful field in
                 any of these files: it is a feed telling you how much to
                 believe it.

`prospects.xml`  The top 400, with a league level. Level is about WHEN a player
                 arrives, not how much he plays once he does, so it sets
                 `pt_role_start` (the timing share) rather than the mixture.

TWO TRAPS, BOTH MEASURED
------------------------

1. The batting orders feed carries STALE LINEUPS alongside current ones. Atlanta
   has two blocks both marked GameType="NORMAL": one reading Swanson / Albies /
   Riley / Ozuna / Duvall / d'Arnaud / Arcia / Pache, and one reading Acuna /
   Baldwin / Olson / Albies / Harris / Dubon. The first is from the era when
   the pitcher batted, and it has EIGHT spots; the current one has nine. That
   is the whole discriminator, and it is decisive: 8-spot orders agree with the
   projection's club for 29.4% of their players, 9-spot orders for 99.6%. The
   same filter quietly resolves a duplicate Washington entry (code WAS, Id 64,
   two 8-spot orders; code WSH, Id 66, two 9-spot ones).

2. Seven clubs have no batting order at all — ARI, CHC, CWS, LAA, LAD, NYM,
   NYY. A player on one of them is not a bench bat, he is unobserved, and the
   mixtures fall back to the depth chart with a width that says so.

IDENTITY
--------

None of these feeds carries an MLBAM id except `prospects.xml`, and that for
261 of its 400. Everything else joins on a normalized name, preferring a player
whose projected club matches the feed's. Measured against the 2027 set: 99.6%
of batting-order hitters, 99.3% of bullpen arms and 92.5% of depth-chart
players resolve, and the hitter pool contains exactly two names shared by two
players. Those two, and any future collision, are dropped with a warning rather
than guessed at — `rosters/player_id_aliases_<year>.csv` is where a human
settles them, and where the handful of genuine misses get fixed.
"""
from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from durability import record_read
from role_taxonomy import (DEPTH_HITTER_ROLE, DEPTH_PITCHER_ROLE,
                           format_role_mix, is_depth_role, role_anchor,
                           role_names)
from team_context import canonical_team, team_id_for_abbr

FEED_DIR = "feeds"
FEED_FILES = {"depth": "depth.xml", "orders": "orders.xml",
              "closers": "closers.xml", "prospects": "prospects.xml"}

# A batting order from the era when the pitcher hit. See TWO TRAPS above.
CURRENT_LINEUP_SPOTS = 9


# How much of a season a prospect at each level is likely to be available for,
# expressed as the timing label the role model already understands. A player
# the feed lists as being in the majors is simply on the roster.
LEVEL_TIMING = {
    "MAJORS": "Opening Day",
    "AAA": "Early Season (~May)",
    "AA": "Mid Season (~July)",
    "A+": "Late Season (~Sept)",
    "A": "Late Season (~Sept)",
    "ROOKIE": "Late Season (~Sept)",
}
# Below this level a prospect is not a 2027 major leaguer in any meaningful
# sense, and giving him a timing share at all would take plate appearances
# from someone who will actually bat.
LEVEL_NO_MLB = {"A", "ROOKIE"}

# WHEN A PROSPECT ARRIVES, AND WHAT HE DOES WHEN HE GETS THERE.
#
# A player one level from the majors is not a depth player, and calling him
# one costs the projection more than it looks. Real first-year position
# players take 8.9% of all league plate appearances — 114 of them a season,
# 3.8 a club, median 92 plate appearances, ninetieth percentile 368 — and
# every one of those was somewhere in this feed the winter before. Leaving
# them at the 1-PA floor does not save that playing time; it hands it to
# incumbents who will not be taking it.
#
# The feed has no ETA field, so arrival is read off the three things it does
# carry: LEVEL (how far away he is), RANK (how good, which decides both how
# soon a club promotes him and how much he plays once up) and AGE (a
# 21-year-old at Triple-A is next year's regular; a 27-year-old at Triple-A
# has been passed over).
#
# The mixtures below are WIDE on purpose, and this is the case the mixture
# machinery was built for: nobody knows whether a top prospect called up in
# May finishes the year as the regular or back on the bus. Expressing that
# as 40% regular / 60% part-time is the honest reading, and it is what makes
# the blended anchor land near the real distribution instead of on either
# tail of it.
PROSPECT_ARRIVAL = {
    # level: ((rank cutoff, timing, mixture), ...) — first match wins
    "AAA": (
        (60, "Early Season (~May)",
         {"Full Time": 0.47, "Strong Side Platoon": 0.25, "Bench Bat": 0.28}),
        (200, "Mid Season (~July)",
         {"Strong Side Platoon": 0.35, "Bench Bat": 0.45,
          "Injury Replacement / 26th Man": 0.20}),
    ),
    "AA": (
        (40, "Mid Season (~July)",
         {"Full Time": 0.34, "Strong Side Platoon": 0.30,
          "Bench Bat": 0.36}),
        (100, "Mid Season (~July)",
         {"Strong Side Platoon": 0.26, "Bench Bat": 0.50,
          "Injury Replacement / 26th Man": 0.24}),
    ),
}

# HOW MUCH WEIGHT SITS ON THE REAL JOB, which is the knob that decides how
# much playing time an arrival actually gets, and it was first set too low.
# Every band above has been moved toward its top role and away from its
# bottom one, and it is a free move in both directions that matter: the
# median arrival goes from 30.9 plate appearances to 38.5 and the ninetieth
# percentile from 280 to 302, while the rank curve IMPROVES from 0.0212 to
# 0.0200.
#
# It stops where it does because of a CLIFF rather than a slope. Pushed
# roughly 15% further the Double-A top-40 band tips from "Bench Bat" to
# "Full Time" as its heaviest role, and the modal role is what picks the
# family a player is ranked in — so the whole band would jump out of the
# bench queue and into the lineup, competing with regulars for the top of
# the club rather than with the last men for the bottom of it. The rank
# curve says so loudly: 0.0200 at the setting below, 0.0457 one step past
# it. Nothing in between is being left on the table; the next step is a
# different claim, not a bigger one.

# WHY THE TABLE STOPS WHERE IT DOES, which is the part that was measured.
#
# It reached further to begin with — unranked Triple-A, Double-A to 150, the
# top of High-A — on the theory that a wider net is a fairer one. It is not.
# Those bands put 34 more names in the projection and gave them a MEDIAN OF
# 3.8 PLATE APPEARANCES, because a September arrival on a 117-plate-
# appearance anchor is a fifth of a season of a job that barely exists, and
# the roster-depth decay then takes what is left. They were not projections;
# they were noise with a name attached, and they cost the rank curve 0.0250
# against 0.0228 for the top 200 and top 40 alone.
#
# The second Double-A band is NOT a retreat from that, and the difference is
# the reason it works. The rejected band was LATE SEASON on an injury-
# replacement mixture — a fifth of a season of nothing. This one is a July
# callup to a bench job, which is a real thing that happens to a top-100
# prospect, and it measures like one: eleven more names at a median of 31
# plate appearances, a minimum of 5.5, and the rank curve IMPROVES to 0.0212.
# The lesson is about what a band claims, not about how far down it reaches.
#
# The league's playing time is fixed, so a name that takes nothing still
# takes a roster place. Better to leave the September cup-of-coffee man at
# the floor — WHICH IS WHAT HE IS — and spend the arrivals on the players a
# club is actually waiting on.
# THE SAME THING FOR ARMS, which is not the same table.
#
# A pitching prospect's first season looks nothing like a hitter's. Real
# first-year pitchers are far more numerous and take far less each: 181 of
# them in the most recent season against 114 position players, but at a
# MEDIAN OF 12.7 INNINGS against a median of 92 plate appearances, and only 6
# of the 181 cleared 100 innings. Together they are 9.1% of all league
# innings, almost exactly the hitters' 8.9%, arrived at completely
# differently.
#
# That median hides two populations. About 23 to 34 a season START games and
# take a median of 72 innings; the other ~170 are relief arms shuttling up
# from Triple-A for a handful of outings each. ONLY THE FIRST GROUP IS HERE,
# and that is the main thing this table decides. Three independent reasons,
# all pointing the same way:
#
#   * Nine innings is not a projection. It is the same judgement that stops
#     the hitter table at Triple-A's top 200 — a name that takes nothing
#     still takes a roster place, and the league's innings are fixed.
#   * The depth ranking already says so. A club carries ~20 relief arms in
#     the projected tier against 8 core bullpen slots, so an arriving
#     reliever sorts 16th to 20th of 20 and the decay hands him ~1 inning
#     whatever the table claims. Modelling him changes nothing except the
#     size of the pool.
#   * The evidence for "reliever" is the weakest evidence in the feed. The
#     classification is innings per appearance, and it is sharp where there
#     is a record — real starters run 4.0 to 5.4, real relievers 0.3 to 3.4 —
#     but a prospect with six career innings reads as a reliever on nothing.
#     A starter classification on a thin record cannot happen by accident;
#     a reliever one happens all the time.
#
# So the key is the pitcher's own record — `role`, which the pipeline derives
# from his game log including the minor-league lines — and it is the thing
# that makes this table possible at all, because THE PROSPECTS FEED CALLS
# EVERY ARM IN IT A BARE "P". There is no SP/RP split to read, and a rookie
# rotation arm and a shuttle reliever differ by a factor of eight in innings.
#
# The mixtures lean on "Innings-Limited Starter", which is the anchor that
# exists for exactly this man, rather than on a rotation slot a 23-year-old
# will not hold for a full season. They stay modal-starter on purpose: the
# modal role picks the family a player is ranked in, and a rookie starter
# competes with his club's rotation for innings, not with its bullpen.
PITCHER_ARRIVAL = {
    # level: ((rank cutoff, timing, mixture), ...) — first match wins
    "AAA": (
        (100, "Early Season (~May)",
         {"Innings-Limited Starter": 0.45, "Swing Arm / Long Relief": 0.30,
          "Bullpen Depth Arm": 0.25}),
        (250, "Mid Season (~July)",
         {"Innings-Limited Starter": 0.47, "Swing Arm / Long Relief": 0.30,
          "Bullpen Depth Arm": 0.23}),
    ),
    "AA": (
        (150, "Mid Season (~July)",
         {"Innings-Limited Starter": 0.46, "Swing Arm / Long Relief": 0.30,
          "Bullpen Depth Arm": 0.24}),
    ),
}

# The same move as the hitters', and it buys much less — a median of 23.2
# innings against 21.8 — because THE MIXTURE IS NOT WHAT BINDS HERE. Every
# arriving arm leaves the anchor stage at the same ~41 innings and then
# differs only by where he ranks: a club carries 12 in the starting family
# against 6 core slots, so the arrivals sort 7th to 12th and the decay hands
# out 0.66, 0.44, 0.29, 0.19, 0.13, 0.08 of it. Rank, not role, is what
# separates Yhoiker Fajardo's 35 innings from Tyler Bremner's 4.
#
# The top band is deliberately left alone. It already carries 0.45 on one
# role, and moving it further would put a single role above half — a
# confidence nobody has about whether a 23-year-old holds a rotation spot,
# and the opposite of what the mixture is for.

# The cutoffs run deeper than the hitters' 200 and 40, and that is not
# generosity — it is that RANK MEANS SOMETHING DIFFERENT HERE. The feed
# publishes one combined top 400, so a pitcher's rank is his standing among
# bats as well as arms, and the 128 arms in the list are spread through it
# rather than clustered at the top. Triple-A's top 100 holds seven of them
# where it holds a dozen hitters. Matching the hitters' numbers would not
# mean matching their strictness; it would mean reaching a third as far.
PITCHER_ARRIVAL_ROLE = "starter"


# Past this, a "prospect" at Triple-A is organisational depth rather than
# someone a club is waiting on, and the arrival table should not promote
# him. The feed's own Triple-A cohort runs 23 to 25, so this cuts the tail
# rather than the middle.
PROSPECT_MAX_AGE = 26

# How much of the arrival table's volume to actually spend. Kept at 1.0:
# once the arrivals were made to COMPETE for their playing time in the
# roster-depth ranking rather than bypassing it, there was nothing left for
# a global haircut to fix. It survives as the lever to reach for if the
# arrivals ever need damping as a group.
PROSPECT_VOLUME_SCALE = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def name_key(first: str, last: str) -> str:
    """Normalized join key. Mirrors `stage_d.norm`, which the DFS side of this
    repo has used against RotoWire feeds for long enough to trust, but is
    reimplemented rather than imported so the season pipeline does not depend
    on the daily one."""
    n = unicodedata.normalize("NFKD", f"{first} {last}")
    n = n.encode("ascii", "ignore").decode().lower()
    n = n.replace(".", "").replace(",", "").replace("'", "")
    n = re.sub(r"\s+", " ", n).strip()
    for suffix in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip()


def _root(path: str | Path) -> ET.Element | None:
    """Parse a feed, tolerating the "This XML file does not appear to have any
    style information" banner a browser prepends when the feed is saved by
    hand — which is how these files actually arrive."""
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    start = text.find("<")
    if start < 0:
        return None
    try:
        return ET.fromstring(text[start:])
    except ET.ParseError as e:
        print(f"  role feeds: {p.name} is not parseable ({e}); ignored")
        return None


def _player_row(el: ET.Element, team_abbr: str | None) -> dict:
    return {
        "name_key": name_key(el.findtext("FirstName") or "",
                             el.findtext("LastName") or ""),
        "feed_name": f"{(el.findtext('FirstName') or '').strip()} "
                     f"{(el.findtext('LastName') or '').strip()}".strip(),
        "rw_id": el.get("Id"),
        "feed_team": team_abbr,
        "feed_team_id": team_id_for_abbr(team_abbr) if team_abbr else None,
        "feed_position": (el.findtext("Position") or "").strip().upper(),
    }


def _teams(root: ET.Element):
    for team in root.iter("Team"):
        code = team.get("Code")
        if not code:
            continue
        yield team, canonical_team(code)


def read_depth(path: str | Path) -> pd.DataFrame:
    """Every club's depth chart: one row per player per position group.

    `depth_rank` is the rank WITHIN that group, so a rank of 1 at SS and a
    rank of 1 at BP mean very different things; the mapping below reads the
    position group first and the rank second.
    """
    root = _root(path)
    if root is None:
        return pd.DataFrame(columns=["name_key", "feed_team", "depth_pos",
                                     "depth_rank"])
    rows = []
    for team, abbr in _teams(root):
        for el in team.findall(".//Players/Player"):
            row = _player_row(el, abbr)
            row["depth_pos"] = row.pop("feed_position")
            row["depth_rank"] = pd.to_numeric(el.findtext("Rank"),
                                              errors="coerce")
            rows.append(row)
    return pd.DataFrame(rows)


def read_batting_orders(path: str | Path) -> pd.DataFrame:
    """The current vs-LHP and vs-RHP batting orders, one row per player.

    Only NINE-spot orders are read. An eight-spot order is a lineup from when
    the pitcher batted, and this feed still carries them: see TWO TRAPS. The
    legacy rows are returned too, flagged `legacy=True`, so the report can say
    how many were dropped rather than silently discarding half the feed.
    """
    root = _root(path)
    cols = ["name_key", "feed_team", "vs_hand", "spot", "legacy"]
    if root is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for team, abbr in _teams(root):
        for order in team.findall(".//BattingOrder"):
            spots = order.findall(".//BattingSpot")
            legacy = len(spots) != CURRENT_LINEUP_SPOTS
            hand = (order.get("OpposingPitcherHandedness") or "").upper()
            for spot in spots:
                el = spot.find("Player")
                if el is None:
                    continue
                row = _player_row(el, abbr)
                row["vs_hand"] = hand
                row["spot"] = pd.to_numeric(spot.get("Position"),
                                            errors="coerce")
                row["legacy"] = legacy
                rows.append(row)
    return pd.DataFrame(rows)


def read_closers(path: str | Path) -> pd.DataFrame:
    """Bullpen pecking order with RotoWire's own role label.

    `stability` is present only on each club's top arm and is the feed telling
    you how much to believe its own Rank-1 call. It is the one field in any of
    these files that is explicitly about uncertainty, and it is carried through
    to the mixture width unchanged.
    """
    root = _root(path)
    cols = ["name_key", "feed_team", "bullpen_rank", "rw_role", "stability"]
    if root is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for team, abbr in _teams(root):
        for el in team.iter("Player"):
            row = _player_row(el, abbr)
            row["bullpen_rank"] = pd.to_numeric(el.get("Rank"),
                                                errors="coerce")
            row["rw_role"] = (el.get("Role") or "").strip()
            row["stability"] = (el.get("Stability") or "").strip()
            rows.append(row)
    return pd.DataFrame(rows)


def read_prospects(path: str | Path) -> pd.DataFrame:
    """The top 400, with the league level that decides WHEN they arrive.

    This is also the only feed carrying an MLBAM id (`MLBId`, on 261 of the
    400), so it seeds the identity crosswalk for everything else.
    """
    root = _root(path)
    cols = ["name_key", "feed_team", "prospect_rank", "level", "PlayerId",
            "birth_year"]
    if root is None:
        return pd.DataFrame(columns=cols)
    rows = []
    for el in root.iter("Player"):
        team = el.find("Team")
        abbr = canonical_team(team.get("Code")) if team is not None else None
        row = _player_row(el, abbr)
        row["prospect_rank"] = pd.to_numeric(el.get("Rank"), errors="coerce")
        row["level"] = (el.findtext("LeagueLevel") or "").strip().upper()
        row["PlayerId"] = pd.to_numeric(el.findtext("MLBId"), errors="coerce")
        # There is no ETA field in this feed — only Rank, LeagueLevel, a
        # draft year and a birth date — so the arrival has to be read off
        # those. Age is the one that separates a prospect from org filler:
        # a 21-year-old at Triple-A is next year's regular and a 27-year-old
        # at Triple-A is a player who has been passed over.
        birth = el.find(".//Birth")
        born = (birth.get("Date") if birth is not None else None) or ""
        row["birth_year"] = pd.to_numeric(born[:4], errors="coerce")
        rows.append(row)
    return pd.DataFrame(rows)


def read_feeds(feed_dir: str | Path = FEED_DIR) -> dict[str, pd.DataFrame]:
    """All four, by name. Each is empty rather than absent when its file is
    missing, so a caller with two of the four still gets the two."""
    d = Path(feed_dir)
    return {
        "depth": read_depth(d / FEED_FILES["depth"]),
        "orders": read_batting_orders(d / FEED_FILES["orders"]),
        "closers": read_closers(d / FEED_FILES["closers"]),
        "prospects": read_prospects(d / FEED_FILES["prospects"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

def load_aliases(path: str | Path) -> dict[str, int]:
    """Hand-written `name_key -> PlayerId` fixes for what the join misses.

    Two columns, `name_key` and `PlayerId`. It exists because a name join is
    the best available and not a good one: the feeds carry no MLBAM id, so
    there will always be a residue of players whose RotoWire spelling and
    whose MLBAM spelling differ, and a file is the right place to settle them
    once rather than re-deriving a fuzzy match every run.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, comment="#")
    except Exception as e:
        print(f"  role feeds: {p.name} unreadable ({type(e).__name__}); ignored")
        return {}
    if not {"name_key", "PlayerId"} <= set(df.columns):
        print(f"  role feeds: {p.name} needs name_key and PlayerId columns")
        return {}
    ids = pd.to_numeric(df["PlayerId"], errors="coerce")
    return {str(k).strip().lower(): int(v)
            for k, v in zip(df["name_key"], ids) if pd.notna(v)}


def resolve_ids(feed: pd.DataFrame, players: pd.DataFrame, *,
                aliases: dict[str, int] | None = None,
                team_col: str = "Pred_target_team_id",
                label: str = "feed") -> tuple[pd.DataFrame, dict]:
    """Attach an MLBAM `PlayerId` to feed rows, by name and then by club.

    The club is a TIE-BREAKER, never a filter. Measured against the 2027 set,
    99.6% of batting-order hitters resolve by name and 99.6% of those are on
    the club the feed says — but the ones that are not are mostly real
    offseason moves, and dropping a player because he has been traded would
    throw away exactly the rows a season projection needs most.

    A name shared by two players in the same pool is resolved by club if that
    settles it and DROPPED if it does not. Guessing would silently hand one
    man's job to another, which is worse than leaving the heuristic default in
    place; the alias file is how a human settles it.
    """
    stats = {"rows": len(feed), "matched": 0, "unmatched": 0, "ambiguous": 0,
             "by_alias": 0, "wrong_club": 0, "missed_names": []}
    if feed.empty or players.empty:
        return feed.assign(PlayerId=pd.Series(dtype=float)), stats

    ref = players[["PlayerId"]].copy()
    ref["name_key"] = players["Name"].astype(str).map(
        lambda s: name_key(*s.split(" ", 1)) if " " in str(s)
        else name_key(s, ""))
    ref["tid"] = (pd.to_numeric(players[team_col], errors="coerce")
                  if team_col in players.columns
                  else pd.Series(np.nan, index=players.index))

    by_name: dict[str, list] = {}
    for pid, key, tid in zip(ref["PlayerId"], ref["name_key"], ref["tid"]):
        by_name.setdefault(key, []).append((int(pid), tid))
    aliases = {k.lower(): v for k, v in (aliases or {}).items()}

    out = feed.copy()
    resolved = []
    for key, tid in zip(out["name_key"], out.get(
            "feed_team_id", pd.Series(np.nan, index=out.index))):
        if key in aliases:
            stats["by_alias"] += 1
            stats["matched"] += 1
            resolved.append(float(aliases[key]))
            continue
        cands = by_name.get(key, [])
        if not cands:
            stats["unmatched"] += 1
            if len(stats["missed_names"]) < 12:
                stats["missed_names"].append(key)
            resolved.append(np.nan)
            continue
        if len(cands) == 1:
            pid, ptid = cands[0]
            if pd.notna(tid) and pd.notna(ptid) and int(tid) != int(ptid):
                stats["wrong_club"] += 1
            stats["matched"] += 1
            resolved.append(float(pid))
            continue
        same = [pid for pid, ptid in cands
                if pd.notna(tid) and pd.notna(ptid) and int(ptid) == int(tid)]
        if len(same) == 1:
            stats["matched"] += 1
            resolved.append(float(same[0]))
        else:
            stats["ambiguous"] += 1
            resolved.append(np.nan)
    out["PlayerId"] = resolved
    stats["label"] = label
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# Feed evidence -> role mixture
# ─────────────────────────────────────────────────────────────────────────────
#
# How confident a mixture is allowed to be, and why each number is what it is.
# These are the only free parameters in the module and they all mean the same
# thing: how much of the mixture's mass the primary role gets.
#
# AGREE is not 1.0 because two feeds agreeing that a man is the left fielder
# still leaves him a season to get hurt in, lose the job in, or be traded out
# of — and because a role mixture that is 100% one role is exactly the claim
# this module exists to stop making.
# A job nobody disputes. Reaching 1.0 is the point: a role mixture answers
# "which job does he hold", and nothing else. It used to stop at 0.85 on the
# grounds that two feeds agreeing about a left fielder "still leaves him a
# season to get hurt in, lose the job in, or be traded out of" — but getting
# hurt is `pt_availability`'s question now, and charging for it here as well
# is the same absence counted twice. That mistake has turned up three times
# in this model (the batting-order multiplier, the evidence factor, and this)
# and it is always the same shape: a second mechanism quietly re-spending a
# signal the first one already spent.
#
# What is left after health moves out is losing the job and being traded, and
# for a player whose own usage CONFIRMS the feeds there is no evidence of
# either. Aaron Judge is a full-time player, full stop; "85% full-time, 15%
# platoon bat" says he might platoon, which nothing in the record suggests.
SETTLED = 1.0
AGREE = 0.85          # both feeds say the same thing, nothing confirms it
# ...except that a lineup spot says how SAFE the job is, and the difference is
# not small. A real club's nine starters run from about 634 plate appearances
# down to about 342 — a factor of 1.85 — and the spot itself only explains 4.65
# against 3.97 a game. The rest is job security: a man batting third is in the
# lineup in September, and a man batting ninth is the one replaced when the
# club trades for a bat or calls somebody up. Giving all nine a 630 anchor and
# letting only the evidence factor separate them put the top five 6-9% above
# the real rank curve while ranks 9-16 ran light.
AGREE_BY_SPOT = {1: 0.90, 2: 0.90, 3: 0.90, 4: 0.85, 5: 0.80,
                 6: 0.75, 7: 0.68, 8: 0.62, 9: 0.58}
# The spot is spent HERE and only here. Spending it a second time as a
# per-game multiplier on the anchor — 4.65 plate appearances for the leadoff
# man against 3.97 for the ninth hitter — is the obvious thing to do and
# measurably makes the projection worse: with it the top five of each club sat
# 6.1% above the real rank curve and the curve's error was 0.061, without it
# 4.7% and 0.053. The per-game gap is 110 plate appearances over a season
# against a real gap of nearly 300, so it was never the main term anyway. One
# mechanism for one signal.
DISAGREE = 0.60       # they say different things; the batting order leads
# No second feed to check the depth chart against — and what that is worth
# differs by kind, because the two charts are claims about different things.
# For a hitter it means his club has no current batting order (seven do not),
# so 0.70 is just "one feed instead of two". For a pitcher, rank 1 in a
# rotation is a far weaker claim about INNINGS than rank 1 at a position is
# about plate appearances: every club has a nominal ace and only about
# eighteen pitchers in baseball clear 180 innings, because a rotation slot is
# governed by health rather than by status. Carrying the hitter's 0.70 over to
# pitchers projected 26 of them past 180.
DEPTH_ONLY = {"hitter": 0.70, "pitcher": 0.50}
ORDER_ONLY = 0.75     # in a lineup but absent from the depth chart


# RotoWire's own Stability rating on a club's top bullpen arm, read as what it
# plainly is: the feed's confidence in its own Rank 1 call. A "Very Low"
# closer keeping 0.92 of a closer's save share would be the model ignoring the
# one field in these files that is explicitly about doubt.
STABILITY = {"VERY HIGH": 0.92, "HIGH": 0.85, "MEDIUM": 0.70,
             "LOW": 0.58, "VERY LOW": 0.50}
STABILITY_UNSTATED = 0.80

# The ladder a role steps DOWN to when a mixture needs somewhere to put the
# remainder. Always the next-smaller job of the same kind, so the spread is
# "he might play less than this" rather than a jump to an unrelated role.
STEP_DOWN_HITTER = {
    "Full Time": "Strong Side Platoon",
    "Everyday DH / 1B-DH": "Bench Bat",
    "Strong Side Platoon": "Bench Bat",
    "Weak Side Platoon": "Bench Bat",
    "Catcher - Primary": "Catcher - Tandem",
    "Catcher - Tandem": "Catcher - Backup",
    "Catcher - Backup": "Bench Bat",
    "Utility IF": "Bench Bat",
    "Utility OF / 4th OF": "Bench Bat",
    "Bench Bat": "Injury Replacement / 26th Man",
    "Injury Replacement / 26th Man": "Depth (no MLB PA)",
}
STEP_DOWN_PITCHER = {
    "Ace (SP1)": "Mid-Rotation Starter (SP2-3)",
    "Mid-Rotation Starter (SP2-3)": "End-of-Rotation Starter (SP4-5)",
    "End-of-Rotation Starter (SP4-5)": "Swing Arm / Long Relief",
    "Innings-Limited Starter": "Swing Arm / Long Relief",
    "Swing Arm / Long Relief": "Middle Relief",
    "Opener": "Middle Relief",
    "Closer": "Late Inning RP (Setup)",
    "Late Inning RP (Setup)": "Middle Relief",
    "Middle Relief": "Bullpen Depth Arm",
    "Bullpen Depth Arm": "Depth (no MLB IP)",
    "Rehab / Injury Return": "Bullpen Depth Arm",
}

EVERYDAY_ROLES = {"Full Time", "Everyday DH / 1B-DH", "Catcher - Primary"}


def _is_everyday(role: str) -> bool:
    return str(role) in EVERYDAY_ROLES


def _confirms(record: str | None, role: str | None) -> bool:
    """Does a player's own usage rate back up the role a feed gave him?

    Matched on WHAT KIND of job it is, not on the name. `record_read` answers
    with three labels because a rate can tell an everyday player from a
    part-time one and cannot tell a designated hitter from a left fielder —
    so asking it for the exact string leaves the feed's more specific answer
    permanently unconfirmable. Mike Trout, whose depth chart says "Everyday
    DH / 1B-DH" and whose 4.46 plate appearances a game say "everyday", was
    left split between a full-time job and a bench job by that mismatch.
    """
    if not record or not role:
        return False
    if _is_everyday(record):
        return _is_everyday(role)
    return record == role


INFIELD = {"1B", "2B", "3B", "SS"}
OUTFIELD = {"LF", "CF", "RF", "OF"}

# What a rank WITHIN a position group is worth, past the rank-1 starter.
# "UTILITY" resolves to the infield or outfield utility role by position.
#
# Calibrated against the real within-club rank curve (`scripts/
# fit_role_anchors.real_rank_curve`), not chosen by eye — and the first guess
# was wrong in a way only measurement shows. Dropping to the 26th man at rank
# 4 and out of the league at rank 5 made the curve far too steep: the top five
# of each club came out 6-9% above the real ones and ranks 9 through 16 ran
# 4-25% light, because a club's genuine 200-plate-appearance bench players
# were being handed 90 or 1. The ladder below runs out two rungs later, which
# is what the real curve asks for.
DEPTH_RANK_ROLE = {
    2: "UTILITY",
    3: "UTILITY",
    4: "UTILITY",
    5: "Bench Bat",
    6: "Bench Bat",
    7: "Injury Replacement / 26th Man",
    8: "Depth (no MLB PA)",
}

# RotoWire's bullpen role vocabulary, mapped onto the taxonomy. A Committee is
# the one entry that is a mixture in the feed's own terms rather than in ours:
# "nobody has this job" is not a role, it is three men splitting one, and it
# is written here as what it is.
RW_BULLPEN = {
    "CLOSER": {"Closer": 1.0},
    "COMMITTEE": {"Closer": 0.40, "Late Inning RP (Setup)": 0.40,
                  "Middle Relief": 0.20},
    "CLOSER OF FUTURE": {"Late Inning RP (Setup)": 0.50, "Closer": 0.30,
                         "Middle Relief": 0.20},
    "SETUP MAN": {"Late Inning RP (Setup)": 1.0},
    "MIDDLE RELIEVER": {"Middle Relief": 1.0},
    "MULTI-INNING": {"Swing Arm / Long Relief": 0.55, "Middle Relief": 0.45},
}


def _mix(primary: str, weight: float, kind: str,
         second: str | None = None) -> dict[str, float]:
    """A primary role and where the rest of the belief goes."""
    ladder = STEP_DOWN_HITTER if kind == "hitter" else STEP_DOWN_PITCHER
    second = second or ladder.get(primary)
    weight = float(np.clip(weight, 0.0, 1.0))
    if not second or second == primary or weight >= 1.0:
        return {primary: 1.0}
    return {primary: weight, second: 1.0 - weight}


def _spread(a: str, wa: float, b: str, kind: str) -> dict[str, float]:
    """Two feeds, two answers. Neither is discarded."""
    if a == b:
        return _mix(a, AGREE, kind)
    return {a: wa, b: 1.0 - wa}


def _not_depth(role: str, kind: str) -> str:
    """The smallest REAL job, for a player the feeds want to bury but whose
    own record says he is a major leaguer."""
    if not is_depth_role(str(role)):
        return role
    return ("Injury Replacement / 26th Man" if kind == "hitter"
            else "Bullpen Depth Arm")


def _neighbour(role: str, kind: str) -> str | None:
    """Somewhere to put the rest of the belief, for a role at the bottom.

    The ladder steps DOWN, and the smallest real job has nowhere below it but
    the depth role — which is the one thing a mixture must not acquire by
    accident, since it routes to the 1-PA floor. So the last rung steps UP
    instead: a 26th man might play a bit more than a 26th man, which is a
    truer thing to say than "he is exactly the 26th man" at probability one.
    """
    ladder = STEP_DOWN_HITTER if kind == "hitter" else STEP_DOWN_PITCHER
    down = ladder.get(role)
    if down and not is_depth_role(down):
        return down
    ups = [a for a, b in ladder.items() if b == role and not is_depth_role(a)]
    return ups[0] if ups else None


def role_volume(role: str, kind: str) -> float:
    """A role's own anchor, for ordering jobs by size."""
    a = role_anchor(str(role), kind)
    if not a:
        return 0.0
    return float(a["pa" if kind == "hitter" else "ip"])


def hitter_role_from_feeds(ev: dict) -> tuple[dict[str, float], str] | None:
    """What the feeds say a position player's job is.

    Returns (mixture, source) or None when no feed covers him, in which case
    the heuristic default stands. The two reads are deliberately computed
    independently and only then compared, so that `_spread` is measuring a
    real disagreement rather than one read contaminated by the other.
    """
    lineup, from_absence = _lineup_read(ev)
    depth = _depth_read(ev)
    record = record_read(ev.get("play_rate"), "hitter")

    # A prospect one level away outranks a depth chart that has him in the
    # farm system, because that is where the depth chart is supposed to have
    # him — being in Triple-A in February is not a statement that he will
    # not play in July. Only where no MLB feed has PLACED him: a man the
    # depth chart lists at a position, or who appears in a batting order, is
    # up, and what the prospect list thought is out of date.
    #
    # Being left out of a nine-man lineup does not count as being placed,
    # which is the whole reason this needs `from_absence`. Twenty-three
    # clubs have a current order, so without it every prospect on one of
    # them was read as "a bench bat who did not make today's lineup" and the
    # arrival never fired at all.
    if (not lineup or from_absence) and (depth is None
                                         or is_depth_role(depth)):
        got = prospect_read(ev)
        if got:
            return got[0], "feed:prospect"

    # A player's OWN usage rate outranks a depth chart that has written him
    # off. Byron Buxton takes 4.30 plate appearances per game played, above
    # the median Full Time hitter's 3.97 — when he is in the lineup he is an
    # everyday centre fielder. RotoWire has him eighth among Minnesota's
    # centre fielders because he is hurt, which the feeds read as a 99-PA
    # bench job, and he projected NINE plate appearances for the season.
    #
    # Being hurt is not a job. Where the record says everyday and a feed says
    # bench, the disagreement is about HEALTH, and health belongs in
    # `pt_availability`, which docks him separately. So the record wins the
    # role OUTRIGHT, and the width that used to be kept here is gone.
    #
    # Keeping it was the double-count again. A quarter of Buxton on a 99-PA
    # role cost him 135 plate appearances on top of an availability dock that
    # had already charged him for the same injury — and it spread him onto
    # "Injury Replacement / 26th Man", which is a statement about health
    # wearing a role's clothes.
    #
    # What this gives up is the other reading of a buried regular: that his
    # club has genuinely moved on rather than waiting for him to get well.
    # That reading is real and this cannot see it, so every player the record
    # overrules is NAMED in the run log, where a person can check him against
    # the roster and write an override if the depth chart was right.
    if record and _is_everyday(record) and not _is_everyday(
            depth or lineup or ""):
        return _mix(record, SETTLED, "hitter"), "feed:record over depth"

    if lineup and depth:
        # Being left out of a nine-man lineup is a CAP, not a job. Read as a
        # positive claim it promoted every prospect in the organisation to a
        # bench bat — Boston's Jake Schaffner, Franklin Arias and Brooks
        # Brannon all came out "Bench Bat 0.6", taking plate appearances from
        # players who will actually bat. So where the depth chart already has
        # a man at or below a bench job, the two feeds AGREE that he is not a
        # regular, and the depth chart's more specific answer is the one kept.
        if from_absence and role_volume(depth, "hitter") <= role_volume(
                lineup, "hitter"):
            return _mix(depth, AGREE, "hitter"), "feed:order+depth"
        if lineup == depth:
            if _confirms(record, lineup):
                return _mix(lineup, SETTLED, "hitter"), "feed:order+depth+record"
            return _mix(lineup, _agree_for(ev), "hitter"), "feed:order+depth"
        return _spread(lineup, DISAGREE, depth, "hitter"), "feed:order+depth"
    if depth:
        # The record agreeing with the only feed that covers him is a second
        # source, and settles a role the depth chart alone could not. Judge
        # is the case: the Yankees have no current batting order, so he came
        # out "Full Time 0.70 / Strong Side Platoon 0.30" — a sentence that
        # says he might be a platoon bat. His own 4.50 plate appearances a
        # game say he is not.
        if _confirms(record, depth):
            return _mix(depth, SETTLED, "hitter"), "feed:depth+record"
        return _mix(depth, DEPTH_ONLY["hitter"], "hitter"), "feed:depth"
    if lineup and not from_absence:
        return _mix(lineup, ORDER_ONLY, "hitter"), "feed:order"
    return None


def _agree_for(ev: dict) -> float:
    """How settled a job is, when both feeds say a man has it.

    `AGREE_BY_SPOT` where the batting order places him, and the flat `AGREE`
    where it does not. The remainder goes down the role ladder, so a ninth
    hitter is 0.58 a regular and 0.42 a bench bat — which is a truer account
    of a ninth hitter's season than 630 plate appearances is.
    """
    spots = [s for s in (ev.get("spot_vs_r"), ev.get("spot_vs_l"))
             if s and np.isfinite(s)]
    if not spots:
        return AGREE
    spot = int(np.clip(round(float(np.mean(spots))), 1, 9))
    return AGREE_BY_SPOT.get(spot, AGREE)


def _lineup_read(ev: dict) -> tuple[str | None, bool]:
    """The batting orders alone, and whether the answer came from ABSENCE.

    Absence is evidence only where the club has a current order to be absent
    from — seven clubs have none, and a player on one of them is unobserved,
    not benched — and even then it is weaker evidence than presence, which is
    what the second return value is for.
    """
    if not ev.get("club_has_orders"):
        return None, False
    in_r, in_l = ev.get("spot_vs_r"), ev.get("spot_vs_l")
    pos = (ev.get("depth_pos") or ev.get("feed_position") or "").upper()
    if in_r and in_l:
        if pos == "C":
            return "Catcher - Primary", False
        return ("Everyday DH / 1B-DH" if pos == "DH" else "Full Time"), False
    if in_r:
        return "Strong Side Platoon", False
    if in_l:
        return "Weak Side Platoon", False
    return "Bench Bat", True


def prospect_read(ev: dict, target_year: int = 2027
                  ) -> tuple[dict[str, float], str] | None:
    """The role and arrival of a player who is not up yet but is close.

    None for anyone the table does not reach, which leaves the depth chart's
    answer — usually the depth role — exactly as it was.
    """
    level = str(ev.get("level") or "").upper()
    bands = PROSPECT_ARRIVAL.get(level)
    if not bands:
        return None
    rank = ev.get("prospect_rank")
    rank = float(rank) if rank is not None and np.isfinite(
        float(rank or np.nan)) else 10_000.0
    born = ev.get("birth_year")
    if born is not None and np.isfinite(float(born or np.nan)):
        if target_year - float(born) > PROSPECT_MAX_AGE:
            return None
    for cutoff, timing, mix in bands:
        if rank <= cutoff:
            return _scaled(mix), timing
    return None


def pitcher_prospect_read(ev: dict, target_year: int = 2027
                          ) -> tuple[dict[str, float], str] | None:
    """The role and arrival of an arm who is not up yet but is close.

    Starters only, keyed on the pitcher's OWN record, because the prospects
    feed calls every arm in it "P" and a rookie rotation piece takes eight
    times the innings of a shuttle reliever. See PITCHER_ARRIVAL for why the
    relievers are left at the floor.
    """
    level = str(ev.get("level") or "").upper()
    bands = PITCHER_ARRIVAL.get(level)
    if not bands:
        return None
    if str(ev.get("own_role") or "").strip().lower() != PITCHER_ARRIVAL_ROLE:
        return None
    rank = ev.get("prospect_rank")
    rank = float(rank) if rank is not None and np.isfinite(
        float(rank or np.nan)) else 10_000.0
    born = ev.get("birth_year")
    if born is not None and np.isfinite(float(born or np.nan)):
        if target_year - float(born) > PROSPECT_MAX_AGE:
            return None
    for cutoff, timing, mix in bands:
        if rank <= cutoff:
            return _scaled(mix), timing
    return None


def _scaled(mix: dict[str, float]) -> dict[str, float]:
    """Spend PROSPECT_VOLUME_SCALE of the arrival, the rest as bench time.

    Expressed by moving weight onto the smallest real job rather than by
    scaling the anchor, so the result is still a mixture over the taxonomy
    and still sums to one.
    """
    k = float(np.clip(PROSPECT_VOLUME_SCALE, 0.0, 1.0))
    if k >= 1.0:
        return dict(mix)
    spare = ("Bullpen Depth Arm" if "Bullpen Depth Arm" in mix
             else "Injury Replacement / 26th Man")
    out = {r: w * k for r, w in mix.items()}
    out[spare] = out.get(spare, 0.0) + (1.0 - k)
    return out


def _depth_read(ev: dict) -> str | None:
    """The depth chart alone: a rank within one position group."""
    pos = (ev.get("depth_pos") or "").upper()
    rank = ev.get("depth_rank")
    if not pos or rank is None or not np.isfinite(rank):
        return None
    if pos == "PROS":
        return "Depth (no MLB PA)"
    rank = int(rank)
    if pos == "C":
        return {1: "Catcher - Primary", 2: "Catcher - Backup"}.get(
            rank, "Depth (no MLB PA)" if rank > 3 else "Bench Bat")
    if rank == 1:
        return "Everyday DH / 1B-DH" if pos == "DH" else "Full Time"
    role = DEPTH_RANK_ROLE.get(rank, DEPTH_RANK_ROLE[max(DEPTH_RANK_ROLE)])
    if role == "UTILITY":
        if pos in INFIELD:
            return "Utility IF"
        return "Utility OF / 4th OF" if pos in OUTFIELD else "Bench Bat"
    return role


def pitcher_role_from_feeds(ev: dict) -> tuple[dict[str, float], str] | None:
    """What the feeds say a pitcher's job is.

    The bullpen feed wins where it speaks, because it is the more specific of
    the two: the depth chart knows a man is in the bullpen, the closers feed
    knows which bullpen job he has and how sure it is. Where both speak and
    disagree — a depth-chart starter who is also in the bullpen pecking order
    — that is a swing arm, and the disagreement is carried rather than
    resolved.
    """
    pen = _bullpen_read(ev)
    rot = _rotation_read(ev)

    # An arm one level away outranks a depth chart that has him in the farm
    # system, because that is where the depth chart is supposed to have him.
    # Only where NEITHER major-league feed has placed him: a pitcher in the
    # rotation's order or in the bullpen's pecking order is up, and what the
    # prospect list thought is out of date.
    #
    # No `from_absence` counterpart here, and none is needed. The hitters'
    # version has one because a batting order is nine names and everyone else
    # on the club reads as absent from it; a club's pitching staff is not
    # published as a list of who is unavailable today.
    if pen is None and (rot is None or is_depth_role(rot)):
        got = pitcher_prospect_read(ev)
        if got:
            return got[0], "feed:prospect"

    if pen and rot:
        mix, w = pen
        heavy = max(mix, key=mix.get)
        return _spread(heavy, DISAGREE, rot, "pitcher"), "feed:pen+depth"
    if pen:
        mix, w = pen
        if len(mix) > 1:            # a committee is already a mixture
            return mix, "feed:closers"
        heavy = max(mix, key=mix.get)
        return _mix(heavy, w, "pitcher"), "feed:closers"
    if rot:
        return _mix(rot, DEPTH_ONLY["pitcher"], "pitcher"), "feed:depth"
    return None


def _bullpen_read(ev: dict) -> tuple[dict[str, float], float] | None:
    """The closers feed: a role label, and the feed's confidence in it."""
    label = str(ev.get("rw_role") or "").strip().upper()
    base = RW_BULLPEN.get(label)
    if base is None:
        return None
    stab = str(ev.get("stability") or "").strip().upper()
    if label == "CLOSER":
        return base, STABILITY.get(stab, STABILITY_UNSTATED)
    # Stability is only published for a club's top arm, so everyone else is
    # held at the unstated default rather than at a confidence nobody stated.
    return base, STABILITY_UNSTATED


def _rotation_read(ev: dict) -> str | None:
    """The depth chart's pitching groups: P is the rotation in order, BP the
    bullpen in order, CL the closer, PROS the farm."""
    pos = str(ev.get("depth_pos") or "").upper()
    rank = ev.get("depth_rank")
    if not pos or rank is None or not np.isfinite(rank):
        return None
    rank = int(rank)
    if pos == "PROS":
        return "Depth (no MLB IP)"
    if pos == "CL":
        return "Closer"
    if pos == "BP":
        if rank <= 2:
            return "Late Inning RP (Setup)"
        return "Middle Relief" if rank <= 5 else "Bullpen Depth Arm"
    if pos != "P":
        return None
    if rank == 1:
        return "Ace (SP1)"
    if rank <= 3:
        return "Mid-Rotation Starter (SP2-3)"
    if rank <= 5:
        return "End-of-Rotation Starter (SP4-5)"
    return "Swing Arm / Long Relief" if rank <= 7 else "Depth (no MLB IP)"


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

HITTER_GROUPS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}
PITCHER_GROUPS = ("CL", "P", "BP")       # in order of how specific they are


def _best_depth_row(rows: pd.DataFrame, kind: str) -> dict:
    """One depth-chart read per player, from however many groups he is in.

    A player listed at three positions is a utility player, and the position
    that describes his job is the one he is highest on — so the lowest rank
    wins. For pitchers the GROUP is more informative than the rank: being the
    club's closer says more than being its sixth-ranked bullpen arm, so CL is
    preferred to P and P to BP regardless of the numbers.
    """
    if rows.empty:
        return {}
    if kind == "hitter":
        real = rows[rows["depth_pos"].isin(HITTER_GROUPS)]
        pick = real if not real.empty else rows
        row = pick.loc[pd.to_numeric(pick["depth_rank"],
                                     errors="coerce").idxmin()]
    else:
        row = None
        for group in PITCHER_GROUPS:
            sub = rows[rows["depth_pos"] == group]
            if not sub.empty:
                row = sub.loc[pd.to_numeric(sub["depth_rank"],
                                            errors="coerce").idxmin()]
                break
        if row is None:
            row = rows.iloc[0]
    return {"depth_pos": row["depth_pos"],
            "depth_rank": pd.to_numeric(row["depth_rank"], errors="coerce"),
            "n_depth_groups": int(rows["depth_pos"].nunique())}


def _for_pool(feeds: dict[str, pd.DataFrame], name: str,
              kind: str) -> pd.DataFrame:
    """The rows of a feed that describe THIS pool, before any id resolution.

    Hitters and pitchers are projected separately, so measuring the closers
    feed against the hitter pool reports a 3% match rate for a file that is
    not about hitters at all — a number that looks like a broken join and is
    really a category error. Each feed is cut to its pool first, so the rates
    the report prints are rates of the thing that was actually attempted.

    The batting orders are also cut to the CURRENT nine-spot lineups here
    rather than later, for the same reason: carrying the legacy orders through
    resolution reported 232 players "on a different club than projected" when
    the current orders disagree about eleven.
    """
    df = feeds.get(name)
    if df is None or df.empty:
        return pd.DataFrame()
    is_hitter = kind == "hitter"
    if name == "orders":
        if not is_hitter:
            return pd.DataFrame()
        df = df[~df["legacy"].astype(bool)]
        return df[df["feed_position"].str.upper() != "P"]
    if name == "closers":
        return pd.DataFrame() if is_hitter else df
    if name == "depth":
        groups = HITTER_GROUPS if is_hitter else set(PITCHER_GROUPS)
        return df[df["depth_pos"].isin(groups | {"PROS"})]
    if name == "prospects":
        p = df["feed_position"].str.upper()
        return df[p.ne("P") if is_hitter else p.eq("P")]
    return df


def build_evidence(players: pd.DataFrame, kind: str,
                   feeds: dict[str, pd.DataFrame], *,
                   aliases: dict[str, int] | None = None,
                   team_col: str = "Pred_target_team_id",
                   ) -> tuple[dict[int, dict], dict]:
    """Everything the four feeds say about each player, keyed by PlayerId."""
    stats: dict = {"resolve": {}}
    ev: dict[int, dict] = {}

    def resolved(name):
        df = _for_pool(feeds, name, kind)
        if df.empty:
            return pd.DataFrame()
        out, s = resolve_ids(df, players, aliases=aliases, team_col=team_col,
                             label=name)
        stats["resolve"][name] = s
        return out[out["PlayerId"].notna()]

    orders = resolved("orders")
    depth = resolved("depth")
    closers = resolved("closers")
    prospects = resolved("prospects")

    # Which clubs have a CURRENT batting order at all. Seven do not, and a
    # player on one of them must not be read as "left out of the lineup".
    with_orders: set[int] = set()
    if not orders.empty:
        cur = orders
        with_orders = {int(t) for t in
                       pd.to_numeric(cur["feed_team_id"],
                                     errors="coerce").dropna().unique()}
        for pid, grp in cur.groupby(cur["PlayerId"].astype(int)):
            spots = {}
            for hand, sub in grp.groupby("vs_hand"):
                s = pd.to_numeric(sub["spot"], errors="coerce").dropna()
                if len(s):
                    spots[str(hand).upper()] = float(s.min())
            ev.setdefault(pid, {}).update(
                spot_vs_r=spots.get("R"), spot_vs_l=spots.get("L"),
                feed_position=grp.iloc[0].get("feed_position"))

    if not depth.empty:
        for pid, grp in depth.groupby(depth["PlayerId"].astype(int)):
            ev.setdefault(pid, {}).update(_best_depth_row(grp, kind))

    if not closers.empty:
        best = closers.sort_values("bullpen_rank").groupby(
            closers["PlayerId"].astype(int)).first()
        for pid, row in best.iterrows():
            ev.setdefault(int(pid), {}).update(
                rw_role=row["rw_role"], stability=row["stability"],
                bullpen_rank=row["bullpen_rank"])

    if not prospects.empty:
        best = prospects.sort_values("prospect_rank").groupby(
            prospects["PlayerId"].astype(int)).first()
        for pid, row in best.iterrows():
            ev.setdefault(int(pid), {}).update(
                level=row["level"], prospect_rank=row["prospect_rank"],
                birth_year=row.get("birth_year"))

    # Starter or reliever, from the pitcher's own game log. Like the usage
    # rate below it this is a source about HIM rather than about his club's
    # plans, and it is the only thing that tells a rookie rotation arm apart
    # from a shuttle reliever — the prospects feed lists both as "P".
    if kind == "pitcher" and "role" in players.columns:
        for pid, r in zip(pd.to_numeric(players["PlayerId"], errors="coerce"),
                          players["role"].astype(str)):
            if pd.notna(pid) and r and r.lower() != "nan":
                ev.setdefault(int(pid), {})["own_role"] = r.strip().lower()

    # The player's own usage rate, which is a source like any other and the
    # only one that is about HIM rather than about his club's plans.
    if "pt_play_rate" in players.columns:
        rate = pd.to_numeric(players["pt_play_rate"], errors="coerce")
        for pid, r in zip(pd.to_numeric(players["PlayerId"], errors="coerce"),
                          rate):
            if pd.notna(pid) and pd.notna(r):
                ev.setdefault(int(pid), {})["play_rate"] = float(r)

    tids = (pd.to_numeric(players[team_col], errors="coerce")
            if team_col in players.columns
            else pd.Series(np.nan, index=players.index))
    club = {int(p): (int(t) if pd.notna(t) else None)
            for p, t in zip(players["PlayerId"], tids)}
    for pid, e in ev.items():
        t = club.get(pid)
        e["club_has_orders"] = t is not None and t in with_orders

    stats["clubs_with_orders"] = len(with_orders)
    stats["players_with_evidence"] = len(ev)
    return ev, stats


def _timing(e: dict, mix: dict[str, float], kind: str,
            source: str = "") -> str | None:
    """When a player arrives, which is what a prospect's LEVEL is about.

    Only ever applied to someone the other feeds do NOT already have in a
    major-league job. A top prospect who is also his club's rank-1 shortstop
    is not a July callup, he is the shortstop, and the depth chart is the
    better-informed feed about that.
    """
    # An arrival the table has already decided: use its own answer rather
    # than the level's generic one, since the table's timing is what its
    # mixture was calibrated against.
    if source == "feed:prospect":
        got = (pitcher_prospect_read(e) if kind == "pitcher"
               else prospect_read(e))
        if got:
            return got[1]
    level = str(e.get("level") or "").upper()
    if not level:
        return None
    heavy = max(mix, key=mix.get) if mix else ""
    established = heavy in {
        "Full Time", "Everyday DH / 1B-DH", "Strong Side Platoon",
        "Catcher - Primary", "Catcher - Tandem",
        "Ace (SP1)", "Mid-Rotation Starter (SP2-3)",
        "End-of-Rotation Starter (SP4-5)", "Closer",
        "Late Inning RP (Setup)"}
    if established:
        return None
    # A DEPTH ROLE CANNOT CARRY A CALLUP. The level alone used to decide the
    # timing, so a Double-A bat the arrival table does not reach came out as
    # "Depth (no MLB PA), Mid Season (~July)" — the engine saying it expects
    # him in July and expects him to do nothing when he gets there. 169
    # hitters and 58 pitchers read that way, Andrew Fischer among them.
    #
    # The two halves have to agree, and which half gives is decided by the
    # arrival table: a player it reaches gets a role AND a timing, a player
    # it does not gets neither. LEVEL_TIMING is a statement about a LEVEL,
    # and "players at Double-A tend to arrive in July" is not a claim that
    # THIS Double-A player arrives at all.
    if is_depth_role(heavy):
        return None
    return LEVEL_TIMING.get(level)


TARGET_YEAR = 2027


def prospect_birth_years(feed_dir: str | Path = FEED_DIR) -> dict[int, int]:
    """MLBAM id -> birth year, for everyone the prospects feed names.

    Pulled out on its own because the MLE stage needs it long before any of
    the role machinery runs: a translated minor leaguer's age decides how the
    aging curve treats him, and Chadwick misses most players who have never
    reached the majors. Returns {} when the feed is not on disk, so the
    pipeline behaves exactly as it did before where there are no feeds.
    """
    try:
        pr = read_prospects(Path(feed_dir) / "prospects.xml")
    except Exception:
        return {}
    if pr.empty or "birth_year" not in pr.columns:
        return {}
    pid = pd.to_numeric(pr["PlayerId"], errors="coerce")
    born = pd.to_numeric(pr["birth_year"], errors="coerce")
    ok = pid.notna() & born.notna()
    return {int(p): int(b) for p, b in zip(pid[ok], born[ok])}


def repair_ages(players: pd.DataFrame, ev: dict[int, dict], *,
                target_year: int = TARGET_YEAR) -> tuple[pd.DataFrame, int]:
    """Replace a defaulted age with the one the prospects feed states.

    Two thirds of the players in the projection carry an age of exactly 25,
    and it is not a coincidence: a player with no major-league history gets
    his age from `mle_translations._age_from_chadwick`, which falls back to
    MLE_DEFAULT_AGE (24) for the translated season whenever the Chadwick
    lookup misses, and the rate models then age him forward one year to the
    target. So "25" on a minor leaguer means "we never found out", and it is
    indistinguishable in the sheet from a real 25 — Andrew Fischer read 25
    while the feed on disk gave his birth date as 2004.

    The prospects feed states a Birth Date for all 400 players in it, which
    `read_prospects` already parses because the arrival table needs it to
    tell a 21-year-old at Triple-A from a 27-year-old at Triple-A. The same
    number answers the age column, so it is used here rather than left to a
    lookup that has already failed.

    Only ever REPLACES a default. A player whose age came from his own
    major-league record keeps it, because that record is the better source
    and the feed's birth date is one more name join that can go wrong.
    """
    if "Age" not in players.columns or not ev:
        return players, 0
    out = players.copy()
    age = pd.to_numeric(out["Age"], errors="coerce")
    # The defaulted value, derived rather than hard-coded, so it follows the
    # config if either constant ever moves: the translated row is written for
    # (target - offset) at MLE_DEFAULT_AGE, and `rate_models` ages it forward
    # by the gap to the target year.
    try:
        from pipeline_config import MLE_DEFAULT_AGE, MLE_SEASON_OFFSET
    except ImportError:
        return out, 0
    defaulted = float(MLE_DEFAULT_AGE) + float(MLE_SEASON_OFFSET)

    fixed = 0
    for i, pid in enumerate(pd.to_numeric(out["PlayerId"], errors="coerce")):
        if pd.isna(pid) or age.iloc[i] != defaulted:
            continue
        born = (ev.get(int(pid)) or {}).get("birth_year")
        if born is None or not np.isfinite(float(born or np.nan)):
            continue
        real = target_year - int(born)
        if real == int(defaulted):
            continue
        out.iloc[i, out.columns.get_loc("Age")] = real
        fixed += 1
    return out, fixed


def apply_feed_roles(players: pd.DataFrame, kind: str, *,
                     feed_dir: str | Path = FEED_DIR,
                     alias_path: str | Path | None = None,
                     feeds: dict[str, pd.DataFrame] | None = None,
                     team_col: str = "Pred_target_team_id",
                     ) -> tuple[pd.DataFrame, dict]:
    """Replace heuristic roles with what the usage feeds say, as MIXTURES.

    Sits between `assign_default_roles` and `apply_role_overrides`: better
    than a guess from last season's plate appearances, and still beaten by a
    human who has typed a row in the override file. A player no feed covers
    keeps the heuristic default, so this never makes coverage worse.

    Writes `pt_role_mix` (which `raw_volumes` already blends into one anchor),
    `pt_role` (the modal role, for anything that reads a single name),
    `pt_role_start` and `pt_role_source`.
    """
    stats: dict = {"applied": 0, "sources": {}, "kind": kind}
    out = players.copy()
    if feeds is None:
        feeds = read_feeds(feed_dir)
    if all(df is None or df.empty for df in feeds.values()):
        stats["note"] = f"no feeds found in {feed_dir}"
        return out, stats

    aliases = load_aliases(alias_path) if alias_path else {}
    ev, estats = build_evidence(out, kind, feeds, aliases=aliases,
                                team_col=team_col)
    stats.update(estats)
    stats["aliases"] = len(aliases)

    if "pt_role_mix" not in out.columns:
        out["pt_role_mix"] = ""

    out, stats["ages_fixed"] = repair_ages(out, ev, target_year=TARGET_YEAR)

    reader = hitter_role_from_feeds if kind == "hitter" \
        else pitcher_role_from_feeds
    known = set(role_names(kind))
    # Whether the heuristic — which reads the player's own MLB volume — had
    # already placed him outside the majors. See the guard below.
    was_depth = out["pt_role"].astype(str).map(is_depth_role).to_numpy()
    mixes, roles, starts, sources = [], [], [], []
    for i, pid in enumerate(pd.to_numeric(out["PlayerId"], errors="coerce")):
        e = ev.get(int(pid)) if pd.notna(pid) else None
        got = reader(e) if e else None
        if not got:
            mixes.append(out["pt_role_mix"].iloc[i])
            roles.append(out["pt_role"].iloc[i])
            starts.append(out["pt_role_start"].iloc[i])
            sources.append(out["pt_role_source"].iloc[i])
            continue
        mix, source = got
        # A role the taxonomy does not know would be silently dropped by
        # parse_role_mix and quietly become a depth player, so it is caught
        # here instead, where the name is still attached to a reason.
        bad = [r for r in mix if r not in known]
        if bad:
            stats.setdefault("unknown_roles", set()).update(bad)
            mix = {r: w for r, w in mix.items() if r in known}
        # A feed may say how MUCH a player plays. Whether he is a major
        # leaguer at all is his own record's to answer, and the heuristic
        # default already read it — so a depth chart that has buried a real
        # player demotes him to the last man on the roster, not out of the
        # league. The distinction is not cosmetic: the depth role routes
        # straight to the 1-PA floor and outside the club's budget, so the
        # feeds could silently delete a hitter's season. Boston has Triston
        # Casas eighth among its first basemen; that is a bench job, not a
        # retirement.
        if not was_depth[i]:
            lifted = {}
            for r, w in mix.items():
                r = _not_depth(r, kind)
                lifted[r] = lifted.get(r, 0.0) + w
            collapsed = len(lifted) < len(mix)
            if collapsed:
                stats["un_floored"] = stats.get("un_floored", 0) + 1
                # Merging two roles onto one leaves a mixture of 1.0 that
                # nothing asserted — an artefact of the lift, not a judgement
                # — so it is spread again. Only then: a mixture that arrived
                # here already settled was MEANT to be 1.0, and re-spreading
                # it put Judge back on a platoon role he had just been taken
                # off.
                if len(lifted) == 1:
                    only = next(iter(lifted))
                    nb = _neighbour(only, kind)
                    lifted = {only: AGREE, nb: 1 - AGREE} if nb else lifted
            mix = lifted
        if not mix:
            mixes.append(out["pt_role_mix"].iloc[i])
            roles.append(out["pt_role"].iloc[i])
            starts.append(out["pt_role_start"].iloc[i])
            sources.append(out["pt_role_source"].iloc[i])
            continue
        total = sum(mix.values())
        mix = {r: w / total for r, w in mix.items()}
        mixes.append(format_role_mix(mix))
        roles.append(max(mix, key=mix.get))
        starts.append(_timing(e, mix, kind, source)
                      or out["pt_role_start"].iloc[i])
        sources.append(source)
        stats["applied"] += 1
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        # Named, not just counted. These are the players whose own usage
        # overruled a depth chart that had them buried, and the one reading
        # this cannot distinguish — a club that has moved on rather than one
        # waiting for a man to get well — is checkable only by a person.
        if source.endswith("over depth"):
            stats.setdefault("overruled", []).append(
                str(out["Name"].iloc[i]) if "Name" in out.columns else "?")

    out["pt_role_mix"] = mixes
    out["pt_role"] = roles
    out["pt_role_start"] = starts
    out["pt_role_source"] = sources
    return out, stats


def feed_report(stats: dict) -> str:
    """What the feeds covered, and what they could not be joined to."""
    if not stats or stats.get("note"):
        return f"  role feeds: {stats.get('note', 'not run')}"
    head = (f"  role feeds ({stats.get('kind', '?')}): "
            f"{stats.get('applied', 0):,} players roled from feeds")
    if stats.get("kind") == "hitter":
        head += (f", {stats.get('clubs_with_orders', 0)} clubs with a current "
                 f"batting order")
    lines = [head]
    for name, s in (stats.get("resolve") or {}).items():
        extra = ""
        if s.get("ambiguous"):
            extra += f", {s['ambiguous']} AMBIGUOUS (dropped)"
        if s.get("by_alias"):
            extra += f", {s['by_alias']} by alias"
        if s.get("wrong_club"):
            extra += f", {s['wrong_club']} on a different club than projected"
        rate = s["matched"] / s["rows"] if s.get("rows") else 0.0
        lines.append(f"    {name:10s} {s['matched']:5d}/{s['rows']:<5d} "
                     f"resolved ({rate:.1%}){extra}")
        if s.get("missed_names"):
            lines.append(f"      unmatched e.g. "
                         f"{', '.join(s['missed_names'][:6])}")
    if stats.get("sources"):
        lines.append("    by evidence: " + ", ".join(
            f"{k} {v}" for k, v in sorted(stats["sources"].items(),
                                          key=lambda kv: -kv[1])))
    if stats.get("overruled"):
        who = stats["overruled"]
        lines.append(f"    {len(who)} player(s) kept an everyday role against "
                     f"a depth chart that buried them, on their own usage "
                     f"rate; the dock is in pt_availability. Check these "
                     f"against the roster:")
        lines.append(f"      {', '.join(who[:10])}"
                     + (f" (+{len(who) - 10} more)" if len(who) > 10 else ""))
    if stats.get("unknown_roles"):
        lines.append(f"    WARNING: roles not in the taxonomy were produced "
                     f"and dropped: {sorted(stats['unknown_roles'])}")
    return "\n".join(lines)

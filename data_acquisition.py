"""
data_acquisition.py
===================
Fetches the raw data the pipeline needs:

1. K%/BB%/HBP%/SF% rate tables from statsapi.mlb.com (hitters and pitchers,
   one row per player-season).
2. The current target-year minus one season of Statcast BIP data from
   baseballsavant (the equivalent of scrape_statcast_savant in the R script).
3. Sprint speed leaderboards from baseballsavant (player_id → speed for each
   year).

Cached fetches are reused on subsequent runs.
"""

from __future__ import annotations

import io
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from pipeline_config import (
    CACHE_DIR, STATSAPI_TIMEOUT, SAVANT_TIMEOUT, SAVANT_DAYS_PER_CHUNK,
    RATE_HIST_START,
)

# statsapi.mlb.com and baseballsavant return 403 to the default python-requests
# User-Agent on some networks; present a browser-like UA for every request.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36"}


# ─────────────────────────────────────────────────────────────────────────────
# statsapi.mlb.com — counting stats for K%/BB%/HBP%/SF%
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_statsapi_one(year: int, group: str) -> pd.DataFrame:
    """Pull one season of hitting (or pitching) stats from statsapi.

    Group is 'hitting' or 'pitching'. Returns a flat dataframe with one row
    per player who appeared that season.
    """
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats":      "season",
        "season":     year,
        "group":      group,
        "playerPool": "ALL",
        "sportId":    1,
        "limit":      5000,
    }
    r = requests.get(url, params=params, timeout=STATSAPI_TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if not data.get("stats"):
        return pd.DataFrame()
    splits = data["stats"][0].get("splits", [])
    rows = []
    for s in splits:
        p = s.get("player", {})
        st = s.get("stat", {})
        team = s.get("team", {})
        rows.append({
            "Season":   year,
            "PlayerId": p.get("id"),
            "Name":     p.get("fullName"),
            "Team":     team.get("abbreviation") or team.get("name", "")[:3],
            "TeamId":   team.get("id"),
            "Age":      st.get("age"),
            "PA":       st.get("plateAppearances", 0),
            "AB":       st.get("atBats", 0),
            "K":        st.get("strikeOuts", 0),
            "BB":       st.get("baseOnBalls", 0),
            "IBB":      st.get("intentionalWalks", 0),
            "HBP":      st.get("hitByPitch", 0),
            "SF":       st.get("sacFlies", 0),
            "SH":       st.get("sacBunts", 0),
            "H":        st.get("hits", 0),
            "HR":       st.get("homeRuns", 0),
            "2B":       st.get("doubles", 0),
            "3B":       st.get("triples", 0),
            "SB":       st.get("stolenBases", 0),
            "CS":       st.get("caughtStealing", 0),
            "R":        st.get("runs", 0),
            "RBI":      st.get("rbi", 0),
            "G":        st.get("gamesPlayed", 0),
            "TBF":      st.get("battersFaced", 0),    # pitching-only
            "IP_str":   st.get("inningsPitched"),     # pitching-only; "X.Y" notation
            "WP":       st.get("wildPitches", 0),     # pitching-only
            "BK":       st.get("balks", 0),           # pitching-only
            "ER":       st.get("earnedRuns", 0),      # pitching-only
            "RA":       st.get("runs", 0) if group == "pitching" else 0,  # runs allowed (pitching)
        })
    df = pd.DataFrame(rows)
    if not df.empty and "IP_str" in df.columns:
        # Convert MLB's "X.Y" notation (Y = 0/1/2 outs past X innings) to a
        # real float. E.g., "187.2" → 187 + 2/3 = 187.667.
        def _ip_to_float(s):
            if s is None or (isinstance(s, float) and pd.isna(s)):
                return 0.0
            try:
                ip_str = str(s)
                if "." in ip_str:
                    whole, frac = ip_str.split(".", 1)
                    outs = int(frac[:1]) if frac else 0
                    return float(whole) + outs / 3.0
                return float(ip_str)
            except Exception:
                return 0.0
        df["IP"] = df["IP_str"].apply(_ip_to_float)
    return df


def fetch_rate_data(target_year: int, start_year: int = RATE_HIST_START,
                    force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hitting_df, pitching_df) for [start_year, target_year-1].

    Always pulls fresh for the most recent year (it may still be in-progress).
    Caches older years.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hit_path = CACHE_DIR / f"statsapi_hitting_{start_year}_to_{target_year - 1}.parquet"
    pit_path = CACHE_DIR / f"statsapi_pitching_{start_year}_to_{target_year - 1}.parquet"

    if hit_path.exists() and pit_path.exists() and not force:
        return pd.read_parquet(hit_path), pd.read_parquet(pit_path)

    hit_rows, pit_rows = [], []
    for y in range(start_year, target_year):
        print(f"  statsapi {y}: hitting", end="", flush=True)
        try:
            hd = _fetch_statsapi_one(y, "hitting")
            hit_rows.append(hd)
            print(f" ({len(hd)})", end="", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
        print(", pitching", end="", flush=True)
        try:
            pd_ = _fetch_statsapi_one(y, "pitching")
            pit_rows.append(pd_)
            print(f" ({len(pd_)})", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
        time.sleep(0.5)  # be polite

    hit_df = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()
    pit_df = pd.concat(pit_rows, ignore_index=True) if pit_rows else pd.DataFrame()

    # Fail loudly (and don't poison the cache with empty tables) if statsapi gave
    # us nothing — otherwise this surfaces later as a cryptic KeyError('PlayerId').
    if hit_df.empty or pit_df.empty or "PlayerId" not in hit_df.columns:
        raise RuntimeError(
            f"statsapi returned no rate data for {start_year}-{target_year - 1}: "
            "every request failed (statsapi.mlb.com is likely blocked / returning "
            "403, or this machine is offline). Stage B cannot build projections "
            "without it. Verify statsapi.mlb.com is reachable from this machine.")

    # Derive rates. Guard against PA=0 to avoid div-by-zero.
    for df, denom_col in [(hit_df, "PA"), (pit_df, "TBF")]:
        if df.empty:
            continue
        # For pitchers, use TBF as the denominator (~PA-equivalent against);
        # statsapi reports battersFaced for pitchers.
        df["denom"] = df[denom_col].clip(lower=1)
        df["K%"]   = df["K"]   / df["denom"]
        df["BB%"]  = df["BB"]  / df["denom"]
        df["HBP%"] = df["HBP"] / df["denom"]
        df["SF%"]  = df["SF"]  / df["denom"]
        df.drop(columns=["denom"], inplace=True)

    hit_df.to_parquet(hit_path)
    pit_df.to_parquet(pit_path)
    return hit_df, pit_df


# ─────────────────────────────────────────────────────────────────────────────
# Team-level runs-per-game from statsapi (one team table per season)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_rpg(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Returns columns: Season, TeamId, Team, G, R, RPG.

    Used to apply a team-context adjustment to R/PA and RBI/PA projections.
    A 2025 Yankees hitter on a 5.2 RPG team should be projected with a
    different run environment than a 2025 Pirates hitter on a 3.6 RPG team.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not seasons:
        return pd.DataFrame()
    out_path = CACHE_DIR / f"team_rpg_{min(seasons)}_to_{max(seasons)}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    rows = []
    for year in seasons:
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/stats?stats=season"
            f"&group=hitting&season={year}&sportIds=1"
        )
        print(f"  team_rpg {year}...", end="", flush=True)
        try:
            r = requests.get(url, timeout=STATSAPI_TIMEOUT, headers=HEADERS)
            if r.status_code != 200:
                print(f" FAILED ({r.status_code})")
                continue
            data = r.json()
        except Exception as e:
            print(f" FAILED: {e}")
            continue
        splits = data.get("stats", [{}])[0].get("splits", [])
        for sp in splits:
            stat = sp.get("stat", {})
            team = sp.get("team", {})
            g = stat.get("gamesPlayed", 0)
            r_ = stat.get("runs", 0)
            if g <= 0:
                continue
            rows.append({
                "Season": year,
                "TeamId": team.get("id"),
                "Team":   team.get("abbreviation") or team.get("name", "")[:3],
                "TeamName": team.get("name"),
                "G":      g,
                "R":      r_,
                "RPG":    r_ / g,
            })
        print(f" {len(splits)} teams")
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Platoon split rate tables from statsapi.mlb.com (vs LHP/RHP, vs LHH/RHH)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_statsapi_splits_one(year: int, group: str, sit_code: str) -> pd.DataFrame:
    """Pull one year × group × split-side from statsapi's statSplits endpoint.

    sit_code is 'vl' (vs Left) or 'vr' (vs Right). For hitters this is vs LHP
    vs RHP; for pitchers it's vs LHB vs RHB.
    """
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats":      "statSplits",
        "season":     year,
        "group":      group,
        "sitCodes":   sit_code,
        "playerPool": "ALL",
        "sportId":    1,
        "limit":      5000,
    }
    r = requests.get(url, params=params, timeout=STATSAPI_TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if not data.get("stats"):
        return pd.DataFrame()
    splits = data["stats"][0].get("splits", [])
    rows = []
    side = "vL" if sit_code == "vl" else "vR"
    for s in splits:
        p  = s.get("player", {})
        st = s.get("stat", {})
        # Pitcher splits return None for plateAppearances; use battersFaced
        pa = st.get("plateAppearances") or st.get("battersFaced", 0) or 0
        rows.append({
            "Season":   year,
            "PlayerId": p.get("id"),
            "Name":     p.get("fullName"),
            "Side":     side,        # 'vL' or 'vR'
            "PA":       pa,
            "AB":       st.get("atBats", 0),
            "K":        st.get("strikeOuts", 0),
            "BB":       st.get("baseOnBalls", 0),
            "IBB":      st.get("intentionalWalks", 0),
            "HBP":      st.get("hitByPitch", 0),
            "SF":       st.get("sacFlies", 0),
            "H":        st.get("hits", 0),
            "HR":       st.get("homeRuns", 0),
            "2B":       st.get("doubles", 0),
            "3B":       st.get("triples", 0),
        })
    return pd.DataFrame(rows)


def fetch_split_rates(target_year: int, start_year: int = RATE_HIST_START,
                      force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch vL and vR platoon splits for hitters and pitchers.

    Returns (hitter_splits_df, pitcher_splits_df) where each row is one
    player-season-side. The 'Side' column is 'vL' (vs Left) or 'vR' (vs Right).

    For hitters: 'vL' = facing LHP, 'vR' = facing RHP.
    For pitchers: 'vL' = facing LHB, 'vR' = facing RHB.

    Used by the splits projection module to project per-side per-PA event
    probabilities for daily/matchup-specific projections. The single-season
    sample sizes per side (~140 PA vL for everyday hitters, ~410 PA vR) are
    smaller than full-season totals, so heavy shrinkage is warranted.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hit_path = CACHE_DIR / f"statsapi_hitting_splits_{start_year}_to_{target_year-1}.parquet"
    pit_path = CACHE_DIR / f"statsapi_pitching_splits_{start_year}_to_{target_year-1}.parquet"
    if hit_path.exists() and pit_path.exists() and not force:
        return pd.read_parquet(hit_path), pd.read_parquet(pit_path)

    hit_rows, pit_rows = [], []
    for y in range(start_year, target_year):
        print(f"  splits {y}:", end="", flush=True)
        for grp, rows in [("hitting", hit_rows), ("pitching", pit_rows)]:
            print(f" {grp}", end="", flush=True)
            for sc in ("vl", "vr"):
                try:
                    d = _fetch_statsapi_splits_one(y, grp, sc)
                    rows.append(d)
                    print(f" {sc}({len(d)})", end="", flush=True)
                except Exception as e:
                    print(f" {sc}-FAIL", end="", flush=True)
                time.sleep(0.3)
        print()

    hit_df = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()
    pit_df = pd.concat(pit_rows, ignore_index=True) if pit_rows else pd.DataFrame()
    if not hit_df.empty:
        hit_df.to_parquet(hit_path)
    if not pit_df.empty:
        pit_df.to_parquet(pit_path)
    return hit_df, pit_df


# ─────────────────────────────────────────────────────────────────────────────
# Player handedness (BatSide / PitchHand)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_player_handedness(player_ids: list[int],
                             force: bool = False) -> pd.DataFrame:
    """Fetch BatSide ('L'/'R'/'S') and PitchHand ('L'/'R') for a set of players.

    Returns DataFrame with PlayerId, BatSide, PitchHand. Both can be None for
    players who only have one side recorded (e.g., a hitter has BatSide but
    not PitchHand). Switch-hitters have BatSide='S'.

    Used to identify the platoon advantage/disadvantage matchup, and to label
    daily projection outputs (a switch-hitter vs LHP is in their RH platoon
    profile by default — they bat from the side opposite the pitcher).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "player_handedness.parquet"
    if out_path.exists() and not force:
        cached = pd.read_parquet(out_path)
        missing = [int(pid) for pid in player_ids
                   if int(pid) not in set(cached["PlayerId"].astype(int))]
        if not missing:
            return cached
        # Append missing
        print(f"  handedness: fetching {len(missing)} new players...", flush=True)
        new_df = _fetch_handedness_batch(missing)
        out = pd.concat([cached, new_df], ignore_index=True).drop_duplicates("PlayerId")
        out.to_parquet(out_path)
        return out

    df = _fetch_handedness_batch(player_ids)
    df.to_parquet(out_path)
    return df


def _fetch_handedness_batch(player_ids: list[int]) -> pd.DataFrame:
    """Fetch handedness for a batch of player IDs via statsapi /people endpoint."""
    rows = []
    chunk_size = 200
    pid_list = sorted(set(int(p) for p in player_ids if p is not None))
    for i in range(0, len(pid_list), chunk_size):
        chunk = pid_list[i:i+chunk_size]
        ids_str = ",".join(str(p) for p in chunk)
        url = "https://statsapi.mlb.com/api/v1/people"
        params = {"personIds": ids_str}
        try:
            r = requests.get(url, params=params, timeout=STATSAPI_TIMEOUT, headers=HEADERS)
            data = r.json()
        except Exception:
            continue
        for p in data.get("people", []):
            rows.append({
                "PlayerId":  p.get("id"),
                "Name":      p.get("fullName"),
                "BatSide":   (p.get("batSide") or {}).get("code"),    # 'L','R','S'
                "PitchHand": (p.get("pitchHand") or {}).get("code"),  # 'L','R'
            })
        time.sleep(0.2)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Statcast park factors from baseballsavant.mlb.com/leaderboard/statcast-park-factors
# ─────────────────────────────────────────────────────────────────────────────

def fetch_park_factors(year: int, rolling: int = 3,
                       force: bool = False) -> pd.DataFrame:
    """Fetch Statcast park factors for venues.

    Returns a DataFrame with columns:
        venue_id, venue_name, main_team_id, team, year_range, n_pa,
        pf_HR, pf_1B, pf_2B, pf_3B, pf_BB, pf_SO, pf_runs, pf_OBP, pf_woba

    `rolling` is the number of years (typically 3 for stable estimates). When
    rolling=3, very new parks (e.g., Sahlen Field in 2024, Sutter Health Park
    in 2025) won't appear because they don't have 3 years of data. We fall
    back to rolling=1 for those.

    Park factors are indexed so 100 = league average. A factor of 127 for HR
    means 27% more HR rate at that park than at other parks (controlled for
    the batters and pitchers playing there).

    Source: https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
    The data is embedded as a JSON array in the page HTML.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"park_factors_{year}_rolling_{rolling}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    import json as _json
    import re as _re

    def _scrape(yr, roll):
        url = (
            f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
            f"?type=year&year={yr}&batSide=All&stat=index_HR"
            f"&condition=All&rolling={roll}&parks=mlb"
        )
        try:
            r = requests.get(url, timeout=STATSAPI_TIMEOUT,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return []
        except Exception:
            return []
        idx = r.text.find("venue_id")
        if idx < 0:
            return []
        start = idx
        for i in range(idx, max(0, idx - 200), -1):
            if r.text[i] == "[":
                start = i
                break
        depth = 0
        end = start
        for i, c in enumerate(r.text[start:], start=start):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            return _json.loads(r.text[start:end])
        except Exception:
            return []

    print(f"  park_factors {year} (rolling={rolling})...", end="", flush=True)
    data = _scrape(year, rolling)
    print(f" {len(data)} venues", end="")

    # Fall back to 1-year rolling for new parks that didn't have 3 years
    if rolling > 1:
        data_1yr = _scrape(year, 1)
        existing_ids = {d["venue_id"] for d in data}
        missing = [d for d in data_1yr if d["venue_id"] not in existing_ids]
        if missing:
            print(f" + {len(missing)} from 1yr fallback", end="")
            data.extend(missing)
    print()

    rows = []
    for d in data:
        rows.append({
            "Season":       year,
            "venue_id":     int(d["venue_id"]),
            "venue_name":   d["venue_name"],
            "TeamId":       int(d["main_team_id"]) if d.get("main_team_id") else None,
            "team":         d.get("name_display_club"),
            "year_range":   d.get("year_range"),
            "n_pa":         int(d.get("n_pa", 0)),
            "pf_runs":      float(d.get("index_runs", 100)) / 100.0,
            "pf_HR":        float(d.get("index_hr", 100))   / 100.0,
            "pf_1B":        float(d.get("index_1b", 100))   / 100.0,
            "pf_2B":        float(d.get("index_2b", 100))   / 100.0,
            "pf_3B":        float(d.get("index_3b", 100))   / 100.0,
            "pf_BB":        float(d.get("index_bb", 100))   / 100.0,
            "pf_SO":        float(d.get("index_so", 100))   / 100.0,
            "pf_hits":      float(d.get("index_hits", 100)) / 100.0,
            "pf_obp":       float(d.get("index_obp", 100))  / 100.0,
            "pf_woba":      float(d.get("index_woba", 100)) / 100.0,
            "pf_bacon":     float(d.get("index_bacon", 100))/ 100.0,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    time.sleep(0.5)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sprint speed leaderboards from baseballsavant
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sprint_speeds(years: list[int], force: bool = False) -> pd.DataFrame:
    """Returns columns: year, player_id, sprint_speed.

    Mirrors the R script's statcast_leaderboards(leaderboard='sprint_speed').
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"sprint_speed_{min(years)}_to_{max(years)}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    frames = []
    for y in years:
        url = (
            "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
            f"?year={y}&position=&team=&min-results=0&csv=true"
        )
        print(f"  sprint_speed {y}...", end="", flush=True)
        try:
            r = requests.get(url, timeout=SAVANT_TIMEOUT, headers=HEADERS)
            if r.status_code != 200 or len(r.content) < 200:
                print(f" FAILED ({r.status_code})")
                continue
            df = pd.read_csv(io.BytesIO(r.content))
            df["year"] = y
            # Pull only what we need
            keep = ["year", "player_id", "sprint_speed"]
            df = df[[c for c in keep if c in df.columns]].copy()
            df = df.dropna(subset=["player_id"])
            df["player_id"] = df["player_id"].astype(int)
            frames.append(df)
            print(f" {len(df)}")
        except Exception as e:
            print(f" FAILED: {e}")
        time.sleep(0.5)

    sp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sp.to_parquet(out_path)
    return sp


# ─────────────────────────────────────────────────────────────────────────────
# Statcast scraping for the target_year - 1 season (e.g., 2026 to predict 2027)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_statcast_season(year: int, force: bool = False) -> pd.DataFrame:
    """Scrape a full season of Statcast BIP data from baseballsavant.

    Returns the post-BIP-filter, post-spray-angle dataframe — same shape as
    the bip CSVs we generate from the RDS files.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"bip_{year}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    # Use pybaseball for the day-by-day chunked scrape (equivalent to the R
    # while-loop in impute_observations.R)
    import pybaseball as pb
    import warnings
    warnings.filterwarnings("ignore")

    start = date(year, 3, 1)
    end   = date.today() if year == date.today().year else date(year, 11, 1)

    frames = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=SAVANT_DAYS_PER_CHUNK - 1), end)
        print(f"  Statcast {cur} → {chunk_end}...", end="", flush=True)
        try:
            df = pb.statcast(start_dt=cur.isoformat(), end_dt=chunk_end.isoformat(),
                             verbose=False)
            if df is not None and len(df):
                # Filter to BIP immediately to save memory
                df = df[df["description"].isin(
                    ["hit_into_play", "hit_into_play_no_out", "hit_into_play_score"]
                )]
                if len(df):
                    frames.append(df[[
                        "batter", "pitcher", "events", "stand",
                        "launch_speed", "launch_angle", "hc_x", "hc_y", "home_team",
                    ]].copy())
            print(f" {len(df) if df is not None else 0} pitches")
        except Exception as e:
            print(f" FAILED: {e}")
        cur = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not frames:
        out = pd.DataFrame(columns=["Season", "batter", "pitcher", "events", "stand",
                                    "launch_speed", "launch_angle", "adjusted_angle",
                                    "home_team"])
        out.to_parquet(out_path)
        return out

    df = pd.concat(frames, ignore_index=True)
    df["Season"] = year

    # Apply spray angle transformation (matches the R script)
    df["x"] = df["hc_x"] - 125.42
    df["y"] = 198.27 - df["hc_y"]
    df["spray_angle"] = np.degrees(np.arctan2(df["x"], df["y"]))
    df["adjusted_angle"] = np.where(df["stand"] == "L",
                                    -df["spray_angle"], df["spray_angle"])

    df = df[["Season", "batter", "pitcher", "events", "stand",
             "launch_speed", "launch_angle", "adjusted_angle", "home_team"]]
    df = df.drop_duplicates().reset_index(drop=True)

    # Coerce id types
    df["batter"]  = df["batter"].astype("Int64")
    df["pitcher"] = df["pitcher"].astype("Int64")

    df.to_parquet(out_path)
    print(f"  Saved {len(df)} BIP rows for {year}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Chadwick player registry (id → name lookup, for the final outputs)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_chadwick_lookup(force: bool = False) -> pd.DataFrame:
    """Returns columns: key_mlbam, name_first, name_last.

    Mirrors chadwick_player_lu() in baseballr. Falls back to an empty
    dataframe if the upstream is unavailable — the pipeline's primary name
    source is statsapi anyway, so this is only used to fill gaps.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "chadwick_lookup.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    try:
        import pybaseball as pb
        import warnings
        warnings.filterwarnings("ignore")
        df = pb.chadwick_register()
        df = df[["key_mlbam", "name_first", "name_last"]].dropna(subset=["key_mlbam"])
        df["key_mlbam"] = df["key_mlbam"].astype(int)
        df.to_parquet(out_path)
        return df
    except Exception as e:
        print(f"  Warning: Chadwick lookup failed ({type(e).__name__}); "
              f"using statsapi names only")
        # Return an empty frame with the expected schema
        return pd.DataFrame({"key_mlbam": pd.Series(dtype="int64"),
                             "name_first": pd.Series(dtype="string"),
                             "name_last":  pd.Series(dtype="string")})


if __name__ == "__main__":
    # Quick smoke test
    print("Testing data acquisition...")
    h, p = fetch_rate_data(2027, 2024)
    print(f"Hitting: {h.shape}  Pitching: {p.shape}")
    sp = fetch_sprint_speeds([2024, 2025])
    print(f"Sprint: {sp.shape}")

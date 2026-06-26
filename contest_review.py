"""
contest_review.py  —  Post-slate grading and performance analysis
=================================================================
Parses a DraftKings contest-standings CSV (the dual-column format DK exports),
finds a user's entries by username, scores them against the actual field, and
runs the projection sims against the actual field composition to understand
how the portfolio should have performed across all simulated outcomes.

Public API
----------
    parse_contest_csv(bytes_or_str)              ->  ContestData
    find_user_entries(contest, username)          ->  list[ContestEntry]
    list_usernames(contest)                       ->  list[str]
    grade_user_entries(contest, user_entries)     ->  GradedResult
    simulate_portfolio_vs_field(contest,
        sim_scores, user_entries)                 ->  PortfolioSimResult
    build_player_sim_table(contest, sim_scores)   ->  pd.DataFrame
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
_POSITIONS = {"1B", "2B", "3B", "C", "OF", "P", "SS"}
_ENTRY_SUFFIX = re.compile(r"\s*\(\d+/\d+\)\s*$")


def _norm(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace(",", "").replace("'", "")
    for s in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(s):
            n = n[: -len(s)]
    return n.strip()


def _extract_username(entry_name: str) -> str:
    return _ENTRY_SUFFIX.sub("", entry_name).strip()


def _parse_lineup(lineup_str: str) -> list[tuple[str, str]]:
    tokens = lineup_str.split()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _POSITIONS:
            pos = tokens[i]
            name_parts: list[str] = []
            i += 1
            while i < len(tokens) and tokens[i] not in _POSITIONS:
                name_parts.append(tokens[i])
                i += 1
            if name_parts:
                out.append((pos, " ".join(name_parts)))
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------
@dataclass
class PlayerActual:
    name: str
    norm_name: str
    position: str
    pct_drafted: float
    fpts: float


@dataclass
class ContestEntry:
    rank: int
    entry_id: str
    entry_name: str
    username: str
    points: float
    players: list[tuple[str, str]]
    norm_players: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.norm_players = [_norm(p) for _, p in self.players]


@dataclass
class ContestData:
    entries: list[ContestEntry]
    player_actuals: dict[str, PlayerActual]
    date: Optional[str] = None

    @property
    def scores(self) -> np.ndarray:
        return np.array([e.points for e in self.entries], dtype=float)

    @property
    def n_entries(self) -> int:
        return len(self.entries)

    def percentile_score(self, pct: float) -> float:
        return float(np.percentile(self.scores, pct * 100))

    def player_actual(self, name: str) -> Optional[PlayerActual]:
        return self.player_actuals.get(_norm(name))


@dataclass
class GradedLineup:
    lineup_id: int
    entry_name: str
    players: list[tuple[str, str]]
    actual_score: float
    field_rank: int
    field_pct: float
    player_scores: dict[str, float]


@dataclass
class GradedResult:
    graded: list[GradedLineup]
    contest: ContestData


# ---- portfolio simulation results ----------------------------------------

@dataclass
class LineupPortfolioStats:
    lineup_id: int
    entry_name: str
    players: list[tuple[str, str]]
    # per-sim aggregates (fraction of sims)
    win_pct: float
    top10_pct: float
    top1pct_pct: float     # top 1% of field
    top10pct_pct: float    # top 10% of field (typical GPP cash line)
    avg_place: float
    # projected score distribution
    proj_mean: float
    proj_p10: float
    proj_p25: float
    proj_p50: float
    proj_p75: float
    proj_p90: float
    # coverage: fraction of players found in sim dict
    sim_coverage: float


@dataclass
class PortfolioSimResult:
    lineup_stats: list[LineupPortfolioStats]
    n_sim: int
    n_field: int
    n_user_lineups: int
    # portfolio-level (any lineup achieves the outcome)
    portfolio_win_pct: float
    portfolio_top10_pct: float
    portfolio_top1pct_pct: float
    portfolio_top10pct_pct: float
    # field score distribution summary (from sims)
    field_proj_mean: float
    field_proj_p90: float    # ≈ top-10% line
    field_proj_p99: float    # ≈ winner range
    # fraction of field lineups that had full sim coverage
    field_sim_coverage: float
    # expected cash/gpp lines from sim
    sim_cash_line: float     # median of field score distribution
    sim_top10_line: float    # 90th pct of field score distribution


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse_contest_csv(raw: bytes | str) -> ContestData:
    """
    Parse the DK contest-standings dual-column CSV.
    Left block:  Rank, EntryId, EntryName, TimeRemaining, Points, Lineup
    Right block: Player, Roster Position, %Drafted, FPTS
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")

    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    cols = list(df.columns)
    blank_idx = [i for i, c in enumerate(cols) if c == ""]
    if blank_idx:
        df = df.rename(columns={cols[blank_idx[0]]: "_sep"})

    entries: list[ContestEntry] = []
    player_actuals: dict[str, PlayerActual] = {}

    for _, row in df.iterrows():
        rank_raw = str(row.get("Rank", "")).strip()
        points_raw = str(row.get("Points", "")).strip()
        lineup_raw = str(row.get("Lineup", "")).strip()
        entry_name = str(row.get("EntryName", "")).strip()

        if rank_raw.lstrip("-").isdigit() and lineup_raw:
            try:
                rank = int(rank_raw)
                points = float(points_raw) if points_raw else 0.0
                players = _parse_lineup(lineup_raw)
                if players:
                    entries.append(ContestEntry(
                        rank=rank,
                        entry_id=str(row.get("EntryId", "")).strip(),
                        entry_name=entry_name,
                        username=_extract_username(entry_name),
                        points=points,
                        players=players,
                    ))
            except (ValueError, TypeError):
                pass

        player_raw = str(row.get("Player", "")).strip()
        fpts_raw = str(row.get("FPTS", "")).strip()
        pos_raw = str(row.get("Roster Position", "")).strip()
        pct_raw = str(row.get("%Drafted", "")).strip()

        if player_raw and fpts_raw and fpts_raw.replace(".", "").replace("-", "").isdigit():
            try:
                pct = float(pct_raw.rstrip("%")) if pct_raw else 0.0
                pa = PlayerActual(
                    name=player_raw,
                    norm_name=_norm(player_raw),
                    position=pos_raw,
                    pct_drafted=pct,
                    fpts=float(fpts_raw),
                )
                player_actuals[pa.norm_name] = pa
            except (ValueError, TypeError):
                pass

    entries.sort(key=lambda e: e.rank)
    return ContestData(entries=entries, player_actuals=player_actuals)


# ---------------------------------------------------------------------------
# username lookup
# ---------------------------------------------------------------------------
def find_user_entries(contest: ContestData, username: str) -> list[ContestEntry]:
    target = username.strip().lower()
    return [e for e in contest.entries if e.username.lower() == target]


def list_usernames(contest: ContestData) -> list[str]:
    seen: dict[str, str] = {}
    for e in contest.entries:
        lc = e.username.lower()
        if lc not in seen:
            seen[lc] = e.username
    return sorted(seen.values(), key=str.lower)


# ---------------------------------------------------------------------------
# actual-results grading (tab 3 — "Your Lineups" actual outcome)
# ---------------------------------------------------------------------------
def grade_user_entries(contest: ContestData,
                       user_entries: list[ContestEntry]) -> GradedResult:
    """Score user entries using actual player FPTS; rank from the standings."""
    graded: list[GradedLineup] = []
    n = contest.n_entries
    for idx, entry in enumerate(user_entries, 1):
        total = 0.0
        pscores: dict[str, float] = {}
        for pos, name in entry.players:
            pa = contest.player_actual(name)
            pts = pa.fpts if pa else 0.0
            total += pts
            pscores[_norm(name)] = pts

        field_rank = entry.rank
        field_pct = 1.0 - (field_rank - 1) / max(1, n)

        graded.append(GradedLineup(
            lineup_id=idx,
            entry_name=entry.entry_name,
            players=entry.players,
            actual_score=total,
            field_rank=field_rank,
            field_pct=field_pct,
            player_scores=pscores,
        ))
    return GradedResult(graded=graded, contest=contest)


# ---------------------------------------------------------------------------
# portfolio simulation against actual field
# ---------------------------------------------------------------------------
def _build_score_matrix(entries: list[ContestEntry],
                        sim_scores: dict[str, np.ndarray],
                        n_sim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (n_entries, n_sim) score matrix by summing per-player sim arrays.
    Also returns a (n_entries,) coverage array: fraction of players found.
    """
    n = len(entries)
    mat = np.zeros((n, n_sim), dtype=np.float32)
    coverage = np.zeros(n, dtype=np.float32)

    for i, entry in enumerate(entries):
        found = 0
        for _, name in entry.players:
            arr = sim_scores.get(_norm(name))
            if arr is not None:
                mat[i] += arr[:n_sim]
                found += 1
        coverage[i] = found / max(1, len(entry.players))

    return mat, coverage


def simulate_portfolio_vs_field(
        contest: ContestData,
        sim_scores: dict[str, np.ndarray],
        user_entries: list[ContestEntry],
        progress_cb=None,
) -> PortfolioSimResult:
    """
    Score the user's submitted lineups against the actual contest field using
    the projection sim distributions, producing Win%/Top10%/Top10pct%/AvgPlace
    for each lineup and at the portfolio level.

    All scoring uses the projected sim arrays — not the actual slate outcome —
    so the results show the *distribution* of expected performance across all
    simulated universes, not just the one that happened.

    progress_cb: optional callable(step, total) for progress reporting
    """
    # Determine n_sim from the sim dict
    sample_arr = next(iter(sim_scores.values()))
    n_sim = len(sample_arr)
    n_field = contest.n_entries

    if progress_cb:
        progress_cb(0, 3)

    # 1. Build field score matrix (n_field × n_sim)
    field_mat, field_cov = _build_score_matrix(contest.entries, sim_scores, n_sim)

    if progress_cb:
        progress_cb(1, 3)

    # 2. Build user score matrix (n_user × n_sim)
    user_mat, user_cov = _build_score_matrix(user_entries, sim_scores, n_sim)
    n_user = len(user_entries)

    if progress_cb:
        progress_cb(2, 3)

    # 3. Compute field distribution summary
    # field_mean_per_sim: (n_sim,) — mean field score each sim
    # We want percentiles of the *per-sim field max* for expected winner range
    field_max_per_sim = field_mat.max(axis=0)      # (n_sim,)
    field_p50_per_sim = np.percentile(field_mat, 50, axis=0)  # median field per sim
    field_p90_per_sim = np.percentile(field_mat, 90, axis=0)  # top-10% line per sim

    # Expected cash/gpp lines: mean across sims of the per-sim percentile
    sim_cash_line = float(field_p50_per_sim.mean())
    sim_top10_line = float(field_p90_per_sim.mean())
    field_proj_mean = float(field_mat.mean())
    field_proj_p90 = float(np.percentile(field_mat, 90))
    field_proj_p99 = float(np.percentile(field_mat, 99))

    # 4. Per-lineup and portfolio stats
    n1pct = max(1, int(n_field * 0.01))   # top-1% count
    n10pct = max(1, int(n_field * 0.10))  # top-10% count

    # Vectorized placement via chunked broadcasting.
    # For each (j, s): place[j, s] = #field_entries_beating_user[j,s] + 1
    #   = (field_mat[:, s] > user_mat[j, s]).sum() + 1  over all j,s
    #
    # Full broadcast (n_user, n_field, n_sim) is too large (~900 M entries).
    # Instead chunk over sims: process SIM_CHUNK sims at a time so the
    # working tensor is (n_user, n_field, SIM_CHUNK) ≈ 50 MB per chunk.
    SIM_CHUNK = 500
    beaten_by_mat = np.zeros((n_user, n_sim), dtype=np.int32)

    for s0 in range(0, n_sim, SIM_CHUNK):
        s1 = min(s0 + SIM_CHUNK, n_sim)
        # field_chunk: (n_field, chunk)   user_chunk: (n_user, chunk)
        fc = field_mat[:, s0:s1]          # (n_field, chunk)
        uc = user_mat[:, s0:s1]           # (n_user,  chunk)
        # Broadcast to (n_user, n_field, chunk): field beats user when > user
        # sum over axis=1 gives # field entries that beat each user lineup
        beaten_by_mat[:, s0:s1] = (
            fc[np.newaxis, :, :] > uc[:, np.newaxis, :]
        ).sum(axis=1, dtype=np.int32)     # (n_user, chunk)

    place_mat = beaten_by_mat + 1   # (n_user, n_sim)

    win_mat    = place_mat == 1
    top10_mat  = place_mat <= 10
    top1p_mat  = place_mat <= n1pct
    top10p_mat = place_mat <= n10pct

    wins_any    = win_mat.any(axis=0)
    top10_any   = top10_mat.any(axis=0)
    top1pct_any = top1p_mat.any(axis=0)
    top10pct_any = top10p_mat.any(axis=0)

    lineup_stats: list[LineupPortfolioStats] = []
    for j in range(n_user):
        u_arr = user_mat[j]
        lineup_stats.append(LineupPortfolioStats(
            lineup_id=j + 1,
            entry_name=user_entries[j].entry_name,
            players=user_entries[j].players,
            win_pct=float(win_mat[j].mean() * 100),
            top10_pct=float(top10_mat[j].mean() * 100),
            top1pct_pct=float(top1p_mat[j].mean() * 100),
            top10pct_pct=float(top10p_mat[j].mean() * 100),
            avg_place=float(place_mat[j].mean()),
            proj_mean=float(u_arr.mean()),
            proj_p10=float(np.percentile(u_arr, 10)),
            proj_p25=float(np.percentile(u_arr, 25)),
            proj_p50=float(np.percentile(u_arr, 50)),
            proj_p75=float(np.percentile(u_arr, 75)),
            proj_p90=float(np.percentile(u_arr, 90)),
            sim_coverage=float(user_cov[j]),
        ))

    if progress_cb:
        progress_cb(3, 3)

    return PortfolioSimResult(
        lineup_stats=lineup_stats,
        n_sim=n_sim,
        n_field=n_field,
        n_user_lineups=n_user,
        portfolio_win_pct=float(wins_any.mean() * 100),
        portfolio_top10_pct=float(top10_any.mean() * 100),
        portfolio_top1pct_pct=float(top1pct_any.mean() * 100),
        portfolio_top10pct_pct=float(top10pct_any.mean() * 100),
        field_proj_mean=field_proj_mean,
        field_proj_p90=field_proj_p90,
        field_proj_p99=field_proj_p99,
        field_sim_coverage=float(field_cov.mean()),
        sim_cash_line=sim_cash_line,
        sim_top10_line=sim_top10_line,
    )


# ---------------------------------------------------------------------------
# player actuals table (used by the Player Actuals sub-tab)
# ---------------------------------------------------------------------------
def build_player_sim_table(contest: ContestData,
                           sim_scores: Optional[dict[str, np.ndarray]] = None,
                           projected: Optional[dict[str, float]] = None,
                           proj_own: Optional[dict[str, float]] = None) -> pd.DataFrame:
    rows = []
    for pa in sorted(contest.player_actuals.values(), key=lambda p: -p.fpts):
        row: dict = {
            "Player": pa.name,
            "Position": pa.position,
            "Actual FPTS": pa.fpts,
            "Ownership %": pa.pct_drafted,
        }
        if projected:
            pv = projected.get(pa.norm_name)
            if pv is not None:
                row["Proj Mean"] = round(pv, 2)
                row["vs Proj"] = round(pa.fpts - pv, 2)
        if proj_own:
            po = proj_own.get(pa.norm_name)
            if po is not None:
                row["Proj Own%"] = round(po, 1)
                row["Own Delta"] = round(pa.pct_drafted - po, 1)
        if sim_scores:
            arr = sim_scores.get(pa.norm_name)
            if arr is not None:
                arr = np.asarray(arr, dtype=float)
                row["Sim Mean"] = round(float(arr.mean()), 2)
                row["Sim P10"] = round(float(np.percentile(arr, 10)), 1)
                row["Sim P50"] = round(float(np.percentile(arr, 50)), 1)
                row["Sim P90"] = round(float(np.percentile(arr, 90)), 1)
                row["Actual %ile"] = round(float(np.mean(arr <= pa.fpts) * 100), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# helper DataFrames for the UI
# ---------------------------------------------------------------------------
def entries_to_df(contest: ContestData, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for e in contest.entries[:top_n]:
        rows.append({
            "Rank": e.rank,
            "Entry": e.entry_name,
            "Score": e.points,
            "Players": ", ".join(p for _, p in e.players),
        })
    return pd.DataFrame(rows)


def graded_to_df(graded: GradedResult) -> pd.DataFrame:
    rows = []
    for g in sorted(graded.graded, key=lambda x: x.field_rank):
        rows.append({
            "Entry": g.entry_name,
            "Actual Score": round(g.actual_score, 2),
            "Field Rank": g.field_rank,
            "Field %ile": f"{g.field_pct * 100:.1f}%",
            "Players": ", ".join(p for _, p in g.players),
        })
    return pd.DataFrame(rows)


def portfolio_sim_to_df(result: PortfolioSimResult) -> pd.DataFrame:
    rows = []
    for ls in sorted(result.lineup_stats, key=lambda x: -x.top10pct_pct):
        rows.append({
            "Entry": ls.entry_name,
            "Win%": round(ls.win_pct, 3),
            "Top-10%": round(ls.top10_pct, 2),
            "Top-1%": round(ls.top1pct_pct, 2),
            "Top-10% field": round(ls.top10pct_pct, 1),
            "Avg Place": round(ls.avg_place, 0),
            "Proj Mean": round(ls.proj_mean, 1),
            "Proj P10": round(ls.proj_p10, 1),
            "Proj P50": round(ls.proj_p50, 1),
            "Proj P90": round(ls.proj_p90, 1),
            "Sim Coverage": f"{ls.sim_coverage * 100:.0f}%",
        })
    return pd.DataFrame(rows)

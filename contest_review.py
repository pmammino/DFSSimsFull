"""
contest_review.py  —  Post-slate grading and performance analysis
=================================================================
Parses a DraftKings contest-standings CSV (the dual-column format DK exports),
scores any candidate lineups against the actual field, and generates a
diagnostic report flagging systematic issues in the original candidate set.

Public API
----------
    parse_contest_csv(bytes_or_str)  ->  ContestData
    grade_lineups(contest, your_lineups) -> GradedResult
    build_diagnosis(contest, graded)  -> list[DiagnosticInsight]
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_POS_RE = re.compile(r"\b(1B|2B|3B|C|OF|P|SS)\b")
_POSITIONS = {"1B", "2B", "3B", "C", "OF", "P", "SS"}


def _norm(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace(",", "").replace("'", "")
    for s in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(s):
            n = n[: -len(s)]
    return n.strip()


def _parse_lineup(lineup_str: str) -> list[tuple[str, str]]:
    """Return [(position, player_name), ...] from a DK lineup string."""
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
    pct_drafted: float      # actual contest ownership %
    fpts: float             # actual DK fantasy points scored


@dataclass
class ContestEntry:
    rank: int
    entry_id: str
    entry_name: str
    points: float
    # list of (position, player_name) pairs
    players: list[tuple[str, str]]
    # normalized player names for fast lookup
    norm_players: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.norm_players = [_norm(p) for _, p in self.players]


@dataclass
class ContestData:
    entries: list[ContestEntry]          # all field entries, sorted by rank
    player_actuals: dict[str, PlayerActual]  # norm_name -> PlayerActual
    date: Optional[str] = None

    # derived convenience properties -------------------------------------------
    @property
    def scores(self) -> np.ndarray:
        return np.array([e.points for e in self.entries], dtype=float)

    @property
    def n_entries(self) -> int:
        return len(self.entries)

    def percentile_score(self, pct: float) -> float:
        """Score that beats pct% of the field (e.g. pct=0.9 → top 10%)."""
        return float(np.percentile(self.scores, pct * 100))

    def player_actual(self, name: str) -> Optional[PlayerActual]:
        return self.player_actuals.get(_norm(name))


@dataclass
class GradedLineup:
    lineup_id: int                       # 1-based index
    players: list[tuple[str, str]]       # [(pos, name), ...]
    actual_score: float
    field_rank: int                      # 1-based rank among ALL entries
    field_pct: float                     # percentile beaten (0–1)
    # per-player breakdown
    player_scores: dict[str, float]      # norm_name -> fpts


@dataclass
class GradedResult:
    graded: list[GradedLineup]
    contest: ContestData


@dataclass
class DiagnosticInsight:
    category: str   # e.g. "Pitcher", "Stack", "Ownership", "Value"
    severity: str   # "high" | "medium" | "low"
    headline: str
    detail: str


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse_contest_csv(raw: bytes | str) -> ContestData:
    """
    Parse a DraftKings contest-standings CSV.

    The file has a dual-column layout where the left block carries contest
    entries (Rank, EntryId, EntryName, TimeRemaining, Points, Lineup) and the
    right block carries per-player scoring (Player, Roster Position, %Drafted,
    FPTS), with a blank separator column in between.  Not every row has both
    halves populated.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")

    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Rename the blank separator column if present
    cols = list(df.columns)
    blank_idx = [i for i, c in enumerate(cols) if c == ""]
    if blank_idx:
        df = df.rename(columns={cols[blank_idx[0]]: "_sep"})

    entries: list[ContestEntry] = []
    player_actuals: dict[str, PlayerActual] = {}

    for _, row in df.iterrows():
        # --- left block: contest entry ---
        rank_raw = str(row.get("Rank", "")).strip()
        points_raw = str(row.get("Points", "")).strip()
        lineup_raw = str(row.get("Lineup", "")).strip()

        if rank_raw.lstrip("-").isdigit() and lineup_raw:
            try:
                rank = int(rank_raw)
                points = float(points_raw) if points_raw else 0.0
                players = _parse_lineup(lineup_raw)
                if players:
                    entries.append(ContestEntry(
                        rank=rank,
                        entry_id=str(row.get("EntryId", "")).strip(),
                        entry_name=str(row.get("EntryName", "")).strip(),
                        points=points,
                        players=players,
                    ))
            except (ValueError, TypeError):
                pass

        # --- right block: player actuals ---
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

    # Sort entries by rank
    entries.sort(key=lambda e: e.rank)

    return ContestData(entries=entries, player_actuals=player_actuals)


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------
def grade_lineups(contest: ContestData,
                  your_lineups: list[list[tuple[str, str]]]) -> GradedResult:
    """
    Score each of your lineups against actual player FPTS and place them
    in the contest field.

    your_lineups: list of [(position, player_name), ...] — one per lineup.
    """
    all_scores = np.sort(contest.scores)[::-1]   # descending

    graded: list[GradedLineup] = []
    for idx, lp in enumerate(your_lineups, 1):
        total = 0.0
        pscores: dict[str, float] = {}
        for pos, name in lp:
            pa = contest.player_actual(name)
            pts = pa.fpts if pa else 0.0
            total += pts
            pscores[_norm(name)] = pts

        # field_rank: 1-based position after inserting your score
        field_rank = int(np.searchsorted(-all_scores, -total, side="left")) + 1
        field_pct = 1.0 - (field_rank - 1) / max(1, len(all_scores))

        graded.append(GradedLineup(
            lineup_id=idx,
            players=lp,
            actual_score=total,
            field_rank=field_rank,
            field_pct=field_pct,
            player_scores=pscores,
        ))

    return GradedResult(graded=graded, contest=contest)


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------
_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_diagnosis(contest: ContestData,
                    graded: Optional[GradedResult] = None,
                    projected: Optional[dict[str, float]] = None,
                    proj_own: Optional[dict[str, float]] = None) -> list[DiagnosticInsight]:
    """
    Generate diagnostic insights about the slate.

    Parameters
    ----------
    contest   : parsed actual contest data
    graded    : (optional) your graded lineups
    projected : (optional) dict of norm_name -> projected_fpts from the sim
    proj_own  : (optional) dict of norm_name -> projected_ownership%
    """
    insights: list[DiagnosticInsight] = []
    actuals = contest.player_actuals
    scores = contest.scores
    n = contest.n_entries
    top10_score = contest.percentile_score(0.90)
    cash_score = contest.percentile_score(0.50)

    # ---- 1. Field score summary (always shown) ----------------------------
    insights.append(DiagnosticInsight(
        category="Field",
        severity="low",
        headline="Contest score distribution",
        detail=(
            f"{n:,} entries · "
            f"Winner: **{scores.max():.2f} pts** · "
            f"Top 10% line: **{top10_score:.2f} pts** · "
            f"Median (cash line): **{cash_score:.2f} pts**"
        ),
    ))

    # ---- 2. Pitcher analysis ---------------------------------------------
    pitchers = [pa for pa in actuals.values() if pa.position == "P"]
    pitchers.sort(key=lambda p: -p.fpts)

    if pitchers:
        best_p = pitchers[0]
        # Count pitcher ownership in the top 10%
        top_entries = [e for e in contest.entries if e.points >= top10_score]
        p_counter: Counter[str] = Counter()
        for e in top_entries:
            for pos, name in e.players:
                if pos == "P":
                    p_counter[_norm(name)] += 1

        if p_counter:
            top_p_norm, top_p_cnt = p_counter.most_common(1)[0]
            top_p_pct_in_top = top_p_cnt / max(1, len(top_entries)) * 100
            top_p_pa = actuals.get(top_p_norm)
            top_p_name = top_p_pa.name if top_p_pa else top_p_norm
            top_p_fpts = top_p_pa.fpts if top_p_pa else 0.0

            # Pitcher was highly correlated with top finishes
            if top_p_pct_in_top > 50:
                insights.append(DiagnosticInsight(
                    category="Pitcher",
                    severity="high",
                    headline=f"{top_p_name} appeared in {top_p_pct_in_top:.0f}% of top-10% lineups",
                    detail=(
                        f"**{top_p_name}** scored **{top_p_fpts:.2f} pts** and was "
                        f"the dominant pitcher in winning lineups. "
                        + (_pitcher_proj_note(top_p_norm, projected, top_p_fpts) if projected else "")
                    ),
                ))

        # Pitchers who scored big but were low-owned
        for p in pitchers[:5]:
            if p.pct_drafted < 25 and p.fpts > 20:
                insights.append(DiagnosticInsight(
                    category="Pitcher",
                    severity="medium",
                    headline=f"Low-owned pitcher {p.name} scored {p.fpts:.2f} pts at {p.pct_drafted:.1f}% ownership",
                    detail=(
                        f"**{p.name}** was drafted by only {p.pct_drafted:.1f}% of the field "
                        f"yet scored {p.fpts:.2f} DK points — a significant leverage "
                        f"opportunity that was largely missed."
                    ),
                ))

    # ---- 3. Ownership accuracy (if projected ownership supplied) ----------
    if proj_own:
        over_owned: list[tuple[str, float, float]] = []
        under_owned: list[tuple[str, float, float]] = []
        for pa in actuals.values():
            p_own = proj_own.get(pa.norm_name)
            if p_own is None:
                continue
            delta = pa.pct_drafted - p_own
            if delta > 15:
                over_owned.append((pa.name, p_own, pa.pct_drafted))
            elif delta < -15:
                under_owned.append((pa.name, p_own, pa.pct_drafted))

        if over_owned:
            over_owned.sort(key=lambda x: -(x[2] - x[1]))
            names = ", ".join(f"**{n}** ({p:.0f}%→{a:.0f}%)"
                              for n, p, a in over_owned[:3])
            insights.append(DiagnosticInsight(
                category="Ownership",
                severity="medium",
                headline=f"{len(over_owned)} player(s) came in significantly over projected ownership",
                detail=f"Underestimated chalk creates unnecessary duplicate risk. Notable: {names}.",
            ))

        if under_owned:
            under_owned.sort(key=lambda x: x[2] - x[1])
            names = ", ".join(f"**{n}** ({p:.0f}%→{a:.0f}%)"
                              for n, p, a in under_owned[:3])
            insights.append(DiagnosticInsight(
                category="Ownership",
                severity="medium",
                headline=f"{len(under_owned)} player(s) came in significantly under projected ownership",
                detail=f"Overestimated chalk leads to false leverage assumptions. Notable: {names}.",
            ))

    # ---- 4. Projection accuracy (if projected FPTS supplied) ---------------
    if projected:
        busts: list[tuple[str, float, float]] = []   # (name, proj, actual)
        booms: list[tuple[str, float, float]] = []
        for pa in actuals.values():
            p_pts = projected.get(pa.norm_name)
            if p_pts is None:
                continue
            err = pa.fpts - p_pts
            if err < -8:
                busts.append((pa.name, p_pts, pa.fpts))
            elif err > 10:
                booms.append((pa.name, p_pts, pa.fpts))

        if busts:
            busts.sort(key=lambda x: x[2] - x[1])
            names = ", ".join(f"**{n}** (proj {p:.1f}, actual {a:.1f})"
                              for n, p, a in busts[:3])
            insights.append(DiagnosticInsight(
                category="Projection",
                severity="high",
                headline=f"{len(busts)} player(s) significantly underperformed projections",
                detail=f"Large busts hurt lineup scores: {names}.",
            ))

        if booms:
            booms.sort(key=lambda x: -(x[2] - x[1]))
            names = ", ".join(f"**{n}** (proj {p:.1f}, actual {a:.1f})"
                              for n, p, a in booms[:3])
            insights.append(DiagnosticInsight(
                category="Projection",
                severity="medium",
                headline=f"{len(booms)} value player(s) far exceeded projections",
                detail=f"Missed upside opportunities: {names}.",
            ))

    # ---- 5. Stack analysis -----------------------------------------------
    # Find teams most represented in top-10% lineups
    top_entries = [e for e in contest.entries if e.points >= top10_score]
    if top_entries:
        # Build team -> how often in top-10% entries
        team_counter: Counter[str] = Counter()
        for e in top_entries:
            team_counts: Counter[str] = Counter()
            for pos, name in e.players:
                pa = actuals.get(_norm(name))
                # we don't have team in ContestData; use first two chars of lineup position as proxy
                # Instead, count player co-occurrences in top lineups
                pass

        # Player co-occurrence in top-10% entries
        pair_counter: Counter[frozenset] = Counter()
        player_top_count: Counter[str] = Counter()
        for e in top_entries:
            norm_names = e.norm_players
            for nm in norm_names:
                player_top_count[nm] += 1
            for i in range(len(norm_names)):
                for j in range(i + 1, len(norm_names)):
                    pair_counter[frozenset([norm_names[i], norm_names[j]])] += 1

        # Top players in top-10% lineups
        top_players_in_top: list[tuple[str, int]] = player_top_count.most_common(5)
        if top_players_in_top:
            parts = []
            for norm_nm, cnt in top_players_in_top:
                pa = actuals.get(norm_nm)
                display = pa.name if pa else norm_nm
                pct_top = cnt / len(top_entries) * 100
                parts.append(f"**{display}** ({pct_top:.0f}%)")
            insights.append(DiagnosticInsight(
                category="Stack",
                severity="low",
                headline="Players most common in top-10% finishes",
                detail="Appearance rate in top-10% lineups: " + ", ".join(parts),
            ))

    # ---- 6. Your lineup grading (if provided) ----------------------------
    if graded:
        gl = graded.graded
        n_gl = len(gl)
        avg_score = np.mean([g.actual_score for g in gl])
        avg_pct = np.mean([g.field_pct for g in gl]) * 100
        best = max(gl, key=lambda g: g.actual_score)
        worst = min(gl, key=lambda g: g.actual_score)

        insights.append(DiagnosticInsight(
            category="Your Lineups",
            severity="low",
            headline=f"{n_gl} candidate lineup(s) graded",
            detail=(
                f"Average actual score: **{avg_score:.2f} pts** "
                f"(beat **{avg_pct:.1f}%** of the field on average). "
                f"Best lineup: **#{best.lineup_id}** at **{best.actual_score:.2f} pts** "
                f"(rank {best.field_rank:,}/{contest.n_entries:,}). "
                f"Weakest: **#{worst.lineup_id}** at **{worst.actual_score:.2f} pts**."
            ),
        ))

        # Identify players that were in your lineups but busted
        if n_gl > 0:
            your_player_scores: dict[str, list[float]] = defaultdict(list)
            for g in gl:
                for pos, name in g.players:
                    nm = _norm(name)
                    pa = actuals.get(nm)
                    your_player_scores[nm].append(pa.fpts if pa else 0.0)

            worst_your_players = []
            for nm, pts_list in your_player_scores.items():
                pa = actuals.get(nm)
                if pa is None:
                    continue
                avg_pts = np.mean(pts_list)
                n_lineups = len(pts_list)
                if avg_pts < 5 and n_lineups >= max(1, n_gl // 4):
                    worst_your_players.append((pa.name, avg_pts, n_lineups))

            if worst_your_players:
                worst_your_players.sort(key=lambda x: x[1])
                parts = [f"**{n}** ({a:.1f} pts, {c} lineup{'s' if c > 1 else ''})"
                         for n, a, c in worst_your_players[:4]]
                insights.append(DiagnosticInsight(
                    category="Your Lineups",
                    severity="high" if len(worst_your_players) >= 2 else "medium",
                    headline="Heavy exposure to low-scoring players dragged down your set",
                    detail=", ".join(parts) + " underperformed while appearing frequently in your lineups.",
                ))

    # Sort by severity
    insights.sort(key=lambda i: _SEV_ORDER.get(i.severity, 9))
    return insights


def _pitcher_proj_note(norm_name: str, projected: dict[str, float], actual: float) -> str:
    p_pts = projected.get(norm_name)
    if p_pts is None:
        return ""
    diff = actual - p_pts
    direction = "exceeded" if diff > 0 else "missed"
    return f"Projection {direction} by {abs(diff):.1f} pts (proj: {p_pts:.1f}, actual: {actual:.1f})."


# ---------------------------------------------------------------------------
# helpers for Streamlit tab
# ---------------------------------------------------------------------------
def parse_your_lineups_csv(raw: bytes | str) -> list[list[tuple[str, str]]]:
    """
    Parse a DK upload CSV (the format the Export tab produces) or a simple
    lineup CSV where each row is a lineup with player names in columns.

    Returns list of [(pos, player_name), ...] per lineup.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")

    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    lineups: list[list[tuple[str, str]]] = []

    # DK upload format: columns like P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
    dk_upload_pos = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    if all(c in df.columns for c in dk_upload_pos):
        for _, row in df.iterrows():
            lp = [(pos, str(row[pos]).strip()) for pos in dk_upload_pos
                  if str(row[pos]).strip()]
            if lp:
                lineups.append(lp)
        return lineups

    # Fallback: try parsing each row as a DK lineup string in first column
    first_col = df.columns[0]
    for _, row in df.iterrows():
        val = str(row[first_col]).strip()
        if val:
            parsed = _parse_lineup(val)
            if parsed:
                lineups.append(parsed)

    return lineups


def actuals_to_df(contest: ContestData) -> pd.DataFrame:
    """Return a DataFrame of player actuals sorted by FPTS desc."""
    rows = []
    for pa in sorted(contest.player_actuals.values(), key=lambda p: -p.fpts):
        rows.append({
            "Player": pa.name,
            "Position": pa.position,
            "Actual FPTS": pa.fpts,
            "Ownership %": pa.pct_drafted,
        })
    return pd.DataFrame(rows)


def entries_to_df(contest: ContestData) -> pd.DataFrame:
    """Return a DataFrame of all contest entries."""
    rows = []
    for e in contest.entries:
        rows.append({
            "Rank": e.rank,
            "Entry": e.entry_name,
            "Score": e.points,
            "Players": ", ".join(p for _, p in e.players),
        })
    return pd.DataFrame(rows)


def graded_to_df(graded: GradedResult) -> pd.DataFrame:
    """Return a DataFrame of your graded lineups."""
    rows = []
    for g in graded.graded:
        rows.append({
            "Lineup #": g.lineup_id,
            "Actual Score": round(g.actual_score, 2),
            "Field Rank": g.field_rank,
            "Field %ile": f"{g.field_pct * 100:.1f}%",
            "Players": ", ".join(p for _, p in g.players),
        })
    return pd.DataFrame(rows)

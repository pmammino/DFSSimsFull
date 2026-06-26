"""
contest_review.py  —  Post-slate grading and performance analysis
=================================================================
Parses a DraftKings contest-standings CSV (the dual-column format DK exports),
finds a user's entries by username, scores them against the actual field, and
generates a diagnostic report comparing projected vs actual performance.

Public API
----------
    parse_contest_csv(bytes_or_str)  ->  ContestData
    find_user_entries(contest, username)  ->  list[ContestEntry]
    grade_user_entries(contest, user_entries)  ->  GradedResult
    compare_sim_vs_actual(contest, sim_scores, n_sim)  ->  SimComparison
    build_diagnosis(contest, graded, projected, proj_own)  ->  list[DiagnosticInsight]
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
    """Strip the ' (X/Y)' multi-entry suffix from a DK entry name."""
    return _ENTRY_SUFFIX.sub("", entry_name).strip()


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
    username: str           # entry_name with (X/Y) stripped
    points: float
    players: list[tuple[str, str]]   # [(position, name), ...]
    norm_players: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.norm_players = [_norm(p) for _, p in self.players]


@dataclass
class ContestData:
    entries: list[ContestEntry]
    player_actuals: dict[str, PlayerActual]   # norm_name -> PlayerActual
    date: Optional[str] = None

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
    lineup_id: int                     # 1-based among user's entries
    entry_name: str                    # original DK entry name
    players: list[tuple[str, str]]
    actual_score: float
    field_rank: int                    # 1-based rank among ALL field entries
    field_pct: float                   # fraction of field beaten (0–1)
    player_scores: dict[str, float]    # norm_name -> fpts


@dataclass
class GradedResult:
    graded: list[GradedLineup]
    contest: ContestData


@dataclass
class PlayerSimComparison:
    name: str
    position: str
    actual_fpts: float
    proj_mean: float
    proj_p10: float
    proj_p50: float
    proj_p90: float
    actual_percentile: float    # where actual fell in sim dist (0–100)
    pct_drafted: float


@dataclass
class LineupSimComparison:
    lineup_id: int
    entry_name: str
    actual_score: float
    proj_mean: float            # mean of (sum of player sims)
    proj_p25: float
    proj_p75: float
    proj_max: float
    actual_percentile: float    # where actual score fell in projected dist


@dataclass
class SimComparison:
    players: list[PlayerSimComparison]
    lineups: list[LineupSimComparison]
    # summary
    avg_player_pct: float   # mean percentile across all slated players
    avg_lineup_pct: float   # mean percentile across user lineups


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse_contest_csv(raw: bytes | str) -> ContestData:
    """
    Parse the DK contest-standings dual-column CSV.

    Left block: Rank, EntryId, EntryName, TimeRemaining, Points, Lineup
    Right block: Player, Roster Position, %Drafted, FPTS
    Blank column separates them.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")

    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Rename blank separator column
    cols = list(df.columns)
    blank_idx = [i for i, c in enumerate(cols) if c == ""]
    if blank_idx:
        df = df.rename(columns={cols[blank_idx[0]]: "_sep"})

    entries: list[ContestEntry] = []
    player_actuals: dict[str, PlayerActual] = {}

    for _, row in df.iterrows():
        # --- left block ---
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

        # --- right block ---
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
    """
    Return all contest entries whose username matches (case-insensitive).
    Matches the stripped username (e.g. "ImaDonk11" matches "ImaDonk11 (15/20)").
    """
    target = username.strip().lower()
    return [e for e in contest.entries if e.username.lower() == target]


def list_usernames(contest: ContestData) -> list[str]:
    """Return sorted unique usernames from the contest."""
    seen: dict[str, str] = {}
    for e in contest.entries:
        lc = e.username.lower()
        if lc not in seen:
            seen[lc] = e.username
    return sorted(seen.values(), key=str.lower)


# ---------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------
def grade_user_entries(contest: ContestData,
                       user_entries: list[ContestEntry]) -> GradedResult:
    """
    Score and rank each of the user's entries using actual player FPTS.
    Field rank is computed by inserting the lineup's actual score into the
    full contest score array.
    """
    all_scores = np.sort(contest.scores)[::-1]  # descending

    graded: list[GradedLineup] = []
    for idx, entry in enumerate(user_entries, 1):
        total = 0.0
        pscores: dict[str, float] = {}
        for pos, name in entry.players:
            pa = contest.player_actual(name)
            pts = pa.fpts if pa else 0.0
            total += pts
            pscores[_norm(name)] = pts

        # The entry already exists in the field so its rank is the entry's rank
        field_rank = entry.rank
        field_pct = 1.0 - (field_rank - 1) / max(1, len(all_scores))

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
# sim vs actual comparison
# ---------------------------------------------------------------------------
def compare_sim_vs_actual(contest: ContestData,
                          sim_scores: dict[str, np.ndarray],
                          graded: Optional[GradedResult] = None) -> SimComparison:
    """
    For each player with both a sim distribution and an actual FPTS, compute
    where the actual result fell in the sim distribution.

    For each graded lineup, build the lineup's projected score distribution
    by summing the sim arrays for its players, then compute where the actual
    score fell in that distribution.

    sim_scores: dict mapping norm_name -> 1-D float array of sim scores
    """
    player_comps: list[PlayerSimComparison] = []

    for pa in sorted(contest.player_actuals.values(), key=lambda p: p.position + p.name):
        arr = sim_scores.get(pa.norm_name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        actual_pct = float(np.mean(arr <= pa.fpts) * 100)
        player_comps.append(PlayerSimComparison(
            name=pa.name,
            position=pa.position,
            actual_fpts=pa.fpts,
            proj_mean=float(arr.mean()),
            proj_p10=float(np.percentile(arr, 10)),
            proj_p50=float(np.percentile(arr, 50)),
            proj_p90=float(np.percentile(arr, 90)),
            actual_percentile=actual_pct,
            pct_drafted=pa.pct_drafted,
        ))

    lineup_comps: list[LineupSimComparison] = []
    if graded:
        for g in graded.graded:
            # Build the lineup's projected score distribution
            lineup_dist: Optional[np.ndarray] = None
            for pos, name in g.players:
                arr = sim_scores.get(_norm(name))
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=float)
                lineup_dist = arr if lineup_dist is None else lineup_dist + arr

            if lineup_dist is None:
                continue

            actual_pct = float(np.mean(lineup_dist <= g.actual_score) * 100)
            lineup_comps.append(LineupSimComparison(
                lineup_id=g.lineup_id,
                entry_name=g.entry_name,
                actual_score=g.actual_score,
                proj_mean=float(lineup_dist.mean()),
                proj_p25=float(np.percentile(lineup_dist, 25)),
                proj_p75=float(np.percentile(lineup_dist, 75)),
                proj_max=float(lineup_dist.max()),
                actual_percentile=actual_pct,
            ))

    avg_player_pct = float(np.mean([p.actual_percentile for p in player_comps])) if player_comps else 0.0
    avg_lineup_pct = float(np.mean([l.actual_percentile for l in lineup_comps])) if lineup_comps else 0.0

    return SimComparison(
        players=player_comps,
        lineups=lineup_comps,
        avg_player_pct=avg_player_pct,
        avg_lineup_pct=avg_lineup_pct,
    )


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------
_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_diagnosis(contest: ContestData,
                    graded: Optional[GradedResult] = None,
                    sim_comparison: Optional[SimComparison] = None,
                    projected: Optional[dict[str, float]] = None,
                    proj_own: Optional[dict[str, float]] = None) -> list[DiagnosticInsight]:
    insights: list[DiagnosticInsight] = []
    actuals = contest.player_actuals
    scores = contest.scores
    top10_score = contest.percentile_score(0.90)
    cash_score = contest.percentile_score(0.50)

    # ---- 1. Field summary ------------------------------------------------
    insights.append(DiagnosticInsight(
        category="Field",
        severity="low",
        headline="Contest score distribution",
        detail=(
            f"{contest.n_entries:,} entries · "
            f"Winner: **{scores.max():.2f} pts** · "
            f"Top-10% line: **{top10_score:.2f} pts** · "
            f"Median / cash line: **{cash_score:.2f} pts**"
        ),
    ))

    # ---- 2. Pitcher analysis ---------------------------------------------
    pitchers = [pa for pa in actuals.values() if pa.position == "P"]
    pitchers.sort(key=lambda p: -p.fpts)

    if pitchers:
        top_entries = [e for e in contest.entries if e.points >= top10_score]
        p_counter: Counter[str] = Counter()
        for e in top_entries:
            for pos, name in e.players:
                if pos == "P":
                    p_counter[_norm(name)] += 1

        if p_counter and top_entries:
            top_p_norm, top_p_cnt = p_counter.most_common(1)[0]
            top_p_pct_in_top = top_p_cnt / len(top_entries) * 100
            top_p_pa = actuals.get(top_p_norm)
            if top_p_pa and top_p_pct_in_top > 40:
                proj_note = ""
                if projected and top_p_norm in projected:
                    diff = top_p_pa.fpts - projected[top_p_norm]
                    direction = "exceeded" if diff > 0 else "missed"
                    proj_note = (f" Projection {direction} by **{abs(diff):.1f} pts** "
                                 f"(proj: {projected[top_p_norm]:.1f}, actual: {top_p_pa.fpts:.1f}).")
                insights.append(DiagnosticInsight(
                    category="Pitcher",
                    severity="high",
                    headline=f"{top_p_pa.name} appeared in {top_p_pct_in_top:.0f}% of top-10% lineups",
                    detail=(
                        f"**{top_p_pa.name}** scored **{top_p_pa.fpts:.2f} pts** at "
                        f"**{top_p_pa.pct_drafted:.1f}%** ownership and was the dominant "
                        f"pitcher in top-finishing lineups.{proj_note}"
                    ),
                ))

        # Low-owned pitchers who blew up
        for p in pitchers[:6]:
            if p.pct_drafted < 25 and p.fpts > 18:
                insights.append(DiagnosticInsight(
                    category="Pitcher",
                    severity="medium",
                    headline=f"Low-owned pitcher {p.name} scored {p.fpts:.2f} pts at {p.pct_drafted:.1f}% ownership",
                    detail=(
                        f"**{p.name}** scored **{p.fpts:.2f} pts** while drafted by only "
                        f"**{p.pct_drafted:.1f}%** of the field — a leverage spot that "
                        f"was broadly missed."
                    ),
                ))

    # ---- 3. Ownership accuracy -------------------------------------------
    if proj_own:
        over_owned, under_owned = [], []
        for pa in actuals.values():
            p_own = proj_own.get(pa.norm_name)
            if p_own is None:
                continue
            delta = pa.pct_drafted - p_own
            if delta > 15:
                over_owned.append((pa.name, p_own, pa.pct_drafted, delta))
            elif delta < -15:
                under_owned.append((pa.name, p_own, pa.pct_drafted, delta))

        if over_owned:
            over_owned.sort(key=lambda x: -x[3])
            names = ", ".join(f"**{n}** (proj {p:.0f}% → actual {a:.0f}%)"
                              for n, p, a, _ in over_owned[:3])
            insights.append(DiagnosticInsight(
                category="Ownership",
                severity="medium",
                headline=f"{len(over_owned)} player(s) came in significantly over projected ownership",
                detail=f"Underestimating chalk creates false uniqueness — you share more lineups than expected. Notable: {names}.",
            ))

        if under_owned:
            under_owned.sort(key=lambda x: x[3])
            names = ", ".join(f"**{n}** (proj {p:.0f}% → actual {a:.0f}%)"
                              for n, p, a, _ in under_owned[:3])
            insights.append(DiagnosticInsight(
                category="Ownership",
                severity="medium",
                headline=f"{len(under_owned)} player(s) came in significantly under projected ownership",
                detail=f"Overestimating chalk leads to false leverage assumptions — players you faded were actually rare. Notable: {names}.",
            ))

    # ---- 4. Projection busts / booms -------------------------------------
    if projected:
        busts, booms = [], []
        for pa in actuals.values():
            p_pts = projected.get(pa.norm_name)
            if p_pts is None:
                continue
            err = pa.fpts - p_pts
            if err < -8:
                busts.append((pa.name, p_pts, pa.fpts, err))
            elif err > 10:
                booms.append((pa.name, p_pts, pa.fpts, err))

        if busts:
            busts.sort(key=lambda x: x[3])
            parts = [f"**{n}** (proj {p:.1f} → actual {a:.1f})" for n, p, a, _ in busts[:4]]
            insights.append(DiagnosticInsight(
                category="Projection",
                severity="high",
                headline=f"{len(busts)} player(s) significantly underperformed projections",
                detail="Large busts: " + ", ".join(parts) + ".",
            ))

        if booms:
            booms.sort(key=lambda x: -x[3])
            parts = [f"**{n}** (proj {p:.1f} → actual {a:.1f})" for n, p, a, _ in booms[:4]]
            insights.append(DiagnosticInsight(
                category="Projection",
                severity="medium",
                headline=f"{len(booms)} value player(s) far exceeded projections",
                detail="Missed upside: " + ", ".join(parts) + ".",
            ))

    # ---- 5. Sim calibration (if sim comparison available) ----------------
    if sim_comparison and sim_comparison.players:
        # Players whose actual was in the bottom 10% of their sim dist
        bad_luck = [p for p in sim_comparison.players if p.actual_percentile < 10]
        good_luck = [p for p in sim_comparison.players if p.actual_percentile > 90]

        if bad_luck:
            bad_luck.sort(key=lambda p: p.actual_percentile)
            parts = [f"**{p.name}** ({p.actual_fpts:.1f} pts, bottom {p.actual_percentile:.0f}%ile of sim)"
                     for p in bad_luck[:3]]
            insights.append(DiagnosticInsight(
                category="Sim Calibration",
                severity="medium",
                headline=f"{len(bad_luck)} player(s) fell in the bottom 10% of their projected distribution",
                detail=(
                    "These outcomes were plausible from the model's perspective but represented "
                    "low-probability outcomes that hurt slate scores: " + ", ".join(parts) + "."
                ),
            ))

        if good_luck:
            good_luck.sort(key=lambda p: -p.actual_percentile)
            parts = [f"**{p.name}** ({p.actual_fpts:.1f} pts, top {100 - p.actual_percentile:.0f}%ile of sim)"
                     for p in good_luck[:3]]
            insights.append(DiagnosticInsight(
                category="Sim Calibration",
                severity="low",
                headline=f"{len(good_luck)} player(s) exceeded 90% of their projected sim outcomes",
                detail="Outperformers relative to projections: " + ", ".join(parts) + ".",
            ))

        # Overall sim calibration
        avg_pct = sim_comparison.avg_player_pct
        if avg_pct < 40:
            insights.append(DiagnosticInsight(
                category="Sim Calibration",
                severity="medium",
                headline=f"Slate ran cold vs projections (avg player at {avg_pct:.0f}th percentile of sim)",
                detail=(
                    "The field-wide average actual performance fell below the projected median. "
                    "This suggests either systematic over-projection or a genuinely low-scoring slate."
                ),
            ))
        elif avg_pct > 60:
            insights.append(DiagnosticInsight(
                category="Sim Calibration",
                severity="low",
                headline=f"Slate ran hot vs projections (avg player at {avg_pct:.0f}th percentile of sim)",
                detail=(
                    "The field-wide average actual performance exceeded the projected median — "
                    "a high-scoring slate overall."
                ),
            ))

    # ---- 6. Stack signals ------------------------------------------------
    top_entries = [e for e in contest.entries if e.points >= top10_score]
    if top_entries:
        player_top_count: Counter[str] = Counter()
        for e in top_entries:
            for nm in e.norm_players:
                player_top_count[nm] += 1

        top_in_top = player_top_count.most_common(6)
        if top_in_top:
            parts = []
            for norm_nm, cnt in top_in_top:
                pa = actuals.get(norm_nm)
                display = pa.name if pa else norm_nm
                pct_top = cnt / len(top_entries) * 100
                parts.append(f"**{display}** ({pct_top:.0f}%)")
            insights.append(DiagnosticInsight(
                category="Stack",
                severity="low",
                headline="Players most common in top-10% finishes",
                detail="Appearance rate in top-10% lineups: " + ", ".join(parts) + ".",
            ))

    # ---- 7. Your lineup performance --------------------------------------
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
            headline=f"{n_gl} lineup(s) evaluated",
            detail=(
                f"Average actual score: **{avg_score:.2f} pts** "
                f"(beat **{avg_pct:.1f}%** of the field on average). "
                f"Best lineup: **{best.entry_name}** at **{best.actual_score:.2f} pts** "
                f"(rank **{best.field_rank:,}** / {contest.n_entries:,}). "
                f"Weakest: **{worst.entry_name}** at **{worst.actual_score:.2f} pts** "
                f"(rank **{worst.field_rank:,}**)."
            ),
        ))

        # Players with heavy exposure that scored poorly
        your_player_scores: dict[str, list[float]] = defaultdict(list)
        for g in gl:
            for pos, name in g.players:
                nm = _norm(name)
                pa = actuals.get(nm)
                your_player_scores[nm].append(pa.fpts if pa else 0.0)

        drag = []
        for nm, pts_list in your_player_scores.items():
            pa = actuals.get(nm)
            if pa is None:
                continue
            avg_pts = np.mean(pts_list)
            exposure = len(pts_list) / n_gl
            if avg_pts < 5 and exposure >= 0.25:
                drag.append((pa.name, avg_pts, exposure * 100))

        if drag:
            drag.sort(key=lambda x: x[1])
            parts = [f"**{n}** ({a:.1f} pts, {e:.0f}% exposure)"
                     for n, a, e in drag[:4]]
            insights.append(DiagnosticInsight(
                category="Your Lineups",
                severity="high",
                headline="High-exposure players who significantly underperformed",
                detail=(
                    "These players appeared in a large share of your lineups and scored poorly, "
                    "dragging down the overall portfolio: " + ", ".join(parts) + "."
                ),
            ))

        # Sim comparison for lineups
        if sim_comparison and sim_comparison.lineups:
            lc_list = sim_comparison.lineups
            avg_lc_pct = np.mean([lc.actual_percentile for lc in lc_list])
            low_luck = [lc for lc in lc_list if lc.actual_percentile < 25]
            if low_luck and avg_lc_pct < 40:
                parts = [f"**{lc.entry_name}** (actual {lc.actual_score:.1f} vs proj median {lc.proj_p25:.1f}–{lc.proj_p75:.1f})"
                         for lc in sorted(low_luck, key=lambda x: x.actual_percentile)[:3]]
                insights.append(DiagnosticInsight(
                    category="Your Lineups",
                    severity="medium",
                    headline=f"Your lineups ran below their projected distribution (avg {avg_lc_pct:.0f}th %ile)",
                    detail=(
                        "These lineups fell in the bottom quartile of their own projected score "
                        "distributions, suggesting poor slate luck rather than lineup construction "
                        "errors: " + ", ".join(parts) + "."
                    ),
                ))

    insights.sort(key=lambda i: _SEV_ORDER.get(i.severity, 9))
    return insights


@dataclass
class DiagnosticInsight:
    category: str
    severity: str   # "high" | "medium" | "low"
    headline: str
    detail: str


# ---------------------------------------------------------------------------
# helpers for the Streamlit tab
# ---------------------------------------------------------------------------
def actuals_to_df(contest: ContestData,
                  sim_comparison: Optional[SimComparison] = None,
                  projected: Optional[dict[str, float]] = None,
                  proj_own: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """Player actuals, optionally enriched with sim/projected columns."""
    # Build base from player_actuals, keyed by norm_name
    pa_map: dict[str, PlayerActual] = {pa.norm_name: pa
                                        for pa in contest.player_actuals.values()}
    # Merge sim comparison
    sim_map: dict[str, PlayerSimComparison] = {}
    if sim_comparison:
        sim_map = {p.name: p for p in sim_comparison.players}

    rows = []
    for pa in sorted(pa_map.values(), key=lambda p: -p.fpts):
        row: dict = {
            "Player": pa.name,
            "Position": pa.position,
            "Actual FPTS": pa.fpts,
            "Ownership %": pa.pct_drafted,
        }
        if projected:
            proj_val = projected.get(pa.norm_name)
            if proj_val is not None:
                row["Proj FPTS"] = round(proj_val, 2)
                row["FPTS vs Proj"] = round(pa.fpts - proj_val, 2)
        if proj_own:
            po = proj_own.get(pa.norm_name)
            if po is not None:
                row["Proj Own%"] = round(po, 1)
                row["Own% Delta"] = round(pa.pct_drafted - po, 1)
        sc = sim_map.get(pa.name)
        if sc is not None:
            row["Sim Mean"] = round(sc.proj_mean, 2)
            row["Sim P10"] = round(sc.proj_p10, 1)
            row["Sim P50"] = round(sc.proj_p50, 1)
            row["Sim P90"] = round(sc.proj_p90, 1)
            row["Actual %ile in Sim"] = round(sc.actual_percentile, 1)
        rows.append(row)
    return pd.DataFrame(rows)


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


def graded_to_df(graded: GradedResult,
                 sim_comparison: Optional[SimComparison] = None) -> pd.DataFrame:
    lc_map: dict[int, LineupSimComparison] = {}
    if sim_comparison:
        lc_map = {lc.lineup_id: lc for lc in sim_comparison.lineups}

    rows = []
    for g in sorted(graded.graded, key=lambda x: x.field_rank):
        row: dict = {
            "Entry": g.entry_name,
            "Actual Score": round(g.actual_score, 2),
            "Field Rank": g.field_rank,
            "Field %ile": f"{g.field_pct * 100:.1f}%",
        }
        lc = lc_map.get(g.lineup_id)
        if lc:
            row["Proj Mean"] = round(lc.proj_mean, 1)
            row["Proj IQR"] = f"{lc.proj_p25:.1f}–{lc.proj_p75:.1f}"
            row["Actual %ile in Sim"] = f"{lc.actual_percentile:.0f}%"
        row["Players"] = ", ".join(p for _, p in g.players)
        rows.append(row)
    return pd.DataFrame(rows)

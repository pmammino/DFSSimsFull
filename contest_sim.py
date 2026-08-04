"""
contest_sim.py — the per-sim contest scoring kernel.

Given the field and candidate score matrices (shape ``(n_sim, n_entries)``), rank
every candidate against the field in each sim and return Win/Top10/Top100 counts,
mean finishing place, a compact place histogram, exact best/worst place, and
(optionally) the field's score at a ladder of finishing places.

This is the hottest loop in a Run. numpy releases the GIL inside ``sort`` and
``searchsorted``, so the sims are split into contiguous chunks and run across a
small thread pool — the result is bit-identical to the serial version (integer
counts/min/max/per-sim rows are order-independent), just faster on multi-core
(e.g. a dedicated-CPU instance). Set ``CONTEST_WORKERS`` to override the worker
count (1 disables threading).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _contest_workers(n_sim):
    """How many threads to score the contest with. Serial for small workloads
    (thread setup isn't worth it) or a single core; capped so we don't
    oversubscribe. ``CONTEST_WORKERS`` env var forces a specific count."""
    env = os.environ.get("CONTEST_WORKERS")
    if env:
        try:
            w = int(env)
        except ValueError:
            w = 0
        if w >= 1:
            return w if n_sim >= 1000 else 1
    cpu = os.cpu_count() or 1
    if cpu <= 1 or n_sim < 2000:
        return 1
    return min(cpu, 8)


def _contest_chunk(field_mat, cand_mat, lo, hi, n_field, edges, nb, take, cut_scores):
    """Score sims [lo, hi). Returns partial (wins, t10, t100, ps, best, worst,
    counts); writes its own rows of the shared ``cut_scores`` (disjoint, so no
    races). Identical math to the original per-sim loop."""
    N = cand_mat.shape[1]
    wins = np.zeros(N, np.int64); t10 = np.zeros(N, np.int64)
    t100 = np.zeros(N, np.int64); ps = np.zeros(N, np.int64)
    best = np.full(N, n_field + 1, np.int64); worst = np.zeros(N, np.int64)
    counts = np.zeros((N, nb), np.int32)
    idx = np.arange(N)
    for s in range(lo, hi):
        fs = np.sort(field_mat[s]); cv = cand_mat[s]
        pl = (n_field - np.searchsorted(fs, cv, side="right")) + 1
        wins += (pl == 1); t10 += (pl <= 10); t100 += (pl <= 100); ps += pl
        np.minimum(best, pl, out=best); np.maximum(worst, pl, out=worst)
        b = np.clip(np.searchsorted(edges, pl, side="right") - 1, 0, nb - 1)
        np.add.at(counts, (idx, b), 1)
        if cut_scores is not None:
            cut_scores[s] = fs[take]
    return wins, t10, t100, ps, best, worst, counts


def run_contest_dist(field_mat, cand_mat, n_sim, n_field, nbins=24, cut_places=None):
    """Score each candidate against the field per sim and capture its
    finishing-place distribution as a compact ~`nbins`-bucket histogram. Returns
    (wins, t10, t100, avg, dist) with exact best/mean/worst places.

    If `cut_places` is given (ascending place indices), the field's score at each
    of those places is also captured per sim into ``dist["field_cut_scores"]``
    (shape ``(n_sim, len(cut_places))``) — piggybacking on the per-sim sort so the
    payout-aware export gets the field placement ladder without a second pass."""
    N = cand_mat.shape[1]
    nb_target = max(6, min(int(nbins), int(n_field)))
    edges = np.unique(np.linspace(1, n_field + 1, nb_target + 1).astype(np.int64))
    nb = len(edges) - 1

    cut_scores = None
    take = None
    if cut_places is not None and len(cut_places):
        cut_places = np.asarray(cut_places, np.int64)
        cut_scores = np.empty((n_sim, len(cut_places)), np.float32)
        # ascending-sorted field: score for place p is the p-th highest total
        take = n_field - cut_places

    workers = _contest_workers(n_sim)
    if workers <= 1:
        parts = [_contest_chunk(field_mat, cand_mat, 0, n_sim, n_field,
                                edges, nb, take, cut_scores)]
    else:
        bounds = np.linspace(0, n_sim, workers + 1).astype(int)
        spans = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with ThreadPoolExecutor(max_workers=len(spans)) as ex:
            parts = list(ex.map(
                lambda sp: _contest_chunk(field_mat, cand_mat, sp[0], sp[1],
                                          n_field, edges, nb, take, cut_scores),
                spans))

    wins = sum(p[0] for p in parts)
    t10 = sum(p[1] for p in parts)
    t100 = sum(p[2] for p in parts)
    ps = sum(p[3] for p in parts)
    best = parts[0][4].copy(); worst = parts[0][5].copy(); counts = parts[0][6].copy()
    for p in parts[1:]:
        np.minimum(best, p[4], out=best)
        np.maximum(worst, p[5], out=worst)
        counts += p[6]

    dist = {"edges": edges, "counts": counts, "best": best, "worst": worst,
            "mean": ps / n_sim}
    if cut_scores is not None:
        dist["field_cut_scores"] = cut_scores
        dist["cut_places"] = cut_places
    return wins, t10, t100, ps / n_sim, dist

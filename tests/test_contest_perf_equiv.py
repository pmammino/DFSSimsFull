"""Equivalence tests for the performance-optimized contest kernels.

Guards that the vectorized/threaded implementations return the SAME results as
the original straightforward loops:

  * stage_d.score_matrix (sparse incidence matmul) == per-player accumulation,
    to float32 rounding.
  * contest_sim.run_contest_dist (threaded) == the original serial per-sim loop,
    exactly (integer counts / min / max / per-sim rows are order-independent).
"""
import numpy as np
import pytest

import stage_d
from contest_sim import run_contest_dist


class _PL:
    def __init__(self, name):
        self.Name = name


def _mk(rng, names, n, ppl=10):
    return [{"players": [_PL(names[i]) for i in rng.choice(len(names), ppl, replace=False)]}
            for _ in range(n)]


def _score_matrix_loop(lineups, score, n_sim):
    out = np.zeros((n_sim, len(lineups)), np.float32)
    for j, lu in enumerate(lineups):
        col = out[:, j]
        for pl in lu["players"]:
            col += score[stage_d.norm(pl.Name)]
    return out


def _run_contest_serial(field_mat, cand_mat, n_sim, n_field, nbins=24, cut_places=None):
    N = cand_mat.shape[1]
    wins = np.zeros(N, np.int64); t10 = np.zeros(N, np.int64)
    t100 = np.zeros(N, np.int64); ps = np.zeros(N, np.int64)
    best = np.full(N, n_field + 1, np.int64); worst = np.zeros(N, np.int64)
    nb_target = max(6, min(int(nbins), int(n_field)))
    edges = np.unique(np.linspace(1, n_field + 1, nb_target + 1).astype(np.int64))
    nb = len(edges) - 1
    counts = np.zeros((N, nb), np.int32); idx = np.arange(N)
    cut_scores = None
    if cut_places is not None and len(cut_places):
        cut_places = np.asarray(cut_places, np.int64)
        cut_scores = np.empty((n_sim, len(cut_places)), np.float32)
        take = n_field - cut_places
    for s in range(n_sim):
        fs = np.sort(field_mat[s]); cv = cand_mat[s]
        pl = (n_field - np.searchsorted(fs, cv, side="right")) + 1
        wins += (pl == 1); t10 += (pl <= 10); t100 += (pl <= 100); ps += pl
        best = np.minimum(best, pl); worst = np.maximum(worst, pl)
        b = np.clip(np.searchsorted(edges, pl, side="right") - 1, 0, nb - 1)
        np.add.at(counts, (idx, b), 1)
        if cut_scores is not None:
            cut_scores[s] = fs[take]
    dist = {"edges": edges, "counts": counts, "best": best, "worst": worst,
            "mean": ps / n_sim}
    if cut_scores is not None:
        dist["field_cut_scores"] = cut_scores
    return wins, t10, t100, ps / n_sim, dist


def test_score_matrix_matches_loop():
    rng = np.random.default_rng(7)
    names = [f"Player {i}" for i in range(120)]
    score = {stage_d.norm(n): rng.gamma(2.0, 5.0, 2000).astype(np.float32) for n in names}
    lineups = _mk(rng, names, 300)
    a = _score_matrix_loop(lineups, score, 2000)
    b = stage_d.score_matrix(lineups, score, 2000)
    assert a.shape == b.shape == (2000, 300)
    assert np.allclose(a, b, rtol=1e-4, atol=1e-2)


def test_score_matrix_empty():
    assert stage_d.score_matrix([], {}, 500).shape == (500, 0)


@pytest.mark.parametrize("workers", ["1", "4"])
def test_run_contest_dist_matches_serial(monkeypatch, workers):
    monkeypatch.setenv("CONTEST_WORKERS", workers)
    rng = np.random.default_rng(11)
    n_sim, n_field, n_cand = 1500, 800, 200
    field_mat = rng.gamma(2, 5, (n_sim, n_field)).astype(np.float32)
    cand_mat = rng.gamma(2, 5, (n_sim, n_cand)).astype(np.float32)
    cut_places = np.array([1, 10, 50, 100, 400], np.int64)

    w0, a0, b0, m0, d0 = _run_contest_serial(
        field_mat, cand_mat, n_sim, n_field, cut_places=cut_places)
    w1, a1, b1, m1, d1 = run_contest_dist(
        field_mat, cand_mat, n_sim, n_field, cut_places=cut_places)

    assert np.array_equal(w0, w1)
    assert np.array_equal(a0, a1)
    assert np.array_equal(b0, b1)
    assert np.array_equal(m0, m1)
    assert np.array_equal(d0["counts"], d1["counts"])
    assert np.array_equal(d0["best"], d1["best"])
    assert np.array_equal(d0["worst"], d1["worst"])
    assert np.array_equal(d0["edges"], d1["edges"])
    assert np.array_equal(d0["field_cut_scores"], d1["field_cut_scores"])

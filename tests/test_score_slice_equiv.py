"""The per-slice scoring path (incidence + score_from_incidence) must equal the
full score_matrix then row-index — that's what keeps the memory optimization
result-identical — and the threaded contest must handle the transposed
(F-contiguous) arrays that path returns."""
import numpy as np

import stage_d
from contest_sim import run_contest_dist, _effective_cpus


class _PL:
    def __init__(self, name):
        self.Name = name


def _mk(rng, names, n, ppl=10):
    return [{"players": [_PL(names[i]) for i in rng.choice(len(names), ppl, replace=False)]}
            for _ in range(n)]


def test_score_from_incidence_matches_full_then_index():
    rng = np.random.default_rng(3)
    names = [f"Player {i}" for i in range(140)]
    score = {stage_d.norm(n): rng.gamma(2.0, 5.0, 3000).astype(np.float32) for n in names}
    lineups = _mk(rng, names, 400)

    full = stage_d.score_matrix(lineups, score, 3000)
    name_index = {nm: i for i, nm in enumerate(score)}
    M = stage_d.incidence(lineups, name_index)

    for idx in (np.arange(3000),                              # all sims
                np.sort(rng.choice(3000, 1200, replace=False)),
                np.array([0, 5, 9, 2999])):
        sliced = stage_d.score_from_incidence(M, lineups, name_index, score, idx)
        assert sliced.shape == (len(idx), len(lineups))
        assert np.array_equal(sliced, full[idx])


def test_score_from_incidence_dense_fallback_matches():
    rng = np.random.default_rng(4)
    names = [f"P{i}" for i in range(60)]
    score = {stage_d.norm(n): rng.gamma(2.0, 5.0, 500).astype(np.float32) for n in names}
    lineups = _mk(rng, names, 50)
    name_index = {nm: i for i, nm in enumerate(score)}
    M = stage_d.incidence(lineups, name_index)
    idx = np.sort(rng.choice(500, 200, replace=False))
    with_scipy = stage_d.score_from_incidence(M, lineups, name_index, score, idx)
    dense = stage_d.score_from_incidence(None, lineups, name_index, score, idx)  # M=None
    assert np.allclose(with_scipy, dense, rtol=1e-4, atol=1e-2)


def test_contest_handles_transposed_view():
    """score_from_incidence returns an (M@S).T view (F-contiguous); the contest
    must give the same answer as on a C-contiguous copy of the same data."""
    rng = np.random.default_rng(5)
    n_sim, n_field, n_cand = 1200, 600, 150
    fld = np.asfortranarray(rng.gamma(2, 5, (n_sim, n_field)).astype(np.float32))
    cnd = np.asfortranarray(rng.gamma(2, 5, (n_sim, n_cand)).astype(np.float32))
    a = run_contest_dist(fld, cnd, n_sim, n_field, cut_places=np.array([1, 10, 100]))
    b = run_contest_dist(np.ascontiguousarray(fld), np.ascontiguousarray(cnd),
                         n_sim, n_field, cut_places=np.array([1, 10, 100]))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])
    assert np.array_equal(a[4]["field_cut_scores"], b[4]["field_cut_scores"])


def test_effective_cpus_reads_cgroup(tmp_path, monkeypatch):
    """A cgroup-v2 quota of half a core should cap effective CPUs at 1 (not the
    host's core count) so we don't oversubscribe."""
    import builtins
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return real_open(tmp_path / "cpu.max", *a, **k)
        return real_open(path, *a, **k)

    (tmp_path / "cpu.max").write_text("50000 100000\n")   # 0.5 vCPU
    monkeypatch.setattr(builtins, "open", fake_open)
    assert _effective_cpus() == 1

    (tmp_path / "cpu.max").write_text("300000 100000\n")  # 3 vCPU
    assert _effective_cpus() == 3

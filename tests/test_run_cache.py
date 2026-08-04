"""Tests for run_cache: signature stability/sensitivity and reuse semantics."""
import numpy as np
import pandas as pd

import run_cache


def _df():
    return pd.DataFrame({"Name": ["a", "b", "c"], "Salary": [4000, 5000, 3000],
                         "Own": [10.0, 20.0, 5.0]})


def test_sig_stable_for_equal_inputs():
    a = run_cache.sig(_df(), {"x": [1, 2], "y": 0.5}, 7, np.arange(4))
    b = run_cache.sig(_df(), {"y": 0.5, "x": [1, 2]}, 7, np.arange(4))  # dict order flip
    assert a == b


def test_sig_changes_on_dataframe_value():
    df2 = _df(); df2.loc[0, "Salary"] = 4001
    assert run_cache.sig(_df()) != run_cache.sig(df2)


def test_sig_changes_on_row_order():
    assert run_cache.sig(_df()) != run_cache.sig(_df().iloc[::-1])


def test_sig_changes_on_scalar_and_array():
    assert run_cache.sig(7) != run_cache.sig(8)
    assert run_cache.sig(np.arange(4)) != run_cache.sig(np.arange(4) + 1)
    assert run_cache.sig(1) != run_cache.sig(1.0)  # type-sensitive


def test_reuse_hit_and_miss():
    store, calls = {}, {"n": 0}

    def build():
        calls["n"] += 1
        return ["lineup"] * 3

    v1, hit1 = run_cache.reuse(store, "k", "sigA", build)
    assert hit1 is False and calls["n"] == 1 and v1 == ["lineup"] * 3

    v2, hit2 = run_cache.reuse(store, "k", "sigA", build)   # same sig -> hit
    assert hit2 is True and calls["n"] == 1 and v2 is v1

    v3, hit3 = run_cache.reuse(store, "k", "sigB", build)   # new sig -> rebuild
    assert hit3 is False and calls["n"] == 2

    # a stale entry is dropped before rebuild (no lingering reference to the old)
    assert store["k"]["sig"] == "sigB"

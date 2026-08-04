"""
run_cache.py — reuse expensive per-Run build artifacts across Runs.

The candidate and field lineups are fully determined by their build inputs plus a
seed, yet the app rebuilds them from scratch on every Run — even when only a
downstream knob changed. This module lets the app reuse them when the *exact*
build inputs are unchanged.

Correctness is by construction: the cache key is a signature of the **immediate
inputs handed to the builder** (the actual pool DataFrames, the params, the
seeds, the sizes) — not the upstream UI knobs — so any change that would alter a
build necessarily changes the signature and forces a rebuild. A hit returns the
identical object a fresh build would have produced, so results are unchanged.
"""
import hashlib

import numpy as np
import pandas as pd


def _update(h, obj):
    """Feed a stable, type-tagged serialization of `obj` into hash `h`."""
    if isinstance(obj, pd.DataFrame):
        h.update(b"DF")
        h.update((",".join(map(str, obj.columns))).encode())
        # index=False so incidental index differences don't cause false misses;
        # row order is preserved (it can affect builder sampling), values do too.
        h.update(pd.util.hash_pandas_object(obj, index=False).values.tobytes())
    elif isinstance(obj, pd.Series):
        h.update(b"SR")
        h.update(pd.util.hash_pandas_object(obj, index=False).values.tobytes())
    elif isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(b"ND"); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    elif isinstance(obj, dict):
        h.update(b"D{")
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            h.update(("K:" + str(k)).encode()); _update(h, obj[k])
        h.update(b"}D")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            _update(h, x)
        h.update(b"]")
    elif obj is None or isinstance(obj, (str, int, float, bool,
                                         np.integer, np.floating)):
        h.update(("S:" + repr(obj)).encode())
    else:
        # Last resort — repr is stable for the simple param objects we sign.
        h.update(("R:" + repr(obj)).encode())


def sig(*parts):
    """A hex signature of all `parts` (DataFrames, arrays, dicts, scalars, …)."""
    h = hashlib.sha1()
    for p in parts:
        _update(h, p)
        h.update(b"|")
    return h.hexdigest()


def reuse(store, key, signature, build_fn):
    """Return (value, hit). On a signature match return the cached value; else
    drop the stale entry (releasing its memory) and rebuild via `build_fn`.

    `store` is a dict-like keyed by `key` (e.g. ``st.session_state``); each entry
    is ``{"sig": ..., "val": ...}``. Only the latest signature per key is kept."""
    ent = store.get(key) if hasattr(store, "get") else (
        store[key] if key in store else None)
    if ent is not None and ent.get("sig") == signature:
        return ent["val"], True
    store[key] = None            # release the old value before building the new
    val = build_fn()
    store[key] = {"sig": signature, "val": val}
    return val, False

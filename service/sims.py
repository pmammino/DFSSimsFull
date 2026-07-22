"""sims.py — warm, in-memory access to the correlated DK-point sims.

The worker is a long-lived process, so unlike the Streamlit app (which reloads
per session via ``st.cache_resource``) we load the sim universe ONCE and keep
it resident. Loading is keyed on the artifact mtimes, so a refresh job that
rewrites the .npy files transparently busts the cache on next access.

Numeric logic is reused verbatim from ``stage_d`` (``load_sims``) and mirrors
``app.py``'s ``cached_player_table`` so the web app is byte-for-byte consistent
with the current Streamlit output.
"""
import threading

import numpy as np

import stage_d  # repo root on sys.path
from . import artifacts

_lock = threading.Lock()
_cache = {"key": None, "H": None, "P": None, "score": None, "n_sim": None,
          "table": None}


def _load(force_sync: bool = False):
    """Return the warm sim bundle, syncing from the object store and reloading
    from disk only when the artifacts have changed. Thread-safe."""
    artifacts.sync_from_store(force=force_sync)
    key = artifacts.sims_mtimes()
    with _lock:
        if _cache["key"] == key and _cache["H"] is not None:
            return _cache
        h, p = artifacts.sim_paths()
        if not h or not p:
            raise FileNotFoundError(
                "Sim artifacts not found. Seed them with "
                "scripts/push_artifacts.py or run the pipeline (run_slate.py). "
                "Looked for deliverables/hitter_dk_sims.npy and "
                "deliverables/pitcher_dk_sims.npy.")
        H, P, score, n_sim = stage_d.load_sims(h, p)
        _cache.update(key=key, H=H, P=P, score=score, n_sim=n_sim, table=None)
        return _cache


def reload():
    """Force a fresh pull + reload (call after a refresh job completes)."""
    return _load(force_sync=True)


def status():
    """Lightweight freshness/inventory info for a health or status endpoint."""
    c = _load()
    stamp = artifacts.build_stamp()
    return {
        "n_sim": c["n_sim"],
        "hitters": len(c["H"]),
        "pitchers": len(c["P"]),
        "remote_store": artifacts.remote_enabled(),
        "build_stamp": stamp,
    }


def player_table():
    """Per-player DK-point threshold table (mean, floor/median/ceiling, min/max,
    std, bust & boom rates) — a direct port of app.py:cached_player_table. Built
    once and cached alongside the sims."""
    c = _load()
    with _lock:
        if _cache["table"] is not None:
            return _cache["table"]
    rows = []
    for typ, D in (("Hitter", c["H"]), ("Pitcher", c["P"])):
        for nm, v in D.items():
            a = np.asarray(v, float)
            m = float(a.mean())
            rows.append({
                "Player": nm, "Type": typ,
                "Proj": round(m, 1),
                "Floor (p10)": round(float(np.percentile(a, 10)), 1),
                "Median": round(float(np.percentile(a, 50)), 1),
                "Ceiling (p90)": round(float(np.percentile(a, 90)), 1),
                "p99": round(float(np.percentile(a, 99)), 1),
                "Min": round(float(a.min()), 1),
                "Max": round(float(a.max()), 1),
                "Std": round(float(a.std()), 1),
                "Bust% (<=0)": round(100 * float((a <= 0).mean()), 1),
                "2x%": round(100 * float((a >= 2 * m).mean()), 1) if m > 0 else 0.0,
                "30+%": round(100 * float((a >= 30).mean()), 1)})
    rows.sort(key=lambda r: r["Proj"], reverse=True)
    with _lock:
        _cache["table"] = rows
    return rows


def _find_player_array(name: str):
    """Return the raw sim array for a player by exact or normalized name."""
    c = _load()
    for D in (c["H"], c["P"]):
        if name in D:
            return np.asarray(D[name], float)
    target = stage_d.norm(name)
    for D in (c["H"], c["P"]):
        for nm, v in D.items():
            if stage_d.norm(nm) == target:
                return np.asarray(v, float)
    return None


def player_distribution(name: str, nbins: int = 40):
    """Histogram of one player's ~10k sampled DK scores, plus summary stats —
    the data behind app.py:player_score_chart. Binning is done server-side so
    the client just renders."""
    arr = _find_player_array(name)
    if arr is None:
        return None
    counts, edges = np.histogram(arr, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "player": name,
        "n_sim": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "p10": round(float(np.percentile(arr, 10)), 2),
        "median": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "bins": [
            {"x": round(float(x), 2), "count": int(cnt)}
            for x, cnt in zip(centers, counts)
        ],
    }

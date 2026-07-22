"""runstore.py — hold recent run payloads in memory, keyed by run_id.

This replaces the Streamlit ``st.session_state["sim"]`` object. Because the
worker is a single long-lived process, keeping the last few runs resident lets
the Results/Export endpoints operate on the numpy arrays (score_pool, dist,
cands) without re-simulating — exactly the property the Streamlit tabs relied
on, but now shareable by run_id across clients.

Bounded LRU so memory stays flat under many runs. (A later phase can spill
payloads to the object store for durable, shareable links.)
"""
import threading
from collections import OrderedDict

_LOCK = threading.Lock()
_MAX = 8                       # keep the last N runs resident
_store: "OrderedDict[str, dict]" = OrderedDict()


def _gen_id(seq: int) -> str:
    # Deterministic, dependency-free id (no uuid/time — resume-safe in tests).
    return f"run_{seq:06d}"


_seq = 0


def put(payload: dict) -> str:
    """Store a run payload, evicting the oldest beyond the cap. Returns run_id."""
    global _seq
    with _LOCK:
        _seq += 1
        rid = _gen_id(_seq)
        _store[rid] = payload
        _store.move_to_end(rid)
        while len(_store) > _MAX:
            _store.popitem(last=False)
    return rid


def get(run_id: str) -> dict | None:
    with _LOCK:
        p = _store.get(run_id)
        if p is not None:
            _store.move_to_end(run_id)
        return p


def ids() -> list[str]:
    with _LOCK:
        return list(_store.keys())

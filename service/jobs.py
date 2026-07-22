"""jobs.py — run the heavy pipeline as a tracked background job.

This replaces the Streamlit app's blocking ``ensure_fresh`` + ``st.status``
spinner. The UI kicks a job (``POST /refresh``), gets a job_id back immediately,
and polls (``GET /refresh/status/{id}``) — so a multi-minute rebuild never
blocks a request. On success the worker publishes the new artifacts to the
object store (if configured) and reloads the warm sims.

The pipeline scripts are launched as subprocesses (exactly like
``app.py:run_script``) using the repo-root Python. NOTE: Stage C
(``run_slate.py``) is numpy-only and runs fine here; Stage B
(``run_pipeline.py``) additionally needs xgboost/scikit-learn/pybaseball, which
are in the repo's root requirements but NOT in the light worker image — run the
full pipeline in an environment that has them (see ARCHITECTURE.md).
"""
import os
import subprocess
import sys
import threading
from collections import OrderedDict, deque

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOCK = threading.Lock()
_JOBS: "OrderedDict[str, dict]" = OrderedDict()
_MAX_JOBS = 20
_seq = 0

# Only one heavy rebuild at a time per process (mirrors the app's RefreshLock
# intent; the cross-instance S3 lock lives in shared_store.RefreshLock).
_run_lock = threading.Lock()


def _new_job(kind: str) -> str:
    global _seq
    with _LOCK:
        _seq += 1
        jid = f"job_{_seq:04d}"
        _JOBS[jid] = {"id": jid, "kind": kind, "state": "queued",
                      "returncode": None, "log": deque(maxlen=200), "error": None}
        _JOBS[jid]["_log_str"] = ""
        while len(_JOBS) > _MAX_JOBS:
            _JOBS.popitem(last=False)
    return jid


def _set(jid: str, **kw):
    with _LOCK:
        job = _JOBS.get(jid)
        if job:
            job.update(kw)


def _append_log(jid: str, line: str):
    with _LOCK:
        job = _JOBS.get(jid)
        if job:
            job["log"].append(line.rstrip("\n"))


def status(jid: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(jid)
        if not job:
            return None
        return {"id": job["id"], "kind": job["kind"], "state": job["state"],
                "returncode": job["returncode"], "error": job["error"],
                "log_tail": list(job["log"])[-40:]}


def _run(jid: str, argv: list[str], on_success):
    if not _run_lock.acquire(blocking=False):
        _set(jid, state="skipped",
             error="Another refresh is already running on this worker.")
        return
    try:
        _set(jid, state="running")
        try:
            proc = subprocess.Popen(
                [sys.executable] + argv, cwd=_REPO_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1)
        except Exception as e:
            _set(jid, state="failed", error=f"could not launch: {e}")
            return
        for line in proc.stdout:            # stream output into the log ring
            _append_log(jid, line)
        proc.wait()
        _set(jid, returncode=proc.returncode)
        if proc.returncode == 0:
            try:
                on_success()
                _set(jid, state="succeeded")
            except Exception as e:
                _set(jid, state="failed", error=f"post-step failed: {e}")
        else:
            _set(jid, state="failed",
                 error=f"pipeline exited with code {proc.returncode}")
    finally:
        _run_lock.release()


def start_refresh(*, team_totals_path: str | None = None,
                  slate_players_path: str | None = None,
                  slate_window_path: str | None = None,
                  full: bool = False) -> str:
    """Launch a Stage C sim rebuild (or the full Stage B+C pipeline when
    ``full``) as a background job. Returns the job_id to poll.

    On success: publish to the object store (if configured) and reload the warm
    sims so the next /players or /run sees the fresh build.
    """
    kind = "refresh_full" if full else "refresh_sims"
    jid = _new_job(kind)

    if full:
        argv = ["refresh_and_run.py"]
    else:
        argv = ["run_slate.py"]
        if team_totals_path:
            argv += ["--team-totals", team_totals_path]
        if slate_players_path:
            argv += ["--slate-players", slate_players_path]
        if slate_window_path:
            argv += ["--slate-window", slate_window_path]

    def on_success():
        # Publish then reload, mirroring ensure_fresh's tail.
        try:
            import shared_store
            if shared_store.enabled():
                shared_store.push()
        except Exception:
            pass
        from service import sims as simsvc
        simsvc.reload()

    threading.Thread(target=_run, args=(jid, argv, on_success), daemon=True).start()
    return jid

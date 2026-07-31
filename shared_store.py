"""
shared_store.py — optional shared persistence of the refreshed artifacts in an
S3-compatible object store, so a live multi-user Streamlit deployment shares ONE
set of sims/projections across all users and survives container restarts.

Configured entirely via Streamlit secrets (or env vars). If no config is found,
every function is a safe no-op, so local single-user runs are unaffected.

Streamlit secrets schema (.streamlit/secrets.toml):

    [shared_store]
    bucket = "my-dfs-bucket"
    prefix = "dfs"                 # optional key prefix
    region = "us-east-1"           # optional
    endpoint_url = "https://..."   # optional — for S3-compatible (R2/MinIO/B2)
    access_key_id = "AKIA..."      # optional if the host provides a role
    secret_access_key = "..."      # optional

The shared state is the minimal set the app needs to be current and scorable:
the two DK-point sim arrays, the per-PA projections, the build stamp (freshness
source of truth), and the slate the sims were built from.
"""
import glob
import json
import os
import socket
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Files (relative to repo root) that make up the shared state.
ARTIFACTS = [
    "deliverables/hitter_dk_sims.npy",
    "deliverables/pitcher_dk_sims.npy",
    "out/hitter_pa_projections_2027.csv",
    "out/pitcher_pa_projections_2027.csv",
    "out/.build_stamp.json",
    "data/slate.json",
    # tiny append-only ownership training log (a few dozen KB/slate). Synced as
    # a single object so it accumulates across builds/instances, independent of
    # the 4-day sim-history pruning below. See ownership_history.py.
    "ownership_history/features.csv",
    # projected (expected) ownership, precomputed in the scheduled refresh and
    # recomputed only when the player pool changes (see scripts/build_ownership.py).
    # The signature travels with it so the pool-change gate survives across runs.
    "deliverables/projected_ownership.csv",
    "deliverables/.ownership_pool_sig.json",
]
STAMP = "out/.build_stamp.json"
LOCK_KEY = "_refresh.lock"

# --------------------------------------------------------------------------- #
# Dated history retention — keep the last N days of each build's slate-specific
# prediction set so simulation accuracy can be reviewed after the fact (compare
# the archived sims/projections against the day's real box scores). Snapshots
# live under history/<slate_date>/ in the same bucket. The canonical latest
# artifacts (above) are untouched — this is purely additive archival.
# --------------------------------------------------------------------------- #
HISTORY_PREFIX = "history"
# Defaults; overridable via secrets/env (see _history_cfg). A day of sims is
# ~25 MB, so 4 days (~100 MB) sits far inside the R2/B2 ~10 GB free tier; the MB
# budget is a hard guard so retention can never blow the free tier even if the
# artifact set grows — if 4 days won't fit, we keep as many recent days as do.
HISTORY_KEEP_DAYS = 4
HISTORY_MAX_MB = 2048

# The per-date review set: the DK sim arrays (the predicted distributions), the
# slate they were built from, the build stamp, and the human-readable projection
# summaries + manifest. Each entry is (local_source, history_basename); dated
# deliverable names are resolved from the slate date at snapshot time so every
# date's folder is self-contained with stable names.
def _history_specs(date):
    pairs = [
        ("deliverables/hitter_dk_sims.npy", "hitter_dk_sims.npy"),
        ("deliverables/pitcher_dk_sims.npy", "pitcher_dk_sims.npy"),
        ("data/slate.json", "slate.json"),
        (STAMP, "build_stamp.json"),
        (f"deliverables/hitter_projections_{date}.csv", "hitter_projections.csv"),
        (f"deliverables/pitcher_projections_{date}.csv", "pitcher_projections.csv"),
        (f"deliverables/sim_manifest_{date}.json", "sim_manifest.json"),
    ]
    return [(src, f"{HISTORY_PREFIX}/{date}/{base}") for src, base in pairs]


def _history_cfg():
    """(keep_days, max_mb) for retention, overridable via secrets/env so the
    window can be tuned without a code change."""
    keep, mb = HISTORY_KEEP_DAYS, HISTORY_MAX_MB
    cfg = _cfg() or {}
    try:
        keep = int(cfg.get("history_days") or os.environ.get(
            "SHARED_STORE_HISTORY_DAYS") or keep)
    except (TypeError, ValueError):
        pass
    try:
        mb = float(cfg.get("history_max_mb") or os.environ.get(
            "SHARED_STORE_HISTORY_MAX_MB") or mb)
    except (TypeError, ValueError):
        pass
    return max(1, keep), max(1.0, mb)

_local_lock = threading.Lock()   # per-process mutual exclusion (one instance)


# --------------------------------------------------------------------------- #
# Config / client
# --------------------------------------------------------------------------- #
def _cfg():
    try:
        import streamlit as st
        if "shared_store" in st.secrets:
            c = dict(st.secrets["shared_store"])
            if c.get("bucket"):
                return c
    except Exception:
        pass
    if os.environ.get("SHARED_STORE_BUCKET"):       # env-var fallback
        return {"bucket": os.environ["SHARED_STORE_BUCKET"],
                "prefix": os.environ.get("SHARED_STORE_PREFIX", ""),
                "region": os.environ.get("AWS_REGION"),
                "endpoint_url": os.environ.get("SHARED_STORE_ENDPOINT"),
                "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
                "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY")}
    return None


def enabled():
    return _cfg() is not None


def _client_and_cfg():
    cfg = _cfg()
    if not cfg:
        return None, None
    try:
        import boto3
    except Exception:
        return None, None
    kw = {}
    if cfg.get("region"):
        kw["region_name"] = cfg["region"]
    if cfg.get("endpoint_url"):
        kw["endpoint_url"] = cfg["endpoint_url"]
    if cfg.get("access_key_id") and cfg.get("secret_access_key"):
        kw["aws_access_key_id"] = cfg["access_key_id"]
        kw["aws_secret_access_key"] = cfg["secret_access_key"]
    try:
        return boto3.client("s3", **kw), cfg
    except Exception:
        return None, None


def _key(cfg, rel):
    p = (cfg.get("prefix") or "").strip("/")
    return f"{p}/{rel}" if p else rel


# --------------------------------------------------------------------------- #
# Stamp helpers (the build timestamp is the shared "version")
# --------------------------------------------------------------------------- #
def _remote_stamp_ts(s3, cfg):
    try:
        obj = s3.get_object(Bucket=cfg["bucket"], Key=_key(cfg, STAMP))
        return float(json.loads(obj["Body"].read()).get("ts", 0) or 0)
    except Exception:
        return None    # no shared build yet (or unreachable)


def _local_stamp_ts():
    p = os.path.join(HERE, STAMP)
    if os.path.exists(p):
        try:
            return float(json.load(open(p)).get("ts", 0) or 0)
        except Exception:
            return 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #
def pull(force=False):
    """Download the shared artifacts if the remote build is newer than local
    (cheap: only the small stamp is read unless a download is warranted).
    Returns True if anything was downloaded."""
    s3, cfg = _client_and_cfg()
    if not s3:
        return False
    rts = _remote_stamp_ts(s3, cfg)
    if rts is None:
        return False                       # nothing shared yet
    if not force and rts <= _local_stamp_ts():
        return False                       # local already at/after shared build
    got = False
    for rel in ARTIFACTS:
        dst = os.path.join(HERE, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".tmp"
            s3.download_file(cfg["bucket"], _key(cfg, rel), tmp)
            os.replace(tmp, dst)
            got = True
        except Exception:
            # artifact may legitimately not exist remotely yet
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
    return got


def push():
    """Upload local artifacts to the shared store. The stamp is written LAST so
    it acts as the commit marker (readers gate on the stamp's timestamp)."""
    s3, cfg = _client_and_cfg()
    if not s3:
        return False
    ok = True
    for rel in [a for a in ARTIFACTS if a != STAMP] + [STAMP]:
        src = os.path.join(HERE, rel)
        if not os.path.exists(src):
            continue
        try:
            s3.upload_file(src, cfg["bucket"], _key(cfg, rel))
        except Exception:
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# Dated history: snapshot + retention
# --------------------------------------------------------------------------- #
def _slate_date():
    """The slate date the current local build is for — the retention key.
    Prefer the build stamp's ``slate_date``; fall back to ``data/slate.json``;
    finally the newest ``deliverables/sim_manifest_<date>.json`` (always written
    by run_slate). Returns 'YYYY-MM-DD' or None."""
    for rel, field in ((STAMP, "slate_date"), ("data/slate.json", "date")):
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            try:
                v = json.load(open(p)).get(field)
                if v:
                    return str(v)
            except Exception:
                pass
    manifests = glob.glob(os.path.join(HERE, "deliverables", "sim_manifest_*.json"))
    if manifests:
        try:
            newest = max(manifests, key=os.path.getmtime)
            v = json.load(open(newest)).get("date")
            if v:
                return str(v)
        except Exception:
            pass
    return None


def _select_history_to_keep(sizes_by_date, keep_days, max_mb):
    """Pick which snapshot dates to KEEP: the most-recent `keep_days`, but never
    exceeding `max_mb` total — if the budget can't hold that many days, keep as
    many of the newest as fit. The single newest date is always kept (so the
    build we just archived survives even if it alone is over budget). Pure /
    S3-free so it is unit-testable. Returns (keep, drop) lists of dates."""
    budget = float(max_mb) * 1e6
    dates = sorted(sizes_by_date, reverse=True)     # newest first
    keep, running = [], 0.0
    for i, d in enumerate(dates):
        sz = float(sizes_by_date[d])
        if i == 0 or (len(keep) < keep_days and running + sz <= budget):
            keep.append(d)
            running += sz
        else:
            break                                    # dates are sorted; stop
    drop = [d for d in dates if d not in set(keep)]
    return keep, drop


def _history_index(s3, cfg):
    """Map snapshot date -> (total_bytes, [object_keys]) under history/."""
    prefix = _key(cfg, HISTORY_PREFIX).rstrip("/") + "/"
    out, token = {}, None
    while True:
        kw = {"Bucket": cfg["bucket"], "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        try:
            resp = s3.list_objects_v2(**kw)
        except Exception:
            break
        for o in resp.get("Contents", []):
            rest = o["Key"][len(prefix):]
            date = rest.split("/", 1)[0]
            if not date:
                continue
            sz, keys = out.get(date, (0, []))
            out[date] = (sz + int(o.get("Size", 0)), keys + [o["Key"]])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def prune_history(s3=None, cfg=None):
    """Enforce the retention window/budget, deleting whole outdated snapshots.
    Returns {'kept': [...], 'dropped': [...]} (dates)."""
    if s3 is None:
        s3, cfg = _client_and_cfg()
    if not s3:
        return {"kept": [], "dropped": []}
    keep_days, max_mb = _history_cfg()
    idx = _history_index(s3, cfg)
    sizes = {d: idx[d][0] for d in idx}
    keep, drop = _select_history_to_keep(sizes, keep_days, max_mb)
    for d in drop:
        for k in idx[d][1]:
            try:
                s3.delete_object(Bucket=cfg["bucket"], Key=k)
            except Exception:
                pass
    return {"kept": keep, "dropped": drop}


def _history_stamp_ts(s3, cfg, date):
    """Build ts already archived for `date` (0 if none), so a re-publish of the
    same slate only re-uploads when it's actually a newer build."""
    try:
        obj = s3.get_object(Bucket=cfg["bucket"],
                            Key=_key(cfg, f"{HISTORY_PREFIX}/{date}/build_stamp.json"))
        return float(json.loads(obj["Body"].read()).get("ts", 0) or 0)
    except Exception:
        return None


def snapshot_history(date=None, force=False):
    """Archive the current build's review set under history/<slate_date>/, then
    prune to the retention window. Idempotent per build: skips the (large) upload
    when this slate date is already archived at an equal-or-newer build ts, unless
    `force`. Retention is enforced either way. No-op when the store is
    unconfigured or the date can't be determined. Returns the date or None."""
    s3, cfg = _client_and_cfg()
    if not s3:
        return None
    date = date or _slate_date()
    if not date:
        return None
    if not force:
        archived = _history_stamp_ts(s3, cfg, date)
        if archived is not None and archived >= _local_stamp_ts():
            prune_history(s3, cfg)          # still keep the window tidy
            return date
    for src, rel in _history_specs(date):
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            continue
        try:
            s3.upload_file(p, cfg["bucket"], _key(cfg, rel))
        except Exception:
            pass
    prune_history(s3, cfg)
    return date


def list_history():
    """Available snapshot dates (newest first) with size + file count, for
    reviewing which days are retained."""
    s3, cfg = _client_and_cfg()
    if not s3:
        return []
    idx = _history_index(s3, cfg)
    return [{"date": d, "size_mb": round(idx[d][0] / 1e6, 2), "files": len(idx[d][1])}
            for d in sorted(idx, reverse=True)]


def pull_history(date, dst_dir):
    """Download a date's archived snapshot into `dst_dir` for offline accuracy
    review. Returns the list of local paths written."""
    s3, cfg = _client_and_cfg()
    if not s3:
        return []
    prefix = _key(cfg, f"{HISTORY_PREFIX}/{date}").rstrip("/") + "/"
    written, token = [], None
    os.makedirs(dst_dir, exist_ok=True)
    while True:
        kw = {"Bucket": cfg["bucket"], "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        try:
            resp = s3.list_objects_v2(**kw)
        except Exception:
            break
        for o in resp.get("Contents", []):
            base = o["Key"][len(prefix):]
            if not base:
                continue
            dst = os.path.join(dst_dir, base)
            try:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                s3.download_file(cfg["bucket"], o["Key"], dst)
                written.append(dst)
            except Exception:
                pass
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return written


# --------------------------------------------------------------------------- #
# Cross-session / cross-instance rebuild lock
# --------------------------------------------------------------------------- #
class RefreshLock:
    """Best-effort mutual exclusion for the heavy rebuild: an in-process
    threading lock (covers concurrent users on one instance) plus an S3 lock
    object with a TTL (covers multiple instances). `.acquired` says whether we
    hold it; if not, the caller should skip rebuilding and use the shared data."""

    def __init__(self, ttl=2400):
        self.ttl = ttl
        self.acquired = False
        self._s3 = None
        self._cfg = None
        self._owner = None

    def acquire(self):
        if not _local_lock.acquire(blocking=False):
            self.acquired = False
            return False
        self._s3, self._cfg = _client_and_cfg()
        if self._s3:
            now = time.time()
            try:
                obj = self._s3.get_object(Bucket=self._cfg["bucket"],
                                          Key=_key(self._cfg, LOCK_KEY))
                cur = json.loads(obj["Body"].read())
                if float(cur.get("expires", 0) or 0) > now:
                    _local_lock.release()      # held by another instance
                    self.acquired = False
                    return False
            except Exception:
                pass                            # no/expired lock object
            self._owner = f"{socket.gethostname()}-{os.getpid()}-{now}"
            try:
                self._s3.put_object(
                    Bucket=self._cfg["bucket"], Key=_key(self._cfg, LOCK_KEY),
                    Body=json.dumps({"owner": self._owner,
                                     "expires": now + self.ttl}).encode())
            except Exception:
                pass
        self.acquired = True
        return True

    def release(self):
        if not self.acquired:
            return
        if self._s3:
            try:
                self._s3.delete_object(Bucket=self._cfg["bucket"],
                                       Key=_key(self._cfg, LOCK_KEY))
            except Exception:
                pass
        try:
            _local_lock.release()
        except Exception:
            pass
        self.acquired = False

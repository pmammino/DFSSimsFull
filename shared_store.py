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
]
STAMP = "out/.build_stamp.json"
LOCK_KEY = "_refresh.lock"

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

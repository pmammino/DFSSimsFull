"""artifacts.py — resolve the pre-computed sim/projection artifacts for the
worker, from object storage (Cloudflare R2 / S3) when configured, otherwise
from the local repo checkout (dev).

This is a thin wrapper over the existing ``shared_store`` module so the
web-facing worker and the pipeline share ONE definition of the artifact set
and ONE pull/push code path. When ``shared_store`` is unconfigured (no bucket
env vars) every call degrades to "use whatever is already on disk", which is
exactly what a local developer wants.

Environment (same schema as shared_store):
    SHARED_STORE_BUCKET      R2/S3 bucket name (enables remote mode)
    SHARED_STORE_PREFIX      optional key prefix
    SHARED_STORE_ENDPOINT    optional S3-compatible endpoint (R2/MinIO/B2)
    AWS_REGION               optional
    AWS_ACCESS_KEY_ID        optional (if the host provides a role)
    AWS_SECRET_ACCESS_KEY    optional
"""
import os

# Repo root = parent of this file's directory.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import shared_store  # noqa: E402  (repo root is on sys.path — see main.py)

HITTER_SIMS = os.path.join(HERE, "deliverables", "hitter_dk_sims.npy")
PITCHER_SIMS = os.path.join(HERE, "deliverables", "pitcher_dk_sims.npy")


def remote_enabled() -> bool:
    """True when an object-store bucket is configured (R2/S3 mode)."""
    return shared_store.enabled()


def sync_from_store(force: bool = False) -> bool:
    """Pull the latest shared artifacts into the local checkout when a newer
    build exists remotely. No-op (returns False) when remote is unconfigured.
    Cheap: only the small build stamp is read unless a download is warranted."""
    try:
        return shared_store.pull(force=force)
    except Exception:
        # Never let a storage hiccup take the API down — fall back to whatever
        # is already on disk. The caller logs; the worker keeps serving.
        return False


def sim_paths():
    """Return (hitter_npy, pitcher_npy) local paths, or (None, None) if either
    is missing after an optional sync."""
    h = HITTER_SIMS if os.path.exists(HITTER_SIMS) else None
    p = PITCHER_SIMS if os.path.exists(PITCHER_SIMS) else None
    return h, p


def sims_mtimes():
    """(hitter_mtime, pitcher_mtime) — used as the warm-cache invalidation key,
    mirroring the Streamlit app's ``cached_sims`` mtime keying."""
    h, p = sim_paths()
    return (os.path.getmtime(h) if h else 0.0,
            os.path.getmtime(p) if p else 0.0)


def build_stamp():
    """The shared freshness record (out/.build_stamp.json), or {} if none."""
    import json
    path = os.path.join(HERE, "out", ".build_stamp.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}

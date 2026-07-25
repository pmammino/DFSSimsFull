"""Dated sim-history retention in shared_store (the S3-free logic).

Covers the retention selection (keep last N days under a storage budget), the
per-date snapshot key layout, the config overrides, and the slate-date resolver.
No object store / boto3 needed.
"""
import io
import json
import os

import shared_store as ss


class _FakeS3:
    """In-memory stand-in for the boto3 S3 client (enough of the surface for the
    history code path) so retention can be tested without a real object store."""

    def __init__(self):
        self.store = {}

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as f:
            self.store[key] = f.read()

    def download_file(self, bucket, key, dst):
        with open(dst, "wb") as f:
            f.write(self.store[key])

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.store[Key])}

    def delete_object(self, Bucket, Key):
        self.store.pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        items = [{"Key": k, "Size": len(v)}
                 for k, v in self.store.items() if k.startswith(Prefix)]
        return {"Contents": items, "IsTruncated": False}


# --------------------------------------------------------------------------- #
# Retention selection
# --------------------------------------------------------------------------- #
def test_keeps_the_newest_n_days():
    sizes = {f"2026-07-{d:02d}": 25_000_000 for d in range(20, 26)}  # 6 days, 25 MB
    keep, drop = ss._select_history_to_keep(sizes, keep_days=4, max_mb=2048)
    assert keep == ["2026-07-25", "2026-07-24", "2026-07-23", "2026-07-22"]
    assert set(drop) == {"2026-07-21", "2026-07-20"}


def test_budget_caps_below_the_day_window():
    sizes = {f"2026-07-{d:02d}": 25_000_000 for d in range(20, 26)}  # 25 MB each
    # 60 MB budget only holds 2 days even though the window asks for 4
    keep, drop = ss._select_history_to_keep(sizes, keep_days=4, max_mb=60)
    assert keep == ["2026-07-25", "2026-07-24"]
    assert len(drop) == 4


def test_newest_day_is_kept_even_if_it_alone_exceeds_budget():
    sizes = {"2026-07-25": 100_000_000, "2026-07-24": 100_000_000}
    keep, drop = ss._select_history_to_keep(sizes, keep_days=4, max_mb=10)
    assert keep == ["2026-07-25"]           # the build we just archived survives
    assert drop == ["2026-07-24"]


def test_fewer_days_than_window_keeps_all():
    sizes = {"2026-07-24": 25_000_000, "2026-07-25": 25_000_000}
    keep, drop = ss._select_history_to_keep(sizes, keep_days=4, max_mb=2048)
    assert set(keep) == set(sizes) and drop == []


# --------------------------------------------------------------------------- #
# Snapshot key layout
# --------------------------------------------------------------------------- #
def test_history_specs_layout():
    specs = dict(ss._history_specs("2026-07-24"))
    # every archived object lands under history/<date>/ with a stable basename
    assert specs["deliverables/hitter_dk_sims.npy"] == "history/2026-07-24/hitter_dk_sims.npy"
    assert specs["data/slate.json"] == "history/2026-07-24/slate.json"
    # dated deliverable names are resolved from the date but stored stably
    assert (specs["deliverables/sim_manifest_2026-07-24.json"]
            == "history/2026-07-24/sim_manifest.json")
    assert (specs["deliverables/hitter_projections_2026-07-24.csv"]
            == "history/2026-07-24/hitter_projections.csv")


# --------------------------------------------------------------------------- #
# Config overrides
# --------------------------------------------------------------------------- #
def test_history_cfg_env_overrides(monkeypatch):
    monkeypatch.setenv("SHARED_STORE_HISTORY_DAYS", "7")
    monkeypatch.setenv("SHARED_STORE_HISTORY_MAX_MB", "500")
    keep, mb = ss._history_cfg()
    assert keep == 7 and mb == 500.0


def test_history_cfg_defaults(monkeypatch):
    monkeypatch.delenv("SHARED_STORE_HISTORY_DAYS", raising=False)
    monkeypatch.delenv("SHARED_STORE_HISTORY_MAX_MB", raising=False)
    keep, mb = ss._history_cfg()
    assert keep == ss.HISTORY_KEEP_DAYS and mb == float(ss.HISTORY_MAX_MB)


# --------------------------------------------------------------------------- #
# Slate-date resolver (retention key)
# --------------------------------------------------------------------------- #
def test_slate_date_prefers_stamp_then_slate_then_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "HERE", str(tmp_path))
    os.makedirs(tmp_path / "out"); os.makedirs(tmp_path / "data")
    os.makedirs(tmp_path / "deliverables")

    # only a manifest -> falls back to it
    (tmp_path / "deliverables" / "sim_manifest_2026-07-22.json").write_text(
        json.dumps({"date": "2026-07-22"}))
    assert ss._slate_date() == "2026-07-22"

    # slate.json present -> preferred over manifest
    (tmp_path / "data" / "slate.json").write_text(json.dumps({"date": "2026-07-23"}))
    assert ss._slate_date() == "2026-07-23"

    # build stamp present -> highest priority
    (tmp_path / "out" / ".build_stamp.json").write_text(
        json.dumps({"slate_date": "2026-07-24", "ts": 1.0}))
    assert ss._slate_date() == "2026-07-24"


# --------------------------------------------------------------------------- #
# End-to-end retention against an in-memory S3
# --------------------------------------------------------------------------- #
def _seed_build(root, date, ts):
    (root / "out").mkdir(exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "deliverables").mkdir(exist_ok=True)
    (root / "out" / ".build_stamp.json").write_text(
        json.dumps({"slate_date": date, "ts": ts}))
    (root / "data" / "slate.json").write_text(json.dumps({"date": date}))
    for nm in ("hitter_dk_sims.npy", "pitcher_dk_sims.npy"):
        (root / "deliverables" / nm).write_bytes(b"S" * 1000)
    for nm in (f"hitter_projections_{date}.csv", f"pitcher_projections_{date}.csv"):
        (root / "deliverables" / nm).write_text("x")
    (root / "deliverables" / f"sim_manifest_{date}.json").write_text(
        json.dumps({"date": date}))


def test_snapshot_prunes_dedups_and_pulls(tmp_path, monkeypatch):
    fake = _FakeS3()
    cfg = {"bucket": "b", "prefix": "dfs"}
    monkeypatch.setattr(ss, "_client_and_cfg", lambda: (fake, cfg))
    monkeypatch.setattr(ss, "_history_cfg", lambda: (4, 2048))
    monkeypatch.setattr(ss, "HERE", str(tmp_path))

    dates = ["2026-07-20", "2026-07-21", "2026-07-22",
             "2026-07-23", "2026-07-24", "2026-07-25"]
    for i, d in enumerate(dates):
        _seed_build(tmp_path, d, 1000 + i)
        ss.snapshot_history()

    # only the last 4 days survive
    assert [h["date"] for h in ss.list_history()] == \
        ["2026-07-25", "2026-07-24", "2026-07-23", "2026-07-22"]

    # re-publishing the same build uploads nothing new
    before = dict(fake.store)
    ss.snapshot_history()
    assert fake.store == before

    # a newer build for the same date overwrites that date's snapshot
    _seed_build(tmp_path, "2026-07-25", 9999)
    (tmp_path / "deliverables" / "hitter_dk_sims.npy").write_bytes(b"NEW")
    ss.snapshot_history()
    assert fake.store[ss._key(cfg, "history/2026-07-25/hitter_dk_sims.npy")] == b"NEW"

    # a date's full review set can be pulled back for scoring
    got = ss.pull_history("2026-07-24", str(tmp_path / "review"))
    assert {os.path.basename(p) for p in got} == {
        "hitter_dk_sims.npy", "pitcher_dk_sims.npy", "slate.json",
        "build_stamp.json", "hitter_projections.csv",
        "pitcher_projections.csv", "sim_manifest.json"}

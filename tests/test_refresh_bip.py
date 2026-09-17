"""
test_refresh_bip.py
===================
Guards the `--refresh-bip` path in `run_pipeline.step3_load_bip_data`.

The bug: the scrape condition was

    if most_recent not in recent and not skip_2026_scrape:

`recent` is populated from `--bip-dir`, so once `bip_inputs/bip_<year-1>.csv`
existed the scrape could never fire — no combination of flags would refresh it.
The committed 2026 file held 54,623 batted balls against ~131,000 for a full
season (42%), and every projection built from it was capped by that while
LOOKING like a fresh build. Power and BABIP are derived from these batted
balls, so it is not a cosmetic staleness.

These tests pin both halves: a refresh actually replaces the local data and
writes it back, and the default path still never touches the network.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_pipeline  # noqa: E402

TARGET = 2027
PRIOR = TARGET - 1


def _bip(n, season=PRIOR, speed=90.0):
    return pd.DataFrame({
        "Season": [season] * n,
        "batter": range(1, n + 1),
        "pitcher": range(1, n + 1),
        "events": ["single"] * n,
        "stand": ["R"] * n,
        "launch_speed": [speed] * n,
        "launch_angle": [15.0] * n,
        "adjusted_angle": [0.0] * n,
    })


@pytest.fixture
def bip_dir(tmp_path):
    """A --bip-dir holding a deliberately PARTIAL prior season."""
    d = tmp_path / "bip_inputs"
    d.mkdir()
    _bip(40, speed=85.0).to_csv(d / f"bip_{PRIOR}.csv", index=False)
    _bip(100, season=TARGET - 2).to_csv(d / f"bip_{TARGET - 2}.csv", index=False)
    _bip(100, season=TARGET - 3).to_csv(d / f"bip_{TARGET - 3}.csv", index=False)
    return d


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything tries to scrape when it should not."""
    def boom(*a, **k):
        raise AssertionError("fetch_statcast_season called unexpectedly")
    monkeypatch.setattr(run_pipeline, "fetch_statcast_season", boom)


def _scraper(monkeypatch, rows=500):
    """Stand in for the Statcast scrape, recording how it was called."""
    calls = {}

    def fake(year, force=False):
        calls["year"] = year
        calls["force"] = force
        return _bip(rows, season=year, speed=95.0)

    monkeypatch.setattr(run_pipeline, "fetch_statcast_season", fake)
    return calls


def test_default_run_reuses_local_data_and_never_scrapes(bip_dir, no_network):
    """The reliable path: committed CSVs only, no network dependency."""
    bip_all, pool = run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=True, force=False)
    assert (bip_all["Season"] == PRIOR).sum() == 40


def test_existing_local_file_blocks_the_scrape_without_the_flag(bip_dir,
                                                                no_network):
    """THE bug. Even with skip_2026_scrape=False, a present local file means no
    scrape — which is why a dedicated flag was needed."""
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=False, force=False)
    assert (bip_all["Season"] == PRIOR).sum() == 40, (
        "a local file must still win when no refresh was requested"
    )


def test_refresh_bip_replaces_the_partial_season(bip_dir, monkeypatch):
    calls = _scraper(monkeypatch, rows=500)
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=True, force=False, refresh_bip=True)

    assert calls["year"] == PRIOR
    assert calls["force"] is True, "a refresh must bypass the parquet cache"
    assert (bip_all["Season"] == PRIOR).sum() == 500, (
        "scraped rows should have replaced the 40 partial ones, not merged"
    )


def test_refresh_bip_overrides_skip_2026_scrape(bip_dir, monkeypatch):
    """An explicit refresh request must win over the reliability flag the
    workflow passes by default."""
    _scraper(monkeypatch, rows=300)
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=True, force=False, refresh_bip=True)
    assert (bip_all["Season"] == PRIOR).sum() == 300


def test_refresh_bip_writes_the_data_back(bip_dir, monkeypatch):
    """Otherwise the next (non-scraping) run falls back to the partial season
    and the refresh buys nothing beyond that single run."""
    _scraper(monkeypatch, rows=500)
    run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=True, force=False, refresh_bip=True)

    written = pd.read_csv(bip_dir / f"bip_{PRIOR}.csv")
    assert len(written) == 500
    # And a following default run now starts from the complete data.
    monkeypatch.setattr(run_pipeline, "fetch_statcast_season",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not scrape")))
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=True, force=False)
    assert (bip_all["Season"] == PRIOR).sum() == 500


def test_an_empty_scrape_keeps_the_existing_data(bip_dir, monkeypatch):
    """A failed or out-of-season scrape must not wipe what we already have."""
    monkeypatch.setattr(run_pipeline, "fetch_statcast_season",
                        lambda year, force=False: pd.DataFrame())
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, bip_dir, skip_2026_scrape=True, force=False, refresh_bip=True)
    assert (bip_all["Season"] == PRIOR).sum() == 40
    assert len(pd.read_csv(bip_dir / f"bip_{PRIOR}.csv")) == 40, (
        "the input CSV must be left alone when the scrape returns nothing"
    )


def test_missing_season_still_scrapes_without_the_flag(tmp_path, monkeypatch):
    """The original 'scrape if absent' behavior must be untouched."""
    d = tmp_path / "bip_inputs"
    d.mkdir()
    _bip(100, season=TARGET - 2).to_csv(d / f"bip_{TARGET - 2}.csv", index=False)
    calls = _scraper(monkeypatch, rows=250)
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, d, skip_2026_scrape=False, force=False)
    assert calls["year"] == PRIOR
    assert (bip_all["Season"] == PRIOR).sum() == 250


def test_refresh_without_a_bip_dir_does_not_write_anything(monkeypatch):
    """No --bip-dir means nowhere to persist to; it must not crash."""
    _scraper(monkeypatch, rows=120)
    bip_all, _ = run_pipeline.step3_load_bip_data(
        TARGET, None, skip_2026_scrape=True, force=False, refresh_bip=True)
    assert (bip_all["Season"] == PRIOR).sum() == 120

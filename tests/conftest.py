"""Shared test setup.

The saved role files in rosters/ (player_roles_{hitters,pitchers}_2027.csv) are
a hand-level override: the playing-time model applies them after every feed, so
they WIN. That is the point of them — and it is exactly why the tests that
measure what the FEEDS do against the real snapshot must not see them. Without
this, `project_playing_time(..., feed_dir=real feeds)` with the default roster
path would be measuring the saved file, not the feeds.

A test that passes its own `roster_path` (a tmp dir, a named folder) is
untouched; only the default "rosters" is redirected to a folder that does not
exist. `tests/test_fg_usage.py::test_the_saved_role_files_are_well_formed` reads
the real files directly.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolate_saved_role_files(monkeypatch):
    import playing_time_model as M
    real = M.role_override_path

    def isolated(kind, target_year, base="rosters"):
        if str(base) == "rosters":
            base = "tests/_no_saved_roles_here"
        return real(kind, target_year, base)

    monkeypatch.setattr(M, "role_override_path", isolated)

"""Depth-chart team context and the role-feed fetcher's safety checks."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_role_feeds as fetch                      # noqa: E402
from role_feeds import depth_team_overrides            # noqa: E402
from team_context import team_id_for_abbr              # noqa: E402

NYY, SF, LAD = (team_id_for_abbr(a) for a in ("NYY", "SF", "LAD"))


def _depth_xml(entries):
    teams = {}
    for code, first, last in entries:
        teams.setdefault(code, []).append((first, last))
    body = "".join(
        f'<Team Id="1" Code="{c}"><Players>' + "".join(
            f"<Player Id='{i}'><FirstName>{f}</FirstName><LastName>{l}</LastName>"
            f"<Position>LF</Position><Rank>1</Rank></Player>"
            for i, (f, l) in enumerate(ps)) + "</Players></Team>"
        for c, ps in teams.items())
    return f"<DepthChart><Teams>{body}</Teams></DepthChart>"


def _history():
    rows = []
    for pid, name, tid in [(1, "Juan Soto", NYY), (2, "Aaron Judge", NYY),
                           (3, "Free Agent Guy", LAD)]:
        rows.append({"PlayerId": pid, "Name": name, "Season": 2026,
                     "TeamId": tid, "PA": 600})
    return pd.DataFrame(rows)


def test_only_changes_are_overrides_and_manual_wins(tmp_path):
    (tmp_path / "depth.xml").write_text(_depth_xml([
        ("SF", "Juan", "Soto"),        # moved NYY -> SF
        ("NYY", "Aaron", "Judge"),     # unchanged
    ]))
    out, moves = depth_team_overrides(_history(), 2027, feed_dir=tmp_path)
    assert out == {1: SF}              # Judge unchanged, FA not on a chart
    assert list(moves["Name"]) == ["Juan Soto"]
    assert moves.iloc[0]["from_team"] == "NYY" and moves.iloc[0]["to_team"] == "SF"

    out, _ = depth_team_overrides(_history(), 2027, feed_dir=tmp_path,
                                  manual={1: LAD})
    assert out[1] == LAD               # the hand-written file beats the feed


def test_player_on_two_clubs_is_dropped(tmp_path):
    (tmp_path / "depth.xml").write_text(_depth_xml([
        ("SF", "Juan", "Soto"), ("LAD", "Juan", "Soto")]))
    out, moves = depth_team_overrides(_history(), 2027, feed_dir=tmp_path)
    assert out == {} and moves.empty


def test_missing_feed_leaves_manual_only(tmp_path):
    out, moves = depth_team_overrides(_history(), 2027, feed_dir=tmp_path,
                                      manual={2: SF})
    assert out == {2: SF} and moves.empty


def test_real_snapshots_pass_validation():
    for name, (_, fname, _) in fetch.FEEDS.items():
        fetch.validate(name, fetch.parse_xml((fetch.FEED_DIR / fname).read_bytes()))


def test_bad_download_never_overwrites_good_file(tmp_path, monkeypatch):
    good = tmp_path / "depth.xml"
    good.write_text("GOOD")
    monkeypatch.setenv("RW_DEPTH_URL", "http://example.invalid/x")
    monkeypatch.setattr(fetch, "REJECT_DIR", tmp_path / ".rejected")
    st = fetch.fetch_one("depth", feed_dir=tmp_path,
                         getter=lambda u: b"<html>403 Forbidden</html>")
    assert not st["ok"] and good.read_text() == "GOOD"
    assert (tmp_path / ".rejected" / "depth.xml").exists()


def test_missing_key_is_reported_not_fetched(monkeypatch):
    monkeypatch.delenv("RW_DEPTH_URL", raising=False)
    st = fetch.fetch_one("depth", getter=lambda u: (_ for _ in ()).throw(AssertionError))
    assert not st["ok"] and "RW_DEPTH_URL" in st["error"]

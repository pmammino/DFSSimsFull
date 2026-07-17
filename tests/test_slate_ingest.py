"""Tests for slate_ingest — focused on double-header handling.

Run: python -m pytest tests/test_slate_ingest.py   (or run this file directly).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slate_ingest as SI


# --- fixtures: a BOS/TB double-header (game 2 = night = on-slate) plus one
#     ordinary single game (LAD@NYY) so non-double-header behavior is exercised.

_CONFIRMED = """<Lineups><Date>2026-07-17</Date><Games>
<Game IsDoubleheader="1" Id="76269"><DoubleheaderId>2</DoubleheaderId><DateTime>2026-07-17T19:10:00-04:00</DateTime><Teams>
<Team IsHome="1" Id="4" Code="BOS"><StartingPitcher Id="1"><FirstName>Eduardo</FirstName><LastName>Rivera</LastName></StartingPitcher><OpenerPitcher/><PrimaryPitcher/><Players/></Team>
<Team IsHome="0" Id="29" Code="TB"><StartingPitcher Id="2"><FirstName>Mason</FirstName><LastName>Englert</LastName></StartingPitcher><OpenerPitcher/><PrimaryPitcher/><Players/></Team>
</Teams></Game>
<Game IsDoubleheader="1" Id="77699"><DoubleheaderId>1</DoubleheaderId><DateTime>2026-07-17T13:35:00-04:00</DateTime><Teams>
<Team IsHome="0" Id="29" Code="TB"><StartingPitcher Id="3"><FirstName>Griffin</FirstName><LastName>Jax</LastName></StartingPitcher><OpenerPitcher/><PrimaryPitcher/><Players/></Team>
<Team IsHome="1" Id="4" Code="BOS"><StartingPitcher Id="4"><FirstName>Jake</FirstName><LastName>Bennett</LastName></StartingPitcher><OpenerPitcher/><PrimaryPitcher/><Players/></Team>
</Teams></Game>
<Game IsDoubleheader="0" Id="76267"><DoubleheaderId>0</DoubleheaderId><DateTime>2026-07-17T19:05:00-04:00</DateTime><Teams>
<Team IsHome="1" Id="18" Code="NY-A"><StartingPitcher Id="5"><FirstName>Gerrit</FirstName><LastName>Cole</LastName></StartingPitcher><OpenerPitcher/><PrimaryPitcher/><Players/></Team>
<Team IsHome="0" Id="14" Code="LA"><StartingPitcher Id="6"><FirstName>Roki</FirstName><LastName>Sasaki</LastName></StartingPitcher><OpenerPitcher/><PrimaryPitcher/><Players/></Team>
</Teams></Game>
</Games></Lineups>"""

_EXPECTED = """<ExpectedLineups><Date>2026-07-17</Date><Games>
<Game Id="76269"><DateTime>2026-07-17T19:10:00-04:00</DateTime><Teams>
<Team IsHome="0" LineupStatus="X" Id="29" Code="TB"><Players>
<Player Id="14811"><FirstName>Ryan</FirstName><LastName>Vilade</LastName><Position>RF</Position><BattingSpot>4</BattingSpot></Player>
</Players></Team>
<Team IsHome="1" LineupStatus="X" Id="4" Code="BOS"><Players>
<Player Id="14047"><FirstName>Willson</FirstName><LastName>Contreras</LastName><Position>1B</Position><BattingSpot>4</BattingSpot></Player>
</Players></Team>
</Teams></Game>
<Game Id="77699"><DateTime>2026-07-17T13:35:00-04:00</DateTime><Teams>
<Team IsHome="0" LineupStatus="X" Id="29" Code="TB"><Players>
<Player Id="14740"><FirstName>Cedric</FirstName><LastName>Mullins</LastName><Position>CF</Position><BattingSpot>4</BattingSpot></Player>
</Players></Team>
<Team IsHome="1" LineupStatus="C" Id="4" Code="BOS"><Players>
<Player Id="18258"><FirstName>Masataka</FirstName><LastName>Yoshida</LastName><Position>DH</Position><BattingSpot>5</BattingSpot></Player>
</Players></Team>
</Teams></Game>
<Game Id="76267"><DateTime>2026-07-17T19:05:00-04:00</DateTime><Teams>
<Team IsHome="0" LineupStatus="X" Id="14" Code="LA"><Players>
<Player Id="12739"><FirstName>Shohei</FirstName><LastName>Ohtani</LastName><Position>DH</Position><BattingSpot>1</BattingSpot></Player>
</Players></Team>
<Team IsHome="1" LineupStatus="X" Id="18" Code="NY-A"><Players>
<Player Id="13816"><FirstName>Trent</FirstName><LastName>Grisham</LastName><Position>CF</Position><BattingSpot>1</BattingSpot></Player>
</Players></Team>
</Teams></Game>
</Games></ExpectedLineups>"""


def _names(g, side):
    return [p['name'] for p in g['lineups'][side]]


def test_doubleheader_uses_on_slate_games_own_expected_lineup():
    """The night game (on-slate) must use ITS expected lineup, never the
    afternoon game's — even though both games share the same two teams."""
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    # off-slate afternoon game dropped, night game kept
    assert set(slate['games']) == {"76269", "76267"}
    g = slate['games']["76269"]
    assert _names(g, 'away') == ["Ryan Vilade"]        # not Cedric Mullins
    assert _names(g, 'home') == ["Willson Contreras"]  # not Masataka Yoshida
    assert g['lineup_source']['away'] == 'expected'
    assert g['lineup_source']['home'] == 'expected'


def test_single_game_expected_fallback_unaffected():
    """Ordinary (non-double-header) games still fall back to expected."""
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    g = slate['games']["76267"]
    assert _names(g, 'away') == ["Shohei Ohtani"]
    assert _names(g, 'home') == ["Trent Grisham"]


def test_confirmed_lineup_takes_priority_over_expected():
    """When the confirmed feed has hitters posted, they win over expected."""
    confirmed = _CONFIRMED.replace(
        '<Team IsHome="1" Id="4" Code="BOS"><StartingPitcher Id="1">'
        '<FirstName>Eduardo</FirstName><LastName>Rivera</LastName></StartingPitcher>'
        '<OpenerPitcher/><PrimaryPitcher/><Players/></Team>',
        '<Team IsHome="1" Id="4" Code="BOS"><StartingPitcher Id="1">'
        '<FirstName>Eduardo</FirstName><LastName>Rivera</LastName></StartingPitcher>'
        '<OpenerPitcher/><PrimaryPitcher/><Players>'
        '<Player Id="99"><FirstName>Roman</FirstName><LastName>Anthony</LastName>'
        '<Position>LF</Position><BattingSpot>1</BattingSpot></Player>'
        '</Players></Team>')
    slate = SI.build_slate(confirmed, _EXPECTED, vegas_json="{}", write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    g = slate['games']["76269"]
    assert _names(g, 'home') == ["Roman Anthony"]
    assert g['lineup_source']['home'] == 'confirmed'


def test_team_code_fallback_when_gids_do_not_match():
    """Historical rebuild: expected feed gids differ from confirmed. The
    team-code index still supplies a fallback lineup (single game per team, so
    no collision)."""
    expected_other_gids = _EXPECTED.replace('Id="76267"', 'Id="99999"')
    # Drop the double-header from confirmed so only the single game remains and
    # every team appears once in expected -> team-code fallback is unambiguous.
    slate = SI.build_slate(_CONFIRMED, expected_other_gids, vegas_json="{}",
                           write=False,
                           slate_players=["Gerrit Cole", "Roki Sasaki"])
    g = slate['games']["76267"]
    assert _names(g, 'away') == ["Shohei Ohtani"]
    assert _names(g, 'home') == ["Trent Grisham"]
    assert g['lineup_source']['away'] == 'expected'


def test_doubleheader_hitters_disambiguate_when_pitchers_tbd():
    """The nightcap's probable pitcher is often still TBD in the DK file, so the
    slate lists only the night game's HITTERS. The on-slate game must still be
    identified from the batting orders — NOT fall back to keeping the earliest
    (afternoon) game, which would surface the off-slate game's pitchers."""
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", write=False,
                           slate_players=["Ryan Vilade", "Willson Contreras"])
    assert "76269" in slate['games']          # night game kept
    assert "77699" not in slate['games']      # afternoon game dropped
    pitchers = {slate['games']["76269"]['pitchers'][s].get(r)
                for s in ('away', 'home') for r in ('starter', 'opener', 'primary')}
    assert "Griffin Jax" not in pitchers      # game-1 (off-slate) pitchers gone
    assert "Jake Bennett" not in pitchers


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")

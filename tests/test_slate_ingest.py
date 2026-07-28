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
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
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
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
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
    slate = SI.build_slate(confirmed, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
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
    slate = SI.build_slate(_CONFIRMED, expected_other_gids, vegas_json="{}", schedule_json="{}",
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
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
                           slate_players=["Ryan Vilade", "Willson Contreras"])
    assert "76269" in slate['games']          # night game kept
    assert "77699" not in slate['games']      # afternoon game dropped
    pitchers = {slate['games']["76269"]['pitchers'][s].get(r)
                for s in ('away', 'home') for r in ('starter', 'opener', 'primary')}
    assert "Griffin Jax" not in pitchers      # game-1 (off-slate) pitchers gone
    assert "Jake Bennett" not in pitchers


def test_parse_dt_naive_handles_feed_shapes():
    p = SI._parse_dt_naive
    assert p("2026-07-17T19:10:00-04:00").hour == 19     # game DateTime (ET)
    assert p("2026-07-17T19:05:00-07:00").hour == 19     # ownership SlateStart
    assert p("07/17/2026 7:05 PM").hour == 19            # salaries SlateStart
    assert p("2026-07-17T13:35:00-04:00").hour == 13
    assert p("") is None and p(None) is None and p("nonsense") is None


def test_window_drops_off_window_doubleheader_game():
    """The DK slate list carries BOTH doubleheader games' players, so only the
    slate time window can tell them apart. The 13:35 game must be dropped and its
    pitchers (Jax/Bennett) must not survive — even when they're in slate_players."""
    window = {"start": "2026-07-17T19:05:00-07:00", "end": "2026-07-17T22:10:00-07:00"}
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
                           slate_players=["Griffin Jax", "Jake Bennett",
                                          "Mason Englert", "Eduardo Rivera"],
                           slate_window=window)
    assert "76269" in slate['games']          # 19:10 nightcap kept
    assert "77699" not in slate['games']      # 13:35 game dropped by window
    assert "76267" in slate['games']          # 19:05 single game kept
    ps = {slate['games']["76269"]['pitchers'][s].get('starter')
          for s in ('away', 'home')}
    assert "Griffin Jax" not in ps and "Jake Bennett" not in ps


def test_window_none_is_a_noop():
    """No window (e.g. CSV upload) leaves the games untouched by the window filter."""
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
                           slate_window=None)
    assert {"76269", "77699", "76267"} <= set(slate['games'])


def test_window_never_empties_the_slate():
    """A window that matches nothing (bad/again-shaped feed) must not wipe the
    slate — all games are kept as a safety fallback."""
    bad = {"start": "2020-01-01T00:00:00-04:00", "end": "2020-01-01T01:00:00-04:00"}
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}", schedule_json="{}", write=False,
                           slate_window=bad)
    assert len(slate['games']) == 3           # nothing dropped


# --- postponement / cancellation detection (filter_slate_postponed) ----------
#
# The StatsAPI schedule feed keys games by full team name; canonical_team folds
# those onto the slate's codes (LA->LAD, NY-A->NYY). A schedule that marks the
# single LAD@NYY game Postponed must drop game 76267 and record why.

def _schedule(*games):
    """Build a minimal StatsAPI-schedule dict. Each game is
    (away_name, home_name, detailedState, coded, reason)."""
    return {"dates": [{"games": [
        {"gamePk": i, "status": {"detailedState": ds, "codedGameState": cd,
                                 "reason": rsn},
         "teams": {"away": {"team": {"name": a}},
                   "home": {"team": {"name": h}}}}
        for i, (a, h, ds, cd, rsn) in enumerate(games)]}]}


def test_postponed_game_is_dropped_and_recorded():
    sched = _schedule(("Los Angeles Dodgers", "New York Yankees",
                       "Postponed", "D", "Rain"))
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}",
                           schedule_json=sched, write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    assert "76267" not in slate['games']          # LAD@NYY dropped
    assert "76269" in slate['games']              # BOS/TB night game unaffected
    pp = slate['postponed']
    assert len(pp) == 1
    assert {pp[0]['away'], pp[0]['home']} == {"LA", "NY-A"}
    assert pp[0]['category'] == "postponed"
    assert pp[0]['reason'] == "Rain"


def test_cancelled_and_suspended_are_dropped():
    for state, coded, cat in [("Cancelled", "C", "cancelled"),
                              ("Suspended", "U", "suspended")]:
        sched = _schedule(("Los Angeles Dodgers", "New York Yankees",
                           state, coded, ""))
        slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}",
                               schedule_json=sched, write=False,
                               slate_players=["Eduardo Rivera", "Mason Englert"])
        assert "76267" not in slate['games'], state
        assert slate['postponed'][0]['category'] == cat


def test_scheduled_game_is_kept():
    """A normal 'Scheduled'/'In Progress' status drops nothing."""
    sched = _schedule(("Los Angeles Dodgers", "New York Yankees",
                       "Scheduled", "S", ""))
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}",
                           schedule_json=sched, write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    assert "76267" in slate['games']
    assert slate['postponed'] == []


def test_empty_schedule_never_drops_games():
    """A StatsAPI outage (empty/best-effort {}) must keep every game."""
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}",
                           schedule_json="{}", write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    assert "76267" in slate['games']
    assert slate['postponed'] == []


def test_drop_postponed_flag_disables_the_filter():
    """drop_postponed=False leaves even a postponed game in place."""
    sched = _schedule(("Los Angeles Dodgers", "New York Yankees",
                       "Postponed", "D", "Rain"))
    slate = SI.build_slate(_CONFIRMED, _EXPECTED, vegas_json="{}",
                           schedule_json=sched, drop_postponed=False, write=False,
                           slate_players=["Eduardo Rivera", "Mason Englert"])
    assert "76267" in slate['games']
    assert slate['postponed'] == []


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")

# RotoWire usage feeds

Four XML files that decide what a player's job is. `role_feeds.py` reads them;
`README_projection_engine.md` explains what it does with them.

| file | what it carries | covers |
|---|---|---|
| `depth.xml` | every club's depth chart, ranked within a position group (`1B`..`RF`, `DH`, `C` for the lineup, `P` the rotation, `BP` the bullpen, `CL` the closer, `PROS` the farm) | all 30 clubs, ~1,690 players |
| `orders.xml` | a batting order against LHP and against RHP | 23 clubs — ARI, CHC, CWS, LAA, LAD, NYM and NYY have none |
| `closers.xml` | bullpen pecking order 1..12, RotoWire's role label, and a Stability rating on each club's top arm | all 30 clubs |
| `prospects.xml` | the top 400 with a league level; the only feed carrying an MLBAM id, and that for 261 of them | — |

## Refreshing them

Save each feed's XML over the file of the same name. Saving straight out of a
browser tab is fine: the parser skips the "This XML file does not appear to
have any style information" banner that adds.

Two things to know, both of which cost real accuracy when missed:

* **`orders.xml` carries stale lineups beside the current ones.** Atlanta has
  two blocks, both marked `GameType="NORMAL"`: one reading Swanson / Albies /
  Riley / Ozuna / Duvall / d'Arnaud / Arcia / Pache, and one reading Acuna /
  Baldwin / Olson / Albies / Harris / Dubon. The old one is from when the
  pitcher batted and has EIGHT spots; the current one has nine, and that is the
  only thing telling them apart. Eight-spot orders agree with the projection's
  club for 29.4% of their players and nine-spot orders for 99.6%, so the parser
  keeps only the nine-spot ones. Nothing needs doing by hand — but if a future
  feed drops the distinction, this is where it will go wrong.

* **Nothing joins on an id.** Only `prospects.xml` has `MLBId`. Everything else
  matches on a normalized name, with the club as a tie-breaker. The run log
  prints the match rate and names what it missed; put the fixes in
  `rosters/player_id_aliases_<target_year>.csv`.

The snapshot committed here is from 2026-09-30 and is what
`tests/test_role_feeds.py` measures against. Delete the directory, or pass
`feed_dir=None`, and the playing-time model falls back to its heuristic roles.

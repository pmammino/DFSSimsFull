# RotoWire usage feeds

Four XML files that decide what a player's job is. `role_feeds.py` reads them;
`README_projection_engine.md` explains what it does with them.

| file | what it carries | covers |
|---|---|---|
| `depth.xml` | every club's depth chart, ranked within a position group (`1B`..`RF`, `DH`, `C` for the lineup, `P` the rotation, `BP` the bullpen, `CL` the closer, `PROS` the farm) | all 30 clubs, ~1,690 players |
| `orders.xml` | a batting order against LHP and against RHP | 23 clubs — ARI, CHC, CWS, LAA, LAD, NYM and NYY have none |
| `closers.xml` | bullpen pecking order 1..12, RotoWire's role label, and a Stability rating on each club's top arm | all 30 clubs |
| `prospects.xml` | the top 400 with a league level; the only feed carrying an MLBAM id, and that for 261 of them | — |

## FanGraphs projected usage (a fifth opinion)

`fg_hitters.csv` and `fg_pitchers.csv` are FanGraphs projection exports, saved by
hand (FanGraphs → Projections → the leaderboard CSV button). Only these columns
are read, and they join on **MLBAM id** (`MLBAMID`), so there is no name matching:

| file | columns used |
|---|---|
| `fg_hitters.csv` | `MLBAMID`, `PA`, `G` (name/team kept for readability) |
| `fg_pitchers.csv` | `MLBAMID`, `IP`, `GS`, `G`, `SV`, `HLD` |

`role_feeds.py` turns a projected PA / IP into the role whose anchor it most
resembles and blends that role into the player's mixture at `FG_WEIGHT` (0.35 for hitters,
0.20 for pitchers).
Where the feeds already agree nothing moves; where they disagree the mixture
**widens**, so the result lands between the two rather than on either. It is the
only evidence for a player no other feed covers. Set the weights to 0 to switch it
off, or delete the files. It is *not* copied into a column and the projected
playing time is not matched: the team closure and depth decay still decide the
final PA / IP.

The committed copy is a trimmed export from 2026-11-17. FanGraphs projections are
the user's own export — keep the repository private or remove these files if that
is a concern.

## Refreshing the RotoWire feeds

Run `python scripts/fetch_role_feeds.py` (also Stage 0 of `refresh_and_run.py`;
skip with `--skip-feeds`). It reads the URLs from environment variables so the
API keys stay out of the repo:

| variable | required | feed |
|---|---|---|
| `RW_DEPTH_URL` | yes | `api.rotowire.com/Baseball/MLB/DepthChart.php?key=...` |
| `RW_CLOSERS_URL` | yes | `api.rotowire.com/Baseball/MLB/Closers.php?key=...` |
| `RW_ORDERS_URL` | no | batting-orders proxy (defaults to the slate-feed host) |
| `RW_PROSPECTS_URL` | no | the keyed `api.rotowire.com` prospects XML (has `MLBId` and birth dates). No default: the public `top-prospects.php` table returns 5 rows of JSON without `MLBId`, so unset keeps the committed file |

A download is validated before it replaces the saved file; a failed or garbled
one leaves the old file alone and stops the run (`--allow-stale` to carry on).
Rejected payloads land in `feeds/.rejected/`. `feeds/fetch_log.json` records
when each feed was last pulled.

`depth.xml` also drives **team context**: a player whose depth-chart club
differs from the club his last season implies is treated as having moved, in
`run_pipeline.py`, for hitters and pitchers. The hand-written
`rosters/team_assignments_<year>.json` still wins. Changes are printed and
written to `out/depth_team_moves_*.csv`.

By hand: save each feed's XML over the file of the same name. Saving straight out of a
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

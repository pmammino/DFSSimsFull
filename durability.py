"""
durability.py — separate HOW OFTEN a player is there from WHAT HE DOES there.

A season's plate appearances are a product of two things that have nothing to
do with each other:

    PA  =  games he was available for  x  plate appearances per game

The first is health and roster time. The second is his job. Measuring only
their product, as `evidence_volume` does, throws the distinction away — and
throwing it away produces exactly the wrong answer for the players it matters
most for.

Byron Buxton is the case that forced this module. He took 542 plate
appearances in 126 games, a rate of 4.30 a game, which is higher than the
median Full Time hitter's 3.97: when he plays, he is an everyday centre
fielder, full stop. But RotoWire's depth chart has him eighth among
Minnesota's centre fielders — because he is hurt — so the feeds called him an
"Injury Replacement / 26th Man", an anchor of 99 plate appearances, and the
roster-depth discount took it from there. He projected NINE plate appearances.

The fix is not to trust the feed less. It is to stop asking one number to
answer two questions:

  * `play_rate` (PA per game played) says WHAT HIS JOB IS, and it defends his
    role against a depth chart that has written him off. Buxton is Full Time.
  * `predicted_games` says HOW MUCH OF THE SEASON HE WILL BE THERE, and it is
    where the injury risk goes. Buxton is docked, Matt Olson is not.

Aaron Judge is the same story with a happier ending: 679 plate appearances in
151 games, 4.50 a game. He is 100% a full-time player with an availability
dock, which is the only honest way to write down a durable star who keeps
getting hurt. Expressing it as "70% full-time, 30% platoon bat" — which is
what a role mixture alone had to do — says he might be a platoon bat, and he
might not.

HOW MUCH THE DOCK IS WORTH, MEASURED
------------------------------------

Not much, and saying so is part of the job. Backtesting the 2025 and 2026
seasons against the three before them, over 557 player-seasons where the
player was a regular at some point in the window:

    predictor          corr    best k    RMSE    vs flat league mean
    weighted mean      0.428     2.00    43.71          8.1% better
    plain mean         0.365     2.50    44.71          6.0%
    minimum            0.290     8.50    45.86          3.6%
    maximum            0.364    13.75    46.96          1.2%

So the 3/2/1 weighted mean of the last three seasons is the best of them, and
it beats simply assuming everyone is league-average by eight percent. Among
the most durable players (140+ games in a prior year) the edge falls to 1.8%.
Games played is weakly predictable, and the regression constant that comes out
of the fit — k=2 against a weight sum of 3, so 60% the player and 40% the
league — is heavy for that reason. Anyone wanting a bigger dock for a fragile
star is asking for more confidence than the record supports.

CENTRED WITHIN THE ROLE
-----------------------

Availability is measured against others holding the SAME JOB, not against the
league. The role anchors were fitted to what real players actually accumulate
— Full Time is 630 plate appearances and real rank-1 hitters average 634 — so
they already contain league-average missed time. Centring on the league
instead would dock everybody for injuries the anchor has already paid for, and
a bench player's 60 games would read as fragility when they are just his job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Seasons back, heaviest first, and the regression constant fitted against
# them. See the table in the module docstring.
GAMES_WEIGHTS = (3.0, 2.0, 1.0)
# How much the player's own record counts for against the league term. The
# fit swept `k` against a prior of ONE UNIT PER SEASON, so this is 3.0 and
# not the sum of the weights above — the weights decide how the three
# seasons are blended with each other, and this decides how much the blend
# as a whole is believed. Using the weight sum (6) by mistake shrank the
# spread between players to three quarters where the fit says three fifths,
# which is a third less regression than the backtest supports.
GAMES_PRIOR = 3.0
GAMES_SHRINK = 2.0

# A club plays 162, and nobody appears in more.
TEAM_GAMES = 162.0

# How far availability may move a player. The spread the fit supports is
# roughly 0.77 to 1.22 at the tenth and ninetieth percentiles, so these bounds
# bite only on the extremes — the iron man who has never missed a game, and
# the player whose last three seasons are mostly rehab.
AVAILABILITY_MIN = 0.55
AVAILABILITY_MAX = 1.15

# Below this many games in his evidence season a player's PA-per-game is
# noise: a September callup with nine games tells you nothing about his job.
MIN_GAMES_FOR_RATE = 20

# How many players a role needs before its own mean can serve as the standard
# its members are judged against.
MIN_COHORT = 8

# What a plate-appearance rate says about the job, measured over the projected
# 2027 hitters (median PA per game played, by assigned role):
#
#     Full Time              3.97      Strong Side Platoon    3.35
#     Everyday DH / 1B-DH    4.07      Utility IF             3.24
#     Catcher - Primary      4.05      Bench Bat              3.09
#     Injury Replacement     3.49      Weak Side Platoon      2.96
#
# The everyday jobs sit at 3.9 and up and the part-time jobs below 3.4, which
# is a cleaner separation than the roles' own anchors give, because it is
# measured per game rather than per season.
RATE_EVERYDAY = 3.90
RATE_PART_TIME = 3.30


def games_by_season(fielding: pd.DataFrame | None,
                    kind: str = "hitter") -> pd.DataFrame:
    """Games APPEARED IN, per player per season, from the fielding history.

    A player has one row per position, so a man who moved from third base to
    left field mid-game is counted twice by a sum and once by a max; the truth
    is between them. Summing and capping at 162 is the closer of the two — 84
    player-seasons of 7,340 exceed the cap, so the overcount is rare and
    small, where taking the max would understate every utility player in the
    league.
    """
    cols = ["Season", "PlayerId", "G"]
    if fielding is None or fielding.empty or not set(cols) <= set(
            fielding.columns):
        return pd.DataFrame(columns=cols)
    f = fielding[fielding["Pos"].astype(str) == "P"] if kind == "pitcher" \
        else fielding[fielding["Pos"].astype(str) != "P"]
    if f.empty:
        return pd.DataFrame(columns=cols)
    g = (f.groupby(["Season", "PlayerId"])["G"].sum()
         .clip(upper=TEAM_GAMES).reset_index())
    return g


def predicted_games(players: pd.DataFrame, games: pd.DataFrame, *,
                    target_year: int) -> pd.Series:
    """Games to expect next season: the recent weighted mean, regressed.

    Returns NaN for a player with no games on record at all, who must not be
    docked for a history he does not have.
    """
    idx = players.index
    if games.empty or "PlayerId" not in players.columns:
        return pd.Series(np.nan, index=idx)
    seasons = [target_year - i for i in range(1, len(GAMES_WEIGHTS) + 1)]
    wide = games[games["Season"].isin(seasons)].pivot(
        index="PlayerId", columns="Season", values="G").reindex(
        columns=seasons)
    if wide.empty:
        return pd.Series(np.nan, index=idx)

    w = np.asarray(GAMES_WEIGHTS, dtype=float)
    have = wide.notna().to_numpy()
    num = np.nansum(wide.to_numpy() * w, axis=1)
    den = (have * w).sum(axis=1)
    wmean = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1),
                               np.nan), index=wide.index)

    # The league term regresses toward players who are REGULARS. A bench
    # player's 60 games are his job, not his health, and averaging them in
    # would drag the reference down for a reason that has nothing to do with
    # durability.
    regular = wide.max(axis=1) >= 100
    league = float(wmean[regular].mean()) if regular.any() \
        else float(wmean.mean())

    # ONLY FOR PLAYERS WHO HAVE SHOWN THEY CAN BE REGULARS. A rookie with
    # thirty games as a September callup has not proved he is fragile, he has
    # proved he was in Triple-A — and docking him for the seasons he spent
    # there is the same error as docking Buxton for being hurt, pointed at a
    # different player. The first run of this made Yohandy Morales and Rafael
    # Flores Jr., who had 47 and 163 plate appearances as rookies, two of the
    # four most "fragile" players in baseball.
    #
    # When a player will arrive is `pt_role_start`'s question, and the
    # prospect feed answers it. This one is only about missing time from a
    # job you already hold.
    wmean = wmean.where(regular)
    if not np.isfinite(league):
        return pd.Series(np.nan, index=idx)
    shrunk = ((wmean * GAMES_PRIOR + league * GAMES_SHRINK)
              / (GAMES_PRIOR + GAMES_SHRINK))
    pid = pd.to_numeric(players["PlayerId"], errors="coerce")
    return pd.Series(pid.map(shrunk).to_numpy(), index=idx)


def play_rate(players: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """Plate appearances (or batters faced) per GAME PLAYED.

    Taken from the one season `evidence_volume` came from, so the numerator
    and denominator describe the same season. This is the number that says
    what a player's job is, independent of how much of the season he was
    there for.
    """
    idx = players.index
    need = {"evidence_volume", "evidence_season", "PlayerId"}
    if games.empty or not need <= set(players.columns):
        return pd.Series(np.nan, index=idx)
    g = games.set_index(["Season", "PlayerId"])["G"]
    season = pd.to_numeric(players["evidence_season"], errors="coerce")
    pid = pd.to_numeric(players["PlayerId"], errors="coerce")
    played = np.array([
        g.get((int(s), int(p)), np.nan)
        if pd.notna(s) and pd.notna(p) else np.nan
        for s, p in zip(season, pid)], dtype=float)
    played = np.where(played >= MIN_GAMES_FOR_RATE, played, np.nan)
    vol = pd.to_numeric(players["evidence_volume"], errors="coerce").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(vol / played, index=idx)


def availability(players: pd.DataFrame, predicted: pd.Series, *,
                 role_col: str = "pt_role") -> pd.Series:
    """How much of the season a player is there for, against his OWN ROLE.

    1.0 for anyone with no games on record, which is the right default: no
    history is not evidence of fragility, and an availability of less than one
    has to be earned.
    """
    idx = players.index
    out = pd.Series(1.0, index=idx)
    if predicted is None or not predicted.notna().any():
        return out
    roles = players[role_col].astype(str) if role_col in players.columns \
        else pd.Series("", index=idx)
    ref = predicted.groupby(roles).transform("mean")
    # A role only a handful of players hold cannot supply a reference, and
    # the pool mean is not a substitute: it is built mostly from players in
    # other jobs, so judging a catching tandem against it would dock them for
    # catching rather than for anything about their health. Not measurable
    # means not docked.
    counts = roles.map(roles.value_counts())
    ref = ref.where(counts >= MIN_COHORT)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = predicted / ref.replace(0, np.nan)
    return rel.fillna(1.0).clip(AVAILABILITY_MIN, AVAILABILITY_MAX)


def record_read(rate: float, kind: str = "hitter") -> str | None:
    """The job a player's OWN usage rate says he holds, or None if unmeasured.

    Three answers, not twelve. A rate can tell an everyday player from a
    part-time one, which is the distinction a depth chart gets wrong about an
    injured regular; it cannot tell a left fielder from a designated hitter,
    and pretending otherwise would be reading precision into one number that
    is not in it.
    """
    if rate is None or not np.isfinite(rate):
        return None
    if kind == "pitcher":
        return None
    if rate >= RATE_EVERYDAY:
        return "Full Time"
    if rate >= RATE_PART_TIME:
        return "Strong Side Platoon"
    return "Bench Bat"


def durability_report(players: pd.DataFrame) -> str:
    """What the dock came to, and for whom."""
    av = pd.to_numeric(players.get("pt_availability"), errors="coerce")
    if av is None or not av.notna().any():
        return "    availability: not computed (no games history)"
    docked = av < 0.995
    lines = [f"    availability: {int(docked.sum()):,} docked, "
             f"p10 {av.quantile(0.10):.3f} median {av.median():.3f} "
             f"p90 {av.quantile(0.90):.3f}"]
    if docked.any() and "Name" in players.columns:
        worst = players.loc[av.nsmallest(4).index, "Name"].tolist()
        lines.append(f"      most docked: {', '.join(map(str, worst))}")
    return "\n".join(lines)

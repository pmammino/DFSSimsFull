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
GAMES_PRIOR = 3.0        # the full prior, for a player with all three seasons

# Innings per appearance regresses THREE TIMES HARDER than appearances do,
# and reusing the games constant for it was simply wrong. Fitted the same
# way over 913 pitcher-seasons — predict next season's innings per
# appearance from the weighted mean of the last three, against the median of
# the same kind of pitcher — the error bottoms out at k = 6.0, where it beats
# the cohort median by 16.8%; at the games constant of 2.0 it beats it by
# only 3.9%, and at no regression at all it is 44% WORSE than just using the
# median.
#
# That ordering is the measurement telling you what the quantity is. How
# often a man is handed the ball is mostly about him; how long he stays once
# he has it is mostly about the job — a starter goes about 5.3 innings and a
# reliever about 1.0, and a pitcher's own deviation from his cohort is
# mostly noise.
IPA_SHRINK = 6.0
GAMES_SHRINK = 2.0

# A club plays 162, and nobody appears in more.
TEAM_GAMES = 162.0

# What counts as having held a regular job, per kind — the bar a player must
# clear before short seasons can be read as MISSED TIME rather than as a
# part-time role.
#
# The pitcher number is not the hitter number, and using one for both is a
# silent failure rather than a loud one. No pitcher in five seasons appeared
# in 100 games; the maximum is 83 and the median 19, so the hitter's bar
# excluded every pitcher in baseball and handed all of them a durability of
# exactly 1.000. The bar below sits just under the tenth percentile of a
# starter's season (19 appearances) so a rotation arm who held his job
# qualifies, while the callups that make up half of all pitcher-seasons do
# not.
#
# Backtested the same way as the hitters, over 631 pitcher-seasons:
# appearances correlate 0.462 with the next season's and beat a flat league
# mean by 11.1%, at k = 2.00 — the same regression the hitter fit chose,
# independently. Innings score higher (0.644, 22.7%) and are the wrong
# measure: a reliever throws 65 of them because he is a reliever, so much of
# that correlation is roles persisting rather than health.
REGULAR_GAMES = {"hitter": 100.0, "pitcher": 18.0}

# ...and pitchers are nonetheless OFF. That is a finding, and the reason is
# not the one first guessed, so both are recorded here.
#
# For a HITTER, games played and playing time are very nearly the same
# quantity: across 2,663 real player-seasons they correlate +0.973. Turning
# up is the whole of it, which is why this works so well on that side — the
# rank curve improves from 0.076 to 0.013.
#
# For a PITCHER they are not the same quantity and are not even pointed the
# same way. Across 2,256 pitcher-seasons, APPEARANCES and INNINGS correlate
# -0.217, because a starter makes about 30 appearances for 170 innings and a
# reliever 65 for 65. Within a role the sign is right (+0.769 for starters,
# +0.865 for relievers), but a pitcher who moves between roles across seasons
# is then judged on a scale belonging to the job he no longer holds: a
# reliever's 65 appearances against a rotation cohort's 30 reads as the
# ceiling of durability, and the reverse reads as a wreck. Switched on, it
# takes pitchers projected past 180 innings from 21 to 29 against a real 18
# and makes the pitcher rank curve worse (0.0264 to 0.0292).
#
# Measuring it on INNINGS instead fixes the decoupling and cannot be used:
# the evidence factor already divides by durability and the raw volume
# multiplies it back, so an innings-based durability cancels itself wherever
# the clip does not bind, and bites only as a widened clip band.
#
# An upstream repair was tried, on the theory that the trouble was
# `evidence_volume` being a best-of-three season and so biased high (+16.6
# innings-worth against what pitchers actually did next). Replacing it with a
# weighted rate times weighted games — a better point estimate by every
# backtest, and close to unbiased — made BOTH metrics worse on its own, 21 to
# 29 over 180 and 0.0264 to 0.0286, because the maximum produces a wider
# spread between players and it is the SHAPE within a club, not the level,
# that survives closure. Individual accuracy and the rank curve disagree here,
# and the rank curve is what the gate measures.
#
# What would actually work is decomposing a pitcher's innings the way his job
# does: starts times innings per start for a rotation arm, appearances times
# innings per appearance for a bullpen one. The model already carries
# Proj_GS and pt_anchor_GS. That is a redesign of the pitcher volume chain
# rather than a constant, and it is not this change.
DURABILITY_KINDS = {"hitter"}

# How far availability may move a player. The spread the fit supports is
# roughly 0.77 to 1.22 at the tenth and ninetieth percentiles, so these bounds
# bite only on the extremes — the iron man who has never missed a game, and
# the player whose last three seasons are mostly rehab.
DURABILITY_MIN = 0.55
DURABILITY_MAX = 1.15

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


def weighted_games(players: pd.DataFrame, games: pd.DataFrame, *,
                   target_year: int) -> pd.Series:
    """The 3/2/1 mean of recent appearances, with NO regression applied.

    `predicted_games` shrinks toward a league mean, which is right for
    durability — that question is "how does he compare with the league" —
    and wrong for projecting a pitcher's appearances, because the league
    mean mixes starters at about 30 with relievers at about 55. Shrinking a
    rotation arm toward it lifted ranks 5 and 6 of the staff 18-21% above
    the real curve while the back of the bullpen ran 10% light.
    """
    idx = players.index
    if games.empty or "PlayerId" not in players.columns:
        return pd.Series(np.nan, index=idx)
    seasons = [target_year - i for i in range(1, len(GAMES_WEIGHTS) + 1)]
    wide = games[games["Season"].isin(seasons)].pivot(
        index="PlayerId", columns="Season", values="G").reindex(columns=seasons)
    if wide.empty:
        return pd.Series(np.nan, index=idx)
    w = np.asarray(GAMES_WEIGHTS, dtype=float)
    have = wide.notna().to_numpy()
    num = np.nansum(wide.to_numpy() * w, axis=1)
    den = (have * w).sum(axis=1)
    wmean = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1),
                               np.nan), index=wide.index)
    pid = pd.to_numeric(players["PlayerId"], errors="coerce")
    return pd.Series(pid.map(wmean).to_numpy(), index=idx)


def predicted_games(players: pd.DataFrame, games: pd.DataFrame, *,
                    target_year: int, kind: str = "hitter") -> pd.Series:
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
    # How many seasons actually stand behind that mean. The regression
    # constant was fitted on players with all THREE, so giving a player with
    # one the same prior treats a single season as though it were three —
    # and one season is exactly where the noise is. Kevin McGonigle, a rookie
    # with 2026 and nothing else, came out at the 1.15 ceiling: the most
    # durable player in baseball on the strength of not yet having had a
    # chance to get hurt.
    seasons = pd.Series(have.sum(axis=1).astype(float), index=wide.index)
    if kind not in DURABILITY_KINDS:
        return pd.Series(np.nan, index=idx)

    # The league term regresses toward players who are REGULARS. A bench
    # player's 60 games are his job, not his health, and averaging them in
    # would drag the reference down for a reason that has nothing to do with
    # durability.
    regular = wide.max(axis=1) >= REGULAR_GAMES.get(kind, 100.0)
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
    prior = seasons.clip(upper=GAMES_PRIOR)
    shrunk = (wmean * prior + league * GAMES_SHRINK) / (prior + GAMES_SHRINK)
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


def durability(players: pd.DataFrame, predicted: pd.Series, *,
               role_col: str = "pt_role") -> pd.Series:
    """How much of the season a player is there for, against his OWN ROLE.

    Centred on 1.0 and allowed to exceed it, which is why this is not
    `pt_availability`. That column is a 0..1 knob a person types — "he will
    miss April" — and the role anchors were fitted to real accumulated
    playing time, so they already contain league-average missed time. A
    player who misses NOTHING beats the average his anchor was built from,
    and the only way to say so is a multiplier above one.

    Clipping this into [0, 1] threw that away silently: Matt Olson, who has
    played 162 games in each of the last three seasons, earned 1.15 and was
    handed 1.00, losing his entire iron-man credit — while Mike Trout's 0.974
    dock passed through untouched. Trout out-projected him by 62 plate
    appearances.

    1.0 for anyone with no games on record, which is the right default: no
    history is not evidence of fragility, and a dock has to be earned.
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
    return rel.fillna(1.0).clip(DURABILITY_MIN, DURABILITY_MAX)


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
    av = pd.to_numeric(players.get("pt_durability"), errors="coerce")
    if av is None or not av.notna().any():
        return "    durability: not computed (no games history)"
    docked = av < 0.995
    lines = [f"    durability: {int(docked.sum()):,} docked, "
             f"p10 {av.quantile(0.10):.3f} median {av.median():.3f} "
             f"p90 {av.quantile(0.90):.3f}"]
    if docked.any() and "Name" in players.columns:
        worst = players.loc[av.nsmallest(4).index, "Name"].tolist()
        lines.append(f"      most docked: {', '.join(map(str, worst))}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pitchers: innings are appearances times innings per appearance
# ─────────────────────────────────────────────────────────────────────────────
#
# A pitcher's season is two numbers multiplied together and the model had
# been carrying only their product. Separating them is what lets durability
# mean anything on this side: availability governs APPEARANCES — how many
# times he is handed the ball — and the job governs how long he stays once
# he has it. Compare a man's appearances against others doing his job and
# the -0.217 that makes a league-wide comparison useless disappears, because
# a reliever's 65 is never weighed against a starter's 30.
#
# Backtested over 983 pitcher-seasons against the next season's innings:
#
#     summary            corr    bias     RMSE
#     best season       0.615   +29.7    54.58
#     weighted innings  0.637    +8.4    42.83
#     apps x IP/app     0.632    +9.6    43.36
#
# The decomposition is level with the weighted innings as a point estimate
# and far better than the best season. That is not why it is here — a better
# point estimate measurably did NOT improve the rank curve when tried on
# volume alone. It is here because it puts availability on the quantity
# availability actually moves.

def innings_per_appearance(players: pd.DataFrame,
                           fielding: pd.DataFrame | None) -> pd.Series:
    """A pitcher's own innings per appearance, weighted over recent seasons.

    NaN for anyone without enough of a record to measure, so the caller can
    fall back to his role's rate rather than to a number invented here.
    """
    idx = players.index
    if fielding is None or fielding.empty or "PlayerId" not in players.columns:
        return pd.Series(np.nan, index=idx)
    need = {"Season", "PlayerId", "G", "Innings", "Pos"}
    if not need <= set(fielding.columns):
        return pd.Series(np.nan, index=idx)

    p = (fielding[fielding["Pos"].astype(str) == "P"]
         .groupby(["Season", "PlayerId"])
         .agg(G=("G", "sum"), IP=("Innings", "sum")).reset_index())
    p = p[p["G"] > 0]
    if p.empty:
        return pd.Series(np.nan, index=idx)

    seasons = sorted(p["Season"].unique())[-len(GAMES_WEIGHTS):]
    w = dict(zip(reversed(seasons), GAMES_WEIGHTS))
    p = p[p["Season"].isin(w)].copy()
    p["_w"] = p["Season"].map(w).astype(float) * p["G"]   # weight by workload
    p["_ipa"] = p["IP"] / p["G"]
    num = p.groupby("PlayerId").apply(
        lambda g: np.average(g["_ipa"], weights=g["_w"]), include_groups=False)
    pid = pd.to_numeric(players["PlayerId"], errors="coerce")
    return pd.Series(pid.map(num).to_numpy(), index=idx)


def expected_ipa(players: pd.DataFrame, own: pd.Series, *,
                 role_col: str = "pt_role") -> pd.Series:
    """Innings per appearance to project, in innings — not as a multiple.

    ABSOLUTE on purpose. Scaling the role anchor's own implied rate was the
    obvious way to write this and it imports an inconsistency the old chain
    never exposed: the Ace anchor is 195 innings over 32 starts, which is
    6.09 an outing, and no modern starter does that — the real figure is
    5.32. Nothing used the `g` anchor for innings before, so the two were
    free to disagree. Taking the rate from the record instead keeps the
    anchor's job to what it is good at, which is saying how often a man
    pitches.

    A player's own rate, regressed toward his role cohort's median by the
    same weight the games fit chose, and the cohort's median outright for
    anyone with no record.
    """
    idx = players.index
    roles = players[role_col].astype(str) if role_col in players.columns \
        else pd.Series("", index=idx)
    if own is None or not own.notna().any():
        return pd.Series(np.nan, index=idx)
    ref = own.groupby(roles).transform("median")
    counts = roles.map(roles.value_counts())
    ref = ref.where(counts >= MIN_COHORT, own.median())
    blended = ((own * GAMES_PRIOR + ref * IPA_SHRINK)
               / (GAMES_PRIOR + IPA_SHRINK))
    return blended.fillna(ref).clip(lower=0.1)


def expected_appearances(players: pd.DataFrame, predicted: pd.Series, *,
                         role_col: str = "pt_role") -> pd.Series:
    """Appearances to project, from the player's own record.

    The ROLE anchors are not used for this, and the reason is the same one
    that kept the anchor out of the rate: they were never fitted for it. The
    Ace anchor says 32 starts and 195 innings, which is 6.09 an outing
    against a real 5.32, and leaning on either number put the projected
    rotation 14% under the real rank curve while the bullpen ran 18% over.
    Nothing had used them this way before, so nothing had caught it.

    So the role supplies only a regression TARGET — the median of the men
    doing that job — and the player's own weighted appearances supply the
    answer. A pitcher with no record gets his cohort's median outright.
    """
    idx = players.index
    roles = players[role_col].astype(str) if role_col in players.columns \
        else pd.Series("", index=idx)
    if predicted is None or not predicted.notna().any():
        return pd.Series(np.nan, index=idx)
    ref = predicted.groupby(roles).transform("median")
    counts = roles.map(roles.value_counts())
    ref = ref.where(counts >= MIN_COHORT, predicted.median())
    # Regressed toward the men doing the same job, by the same weight the
    # games fit chose. Toward the LEAGUE would mix a starter's thirty
    # appearances with a reliever's fifty-five, which is the error this
    # whole decomposition exists to stop making.
    blended = ((predicted * GAMES_PRIOR + ref * GAMES_SHRINK)
               / (GAMES_PRIOR + GAMES_SHRINK))
    return blended.fillna(ref).clip(lower=1.0)

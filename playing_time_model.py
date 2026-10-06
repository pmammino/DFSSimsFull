"""
playing_time_model.py
=====================
The baseline playing-time model: a default role for every player, turned into
PA / IP by a constrained team allocation.

This implements the `PlayingTimeModel` protocol in playing_time.py and fills
the hole that module was written around — 715 hitters and 946 pitchers whose
`Proj_PA` / `Proj_IP` were deliberately NaN rather than a guess. Roughly 26 of
the 54 requested season categories are rate x volume, so nothing downstream
could produce a counting stat until this existed.

The shape of the problem
------------------------
Playing time is **not** a per-player regression. A team bats about
`162 x PA_PER_TEAM_GAME` (~6,156) times and throws `162 x 9` (1,458) innings,
and those totals do not care how many players we like. Project every player
independently and the sum lands wherever it lands; the league then scores more
runs than it allows and the wins model has nothing to stand on.

So it is an **allocation**:

    1. every player gets a ROLE                     (a job)
    2. role -> a raw volume                         anchor x timing x availability
    3. each team's raw volumes are SCALED to close  on its real budget
    4. the scale factor is reported, not hidden     it measures anchor error

Step 4 matters as much as the other three. A team whose assigned roles sum to
1,100 innings against a 1,458 budget gets everyone scaled up 33%, and that is
not a fact about the team — it is a fact about the anchors being too small.
Printing the factor turns a silent distortion into a number someone can fit
against (`fit_role_anchors`).

Everything here is a default
----------------------------
Every role, timing and availability value is overridable per player, from
`rosters/player_roles_{hitters,pitchers}_{year}.csv` or the generated .xlsx
workbook. The model's job is to make sure nobody starts from nothing, not to
have opinions that cannot be corrected. `load_role_overrides` reports how many
defaults each file replaced, so an override file that silently matched nothing
(the usual failure: wrong id column, wrong year) is visible.

What it does NOT do yet
-----------------------
* **Typed slots.** A team's nine lineup spots have positions; this allocates a
  single undifferentiated PA pool. Two first basemen can both be Full Time
  here, which no real team does. Position data now exists (the fielding fetch),
  so this is the next step and it is a modelling step, not a data gap.
* **Injury coupling.** "Injury Replacement" volume is conditional on OTHER
  players getting hurt. A point estimate cannot express that; it needs a
  simulation over the roster.
* **Fitted anchors.** See above.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_config import PT_FLOOR_IP, PT_FLOOR_PA
from playing_time import SOURCE_FLOOR, SOURCE_MODEL, TIER_FLOOR, TIER_PROJECTED
from role_taxonomy import (
    DEFAULT_AVAILABILITY,
    MIX_TOLERANCE,
    blend_anchors,
    format_role_mix,
    mix_volume_sd,
    modal_role,
    parse_role_mix,
    FAMILY_CORE_SLOTS,
    DEFAULT_TIMING,
    DEPTH_HITTER_ROLE,
    DEPTH_PITCHER_ROLE,
    is_depth_role,
    full_time_reference,
    primary_positions,
    role_depth_order,
    role_family,
    role_anchor,
    role_names,
    suggest_hitter_role,
    suggest_pitcher_role,
    timing_share,
)
from durability import (availability, games_by_season, play_rate,
                        predicted_games)
from team_context import FREE_AGENT_TEAM_ID, PA_PER_TEAM_GAME

# Team budgets over a 162-game season.
TEAM_PA_BUDGET = 162.0 * PA_PER_TEAM_GAME     # ~6,156
TEAM_IP_BUDGET = 162.0 * 9.0                  # 1,458
# Starts are the hardest budget of the three and the only one that was not
# being enforced: a club plays 162 games and each one has exactly one
# starting pitcher. Unclosed, the staff collectively started 178.5 games a
# club — 5,354 across the league against the 4,859 that exist.
TEAM_GS_BUDGET = 162.0

# Where the RotoWire usage feeds live. See `role_feeds`: depth charts, batting
# orders, bullpen pecking orders and the prospect list, which between them say
# what a player's job is far better than his own volume history can.
FEED_DIR = "feeds"

# How much playing time an UNSIGNED player gets, and who pays for it.
#
# All three modes give him his ROLE at the league's own scale — a full-time
# role is a full-time season whether or not anybody has signed him. They
# differ on which clubs come up short to make room.
#
#   "open"  — nobody does. Every club is projected at its full budget with the
#             players it actually has, the unsigned class is a pool beside the
#             thirty, and the league reads 30 x budget PLUS that pool.
#
#   "share" — all thirty give up an equal slice of exactly what the unsigned
#             class holds (or an unequal one, if the roster file says which
#             clubs expect to sign). The league reads 30 x budget.
#
#   "pool"  — the clubs give up what the roster file declared and the free
#             agents divide that, however little it is. A different question,
#             and sometimes a real one: "only 2% of league plate appearances
#             will go to players who are unsigned today."
#
# "open" is the default because a mid-offseason league HAS more claimants than
# it has playing time, and that is a fact about the offseason rather than an
# error to be hidden. "share" hides it by docking twenty-nine clubs for a
# signing they will not make — with eighteen bats unsigned, every club gave up
# 268 PA so that one of them could have Judge. "open" leaves each club correct
# for the roster it actually has, leaves Judge correct for the role he will
# actually play, and lets the sum say what is true: this offseason is not
# finished. When he signs, he moves into that club's budget and its incumbents
# compress at that moment, with nothing to adjust by hand.
#
# What made docking look necessary was the idea that an incumbent would soak
# up the plate appearances headed for the free agent. That is a worry about
# clubs being filled UP to their budget, and they are not: the median club
# carries 7,639 PA of role volume against a 6,156 budget, so the closure is
# already compressing every incumbent by a fifth. Docking adds a second,
# smaller squeeze on top, aimed at an arbitrarily chosen victim.
FREE_AGENT_PLAYING_TIME = "open"

# Per-player ceilings, as physical bounds rather than opinions. The most PA
# anyone has taken in a season is ~778 (Jimmy Rollins 2007); the most IP in the
# modern era is ~250. These exist so a shallow roster cannot scale one player
# to an impossible workload during closure — without them, a team with few
# projected players hands its whole budget to whoever is there.
#
# PT_MAX_IP is the one that had to come down. 230 is a bound from an era
# that ended: the real league leader threw 208.7, 207.0 and 214.0 innings in
# 2024, 2025 and 2026, and exactly three pitchers a season clear 200. A
# ceiling set 7% above anything that has happened recently is not a physical
# bound, it is slack, and the closure spent it — the projected leader sat at
# 228.2, above every real season in the sample. 215 still allows a workload
# nobody has reached since 2023 while stopping the closure inventing one.
#
# The same applies to starts. A five-man rotation turns over 32 or 33 times
# in 162 games, and the real maxima in 2024-26 were 33, 34 and 34, with
# nobody reaching 35 and about seven pitchers a season clearing 33. The
# clip was 40, which let the closure hand out 37.4 and 39.2 starts —
# impossible, and visible on the face of the spreadsheet. Appearances are
# left where they were: the real maxima are 79, 81 and 83 against a clip
# of 82, which is the right kind of close.
PT_MAX_PA = 760.0
PT_MAX_IP = 215.0
PT_MAX_GS = 34.0
PT_MAX_G = 82.0

# Closure is an iterative proportional fit: scale, clip anyone over the
# ceiling, redistribute the remainder, repeat. Converges in two or three passes
# because the ceiling binds for very few players; the cap stops a pathological
# roster from looping.
_CLOSURE_PASSES = 6


# ─────────────────────────────────────────────────────────────────────────────
# Overrides
# ─────────────────────────────────────────────────────────────────────────────

def role_override_path(kind: str, target_year: int,
                       base: str | Path = "rosters") -> Path:
    plural = "hitters" if kind == "hitter" else "pitchers"
    return Path(base) / f"player_roles_{plural}_{target_year}.csv"


def load_role_overrides(path: str | Path, kind: str) -> pd.DataFrame:
    """Per-player role / timing / availability overrides.

    A CSV keyed by `PlayerId` with any of:

        Role            a name from the taxonomy
        Role Start      a Timing label ("Opening Day", "Mid Season (~July)", ..)
        Availability    0..1, share of the season healthy and on a roster

    Only the columns present are applied, so a file that sets nothing but
    Availability for three players is valid and leaves every other default
    alone. Unknown role names are dropped WITH A WARNING naming them rather
    than silently benching the player — a typo in a hand-edited file is the
    most likely failure here, and the second most likely is a file that matches
    nobody, which `apply_role_overrides` reports as a count.

    Returns an empty frame when the file is absent: overrides are optional by
    design, and the baseline must work with none of them.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["PlayerId"])
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  role overrides {p.name}: UNREADABLE ({type(e).__name__}); "
              "using defaults")
        return pd.DataFrame(columns=["PlayerId"])
    if "PlayerId" not in df.columns:
        print(f"  role overrides {p.name}: no PlayerId column; ignored")
        return pd.DataFrame(columns=["PlayerId"])

    df = df.dropna(subset=["PlayerId"]).copy()
    df["PlayerId"] = pd.to_numeric(df["PlayerId"], errors="coerce")
    df = df.dropna(subset=["PlayerId"])
    df["PlayerId"] = df["PlayerId"].astype(int)

    valid = set(role_names(kind))
    if "Role" in df.columns:
        bad = sorted(set(df.loc[df["Role"].notna(), "Role"]) - valid)
        if bad:
            print(f"  role overrides {p.name}: {len(bad)} unknown role name(s) "
                  f"ignored: {bad[:5]}")
            df.loc[df["Role"].isin(bad), "Role"] = np.nan
    if "Availability" in df.columns:
        df["Availability"] = pd.to_numeric(
            df["Availability"], errors="coerce").clip(0.0, 1.0)
    # Role MIXTURES. A player in a job battle is not 100% anything, and the
    # workbook has carried one probability column per role — plus a
    # `Role Prob Sum` check — since it was written. Accept either shape: a
    # ready-made "Role Mix" string, or the workbook's own per-role columns.
    mix = _read_role_mix(df, kind, p.name)
    if mix is not None:
        df["Role Mix"] = mix

    keep = ["PlayerId"] + [c for c in ("Role", "Role Mix", "Role Start",
                                       "Availability") if c in df.columns]
    return df[keep].drop_duplicates("PlayerId", keep="last")


def _read_role_mix(df: pd.DataFrame, kind: str, name: str):
    """Normalize whichever mixture shape the file uses into one string column.

    Returns None when the file carries no mixture at all, which is the common
    case: a file that only renames three players' roles is still valid.
    """
    if "Role Mix" in df.columns:
        parsed = df["Role Mix"].map(lambda v: parse_role_mix(v, kind))
    else:
        cols = [c for c in role_names(kind) if c in df.columns]
        if not cols:
            return None
        block = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # A hand-typed set of probabilities that does not sum to 1 is worth
        # saying out loud before it is normalized away: it usually means a
        # column was missed, not that the person meant these ratios.
        totals = block.sum(axis=1)
        off = totals[(totals > 0) & ((totals - 1.0).abs() > MIX_TOLERANCE)]
        if len(off):
            print(f"  role overrides {name}: {len(off)} row(s) whose role "
                  f"probabilities sum to {off.iloc[0]:.2f} rather than 1.00 "
                  f"— normalized, but check them")
        parsed = [
            parse_role_mix(
                "|".join(f"{c}:{v}" for c, v in zip(cols, row) if v > 0), kind)
            for row in block.to_numpy()
        ]
        parsed = pd.Series(parsed, index=df.index)
    out = parsed.map(lambda m: format_role_mix(m) if m else np.nan)
    return out if out.notna().any() else None


def apply_role_overrides(players: pd.DataFrame, overrides: pd.DataFrame, *,
                         kind: str = "hitter") -> tuple[pd.DataFrame, dict]:
    """Let explicit assignments win over the derived defaults."""
    out = players.copy()
    stats = {"matched": 0, "role": 0, "mix": 0, "timing": 0,
             "availability": 0, "unmatched": 0}
    if overrides is None or overrides.empty:
        return out, stats

    known = set(out["PlayerId"])
    stats["unmatched"] = int((~overrides["PlayerId"].isin(known)).sum())
    ov = overrides[overrides["PlayerId"].isin(known)].set_index("PlayerId")
    stats["matched"] = len(ov)

    # The mixture goes on first so an explicit single `Role` in the same file
    # still wins: naming one role is the more specific statement.
    if "Role Mix" in ov.columns:
        vals = ov["Role Mix"].dropna()
        if not vals.empty:
            idx = out["PlayerId"].map(vals)
            mask = idx.notna()
            out.loc[mask, "pt_role_mix"] = idx[mask].to_numpy()
            # Everything that cannot be averaged — roster family, depth
            # order, the name beside him on the sheet — uses the heaviest
            # role. A 60/40 split still has to occupy one roster spot.
            modal = idx[mask].map(
                lambda v: modal_role(parse_role_mix(v, kind) or {}))
            keep = modal.notna()
            out.loc[modal[keep].index, "pt_role"] = modal[keep].to_numpy()
            out.loc[mask, "pt_role_source"] = "override"
            stats["mix"] = int(mask.sum())

    for col, dest, key in (("Role", "pt_role", "role"),
                           ("Role Start", "pt_role_start", "timing"),
                           ("Availability", "pt_availability", "availability")):
        if col not in ov.columns:
            continue
        vals = ov[col].dropna()
        if vals.empty:
            continue
        idx = out["PlayerId"].map(vals)
        mask = idx.notna()
        out.loc[mask, dest] = idx[mask].to_numpy()
        out.loc[mask, "pt_role_source"] = "override"
        stats[key] = int(mask.sum())
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# Default role assignment
# ─────────────────────────────────────────────────────────────────────────────

def _staff_ranks(players: pd.DataFrame, *, team_col: str) -> pd.Series:
    """RA9 rank within each staff, 1 = best.

    A bullpen role is a within-team standing — the best arm on a staff is the
    likeliest closer, and "best reliever in baseball" is not a job. Ranking
    league-wide instead produced 30 closers on a handful of teams the first
    time this was tried.
    """
    if "RA9" not in players.columns or team_col not in players.columns:
        return pd.Series(np.nan, index=players.index)
    ra9 = pd.to_numeric(players["RA9"], errors="coerce")
    relievers = players["role"].astype(str).str.lower() != "starter" \
        if "role" in players.columns else pd.Series(True, index=players.index)
    rank = pd.Series(np.nan, index=players.index)
    # PROJECTED TIER ONLY. Floor-tier arms are MLE-translated minor leaguers,
    # and a translated line shrunk toward the league mean can out-rank every
    # real reliever on the staff. Including them put a Double-A arm at rank 1
    # on all 30 clubs — 30 floor-tier "closers" — and pushed the actual
    # bullpen past rank 6 into the depth role. Ranking is a standing among
    # players who will pitch.
    eligible = relievers & ra9.notna() & players[team_col].notna()
    if "pt_tier" in players.columns:
        eligible &= players["pt_tier"].astype(str) == TIER_PROJECTED

    # A leverage job also needs enough evidence for RA9 to MEAN anything.
    # Without this bar the closer's job went to whoever got lucky in a tiny
    # sample: Oakland's was Michel Otañez on 28 batters faced — about seven
    # innings — where RA9 is noise. Three clubs had a closer under 100 TBF.
    # Below the bar a pitcher still gets a bullpen role, just not the ninth
    # inning, which is the honest distinction: we do not know he is good, we
    # only failed to see him be bad.
    if "evidence_volume" in players.columns:
        ev = pd.to_numeric(players["evidence_volume"], errors="coerce")
        seasoned = eligible & (ev >= BULLPEN_ROLE_MIN_TBF)
        # Unless nobody on the staff clears it — then rank whoever is there,
        # because every club does have a closer and leaving one team without
        # any leverage arm is a worse answer than an uncertain one.
        by_team = seasoned.groupby(players[team_col]).transform("any")
        eligible = np.where(by_team.fillna(False), seasoned, eligible)
        eligible = pd.Series(eligible, index=players.index)
    rank[eligible] = (ra9[eligible]
                      .groupby(players.loc[eligible, team_col])
                      .rank(method="first", ascending=True))
    return rank


def assign_default_roles(players: pd.DataFrame, kind: str, *,
                         fielding: pd.DataFrame | None = None,
                         team_col: str = "Pred_target_team_id",
                         target_year: int = 2027,
                         ) -> pd.DataFrame:
    """Give every player a role, a start time and an availability.

    Defaults only — `apply_role_overrides` replaces any of them. Floor-tier
    players get the depth role, which is what routes them to the 1 PA / 1 IP
    floor instead of into the team's allocation.
    """
    out = players.copy()
    vol_col = "evidence_volume" if "evidence_volume" in out.columns else (
        "Last_PA" if "Last_PA" in out.columns else None)
    reference = full_time_reference(out[vol_col]) if vol_col else 1.0

    if kind == "hitter":
        pos = primary_positions(fielding) if fielding is not None else {}
        out["pt_position"] = out["PlayerId"].map(pos)
        # vL_share is the fraction of a player's PA that came against
        # left-handed pitching. It is what makes a platoon bat identifiable:
        # the role is about usage, and this measures the usage directly.
        vl = (pd.to_numeric(out["vL_share"], errors="coerce")
              if "vL_share" in out.columns
              else pd.Series(np.nan, index=out.index))
        roles = [
            suggest_hitter_role(r, reference, position=r.get("pt_position"),
                                vl_share=vl.iloc[i])
            for i, (_, r) in enumerate(out.iterrows())
        ]
    else:
        ranks = _staff_ranks(out, team_col=team_col)
        roles = [
            suggest_pitcher_role(
                r, reference,
                staff_rank=(int(ranks.iloc[i]) if pd.notna(ranks.iloc[i])
                            else None))
            for i, (_, r) in enumerate(out.iterrows())
        ]

    out["pt_role"] = roles
    out["pt_role_start"] = DEFAULT_TIMING
    out["pt_role_source"] = "default"

    # How often he is there, and what he does when he is. Kept apart on
    # purpose — see `durability`, and Byron Buxton, who takes 4.30 plate
    # appearances a game and was projected for nine of them all season
    # because one number was being asked to answer both questions.
    games = games_by_season(fielding, kind)
    out["pt_games_pred"] = predicted_games(out, games, target_year=target_year)
    out["pt_play_rate"] = play_rate(out, games)
    out["pt_availability"] = availability(out, out["pt_games_pred"])
    # Always present, empty for a settled player, so a consumer never has to
    # ask whether the column exists.
    out["pt_role_mix"] = ""
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Role -> raw volume
# ─────────────────────────────────────────────────────────────────────────────

# How far a player's own history may move him within his role. The role sets
# the TIER; evidence sets his position inside it. Without this every player in
# a role receives an identical number — the first run of this model gave Aaron
# Judge, Ben Rice, Ryan McMahon, Trent Grisham and Heliot Ramos exactly 418.1
# PA each, which is not a projection anyone can use.
#
# Bounded on both sides on purpose. Below the floor the evidence would be
# overriding the role rather than modulating it (a player assigned Full Time is
# a full-time player, whatever last year looked like); above the ceiling the
# player is probably mis-roled, and the override file is the right fix for that
# rather than letting one season's volume silently promote him.
# How far a player's own volume history may move him off his role's anchor.
# Narrowed from 0.45-1.30 against the real rank curve (see
# scripts/fit_role_anchors.py): the wider band was letting team talent push
# aces apart far more than real workloads do. Real #1 starters cluster
# tightly — p25 171, median 179, p75 188 — because a rotation slot is
# governed by health and a five-man turn, not by how good the pitcher is.
# The projections spanned 152 to 228. Tightening improves the weighted
# curve error on both sides (pitchers 12.8% -> 12.2%, hitters 10.9% ->
# 10.7%), so it is not a trade.
EVIDENCE_FACTOR_MIN = 0.55
EVIDENCE_FACTOR_MAX = 1.22


def _evidence_factor(players: pd.DataFrame, kind: str) -> pd.Series:
    """Within-role differentiation from the player's own volume history.

    Hitter evidence is already PA, directly comparable to a PA anchor. Pitcher
    evidence is BATTERS FACED, which is not comparable to an IP anchor at all
    — converting through each player's own `TBF_per_IP` keeps a reliever who
    faces many batters per inning from being credited with a starter's innings.
    """
    ev = pd.to_numeric(players.get("evidence_volume"), errors="coerce") \
        if "evidence_volume" in players.columns else pd.Series(
            np.nan, index=players.index)
    if kind == "pitcher":
        tbf_per_ip = pd.to_numeric(players.get("TBF_per_IP"),
                                   errors="coerce") \
            if "TBF_per_IP" in players.columns else pd.Series(
                np.nan, index=players.index)
        # 4.3 is the league rate; used only where a player has no own value.
        tbf_per_ip = tbf_per_ip.where(tbf_per_ip > 1.0, 4.3)
        ev = ev / tbf_per_ip
    anchor = pd.to_numeric(players["pt_anchor"], errors="coerce")

    # NET OF THE GAMES HE MISSED, where we know how many. `evidence_volume`
    # is a season's total, so it falls when a player is hurt — and
    # `pt_availability` now docks him for exactly the same absence. Spending
    # it twice is the mistake the batting-order factor made in `role_feeds`,
    # and it is worse here, because the two compound: a player who misses a
    # fifth of a season would lose a fifth twice over. Dividing the evidence
    # by his own availability restores what he WOULD have accumulated at that
    # rate over a full season, which is the thing the anchor is comparable to.
    avail = (pd.to_numeric(players.get("pt_availability"), errors="coerce")
             if "pt_availability" in players.columns
             else pd.Series(np.nan, index=players.index))
    ev = ev / avail.where(avail > 0).fillna(1.0)

    factor = (ev / anchor.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    # No evidence -> 1.0, i.e. take the role at face value.
    return factor.fillna(1.0).clip(EVIDENCE_FACTOR_MIN, EVIDENCE_FACTOR_MAX)


# A 26-man roster is 13 position players and 13 pitchers. The projected tier
# is far more generous than that — it admits anyone with 25+ PA inside the
# lookback, which came to 23.8 hitters and 31.5 pitchers per club. Those extra
# players are real major leaguers who will play, but NOT all of them for this
# team and not all season, and their nominal jobs over-subscribe the budget by
# a third.
#
# Left alone, closure spreads that shortfall evenly and compresses everybody:
# Aaron Judge to 460 PA, an ace to 117 innings, a team's top nine to 61% of its
# plate appearances against a real 77%. Concentrating the allocation instead
# (scaling by raw^gamma) fixes the top and wrecks the bottom — at the exponent
# that reproduces the top-nine share, the bench falls to 9 PA and the CLOSER to
# 22 innings, because bullpen anchors are deliberately flat and an exponent on
# volume punishes precisely the roles that should not scale with it.
#
# So the shortfall belongs where the over-subscription is: the players beyond
# roster depth. Each rank past the core keeps `ROSTER_DEPTH_DECAY` of the
# previous one's volume, which is what "he is up and down from Triple-A" means
# expressed as playing time. Overridable per player like every other default.
# Superseded by FAMILY_CORE_SLOTS in role_taxonomy: a flat 13 per club ranked
# starters against relievers and lineup regulars against bench bats, which is
# the wrong competition. Kept only so an external caller referencing it does
# not break; nothing here reads it.
ROSTER_DEPTH_CORE = {"hitter": 13, "pitcher": 13}   # deprecated
#
# Fitted per side against five seasons of real playing time in
# out/fielding_history_<year>.csv; see scripts/fit_role_anchors.py, which
# builds the real within-club rank curve and scores a candidate against it.
#
# One shared 0.78 was leaving the pitching staff far too flat. Weighted by
# the real innings at each rank — a 15% miss on an ace matters, the same
# miss on the 38th arm does not — against the real 2024-26 curve:
#
#     decay   wt err   ace median   league max   arms over 180 IP
#     real                   178.8        214.0                 17
#     0.78     12.9%         165.9        228.2                 10
#     0.70     12.7%         174.1        215.0                 16
#     0.66     12.8%         175.8        215.0                 16
#     0.60     14.7%         181.7        230.0                 25
#
# 0.66 reproduces the ace median and the number of workhorse starters, at
# the same global error as 0.78. The innings it takes back come from ranks
# 13-20, which were running 10-18% heavy — a club was spreading its budget
# over 32 arms where a real club uses 29, and the surplus came off the top
# of the rotation.
#
# The hitter side was already within a few percent of its real curve, and
# the sweep moved it by 0.5 points with the top of the roster getting
# WORSE, so it keeps the value it had. Do not re-merge these into one
# constant: the two sides are not the same shape.
ROSTER_DEPTH_DECAY_HITTER = 0.78
ROSTER_DEPTH_DECAY_PITCHER = 0.66
ROSTER_DEPTH_DECAY = ROSTER_DEPTH_DECAY_HITTER   # back-compat alias
ROSTER_DEPTH_FLOOR = 0.04

# Minimum evidence (batters faced) before a reliever is considered for a
# LEVERAGE role — closer or setup. ~100 TBF is about 25 innings; under that,
# a staff-best RA9 is a small-sample accident rather than a reason to hand
# someone the ninth. Below the bar a pitcher still gets a bullpen role.
BULLPEN_ROLE_MIN_TBF = 100.0


def apply_roster_depth(players: pd.DataFrame, kind: str, *,
                       team_col: str = "Pred_target_team_id") -> pd.DataFrame:
    """Discount players ranked beyond their club's core roster.

    Ranked among the PROJECTED TIER ONLY: the floor tier is organizational
    depth that never enters the allocation, and including it pushed real
    relievers past rank 30 on every staff.

    FREE AGENTS are excluded as well, and for a sharper reason: the discount
    means "he is the eighth arm on this staff", and an unsigned player is not
    on a staff. Grouping them by `team_col` made every free agent in baseball
    one pseudo-club with nine lineup slots and six rotation spots, so they
    were ranked against each other and decayed at 0.78 a rank down to the
    0.04 floor. Forty unsigned regulars came out at a MEDIAN of 39.7 plate
    appearances, with the 37th keeping 24.7 of a 630 anchor — not because
    anything was known about him, but because 36 other unsigned players
    sorted above him on a roster that does not exist.
    """
    out = players.copy()
    out["pt_depth_rank"] = np.nan
    out["pt_depth_factor"] = 1.0
    out["pt_family"] = out["pt_role"].astype(str).map(role_family)
    if team_col not in out.columns:
        return out
    tier = out.get("pt_tier", pd.Series(TIER_PROJECTED, index=out.index))
    teams = pd.to_numeric(out[team_col], errors="coerce")
    eligible = ((tier.astype(str) == TIER_PROJECTED) & teams.notna()
                & (teams != FREE_AGENT_TEAM_ID))
    if not eligible.any():
        return out

    # Rank within (club, FAMILY), not across the whole staff. Ranking by raw
    # innings put 14 of 30 closers outside the core and left one projecting
    # 1.4 innings with 22 saves: a closer's 62-inning anchor loses to any
    # marginal starter's 130, however certain his job is. Starters compete
    # with starters for the rotation's slots and relievers with relievers for
    # the bullpen's, which is how a staff is actually built.
    # Within a family, rank on the ROLE's seniority first and raw volume only
    # as the tie-break. Ranking on volume alone put a closer 13th among his own
    # bullpen — his 62-inning anchor loses to any setup man with a better
    # evidence factor — and left him nine innings while still holding the
    # ninth. Sorting on (order, -raw) and numbering the result gives the same
    # thing a rank over a composite key would, without inventing a scale that
    # mixes the two.
    sub = out.loc[eligible].copy()
    sub["_order"] = sub["pt_role"].astype(str).map(role_depth_order)
    sub = sub.sort_values(["_order", "pt_raw"], ascending=[True, False])
    rank = (sub.groupby([sub[team_col], sub["pt_family"]]).cumcount() + 1
            ).reindex(out.loc[eligible].index)
    out.loc[eligible, "pt_depth_rank"] = rank
    slots = sub["pt_family"].map(FAMILY_CORE_SLOTS).fillna(0).astype(float)
    over = rank - slots
    beyond = over[over > 0]
    if len(beyond):
        decay = (ROSTER_DEPTH_DECAY_PITCHER if kind == "pitcher"
                 else ROSTER_DEPTH_DECAY_HITTER)
        factor = np.maximum(decay ** beyond, ROSTER_DEPTH_FLOOR)
        out.loc[beyond.index, "pt_depth_factor"] = factor.to_numpy()
    out["pt_raw"] = out["pt_raw"] * out["pt_depth_factor"]
    return out


def raw_volumes(players: pd.DataFrame, kind: str,
                team_col: str = "Pred_target_team_id") -> pd.DataFrame:
    """anchor x timing x availability x evidence, then roster-depth discount."""
    out = players.copy()
    key = "pa" if kind == "hitter" else "ip"
    anchors, gs, g, sv, hld, vl, sd = [], [], [], [], [], [], []
    unknown: set[str] = set()

    bats_col = "BatSide" if "BatSide" in out.columns else None
    has_mix = "pt_role_mix" in out.columns
    for _, r in out.iterrows():
        # A mixture blends the anchors it names; everything downstream then
        # works on one blended anchor exactly as it did on a single role's.
        mix = parse_role_mix(r.get("pt_role_mix"), kind) if has_mix else None
        a = blend_anchors(mix, kind) if mix else None
        sd.append(mix_volume_sd(mix, kind) if mix else 0.0)
        if a is None:
            a = role_anchor(str(r.get("pt_role")), kind)
        if a is None:
            unknown.add(str(r.get("pt_role")))
            a = role_anchor(DEPTH_HITTER_ROLE if kind == "hitter"
                            else DEPTH_PITCHER_ROLE, kind)
        anchors.append(float(a[key]))
        if kind == "pitcher":
            gs.append(float(a["gs"])); g.append(float(a["g"]))
            sv.append(float(a["sv"])); hld.append(float(a["hld"]))
        else:
            left = str(r.get(bats_col, "") or "").upper().startswith("L") \
                if bats_col else False
            vl.append(float(a["vl_lhb"] if left else a["vl_rhb"]))

    if unknown:
        print(f"  WARNING: {len(unknown)} unrecognized role name(s) fell back "
              f"to the depth anchor: {sorted(unknown)[:5]}")

    share = out["pt_role_start"].map(timing_share).astype(float)
    avail = pd.to_numeric(out["pt_availability"],
                          errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["pt_anchor"] = np.asarray(anchors, dtype=float)
    # The spread BETWEEN the roles a person named — the uncertainty they
    # expressed by declining to pick one. Zero for a settled player.
    out["pt_role_sd"] = np.asarray(sd, dtype=float)
    out["pt_season_share"] = share * avail
    out["pt_evidence_factor"] = _evidence_factor(out, kind)
    out["pt_raw"] = (out["pt_anchor"] * out["pt_season_share"]
                     * out["pt_evidence_factor"])
    out = apply_roster_depth(out, kind, team_col=team_col)
    if kind == "pitcher":
        out["pt_anchor_GS"] = gs
        out["pt_anchor_G"] = g
        out["pt_save_share"] = sv
        out["pt_hold_share"] = hld
    else:
        out["Proj_vL_share"] = vl
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Team closure
# ─────────────────────────────────────────────────────────────────────────────

def _close_one_team(raw: np.ndarray, target: float,
                    ceiling: float) -> np.ndarray:
    """Scale `raw` so it sums to `target`, with nobody above `ceiling`.

    Iterative proportional fit: scale everyone, clip whoever exceeds the
    ceiling, then re-scale only the unclipped to absorb the remainder. Without
    the redistribution a clipped player's surplus would simply vanish and the
    team would under-close.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0 or target <= 0:
        return np.zeros_like(raw)
    total = raw.sum()
    if total <= 0:
        # No role information at all: split the budget evenly rather than
        # returning zeros, which would delete the team's offense.
        return np.full_like(raw, min(target / raw.size, ceiling))

    out = raw * (target / total)
    for _ in range(_CLOSURE_PASSES):
        over = out > ceiling
        if not over.any():
            break
        out[over] = ceiling
        slack = target - out[over].sum()
        free = ~over
        if not free.any() or slack <= 0:
            break
        s = out[free].sum()
        if s <= 0:
            out[free] = min(slack / free.sum(), ceiling)
            break
        out[free] = out[free] * (slack / s)
    return out


def free_agent_pool(fa_raw: float, club_raw: float, league: float,
                    declared_pool: float = 0.0,
                    mode: str | None = None) -> float:
    """How much playing time the unsigned class holds, in total.

    A free agent is scaled at the rate a COMPLETE league runs at — the one
    the thirty clubs would close at if every unsigned player were already
    spread across them:

        s = league / (club_raw + fa_raw)

    which is the rate he will in fact be paid at, once he signs and his new
    club's roster closes around him. The pool is `s * fa_raw`, the unsigned
    class's share of the league's raw role volume.

    Deliberately NOT the rate the depleted clubs are closing at. In "open"
    mode the thirty clubs close on their full budgets with the players they
    still have, so unsigning people INFLATES what the survivors get — nine
    bats covering 6,156 plate appearances are each credited with some of the
    absent man's. Paying free agents at that inflated rate would make every
    unsigned player's projection rise with the size of the free agent class,
    which is the original disease wearing different clothes. In a toy league
    with twenty of forty-eight players unsigned it pinned all twenty at the
    760-PA ceiling.

    The same formula also resolves "share" mode's circularity, where the
    clubs give the pool up and so the rate depends on the pool that depends
    on the rate. What the clubs have left is the league net of the pool:

        s = (league - s*fa_raw) / club_raw   =>   s = league / (club_raw + fa_raw)

    the same expression, arrived at from the other end. It is self-limiting:
    the clubs keep `league * club_raw / (club_raw + fa_raw)`, positive
    whenever anyone is signed at all, so no arbitrary ceiling is needed to
    stop free agents eating the league.

    In "pool" mode the declared reserve IS the answer, and a free agent gets
    whatever share of it his role earns him.
    """
    fa_raw = max(float(fa_raw), 0.0)
    club_raw = max(float(club_raw), 0.0)
    # Resolved here rather than as a default argument, which would bind the
    # module constant once at import and ignore anything set afterwards.
    mode = str(mode or FREE_AGENT_PLAYING_TIME).lower()
    if mode == "pool":
        return max(float(declared_pool), 0.0)
    if fa_raw <= 0:
        return 0.0
    if club_raw <= 0:
        # Nobody is signed. There is no league scale to match, so the pool is
        # the league: whatever these players are, they are all of it.
        return float(league)
    return float(league) * fa_raw / (club_raw + fa_raw)


def club_reserve_total(pool: float, declared_pool: float,
                       mode: str | None = None) -> float:
    """How much the thirty clubs give up between them.

    The whole of the difference between the modes lives here, which is why it
    is one function and not a branch buried in the allocator.
    """
    mode = str(mode or FREE_AGENT_PLAYING_TIME).lower()
    if mode == "pool":
        return max(float(declared_pool), 0.0)
    if mode == "share":
        return max(float(pool), 0.0)
    return 0.0


def _reserve_by_club(reserved: float, declared: dict[int, float],
                     budget: float) -> dict[int, float]:
    """Split what the clubs give up across the clubs that give it up.

    A declared `pa_share`/`ip_share` is read as WHICH clubs expect to sign and
    in what proportion, not how much any unsigned player gets to play — the
    size is already settled by `free_agent_pool`. With nothing declared, free
    agency is nobody's problem in particular and every club gives up an equal
    slice. In the default "open" mode nothing is given up at all and this
    returns zeros.
    """
    if not declared or reserved <= 0:
        return {t: 0.0 if reserved <= 0 else reserved / len(declared)
                for t in declared}
    weights = {t: budget * max(float(s), 0.0) for t, s in declared.items()}
    total = sum(weights.values())
    if total <= 0:
        return {t: reserved / len(declared) for t in declared}
    return {t: reserved * w / total for t, w in weights.items()}


def allocate_playing_time(players: pd.DataFrame, kind: str, *,
                          team_col: str = "Pred_target_team_id",
                          reserves: dict | None = None,
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close each team's raw volumes on its real budget.

    Returns (players with Proj_PA/Proj_IP, per-team diagnostics).

    Floor-tier and depth-role players are held at exactly PT_FLOOR_PA /
    PT_FLOOR_IP and sit OUTSIDE the budget: the projected players close on
    the whole of it, and the floor rows are carried alongside. The team's
    rows therefore sum to `budget + n_floor`, by design.

    They used to be subtracted from the budget instead, which took their
    plate appearances off the major-league roster — the opposite of what the
    floor is for, and what `team_context.roster_volume_weights` says it is
    for in as many words: "the 1-PA floor exists so a player is present,
    ranked and joinable — not so he takes playing time away from the
    major-league roster."

    The cost of getting that backwards was not small, and it was worst where
    it was least visible. How many floor players a club carries is a fact
    about DATA COVERAGE — how many of its farmhands the MLE step managed to
    translate — not about baseball, and it ranged from 85 to 136 pitchers.
    So a club's real pitchers were handed 5.8% to 9.3% fewer innings than
    its budget, with a 3.5-point spread BETWEEN clubs that tracked nothing
    but the depth of the feed. Hitters lost 1.0% to 1.4% the same way.

    Free agents and players with no club do not close onto any club — folding
    them into a team they are not on would take that team's plate appearances
    away from the players who will actually bat. They close onto a league-wide
    pool instead, sized by `free_agent_pool` so that they are scaled at the
    same rate everyone else is, and the clubs reserve exactly that much. See
    `FREE_AGENT_PLAYING_TIME`.
    """
    out = players.copy()
    is_pa = kind == "hitter"
    vol_out = "Proj_PA" if is_pa else "Proj_IP"
    budget = TEAM_PA_BUDGET if is_pa else TEAM_IP_BUDGET
    floor_v = PT_FLOOR_PA if is_pa else PT_FLOOR_IP
    ceiling = PT_MAX_PA if is_pa else PT_MAX_IP
    share_key = "pa_share" if is_pa else "ip_share"
    reserves = reserves or {}

    tier = out.get("pt_tier", pd.Series(TIER_PROJECTED, index=out.index))
    at_floor = (tier.astype(str) == TIER_FLOOR) | \
        out["pt_role"].astype(str).map(is_depth_role)

    out[vol_out] = np.nan
    out.loc[at_floor, vol_out] = floor_v
    out["pt_source"] = np.where(at_floor, SOURCE_FLOOR, SOURCE_MODEL)
    out["pt_tier"] = np.where(at_floor, TIER_FLOOR, TIER_PROJECTED)

    teams = out[team_col] if team_col in out.columns else pd.Series(
        np.nan, index=out.index)
    on_a_club = teams.notna() & (teams != FREE_AGENT_TEAM_ID)
    off = ~on_a_club & ~at_floor

    # ── What the unsigned class holds, settled BEFORE any club closes ────
    #
    # The order matters and it is the fix. Deciding the clubs' targets first
    # and handing the free agents what was left over is what made an everyday
    # regular's season depend on a number in the roster file rather than on
    # his role. The pool is derived from the free agents' own role volume
    # (`free_agent_pool`), so a full-time role is a full-time season whoever
    # is or is not paying for it — and by default nobody is: the clubs keep
    # their full budgets and the pool sits beside them.
    def _raw(mask) -> np.ndarray:
        if not mask.any():
            return np.zeros(0)
        return pd.to_numeric(out.loc[mask, "pt_raw"],
                             errors="coerce").fillna(floor_v).to_numpy(float)

    fa_raw = _raw(off)
    club_ids = [int(t) for t in sorted(teams[on_a_club].dropna().unique())]
    declared = {t: float((reserves.get(t, {}) or {}).get(share_key, 0.0) or 0.0)
                for t in club_ids}
    declared_pool = sum(budget * s for s in declared.values())
    fa_pool = free_agent_pool(fa_raw.sum(), _raw(on_a_club & ~at_floor).sum(),
                              budget * len(club_ids), declared_pool)
    reserved_by_club = _reserve_by_club(
        club_reserve_total(fa_pool, declared_pool), declared, budget)

    rows = []
    for team_id, idx in out[on_a_club].groupby(teams[on_a_club]).groups.items():
        block = out.loc[idx]
        floor_rows = at_floor.loc[idx]
        reserved = float(reserved_by_club.get(int(team_id), 0.0))
        reserve = reserved / budget if budget else 0.0
        # The floor rows are NOT deducted: they sit outside the budget.
        target = budget - reserved
        proj_idx = block.index[~floor_rows]
        if len(proj_idx) == 0 or target <= 0:
            rows.append({"team_id": int(team_id), "n": len(block),
                         "n_projected": 0, "raw": 0.0, "target": target,
                         "scale": np.nan, "reserved_share": reserve})
            continue
        raw = pd.to_numeric(out.loc[proj_idx, "pt_raw"],
                            errors="coerce").fillna(0.0).to_numpy()
        closed = _close_one_team(raw, target, ceiling)
        out.loc[proj_idx, vol_out] = closed
        rows.append({
            "team_id": int(team_id), "n": len(block),
            "n_projected": len(proj_idx), "raw": float(raw.sum()),
            "target": float(target),
            "scale": float(target / raw.sum()) if raw.sum() > 0 else np.nan,
            "reserved_share": reserve,
        })

    # ── Free agents ──────────────────────────────────────────────────────
    #
    # An unsigned player will play somewhere, so he gets a projection; what he
    # cannot do is play in addition to a league that is already full. The
    # league has exactly 30 x budget of playing time and the 30 clubs close on
    # all of it, so his plate appearances have to come out of what the clubs
    # hold back — and they held back `fa_pool`, which was sized from these
    # players' own roles a hundred lines above. The two halves meet here.
    #
    # Closing onto that pool is the same operation a club's players get, at
    # (near enough) the same scale, so the answer to "what does this free
    # agent do" is his role, same as for anybody else.
    fa_total = 0.0
    if off.any():
        closed = _close_one_team(fa_raw, fa_pool, ceiling) if fa_pool > 0 \
            else np.clip(fa_raw, None, ceiling)
        out.loc[off, vol_out] = closed
        fa_total = float(np.sum(closed))
        rows.append({
            "team_id": int(FREE_AGENT_TEAM_ID), "n": int(off.sum()),
            "n_projected": int(off.sum()), "raw": float(fa_raw.sum()),
            "target": fa_pool if fa_pool > 0 else np.nan,
            "scale": (closed.sum() / fa_raw.sum()) if fa_raw.sum() > 0
            else np.nan,
            "reserved_share": np.nan,
        })

    # Save and hold shares are per-ROLE WEIGHTS, not an allocation: a closer's
    # 0.65 says "a closer takes about 65% of a save pool", which is a fact
    # about the role and not about how many pitchers a club happens to carry.
    # Summed over a real staff they come to 1.16 (saves) and 1.85 (holds), so
    # a consumer who multiplies them straight into a team pool over-allocates
    # — league saves read 1,412 against a pool of 1,215, and holds 4,270
    # against 2,308. Normalising per team turns the weights into shares that
    # sum to 1, which is what a pool needs. The raw role weight is kept beside
    # them because it is still the role's own attribute.
    if not is_pa:
        # The weight is scaled by how much of his role's innings a pitcher
        # ACTUALLY gets. A save is an appearance, so a reliever who throws a
        # fifth of a closer's innings cannot convert a closer's save pool —
        # before this, one projected 1.4 innings and 22.3 saves, which is not
        # a thing that can happen. The innings he does not throw hand their
        # share to the rest of the staff through the normalisation below.
        anchor = pd.to_numeric(out["pt_anchor"], errors="coerce")
        realised = (pd.to_numeric(out[vol_out], errors="coerce")
                    / anchor.replace(0, np.nan))
        realised = realised.replace([np.inf, -np.inf], np.nan).fillna(0.0) \
            .clip(0.0, 1.0)
        for src, dest in (("pt_save_share", "Proj_SV_share"),
                          ("pt_hold_share", "Proj_HLD_share")):
            w = pd.to_numeric(out[src], errors="coerce").fillna(0.0) * realised
            # Only players who will pitch compete for the pool.
            w = w.where(~at_floor, 0.0)
            tot = w.groupby(teams).transform("sum")
            out[dest] = np.where(tot > 0, w / tot, 0.0)

    # Games / starts follow the same scaling as volume, so a player scaled up
    # 30% is credited with proportionally more appearances rather than pitching
    # 250 innings across 32 starts.
    if not is_pa:
        ratio = (pd.to_numeric(out[vol_out], errors="coerce")
                 / pd.to_numeric(out["pt_anchor"], errors="coerce")
                 .replace(0, np.nan))
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["Proj_GS"] = (out["pt_anchor_GS"] * ratio).clip(0, PT_MAX_GS)
        # Close them on the 162 starts the club actually has, by the same
        # iterative fit the innings use. Without it the anchors simply added
        # up to more than a season: one ace, two mid-rotation and two
        # end-of-rotation anchors alone come to 168 before anybody is scaled.
        gs = out["Proj_GS"].to_numpy(float).copy()
        for team_id, idx in out[on_a_club].groupby(teams[on_a_club]).groups.items():
            pos = out.index.get_indexer(idx)
            block = gs[pos]
            if block.sum() <= 0:
                continue
            gs[pos] = _close_one_team(block, TEAM_GS_BUDGET, PT_MAX_GS)
        # The ceiling binds on most staffs here — several starters sit at it
        # at once — so the fit can run out of passes a hair above it. A
        # rotation where somebody starts 34.1 games is wrong in a way a
        # reader notices; being a start short of 162 is not.
        out["Proj_GS"] = np.round(np.minimum(gs, PT_MAX_GS), 1)
        out["Proj_G"] = (out["pt_anchor_G"] * ratio).clip(0, PT_MAX_G).round(1)
    else:
        pa = pd.to_numeric(out[vol_out], errors="coerce")
        out["Proj_G"] = (pa / PA_PER_TEAM_GAME * 9.0).clip(0, 162).round(1)

    return out, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def project_playing_time(players: pd.DataFrame, kind: str, *,
                         target_year: int,
                         fielding: pd.DataFrame | None = None,
                         roster_path: str | Path = "rosters",
                         reserves: dict | None = None,
                         team_col: str = "Pred_target_team_id",
                         feed_dir: str | Path | None = FEED_DIR,
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Defaults -> usage feeds -> overrides -> raw volume -> team closure.

    Satisfies `playing_time.PlayingTimeModel`: returns Proj_PA/Proj_IP, Proj_G,
    pt_tier and pt_source per player, closing on the team and hence the league
    budget.

    The feeds sit in the middle on purpose. A depth chart and a batting order
    know more about a player's job than a guess from last season's volume
    does, and less than a person who has typed a row in the override file —
    so they beat the first and lose to the second. Pass `feed_dir=None` to
    skip them; a player no feed covers keeps his heuristic default either way.
    """
    stats: dict = {}
    out = assign_default_roles(players, kind, fielding=fielding,
                               team_col=team_col, target_year=target_year)
    if feed_dir is not None:
        from role_feeds import apply_feed_roles
        out, stats["feeds"] = apply_feed_roles(
            out, kind, feed_dir=feed_dir, team_col=team_col,
            alias_path=Path(roster_path) / f"player_id_aliases_"
                                           f"{target_year}.csv")
    ov = load_role_overrides(role_override_path(kind, target_year, roster_path),
                             kind)
    # Availability is measured against others holding the SAME job, so it has
    # to be recomputed once the feeds have settled what the job is. Buxton
    # judged as a 26th man looks more durable than average; judged as the
    # everyday centre fielder he is, he is correctly docked.
    if "pt_games_pred" in out.columns:
        out["pt_availability"] = availability(out, out["pt_games_pred"])
    out, stats["overrides"] = apply_role_overrides(out, ov, kind=kind)
    out = raw_volumes(out, kind, team_col=team_col)
    out, team_diag = allocate_playing_time(out, kind, team_col=team_col,
                                           reserves=reserves)
    stats["roles"] = out["pt_role"].value_counts().to_dict()
    return out, team_diag, stats


def allocate_opportunity(shares, pool: float, capacity) -> np.ndarray:
    """Turn shares of a team pool into counts nobody could not have recorded.

    A save and a hold are both APPEARANCES, so a pitcher cannot record more of
    them than games he pitched in. The pool is a team quantity and the shares
    are normalised across the staff, so a thin closer on a thin staff gets
    normalised back up to a full share of a full pool — Félix Bautista drew
    29.6 saves and 1.2 holds from 28.3 appearances.

    Capping alone would quietly lose those saves. The pool is the club's
    opportunity and somebody records it, so the surplus is redistributed to
    team-mates who still have room, by the same iterative proportional fit
    that closes playing time. If the whole staff is saturated the remainder is
    dropped and the caller can see it in the totals — that is a real statement
    about a roster too thin to finish its own games, not an error to hide.
    """
    shares = np.asarray(shares, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    counts = shares * float(pool)
    for _ in range(_CLOSURE_PASSES):
        over = counts > capacity
        if not over.any():
            break
        surplus = float((counts[over] - capacity[over]).sum())
        counts[over] = capacity[over]
        room = capacity - counts
        free = room > 1e-9
        if not free.any() or surplus <= 1e-9:
            break
        w = counts[free]
        # Spread by current allocation where there is any, else by headroom:
        # a pitcher already taking saves is the likelier one to take more.
        basis = w if w.sum() > 0 else room[free]
        counts[free] = counts[free] + surplus * basis / basis.sum()
    return np.minimum(counts, capacity)


def playing_time_report(out: pd.DataFrame, team_diag: pd.DataFrame,
                        stats: dict, kind: str) -> str:
    """What the allocation did, and how much the anchors had to be stretched."""
    is_pa = kind == "hitter"
    vol = "Proj_PA" if is_pa else "Proj_IP"
    budget = TEAM_PA_BUDGET if is_pa else TEAM_IP_BUDGET
    lines = [f"  {kind}s: {len(out)} players"]

    if stats.get("feeds"):
        from role_feeds import feed_report
        lines.append(feed_report(stats["feeds"]).replace("\n  ", "\n    "))

    ov = stats.get("overrides", {})
    if ov.get("matched") or ov.get("unmatched"):
        lines.append(f"    overrides: {ov.get('matched', 0)} matched "
                     f"({ov.get('role', 0)} role, {ov.get('timing', 0)} timing, "
                     f"{ov.get('availability', 0)} availability), "
                     f"{ov.get('unmatched', 0)} unmatched")
    else:
        lines.append("    overrides: none (all roles are defaults)")

    v = pd.to_numeric(out[vol], errors="coerce")
    lines.append(f"    {vol}: sum {v.sum():,.0f}  "
                 f"max {v.max():,.1f}  median {v.median():,.1f}")

    if not team_diag.empty:
        # The free agents are a row in here too, and they are not a team:
        # averaging their scale into the clubs' would make the one number
        # that is supposed to expose a mis-sized anchor depend on how many
        # players happen to be unsigned.
        clubs = team_diag[team_diag["team_id"] != FREE_AGENT_TEAM_ID]
        sc = pd.to_numeric(clubs["scale"], errors="coerce").dropna()
        if len(sc):
            lines.append(f"    team closure: {len(clubs)} teams, "
                         f"budget {budget:,.0f} each")
            lines.append(f"    anchor scale: mean {sc.mean():.3f}  "
                         f"min {sc.min():.3f}  max {sc.max():.3f}")
            if abs(sc.mean() - 1.0) > 0.15:
                lines.append(
                    f"    ^ mean scale is {sc.mean():.2f}, not ~1.0: the role "
                    f"ANCHORS are mis-sized for a real roster, not the teams. "
                    f"This is what fit_role_anchors should correct.")
        # Free agents, which are a row in the same diagnostics keyed by
        # FREE_AGENT_TEAM_ID. The thing worth reading here is their scale
        # beside the clubs': the same number means an unsigned regular is
        # getting a regular's season, which is the entire point of sizing the
        # pool from their roles instead of from the roster file.
        fa = team_diag[team_diag["team_id"] == FREE_AGENT_TEAM_ID]
        res = pd.to_numeric(clubs["reserved_share"], errors="coerce").fillna(0.0)
        unit = vol.replace("Proj_", "")
        if len(fa):
            r = fa.iloc[0]
            got = pd.to_numeric(out.loc[
                pd.to_numeric(out.get(vol), errors="coerce").notna()
                & (pd.to_numeric(out.get("Pred_target_team_id"),
                                 errors="coerce") == FREE_AGENT_TEAM_ID), vol],
                errors="coerce")
            line = (f"    free agents: {int(r['n'])} players, "
                    f"{got.sum():,.0f} {unit} (median {got.median():,.1f})")
            if len(sc) and pd.notna(r["scale"]):
                line += (f", scale {r['scale']:.3f} against the clubs' "
                         f"{sc.mean():.3f}")
            lines.append(line)
            if (res > 0).any():
                lines.append(
                    f"    ^ the clubs give up {(res * budget).sum():,.0f} "
                    f"{unit} between them ({res.mean():.2%} of a budget "
                    f"each), so the league closes at {budget * len(clubs):,.0f}"
                    f" ({FREE_AGENT_PLAYING_TIME!r} mode).")
            else:
                lines.append(
                    f"    ^ beside the clubs, not inside them: every club is "
                    f"projected at its full {budget:,.0f}, so the league reads "
                    f"{budget * len(clubs) + got.sum():,.0f} — a season plus "
                    f"an offseason that has not happened yet.")
        elif (res > 0).any():
            lines.append(f"    reserved for signings: {(res > 0).sum()} "
                         f"team(s), no free agents in the set to take it")
        empty = team_diag[team_diag["n_projected"] == 0]
        if len(empty):
            # Such a club cannot reach its budget — there is nobody to give it
            # to — so the league total will fall short. That is a roster or
            # tier problem, not an allocation one, and it must not be silent.
            lines.append(
                f"    WARNING: {len(empty)} team(s) have no projected players "
                f"and cannot close: {sorted(empty['team_id'].tolist())[:5]}")

    top = stats.get("roles", {})
    if top:
        shown = sorted(top.items(), key=lambda kv: -kv[1])[:6]
        lines.append("    roles: " + ", ".join(f"{k} {v}" for k, v in shown))
    return "\n".join(lines)

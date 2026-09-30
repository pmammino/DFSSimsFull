"""
role_taxonomy.py
================
The role vocabulary, its playing-time anchors, and the default assignment.

ONE definition, shared by the two things that must agree about it:

    scripts/build_role_templates.py   the human-editable workbook
    playing_time_model.py             the model that turns roles into PA/IP

These lived in the workbook script. Leaving them there would have meant the
model re-declaring its own copy, and two copies of a table nobody diffs is a
drift waiting to happen — a role renamed in one place and not the other fails
as a silent lookup miss, which reads downstream as "that player has no role"
rather than as an error.

What a role is
--------------
A role is a *job*, and the anchor is the playing time that job carries over a
full 162-game season at full availability. Two further dimensions scale it:

    Role Start    when the player takes the job (Opening Day .. Late Season)
    Availability  the share of the season he is healthy and on an MLB roster

so volume is `anchor x timing_share x availability`, before any team-level
closure. Splitting timing out this way unifies two things that are the same
quantity measured from different ends: a callup arriving in July and a starter
returning from the IL in July both hold the job for half a season.

The anchors are STARTING POINTS, not measurements
-------------------------------------------------
They are published-consensus figures and reasoning about roster construction,
not a fit. `fit_role_anchors` is the function that should replace them once
there is a season of assignments to fit against. Until then they are wrong in
the third digit and roughly right in the first, which is the correct
ambition for a baseline.

Two of them encode a fact worth keeping visible:

  * **Catchers have their own ladder.** A full-time catcher is ~500 PA, not
    630. Applying the Full Time anchor to catchers over-projects every catcher
    in baseball by ~120 PA, and there are 30 of them.
  * **Bullpen IP is nearly flat.** Closer 62, Setup 65, Middle 60. What
    separates those roles is save and hold context, not innings — which is
    exactly why they are worth distinguishing at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# The roles
# ─────────────────────────────────────────────────────────────────────────────
# PA / IP anchors are per 162 team games, at full availability, with the role
# held all season.
#
# vL Share is the fraction of plate appearances against left-handed pitching,
# given separately for left- and right-handed batters because the same role
# implies very different exposure depending on which side of a platoon the
# player is on. This is the column that fixes a real bug: the pipeline derives
# vL_share from a player's PAST usage (splits_model.py), so a hitter moving
# into a platoon job keeps a stale league-average share.

HITTER_ROLES: list[dict] = [
    dict(role="Full Time", pa=630, vl_lhb=0.26, vl_rhb=0.29,
         note="Everyday regular, ~150 games. Sees the league handedness mix."),
    dict(role="Everyday DH / 1B-DH", pa=600, vl_lhb=0.26, vl_rhb=0.29,
         note="Rarely rested, no defensive injury exposure."),
    dict(role="Strong Side Platoon", pa=480, vl_lhb=0.12, vl_rhb=0.45,
         note="The high-volume half of a platoon — usually a LHB who sits "
              "against LHP. Plays the majority of games."),
    dict(role="Weak Side Platoon", pa=230, vl_lhb=0.08, vl_rhb=0.68,
         note="The low-volume half — usually a RHB who plays mostly against "
              "LHP. Only ~28% of league PA are vs LHP, which caps his volume."),
    dict(role="Catcher - Primary", pa=500, vl_lhb=0.26, vl_rhb=0.29,
         note="CATCHERS HAVE THEIR OWN LADDER. A full-time catcher is ~500 PA, "
              "not 630 — applying the Full Time anchor over-projects every "
              "catcher in baseball by ~120 PA."),
    dict(role="Catcher - Tandem", pa=340, vl_lhb=0.22, vl_rhb=0.38,
         note="Roughly even split with a partner; often matchup-managed."),
    dict(role="Catcher - Backup", pa=180, vl_lhb=0.24, vl_rhb=0.33,
         note="Clear #2, starts ~45-55 games."),
    dict(role="Utility IF", pa=290, vl_lhb=0.26, vl_rhb=0.31,
         note="Covers multiple infield spots; volume depends on incumbents' "
              "health."),
    dict(role="Utility OF / 4th OF", pa=300, vl_lhb=0.24, vl_rhb=0.33,
         note="Distinct from Utility IF — different volume and platoon usage."),
    dict(role="Bench Bat", pa=150, vl_lhb=0.22, vl_rhb=0.36,
         note="Pinch-hits and spot starts."),
    dict(role="Injury Replacement / 26th Man", pa=90, vl_lhb=0.26, vl_rhb=0.29,
         note="Plays only when someone ahead of him is hurt. His volume is "
              "conditional on OTHER players' injuries — a team-level coupling "
              "that ultimately needs simulating, not a point estimate."),
    dict(role="Depth (no MLB PA)", pa=1, vl_lhb=0.26, vl_rhb=0.29,
         note="Organizational depth. Hits the 1 PA floor (PT_FLOOR_PA) so he "
              "stays present and joinable without moving any aggregate."),
]

PITCHER_ROLES: list[dict] = [
    dict(role="Ace (SP1)", ip=195, gs=32, g=32, sv=0.00, hld=0.00,
         note="Front-line starter, ~6.1 IP per start."),
    dict(role="Mid-Rotation Starter (SP2-3)", ip=170, gs=30, g=30,
         sv=0.00, hld=0.00, note="~5.2 IP per start."),
    dict(role="End-of-Rotation Starter (SP4-5)", ip=130, gs=25, g=26,
         sv=0.00, hld=0.01,
         note="Shorter leash; occasional bullpen appearance."),
    dict(role="Innings-Limited Starter", ip=110, gs=22, g=22,
         sv=0.00, hld=0.00,
         note="Young arm on a workload cap, or a post-surgery ramp. Common "
              "now and materially different from End-of-Rotation."),
    dict(role="Swing Arm / Long Relief", ip=95, gs=8, g=30,
         sv=0.01, hld=0.04, note="Moves between rotation and bullpen."),
    dict(role="Opener", ip=55, gs=18, g=45, sv=0.00, hld=0.03,
         note="Already modelled on the daily side (OPENER_BF_MEAN = 4.6 in "
              "slate_config), so it belongs here for consistency. Takes the "
              "start but faces ~4-5 batters."),
    dict(role="Closer", ip=62, gs=0, g=60, sv=0.65, hld=0.05,
         note="Takes the large majority of the team's save pool. Note IP is "
              "NEARLY FLAT across bullpen roles — what differs is save and "
              "hold context, which is why these roles are worth separating."),
    dict(role="Late Inning RP (Setup)", ip=65, gs=0, g=65, sv=0.12, hld=0.28,
         note="Primary hold earner; fills in for the closer."),
    dict(role="Middle Relief", ip=60, gs=0, g=58, sv=0.04, hld=0.14,
         note="Bridge innings, lower leverage."),
    dict(role="Bullpen Depth Arm", ip=40, gs=0, g=35, sv=0.01, hld=0.05,
         note="Up and down from AAA; mop-up and spot duty."),
    dict(role="Rehab / Injury Return", ip=60, gs=10, g=14, sv=0.02, hld=0.03,
         note="Expected back mid-season. Pair with a Role Start of Mid or "
              "Late rather than discounting the anchor twice."),
    dict(role="Depth (no MLB IP)", ip=1, gs=0, g=1, sv=0.00, hld=0.00,
         note="Organizational depth. Hits the 1 IP floor (PT_FLOOR_IP)."),
]

# Role Start -> share of the season the role is held. This replaces the
# Early/Mid/Late "callup" roles with a strictly more expressive parameter, and
# unifies callup timing with return-from-injury timing — they are the same
# quantity.
TIMING = [
    ("Opening Day", 1.00, "On the roster from day one."),
    ("Early Season (~May)", 0.80, "Called up or activated in April/May."),
    ("Mid Season (~July)", 0.50, "Mid-season callup, or back from a long IL stay."),
    ("Late Season (~Sept)", 0.20, "September callup / late activation."),
]

DEFAULT_TIMING = "Opening Day"
DEFAULT_AVAILABILITY = 1.00

# Role FAMILY, and how many of each a 26-man roster actually carries.
#
# This exists because ranking a club's players by raw innings to decide who is
# "core" ranks them on the wrong axis. A staff carries five or six starters and
# seven or eight relievers; it does not carry thirteen arms sorted by volume.
# Ranking by volume put 14 of 30 CLOSERS outside the core 13 and crushed them
# with the depth discount — Justin Martinez projected 1.4 innings while
# recording 22 saves, behind three End-of-Rotation arms sitting at the evidence
# floor whose only qualification was a bigger anchor.
#
# The taxonomy already said why volume cannot be the axis: bullpen IP is nearly
# flat across roles (closer 62, setup 65, middle 60), and what separates those
# jobs is save and hold context. So a closer loses an innings race to any
# marginal starter, every time, however certain his job is.
#
# Ranking within FAMILY fixes it: starters compete with starters for six slots,
# relievers with relievers for eight, and the closer is core by construction.
ROLE_FAMILY: dict[str, str] = {
    # hitters — nine lineup spots plus a four-man bench
    "Full Time": "LINEUP", "Everyday DH / 1B-DH": "LINEUP",
    "Strong Side Platoon": "LINEUP", "Catcher - Primary": "LINEUP",
    "Catcher - Tandem": "LINEUP",
    "Weak Side Platoon": "BENCH", "Catcher - Backup": "BENCH",
    "Utility IF": "BENCH", "Utility OF / 4th OF": "BENCH",
    "Bench Bat": "BENCH", "Injury Replacement / 26th Man": "BENCH",
    "Depth (no MLB PA)": "DEPTH",
    # pitchers — a rotation and a bullpen
    "Ace (SP1)": "SP", "Mid-Rotation Starter (SP2-3)": "SP",
    "End-of-Rotation Starter (SP4-5)": "SP", "Innings-Limited Starter": "SP",
    "Opener": "SP", "Swing Arm / Long Relief": "SP",
    "Closer": "RP", "Late Inning RP (Setup)": "RP", "Middle Relief": "RP",
    "Bullpen Depth Arm": "RP", "Rehab / Injury Return": "RP",
    "Depth (no MLB IP)": "DEPTH",
}

# Core slots per family. Hitters: nine in the lineup, four on the bench.
# Pitchers: six starters (five plus the sixth who covers doubleheaders and
# injuries) and eight relievers. 13 + 13 = the 26-man roster.
FAMILY_CORE_SLOTS: dict[str, int] = {
    "LINEUP": 9, "BENCH": 4, "SP": 6, "RP": 8, "DEPTH": 0,
}


def role_family(role: str) -> str:
    """Which part of the roster a role belongs to. Unknown roles are DEPTH,
    the conservative answer: they get the discount rather than a core slot."""
    return ROLE_FAMILY.get(role, "DEPTH")


# Seniority WITHIN a family — who holds a core slot when there are more
# candidates than slots. Lower is more senior.
#
# The role already encodes how secure a job is, so ranking on it is strictly
# better than ranking on innings. Volume-ranking inside the bullpen put a
# closer 13th among his own relievers and left him 9 innings: his 62-inning
# anchor loses to any setup man with a better evidence factor, even though the
# ninth inning is the most certain job in the pen. A closer is core by
# definition; a depth arm is not, whatever his numbers look like.
#
# Ties are broken on raw volume, so within a role the better-established
# player keeps the slot.
ROLE_DEPTH_ORDER: dict[str, int] = {
    # rotation
    "Ace (SP1)": 1, "Mid-Rotation Starter (SP2-3)": 2,
    "End-of-Rotation Starter (SP4-5)": 3, "Innings-Limited Starter": 4,
    "Swing Arm / Long Relief": 5, "Opener": 6,
    # bullpen
    "Closer": 1, "Late Inning RP (Setup)": 2, "Middle Relief": 3,
    "Rehab / Injury Return": 4, "Bullpen Depth Arm": 5,
    # lineup
    "Full Time": 1, "Everyday DH / 1B-DH": 1, "Catcher - Primary": 2,
    "Strong Side Platoon": 3, "Catcher - Tandem": 4,
    # bench
    "Utility IF": 1, "Utility OF / 4th OF": 1, "Catcher - Backup": 2,
    "Weak Side Platoon": 3, "Bench Bat": 4,
    "Injury Replacement / 26th Man": 5,
    # depth
    "Depth (no MLB PA)": 9, "Depth (no MLB IP)": 9,
}


def role_depth_order(role: str) -> int:
    """Seniority within the family; unknown roles sort last."""
    return ROLE_DEPTH_ORDER.get(role, 9)

# The two roles that mean "organizational depth" — the ones that must land on
# the 1 PA / 1 IP floor rather than in the team's allocation.
DEPTH_HITTER_ROLE = "Depth (no MLB PA)"
DEPTH_PITCHER_ROLE = "Depth (no MLB IP)"


# ─────────────────────────────────────────────────────────────────────────────
# Lookups
# ─────────────────────────────────────────────────────────────────────────────

def role_names(kind: str) -> list[str]:
    return [r["role"] for r in _defs(kind)]


def _defs(kind: str) -> list[dict]:
    if kind == "hitter":
        return HITTER_ROLES
    if kind == "pitcher":
        return PITCHER_ROLES
    raise ValueError(f"kind must be 'hitter' or 'pitcher', got {kind!r}")


def role_anchor(role: str, kind: str) -> dict | None:
    """The anchor row for `role`, or None if the name is not in the taxonomy.

    Returns None rather than raising so a typo in a hand-edited override file
    can be REPORTED with the offending name instead of aborting a pipeline run
    — but callers must not silently treat None as zero playing time, which
    would turn a typo into an invisible benching.
    """
    for r in _defs(kind):
        if r["role"] == role:
            return r
    return None


def timing_share(label: str | float | None) -> float:
    """Season share for a Role Start label. Accepts a raw number too.

    Unknown labels fall back to a full season rather than zero: a role held
    for an unknown fraction of the year is much more likely to be a full one
    (that is the common case and the default) than none of it, and zero would
    silently delete the player from every total.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return 1.0
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return float(np.clip(float(label), 0.0, 1.0))
    for name, share, _note in TIMING:
        if str(label).strip().lower() == name.lower():
            return share
    return 1.0


def is_depth_role(role: str) -> bool:
    return role in (DEPTH_HITTER_ROLE, DEPTH_PITCHER_ROLE)


# ─────────────────────────────────────────────────────────────────────────────
# Primary position — what finally lets a catcher be treated as a catcher
# ─────────────────────────────────────────────────────────────────────────────

def primary_positions(fielding: pd.DataFrame, *,
                      id_col: str = "PlayerId") -> dict[int, str]:
    """{PlayerId: primary position} from the fielding history.

    The position a player logged the most INNINGS at, in the most recent
    season he has, which is the definition that survives a mid-career move:
    a shortstop who became a second baseman last year is a second baseman.

    This is the data the role heuristic never had. The workbook's own
    docstring said so — "it knows nothing about position, so it never suggests
    a catcher role — fill those in by hand" — and the consequence was that
    every catcher defaulted to a hitter ladder topping out at 630 PA against a
    real ~500. The fielding fetch supplies one row per (player, position) with
    innings, so the gap is closed.

    DH is deliberately NOT a position here: it tells you where a player does
    not field, not what job he holds, and a DH-heavy catcher is still a
    catcher for playing-time purposes.
    """
    need = {id_col, "Pos", "Innings", "Season"}
    if fielding is None or fielding.empty or not need <= set(fielding.columns):
        return {}
    f = fielding[fielding["Pos"].astype(str).str.upper() != "DH"].copy()
    f["Innings"] = pd.to_numeric(f["Innings"], errors="coerce").fillna(0.0)
    f = f[f["Innings"] > 0]
    if f.empty:
        return {}
    latest = f.groupby(id_col)["Season"].transform("max")
    f = f[f["Season"] == latest]
    # Most innings wins; ties break on position name so the result is
    # deterministic regardless of row order.
    f = f.sort_values([id_col, "Innings", "Pos"], ascending=[True, False, True])
    picked = f.groupby(id_col).first().reset_index()
    return {int(r[id_col]): str(r["Pos"]) for _, r in picked.iterrows()}


# ─────────────────────────────────────────────────────────────────────────────
# Default role assignment
# ─────────────────────────────────────────────────────────────────────────────

def full_time_reference(volume) -> float:
    """What counts as a full-time workload IN THIS FILE.

    The 95th percentile of the observed volume, not an absolute PA threshold,
    so the heuristic survives a partial-season source: a file covering half a
    year has half-sized volumes throughout, and judging shares against its own
    top end keeps the role distribution stable either way.

    Shared, because the workbook's "Suggested Role" column and the model's
    default assignment must agree about what a full-time workload is — a
    reader who accepts the workbook's suggestion should get the workbook's
    role. `volume` may be a Series or a frame carrying one of the volume
    columns.
    """
    if isinstance(volume, pd.DataFrame):
        col = next((c for c in ("evidence_volume", "Last_PA", "PA", "TBF")
                    if c in volume.columns), None)
        if col is None:
            return 1.0
        volume = volume[col]
    v = pd.to_numeric(volume, errors="coerce").dropna()
    v = v[v > 0]
    if v.empty:
        return 1.0
    ref = float(np.percentile(v, 95))
    return ref if ref > 0 else 1.0


def suggest_hitter_role(row, reference: float, position: str | None = None):
    """A default, NOT a projection.

    Scaled to a full-season equivalent so it survives a partial-season source
    file: `reference` is the volume that counts as a full-time workload, and
    everything is judged as a share of it rather than against an absolute PA
    threshold that a half-season file would fail.

    Catchers get the catcher ladder when position is known — see
    `primary_positions`.
    """
    if row.get("pt_tier") == "floor":
        return DEPTH_HITTER_ROLE
    pa = row.get("evidence_volume")
    if pa is None or pd.isna(pa):
        pa = row.get("Last_PA", 0) or 0
    share = float(pa) / max(reference, 1.0)      # 1.0 = a full-time workload

    if str(position or "").upper() == "C":
        # A catcher's ladder is compressed: the everyday job is ~500 PA, so
        # the same share of a full-time workload means a different role.
        if share >= 0.62:
            return "Catcher - Primary"
        if share >= 0.38:
            return "Catcher - Tandem"
        return "Catcher - Backup"

    if share >= 0.80:
        return "Full Time"
    if share >= 0.55:
        return "Strong Side Platoon"
    if share >= 0.35:
        return "Utility OF / 4th OF"
    # Bench Bat is the FLOOR for a projected-tier player, not the depth role.
    # The depth role routes a player to the 1 PA placeholder, which would
    # contradict the tier he already cleared: `pt_tier == "projected"` means he
    # had real MLB volume inside the lookback. Returning depth here assigned
    # 144 such hitters a role that benched them to 1 PA.
    return "Bench Bat"


def suggest_pitcher_role(row, reference: float, staff_rank=None):
    """Starters split by IP per game (rate-based, so partial seasons are fine).

    Relievers split by their RA9 rank within their own staff, because a
    bullpen role is inherently a within-team standing — the best arm on the
    staff is the likeliest closer. A real depth chart knows things RA9 does
    not (contract, handedness, the manager's preferences), which is what the
    override file is for.
    """
    if row.get("pt_tier") == "floor":
        return DEPTH_PITCHER_ROLE
    if str(row.get("role", "")).lower() == "starter":
        ip_g = row.get("weighted_IP_per_G", 0) or 0
        if ip_g >= 5.8:
            return "Ace (SP1)"
        if ip_g >= 5.0:
            return "Mid-Rotation Starter (SP2-3)"
        if ip_g >= 3.5:
            return "End-of-Rotation Starter (SP4-5)"
        return "Swing Arm / Long Relief"
    if staff_rank == 1:
        return "Closer"
    if staff_rank in (2, 3):
        return "Late Inning RP (Setup)"
    if staff_rank is not None and staff_rank <= 6:
        return "Middle Relief"
    return "Bullpen Depth Arm"

"""
market_odds.py
==============
Betting-market team strength as a prior on the season projection.

The market is the best single forward-looking estimate of team talent
available, and it knows things a roster aggregate cannot: front-office intent,
depth behind the starters, managerial quality, spring injuries, and every
signing that has not shown up in a stat line yet. Folding it in is the cheapest
large accuracy gain available to the team layer.

What it feeds
-------------
    market -> expected wins -> win% -> save & hold opportunity pools
                            -> run differential -> RS/RA magnitude

Wins, and therefore the save and hold pools, are the natural consumers. Run
scoring is a partial consumer, and the distinction matters:

**The market constrains the run DIFFERENTIAL, not the RS/RA split.** A 95-win
team could be 5.2 RS / 4.2 RA or 4.3 / 3.3 — futures prices cannot tell those
apart, and nothing in this module pretends otherwise. So the market supplies
the magnitude and the bottom-up roster factors supply the split. That is the
whole design of `apply_market_to_run_environment`.

Which market to use
-------------------
In descending order of usefulness:

1. **Season win totals** (`over/under 88.5`) — a DIRECT read on expected wins.
   Use these when you can get them. `WinTotalLine` handles them natively and
   needs none of the inference below.
2. **Division / pennant odds** — one playoff round removed, still fairly tight.
3. **World Series futures** — what this module was asked for, and the loosest
   of the three, because a championship is four short series deep. Playoff
   randomness compresses everything: even the best team in baseball only wins
   the title 15-20% of the time, so the top of the odds board saturates and
   carries less information about talent than the middle does.

None of this makes WS odds unusable. It means they should be used for what they
are reliably good at — the ORDERING and relative spacing of teams — with the
absolute scale supplied from elsewhere. See `implied_wins_from_probabilities`.

Known market biases
-------------------
* **Favorite-longshot bias.** Longshots are systematically overbet, so their
  de-vigged probability overstates their true chance. The `power` de-vig method
  corrects for this; `proportional` does not. Default is `power`.
* **Public teams.** Large-market clubs carry shorter prices than talent alone
  justifies. Nothing here corrects for that; treat a big-market team's
  market-implied wins as a mild over-estimate.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from team_context import (
    TEAM_ABBR_BY_ID,
    ROSTER_DIR,
    abbr_for_team_id,
    team_id_for_abbr,
)

# League win distribution, used to set the SCALE when inferring wins from
# championship odds. Both are stable historical facts about MLB, not estimates
# from the odds: 30 teams play 162 games, so the mean is exactly 81, and the
# spread of team win totals has sat near 11-12 wins for decades.
LEAGUE_MEAN_WINS = 81.0
LEAGUE_WIN_SD = 11.5

# Guard rails on inferred wins. No modern team has been credibly projected
# outside this band pre-season, and clipping keeps a mangled odds board from
# producing a 130-win club.
MIN_PROJECTED_WINS = 52.0
MAX_PROJECTED_WINS = 110.0

# Default weight on the market when blending with the bottom-up Pythagenpat.
#
# Market-specific, because the markets are not equally informative about wins.
# A win total is a direct estimate of the quantity we want, so it deserves to
# dominate. A championship future is four short series removed from it and is
# mostly useful for ordering teams, so it deserves less.
#
# None of these are fitted. The case for weighting the market higher generally:
# it is a real forecast with money behind it and sees things a roster aggregate
# cannot. The case for weighting it lower: the bottom-up estimate is built from
# the actual projected players and is not subject to public-team bias. Re-tune
# against a walk-forward backtest once one exists.
MARKET_WEIGHT_BY_MARKET = {
    "win_total":    0.70,   # a direct read on expected wins
    "division":     0.55,
    "pennant":      0.50,
    "world_series": 0.40,   # loosest; see implied_wins_from_probabilities
}
MARKET_WEIGHT = MARKET_WEIGHT_BY_MARKET["win_total"]


def default_market_weight(market: str | None) -> float:
    """Blend weight appropriate to the market type."""
    return MARKET_WEIGHT_BY_MARKET.get(
        str(market or "").lower(), MARKET_WEIGHT_BY_MARKET["world_series"])


# Standard deviation of a team's ACTUAL win total around its true expectation.
# Wins are Binomial(162, p), so the SD is sqrt(162 p (1-p)): 6.36 at p = .500
# and 6.23 at p = .600, i.e. essentially flat across the realistic range. Used
# to convert an over/under price into a shift off the posted line.
WIN_OUTCOME_SD = 6.35


# ─────────────────────────────────────────────────────────────────────────────
# Odds parsing
# ─────────────────────────────────────────────────────────────────────────────

def american_to_probability(odds: float) -> float:
    """Convert American odds to an implied (vigged) probability.

        +450  ->  100 / (450 + 100)      = 0.1818
        -150  ->  150 / (150 + 100)      = 0.6000

    These do NOT sum to 1 across a market — the excess is the book's margin.
    Remove it with `devig`.
    """
    o = float(odds)
    if o == 0:
        raise ValueError("American odds of 0 are not meaningful")
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def decimal_to_probability(odds: float) -> float:
    """Convert decimal/European odds to an implied (vigged) probability."""
    o = float(odds)
    if o <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {o}")
    return 1.0 / o


def to_probability(value: float, fmt: str = "american") -> float:
    """Dispatch on odds format: 'american', 'decimal', or 'probability'."""
    fmt = (fmt or "american").lower()
    if fmt in ("american", "us"):
        return american_to_probability(value)
    if fmt in ("decimal", "european", "eu"):
        return decimal_to_probability(value)
    if fmt in ("probability", "prob", "implied"):
        p = float(value)
        # Accept either 0-1 or a percentage.
        return p / 100.0 if p > 1.0 else p
    raise ValueError(f"unknown odds format {fmt!r}")


# ─────────────────────────────────────────────────────────────────────────────
# De-vigging
# ─────────────────────────────────────────────────────────────────────────────

def devig(probs: Mapping[str, float], method: str = "power",
          ) -> tuple[dict[str, float], float]:
    """Strip the book's margin so probabilities sum to 1.

    Returns (fair_probabilities, overround) where overround is the raw sum —
    a WS futures board typically comes in around 1.15-1.35, i.e. a 15-35%
    margin, which is far larger than a game line and worth reporting.

    Methods:
      ``proportional``  divide every probability by the sum. Simple, and the
                        usual default elsewhere, but it preserves the
                        favorite-longshot bias: longshots are overbet, so
                        scaling everything by one factor leaves them
                        overstated.
      ``power``         solve for k in  sum(p_i ** k) == 1. Because p < 1,
                        raising to k > 1 shrinks small probabilities harder
                        than large ones, which is the right direction for a
                        longshot-heavy market like WS futures. Default.
    """
    keys = list(probs)
    p = np.array([float(probs[k]) for k in keys], dtype=float)
    if len(p) == 0:
        return {}, 0.0
    if (p <= 0).any() or (p >= 1).any():
        raise ValueError("implied probabilities must lie strictly in (0, 1)")

    overround = float(p.sum())
    method = (method or "power").lower()

    if method == "proportional":
        fair = p / overround
    elif method == "power":
        # Bisect on k. k > 1 when the board is overround (the normal case).
        lo, hi = 0.2, 10.0
        for _ in range(200):
            k = 0.5 * (lo + hi)
            s = float(np.sum(p ** k))
            if abs(s - 1.0) < 1e-13:
                break
            # sum(p**k) decreases as k grows, since every p < 1.
            if s > 1.0:
                lo = k
            else:
                hi = k
        fair = p ** k
        fair = fair / fair.sum()        # tidy up residual float error
    else:
        raise ValueError(f"unknown devig method {method!r}")

    return {key: float(v) for key, v in zip(keys, fair)}, overround


# ─────────────────────────────────────────────────────────────────────────────
# Championship probability -> expected wins
# ─────────────────────────────────────────────────────────────────────────────

def implied_wins_from_probabilities(
    fair_probs: Mapping[int, float],
    *,
    league_mean: float = LEAGUE_MEAN_WINS,
    league_sd: float = LEAGUE_WIN_SD,
) -> dict[int, float]:
    """Infer expected wins from fair championship probabilities.

    The honest problem: there is no reliable closed-form map from a title
    probability to a win total. The relationship saturates badly at the top —
    a 100-win team and a 106-win team have nearly the same championship odds,
    because four rounds of short series wash out the difference — so inverting
    an assumed curve would invent precision that is not in the prices.

    What the market IS reliable about is the ordering of teams and the relative
    size of the gaps between them. So this maps the market's log-probability
    SPACING onto the known league win distribution:

        score_i = ln(p_i)                     monotone in talent
        z_i     = (score_i - mean) / sd       standardize the spacing
        wins_i  = league_mean + z_i * league_sd

    The market supplies the shape; MLB's own long-run win distribution (mean
    exactly 81, SD ~11.5) supplies the scale. Two consequences worth knowing:

      * League wins sum to 30 x 81 = 2430 by construction, which is the closure
        constraint `team_wins` also enforces.
      * The result is a RANK-and-SPACING transform, so it is insensitive to the
        absolute level of the odds board and to the de-vig method's effect on
        that level. It is still sensitive to relative distortion, which is why
        the power de-vig matters.

    If you have win-total LINES, do not use this at all — `WinTotalLine` reads
    expected wins straight off the market.
    """
    keys = list(fair_probs)
    if not keys:
        return {}
    p = np.array([float(fair_probs[k]) for k in keys], dtype=float)
    if (p <= 0).any():
        raise ValueError("fair probabilities must be positive to take a log")

    score = np.log(p)
    sd = float(score.std())
    z = np.zeros_like(score) if sd <= 0 else (score - score.mean()) / sd
    wins = np.clip(league_mean + z * league_sd,
                   MIN_PROJECTED_WINS, MAX_PROJECTED_WINS)
    # Re-centre after clipping so the league still totals 30 x league_mean.
    wins = wins + (league_mean - wins.mean())
    return {int(k): float(w) for k, w in zip(keys, wins)}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def win_total_expectation(line: float, over: float | None = None,
                          under: float | None = None,
                          *, odds_format: str = "american",
                          outcome_sd: float = WIN_OUTCOME_SD) -> float:
    """True expected wins implied by a win-total line AND its prices.

    A posted total is not the market's expectation — the PRICE tells you which
    side of the line the expectation sits on. "88.5, over -130 / under +105"
    means the market thinks 88.5 is low; taking the line at face value throws
    that away, and the discarded information is worth up to a win and a half.

    A team's actual win total is Binomial(162, p) around its true mean, so with
    outcome SD sigma:

        P(over L) = P(W > L) = Phi((mu - L) / sigma)
        =>  mu = L + sigma * Phi^-1(P(over))

    The two prices are de-vigged against each other first, since a two-way
    market's raw probabilities sum to well over 1.

    With no prices this returns the line unchanged, which is the honest
    fallback: a bare line IS the best available estimate when the juice is
    unknown.
    """
    L = float(line)
    if over is None and under is None:
        return L

    # De-vig the two-way market.
    p_over = to_probability(over, odds_format) if over is not None else None
    p_under = to_probability(under, odds_format) if under is not None else None
    if p_over is not None and p_under is not None:
        total = p_over + p_under
        if total <= 0:
            return L
        p_over = p_over / total
    elif p_over is None:
        # Only the under was given; the fair over is its complement, but with
        # no pair to de-vig against the margin is still baked in. Strip a
        # typical half-margin so a one-sided price is not read as if it were
        # fair. Crude, and the file format asks for both sides for this reason.
        p_over = 1.0 - p_under / max(p_under + (1 - p_under) * 1.045, 1e-9)
    else:
        p_over = p_over / max(p_over + (1 - p_over) * 1.045, 1e-9)

    p_over = float(np.clip(p_over, 1e-4, 1 - 1e-4))
    # Inverse normal CDF without pulling in scipy: Acklam's rational
    # approximation, accurate to ~1e-9 over the range that matters here.
    z = _norm_ppf(p_over)
    return L + outcome_sd * z


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Hand-rolled to keep this module free of a scipy import for one function;
    scipy is a project dependency but market odds should stay importable in a
    minimal environment.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = np.sqrt(-2 * np.log(1 - p))
        return -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                  + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1)


def MARKET_ODDS_PATH(target_year: int) -> Path:
    """Conventional location of the market-odds file for a target year."""
    return ROSTER_DIR / f"market_odds_{int(target_year)}.json"


def load_market_odds(path: str | Path) -> pd.DataFrame:
    """Load a market-odds file into [team_id, team_abbr, source, value].

    Format (see rosters/market_odds.example.json):

        {
          "target_year": 2027,
          "odds_format": "american",
          "market": "world_series",
          "as_of": "2026-11-15",
          "book": "consensus",
          "teams": {"LAD": 450, "NYY": 600, "COL": 30000, ...}
        }

    `market` is `world_series` (or `pennant` / `division`) for futures prices,
    or `win_total` for season win-total lines — in which case `teams` holds the
    line itself (e.g. `{"LAD": 95.5}`) and no inference is needed.

    A missing file returns an empty frame, so the market layer is optional.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["team_id", "team_abbr", "value"])

    payload = json.loads(p.read_text())
    teams = payload.get("teams", {})
    fmt = payload.get("odds_format", "american")
    # win_total is the default because it is the market worth using: a direct
    # read on expected wins rather than something four playoff rounds removed.
    market = str(payload.get("market", "win_total")).lower()

    rows = []
    for raw_team, value in teams.items():
        tid = team_id_for_abbr(raw_team)
        if tid is None or tid not in TEAM_ABBR_BY_ID:
            warnings.warn(f"market odds for unknown team {raw_team!r}; skipped")
            continue
        if value is None:
            continue
        # A team entry is either a bare number, or an object carrying the line
        # plus its over/under prices. The prices are what let the posted line
        # be corrected to a true expectation.
        over = under = None
        if isinstance(value, dict):
            line = value.get("line", value.get("value"))
            if line is None:
                warnings.warn(f"market entry for {raw_team!r} has no line; skipped")
                continue
            over, under = value.get("over"), value.get("under")
            value = line
        rows.append({"team_id": tid, "team_abbr": abbr_for_team_id(tid),
                     "value": float(value),
                     "over": None if over is None else float(over),
                     "under": None if under is None else float(under)})

    out = pd.DataFrame(rows)
    out.attrs["market"] = market
    out.attrs["odds_format"] = fmt
    out.attrs["as_of"] = payload.get("as_of")
    out.attrs["book"] = payload.get("book")
    if len(out) and len(out) < len(TEAM_ABBR_BY_ID):
        warnings.warn(
            f"market odds cover only {len(out)}/30 teams. Missing clubs fall "
            "back to the bottom-up projection, which mixes two scales — "
            "prefer a complete board."
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The market layer
# ─────────────────────────────────────────────────────────────────────────────

def market_expected_wins(
    odds: pd.DataFrame,
    *,
    devig_method: str = "power",
    league_sd: float = LEAGUE_WIN_SD,
) -> pd.DataFrame:
    """Turn a loaded odds board into per-team expected wins.

    Handles both shapes: a `win_total` market is read directly, anything else
    is treated as a futures price and routed through de-vig plus the
    spacing map.

    Returns [team_id, team_abbr, market_prob, market_wins, market_source]
    (`market_prob` is NaN for win-total lines, which carry no title
    probability).
    """
    if odds.empty:
        return pd.DataFrame(columns=["team_id", "team_abbr", "market_prob",
                                     "market_wins", "market_source"])

    market = odds.attrs.get("market", "world_series")
    fmt = odds.attrs.get("odds_format", "american")
    out = odds[["team_id", "team_abbr"]].copy()

    if market == "win_total":
        # A direct read — no championship-probability inference needed. The
        # only work is correcting each posted line to a true expectation using
        # its over/under price, which is where the remaining information is.
        has_price = ("over" in odds.columns
                     and odds["over"].notna().any()) or (
                     "under" in odds.columns and odds["under"].notna().any())
        out["market_prob"] = np.nan
        out["market_wins"] = [
            win_total_expectation(
                r.value,
                getattr(r, "over", None) if pd.notna(getattr(r, "over", np.nan)) else None,
                getattr(r, "under", None) if pd.notna(getattr(r, "under", np.nan)) else None,
                odds_format=fmt,
            )
            for r in odds.itertuples()
        ]
        out["posted_line"] = odds["value"].to_numpy(float)
        out["market_source"] = "win_total"
        out.attrs["priced"] = bool(has_price)

        # Books shade totals so the book balances action, and the shading is
        # not symmetric across a board, so the league mean drifts off 81.
        # Re-centre to keep closure — every game has one winner.
        shift = LEAGUE_MEAN_WINS - float(out["market_wins"].mean())
        if abs(shift) > 0.05 and len(out) >= 25:
            out["market_wins"] = out["market_wins"] + shift
            out.attrs["recentre_shift"] = shift
        return out.sort_values("market_wins", ascending=False).reset_index(drop=True)

    vigged = {int(r.team_id): to_probability(r.value, fmt)
              for r in odds.itertuples()}
    fair, overround = devig(vigged, method=devig_method)
    wins = implied_wins_from_probabilities(fair, league_sd=league_sd)

    out["market_prob"] = [fair[int(t)] for t in out["team_id"]]
    out["market_wins"] = [wins[int(t)] for t in out["team_id"]]
    out["market_source"] = market
    out.attrs["overround"] = overround
    out.attrs["devig_method"] = devig_method
    return out.sort_values("market_wins", ascending=False).reset_index(drop=True)


def blend_market_wins(
    wins: pd.DataFrame,
    market: pd.DataFrame,
    *,
    team_col: str = "team_id",
    market_weight: float = MARKET_WEIGHT,
) -> pd.DataFrame:
    """Blend market-implied wins with the bottom-up Pythagenpat wins.

    Teams absent from the market board keep their bottom-up value. The blend is
    re-centred so the league still totals exactly 2430 — the same closure
    constraint the rest of the team layer enforces, and the reason a partial
    board is a warning rather than a silent distortion.

    Adds [market_wins, blended_wins, market_weight] and rewrites
    `expected_wins` / `win_pct` so every downstream consumer — the save and
    hold pools above all — picks the blend up automatically.
    """
    from team_wins import GAMES_PER_SEASON, TOTAL_LEAGUE_WINS

    out = wins.copy()
    if market.empty:
        out["market_wins"] = np.nan
        out["blended_wins"] = out["expected_wins"]
        out["market_weight"] = 0.0
        return out

    w = float(np.clip(market_weight, 0.0, 1.0))
    lookup = dict(zip(market[team_col].astype(int),
                      market["market_wins"].astype(float)))
    out["market_wins"] = [lookup.get(int(t), np.nan) for t in out[team_col]]
    # Preserve the pre-blend bottom-up value. `expected_wins` is overwritten
    # below so downstream consumers pick the blend up automatically, which
    # would otherwise destroy the only record of what the market actually
    # moved — and that difference is the most interesting output here.
    out["roster_wins"] = out["expected_wins"].to_numpy(float)

    have = out["market_wins"].notna()
    blended = out["expected_wins"].to_numpy(float).copy()
    blended[have] = (w * out.loc[have, "market_wins"].to_numpy(float)
                     + (1.0 - w) * out.loc[have, "expected_wins"].to_numpy(float))

    # Re-centre to the league total. Every game has one winner, so this is a
    # requirement, not a nicety.
    total = blended.sum()
    if total > 0:
        blended = blended * (TOTAL_LEAGUE_WINS / total)

    out["blended_wins"] = blended
    out["market_weight"] = np.where(have, w, 0.0)
    out["expected_wins"] = blended
    out["win_pct"] = blended / GAMES_PER_SEASON
    return out.sort_values("expected_wins", ascending=False).reset_index(drop=True)


def apply_market_to_run_environment(
    wins: pd.DataFrame,
    *,
    team_col: str = "team_id",
    exponent: float = 0.287,
) -> pd.DataFrame:
    """Re-derive RS/RA so they imply the blended win total.

    This is the "possibly the run scoring inputs as well" piece, and it needs
    care, because **the market says nothing about the RS/RA split**. A 95-win
    club could be 5.2/4.2 or 4.3/3.3 and the futures price is identical.

    So the two sources are used for what each actually knows:

        market  ->  the MAGNITUDE of the run differential (via wins)
        roster  ->  how that differential splits between offense and defense

    Concretely: invert Pythagenpat to find the RS/RA ratio the blended win
    percentage requires, then rotate the team's existing RS and RA to that
    ratio while holding their SUM fixed. Holding the sum fixed is what keeps
    the team in its own run environment — a good pitching team stays
    low-scoring on both sides rather than being handed a generic one.

    Adds [rs_per_game_market, ra_per_game_market, run_diff_market] and leaves
    the originals in place so the market's effect stays auditable.
    """
    out = wins.copy()
    if out.empty or "win_pct" not in out.columns:
        return out

    rs = out["rs_per_game"].to_numpy(float)
    ra = out["ra_per_game"].to_numpy(float)
    total = rs + ra
    wp = np.clip(out["win_pct"].to_numpy(float), 1e-4, 1 - 1e-4)

    # Pythagenpat: wp = RS^e / (RS^e + RA^e), so (RS/RA)^e = wp / (1 - wp).
    e = np.power(np.clip(total, 1e-6, None), exponent)
    ratio = np.power(wp / (1.0 - wp), 1.0 / np.clip(e, 1e-6, None))

    # Hold RS + RA fixed and solve for the pair with that ratio.
    ra_new = total / (1.0 + ratio)
    rs_new = total - ra_new

    out["rs_per_game_market"] = rs_new
    out["ra_per_game_market"] = ra_new
    out["run_diff_market"] = (rs_new - ra_new) * 162.0
    return out


def market_report(market: pd.DataFrame, blended: pd.DataFrame | None = None,
                  *, top: int = 10) -> str:
    """Readable summary of the market layer and what it moved."""
    if market.empty:
        return "no market odds loaded"
    lines = []
    over = market.attrs.get("overround")
    if over:
        lines.append(f"  board overround {over:.3f} "
                     f"({(over - 1) * 100:.1f}% margin), de-vig "
                     f"{market.attrs.get('devig_method')}")
    has_blend = (blended is not None
                 and {"roster_wins", "blended_wins"} <= set(blended.columns))
    head = f"{'team':<6}{'P(WS)':>8}{'mkt W':>8}"
    if has_blend:
        head += f"{'roster W':>10}{'blended':>9}{'vs roster':>11}"
    lines.append(head)

    merged = market
    if has_blend:
        merged = market.merge(
            blended[["team_id", "roster_wins", "blended_wins"]],
            on="team_id", how="left",
        ).sort_values("blended_wins", ascending=False)

    for _, r in merged.head(top).iterrows():
        prob = r.get("market_prob")
        row = (f"{str(r.get('team_abbr')):<6}"
               f"{(f'{prob:.3f}' if pd.notna(prob) else '-'):>8}"
               f"{r['market_wins']:>8.1f}")
        if has_blend and pd.notna(r.get("blended_wins")):
            row += (f"{r['roster_wins']:>10.1f}"
                    f"{r['blended_wins']:>9.1f}"
                    f"{r['blended_wins'] - r['roster_wins']:>+11.1f}")
        lines.append(row)
    if has_blend:
        shift = (merged["blended_wins"] - merged["roster_wins"]).abs()
        lines.append(f"  mean |market shift| {shift.mean():.1f} wins, "
                     f"max {shift.max():.1f}")
    return "\n".join(lines)

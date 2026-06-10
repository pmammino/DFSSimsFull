"""
park_factors.py
===============
Apply Statcast park factors to per-PA event projections.

Pipeline produces two parallel sets of projections:
    1. NEUTRAL — what the player would do at a league-average park
       (the existing `P_K`, `P_BB`, `P_HR`, etc.)
    2. PARK-ADJUSTED — what the player is expected to do given their home
       park, accounting for half-season home games and half-season road games
       (the new `_park` columns)

Why both? The neutral projection isolates true player skill — useful for
comparing players across teams, for free-agent valuation, and for forecasting
mid-season trades. The park-adjusted projection answers "what will this
specific player produce on this specific team" — what fantasy users want.

Methodology
-----------
Statcast park factors are indexed so 100 = league average and represent the
within-park effect for that metric. A factor of 127 for HR means 27% more
HR per PA at that park than at other parks (controlling for batters and
pitchers who played at both).

Since a player plays roughly 50% of games at home and 50% on the road, and
the road games are split across many parks (averaging near-neutral), the
effective full-season factor is:

    effective_factor = 0.5 * home_park_factor + 0.5 * road_park_factor_avg
                     ≈ 0.5 * home_park_factor + 0.5 * 1.0
                     = 0.5 * (home_park_factor + 1.0)

So a 1.27 HR factor at the home park becomes 1.135 over the full season —
13.5% more HR than at a neutral park. The 0.5 home-share weight is a tunable
parameter (`home_share`); the default of 0.5 matches MLB's 81-81 schedule.

What about events without park factors?
    - HBP, K, BB, 1B, 2B, 3B, HR all have park factors from Statcast.
    - SF (sacrifice flies) doesn't have a published factor — treated as
      neutral (factor 1.0). SF is a tiny fraction of PA anyway.
    - P_BIPOut serves as the residual: after factoring all named events,
      P_BIPOut = 1 - sum(others). This automatically conserves probability.

Renormalization
---------------
After applying factors, the sum of probabilities may differ from 1. We
handle this by setting P_BIPOut as the residual. If the sum of factored
non-BIPOut events exceeds 1 (rare but possible), we scale them all down to
sum to (1 - min_BIPOut) and clip P_BIPOut to min_BIPOut. This preserves the
RATIOS of the factored events but caps the total.

Should park factors apply BEFORE the SB and R/RBI models?
---------------------------------------------------------
For SB: YES. The SB equation is
    P_SB_ATTEMPT = (P_1B + P_BB + P_HBP) × Pred_attempts_per_opp
Park affects P_1B and P_BB (the opportunity rate), which in turn changes the
projected SB and CS rates. The player's attempt rate per opportunity is a
behavioral choice (a function of speed and team philosophy) and isn't itself
park-driven, so it stays constant.

For R/RBI: NO. The team_RPG factor already encodes park effects implicitly
— a team playing in a hitter-friendly park naturally has higher RPG. The
team factor is applied to a team-NEUTRALIZED player projection in the R/RBI
model, which gives the right answer. Applying park factors on top of the
team_RPG adjustment would double-count the park effect.

Pitchers
--------
Pitchers also pitch ~50% of innings at home and ~50% on the road, so the
same effective-factor formula applies. A pitcher on a HR-friendly park will
see his per-PA HR rate increase relative to neutral. Some pipelines flip
the factor inversion for pitchers (since "good for batters = bad for
pitchers"), but the per-PA event rates themselves are symmetric — a HR-prone
park increases HR-per-PA for both sides of the matchup. So the same factor
applies; the interpretation just shifts (more HR = worse for the pitcher).
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# Events that get park-factor adjustments. P_HBP and P_SF have no published
# factor; they stay at 1.0. P_BIPOut is the residual and isn't directly
# factored.
PARK_FACTORED_EVENTS = {
    "P_K":  "pf_SO",
    "P_BB": "pf_BB",
    "P_HR": "pf_HR",
    "P_1B": "pf_1B",
    "P_2B": "pf_2B",
    "P_3B": "pf_3B",
}

# Events that pass through unchanged (no factor applied).
PARK_PASSTHROUGH_EVENTS = ["P_HBP", "P_SF"]

# Residual event that absorbs whatever probability is left after factors.
PARK_RESIDUAL_EVENT = "P_BIPOut"

# Default: 50% home games, 50% road games (matching MLB's 81-81 schedule).
DEFAULT_HOME_SHARE = 0.5


def compute_effective_factors(park_factor_row: pd.Series,
                              home_share: float = DEFAULT_HOME_SHARE
                              ) -> dict[str, float]:
    """Convert a park's raw factors into full-season effective factors.

    Each factor is shrunk by:  eff = home_share * factor + (1 - home_share) * 1.0
    so a 1.27 HR factor with home_share=0.5 becomes 1.135.

    Returns dict keyed by P_<event>, e.g., {'P_HR': 1.135, 'P_1B': 0.96, ...}
    """
    out = {}
    for event_col, pf_col in PARK_FACTORED_EVENTS.items():
        raw = float(park_factor_row.get(pf_col, 1.0))
        eff = home_share * raw + (1 - home_share) * 1.0
        out[event_col] = eff
    return out


def apply_park_adjustment_to_row(row: pd.Series, eff_factors: dict[str, float],
                                  min_bipout: float = 0.05
                                  ) -> dict[str, float]:
    """Apply effective factors to a single player's probability vector and
    renormalize so probabilities sum to 1.

    Strategy:
        1. Multiply each factored event probability by its effective factor.
        2. P_BIPOut absorbs the residual: P_BIPOut_new = 1 - sum(others).
        3. If P_BIPOut_new < min_bipout, scale the factored events down so
           that P_BIPOut = min_bipout. (This is a safety net for extreme
           park factors; in practice almost never triggers.)
    """
    new_probs = {}
    factored_sum = 0.0

    for event_col in PARK_FACTORED_EVENTS:
        p = float(row.get(event_col, 0.0))
        eff = eff_factors.get(event_col, 1.0)
        new_p = p * eff
        new_probs[event_col] = new_p
        factored_sum += new_p

    # Add passthrough events (no factor, no change)
    for event_col in PARK_PASSTHROUGH_EVENTS:
        p = float(row.get(event_col, 0.0))
        new_probs[event_col] = p
        factored_sum += p

    # Set residual
    bipout = 1.0 - factored_sum
    if bipout < min_bipout:
        # Scale all non-BIPOut events down so BIPOut = min_bipout
        target = 1.0 - min_bipout
        scale = target / factored_sum if factored_sum > 0 else 1.0
        for k in list(new_probs.keys()):
            new_probs[k] *= scale
        bipout = min_bipout
    new_probs[PARK_RESIDUAL_EVENT] = float(bipout)

    return new_probs


def apply_park_factors_to_projections(proj_df: pd.DataFrame,
                                       park_df: pd.DataFrame,
                                       team_id_col: str = "Pred_target_team_id",
                                       home_share: float = DEFAULT_HOME_SHARE,
                                       fallback_team_id_col: str = "Team",
                                       suffix: str = "_park",
                                       ) -> pd.DataFrame:
    """Add park-adjusted projections to a player projection frame.

    Parameters
    ----------
    proj_df : pd.DataFrame
        Must have the per-PA probability columns (P_K, P_BB, P_HR, P_1B, P_2B,
        P_3B, P_HBP, P_SF, P_BIPOut) and a team identifier column.
    park_df : pd.DataFrame
        From `fetch_park_factors`, with TeamId and pf_* columns. We use
        Statcast factors which are indexed (1.0 = average; 1.27 = +27%).
    team_id_col : str
        Column name in proj_df that holds the player's home TeamId. For
        hitters this is `Pred_target_team_id` from the R/RBI step. For
        pitchers we use the latest team from their history (we add this
        column ourselves if not present).
    home_share : float
        Fraction of games played at home park. Default 0.5 = 81-81 schedule.
    suffix : str
        Appended to each event-probability column for the park-adjusted
        version (e.g., 'P_HR' → 'P_HR_park').

    Returns
    -------
    A copy of proj_df with new columns:
        P_K_park, P_BB_park, P_HR_park, P_1B_park, P_2B_park, P_3B_park,
        P_HBP_park, P_SF_park, P_BIPOut_park,
        home_park_HR_pf, home_park_1B_pf, ..., (raw factors for transparency)
        eff_HR_pf, ... (full-season effective factors after home_share blend)
        home_park_team_id (the team_id used for lookup)
    """
    out = proj_df.copy()
    # Map team_id → park factors row (one row per team)
    park_idx = park_df.set_index("TeamId").to_dict(orient="index")

    # Determine the team_id for each row
    if team_id_col in out.columns:
        team_ids = out[team_id_col].astype("Int64")
    else:
        team_ids = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

    new_event_cols = {f"{ev}{suffix}": [] for ev in
                       list(PARK_FACTORED_EVENTS) + PARK_PASSTHROUGH_EVENTS
                       + [PARK_RESIDUAL_EVENT]}
    pf_raw_cols  = {f"pf_{pf.split('_')[1]}": [] for pf in PARK_FACTORED_EVENTS.values()}
    pf_eff_cols  = {f"eff_{pf.split('_')[1]}": [] for pf in PARK_FACTORED_EVENTS.values()}
    home_team_col: list = []

    for i, row in out.iterrows():
        tid = team_ids.iloc[out.index.get_loc(i)] if len(team_ids) else None
        # Resolve park factor row
        pf_row = None
        if pd.notna(tid) and int(tid) in park_idx:
            pf_row = pd.Series(park_idx[int(tid)])
        if pf_row is None:
            # Neutral fallback — all factors = 1.0
            pf_row = pd.Series({pf: 1.0 for pf in PARK_FACTORED_EVENTS.values()})

        eff = compute_effective_factors(pf_row, home_share=home_share)
        new_probs = apply_park_adjustment_to_row(row, eff)
        for k, v in new_probs.items():
            new_event_cols[f"{k}{suffix}"].append(v)
        for pf_col in PARK_FACTORED_EVENTS.values():
            # short suffix: 'pf_HR' → 'HR'
            short = pf_col.split("_", 1)[1]
            pf_raw_cols[f"pf_{short}"].append(float(pf_row.get(pf_col, 1.0)))
            pf_eff_cols[f"eff_{short}"].append(eff[
                next(k for k, v in PARK_FACTORED_EVENTS.items() if v == pf_col)
            ])
        home_team_col.append(int(tid) if pd.notna(tid) else None)

    # Attach to output
    for col, vals in new_event_cols.items():
        out[col] = vals
    for col, vals in pf_raw_cols.items():
        out[col] = vals
    for col, vals in pf_eff_cols.items():
        out[col] = vals
    out["home_park_team_id"] = home_team_col
    return out


def derive_park_adjusted_sb(proj_df: pd.DataFrame,
                            attempts_col: str = "Pred_attempts_per_opp",
                            success_col: str  = "Pred_success_rate",
                            k_attempts: float = 50.0,
                            k_success: float = 200.0,
                            suffix: str = "_park",
                            ) -> pd.DataFrame:
    """Recompute SB projections using park-adjusted opportunity rates.

    Park affects which events happen (1B, BB, HBP) but NOT the player's
    behavioral attempt rate or success rate. So we reuse `Pred_attempts_per_opp`
    and `Pred_success_rate` and just plug in the park-adjusted 1B/BB/HBP.

        P_SB_ATTEMPT_park = (P_1B_park + P_BB_park + P_HBP_park)
                            × Pred_attempts_per_opp
        P_SB_park         = P_SB_ATTEMPT_park × Pred_success_rate
        P_CS_park         = P_SB_ATTEMPT_park × (1 - Pred_success_rate)
    """
    out = proj_df.copy()
    if attempts_col not in out.columns:
        return out  # SB columns not present yet — skip

    opp = (out.get(f"P_1B{suffix}", 0).fillna(0)
           + out.get(f"P_BB{suffix}", 0).fillna(0)
           + out.get(f"P_HBP{suffix}", 0).fillna(0))
    att_per_opp = out[attempts_col].fillna(0.0)
    succ = out[success_col].fillna(0.78)

    out[f"P_SB_ATTEMPT{suffix}"] = opp * att_per_opp
    out[f"P_SB{suffix}"]         = out[f"P_SB_ATTEMPT{suffix}"] * succ
    out[f"P_CS{suffix}"]         = out[f"P_SB_ATTEMPT{suffix}"] * (1 - succ)
    out[f"Pred_steal_opp_per_PA{suffix}"] = opp
    return out


def derive_park_adjusted_summary_stats(proj_df: pd.DataFrame,
                                        suffix: str = "_park") -> pd.DataFrame:
    """Compute park-adjusted AVG, OBP, BABIP, AVG_against, BABIP_against.

    Mirrors the neutral summary stats computation. Uses _park columns.
    """
    out = proj_df.copy()

    p_1B = out.get(f"P_1B{suffix}", 0).fillna(0)
    p_2B = out.get(f"P_2B{suffix}", 0).fillna(0)
    p_3B = out.get(f"P_3B{suffix}", 0).fillna(0)
    p_HR = out.get(f"P_HR{suffix}", 0).fillna(0)
    p_K  = out.get(f"P_K{suffix}", 0).fillna(0)
    p_BB = out.get(f"P_BB{suffix}", 0).fillna(0)
    p_HBP = out.get(f"P_HBP{suffix}", 0).fillna(0)
    p_SF  = out.get(f"P_SF{suffix}", 0).fillna(0)
    p_BIPOut = out.get(f"P_BIPOut{suffix}", 0).fillna(0)

    hits = p_1B + p_2B + p_3B + p_HR
    ab = hits + p_K + p_BIPOut + p_SF  # AB = PA - BB - HBP - SF... but SF is in PA
    # AB = PA - BB - HBP - SF (where SF is sac fly)
    ab_proxy = 1.0 - p_BB - p_HBP - p_SF
    on_base = hits + p_BB + p_HBP

    out[f"AVG{suffix}"]   = hits / ab_proxy.replace(0, np.nan)
    out[f"OBP{suffix}"]   = on_base
    bip = p_1B + p_2B + p_3B + p_BIPOut
    out[f"BABIP{suffix}"] = (p_1B + p_2B + p_3B) / bip.replace(0, np.nan)

    # For pitchers (no P_HBP column on pitchers): same formula works
    out[f"AVG_against{suffix}"]   = out[f"AVG{suffix}"]
    out[f"BABIP_against{suffix}"] = out[f"BABIP{suffix}"]
    return out

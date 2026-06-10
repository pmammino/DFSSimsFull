"""
sb_model.py
===========
Stolen-base-per-PA AND SB-attempts-per-PA projection.

Architecture (two-stage decomposition):
    P_SB_ATTEMPT = (P_1B + P_BB + P_HBP) × Pred_attempts_per_opp
    P_SB         = P_SB_ATTEMPT × Pred_success_rate
    P_CS         = P_SB_ATTEMPT × (1 - Pred_success_rate)

Stage 1: "Steal opportunity per PA" — comes from the rate model's existing
projections of P_1B + P_BB + P_HBP. We use the pipeline's predictions so the
SB model is consistent with the rest of the projections.

Stage 2a: "Attempts per opportunity" — the player-controlled decision to try
to steal, projected via PA-weighted recency-decay shrinkage plus a sprint-
speed adjustment. This is the MOST stable signal in the SB chain (YoY r=0.81).

Stage 2b: "Success rate" — what fraction of attempts succeed. Per-player
success rate has very low YoY stability (r≈0.12-0.20), so we shrink hard to
the league average (~78.4%). Only well-established baserunners with many
career attempts deviate noticeably from league.

Why project attempts first rather than SB directly:
    - Attempts/opp YoY r = 0.81 (highest among any SB-related metric)
    - SB/opp YoY r = 0.76
    - The attempts decision is the player skill; success is mostly situational

This decomposition gives the user both SB and SB-attempts as outputs, which
is more useful than SB alone (the fantasy/strategy value of an attempt-prone
runner is different from a high-success-rate runner with few attempts).

Key empirical findings driving the parameter choices:
    - Sprint speed YoY r = 0.92 (essentially a fixed physical attribute)
    - Sprint speed correlates r = +0.56 with SB/PA across players
    - OBP / 1B% / BB% only correlate r ≈ 0.05 with SB/PA, because SB/PA is
      dominated by attempt rate not opportunity rate. We let stage 1 (which
      uses the rate model's OBP-equivalent outputs) handle the opportunity
      side and focus stage 2 on the attempt-per-opportunity signal.
    - Adding age as a feature: no marginal value once sprint is included.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# League-average baselines used as fallbacks when a player has no history
# at all. These match the 2024-2025 aggregate; will be recomputed at
# inference time from actual data.
_DEFAULT_LEAGUE_ATTEMPTS_PER_OPP = 0.1044
_DEFAULT_LEAGUE_SUCCESS_RATE     = 0.7835


def _get_sprint_speed(pid: int, year: int, sp_df: pd.DataFrame) -> float:
    """Most recent available sprint speed for this player up to `year`.

    If we have no sprint speed measurement for this player at all, returns NaN.
    Sprint speed is highly stable YoY (r=0.92), so backfilling with the most
    recent available season is fine.
    """
    pid_data = sp_df[sp_df["player_id"] == pid]
    if pid_data.empty:
        return float("nan")
    eligible = pid_data[pid_data["year"] <= year]
    if eligible.empty:
        eligible = pid_data
    return float(eligible.sort_values("year", ascending=False).iloc[0]["sprint_speed"])


def _prepare_history(rate_df: pd.DataFrame, target_year: int,
                     max_history_years: int) -> tuple[pd.DataFrame, float, float]:
    """Augment rate_df with the columns we need, slice to the relevant window,
    and return league baselines computed on that same window.
    """
    df = rate_df.copy()
    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["steal_opp"] = df["1B"] + df["BB"] + df["HBP"]
    df["attempts"] = df["SB"] + df["CS"]

    prior = df[(df["Season"] < target_year) &
               (df["Season"] >= target_year - max_history_years) &
               (df["PA"] >= 25)]

    if prior.empty or prior["steal_opp"].sum() == 0:
        return prior, _DEFAULT_LEAGUE_ATTEMPTS_PER_OPP, _DEFAULT_LEAGUE_SUCCESS_RATE

    league_att_per_opp = float(prior["attempts"].sum() / max(prior["steal_opp"].sum(), 1))
    total_att = prior["attempts"].sum()
    league_success = (float(prior["SB"].sum() / total_att)
                      if total_att > 0 else _DEFAULT_LEAGUE_SUCCESS_RATE)
    return prior, league_att_per_opp, league_success


def project_attempts_per_opp(rate_df: pd.DataFrame, target_year: int,
                              sprint_df: pd.DataFrame,
                              k_attempts: float = 50.0,
                              sprint_coef: float = 0.005,
                              sprint_baseline: float = 27.0,
                              max_history_years: int = 5,
                              decay: float = 0.85) -> pd.DataFrame:
    """Per-player projection of SB attempts per steal opportunity.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Historical hitter stats with at least: Season, PlayerId, PA, SB, CS,
        BB, HBP, H, 2B, 3B, HR.
    target_year : int
        Year to project.
    sprint_df : pd.DataFrame
        Sprint speed leaderboard with columns: year, player_id, sprint_speed.
    k_attempts : float
        Shrinkage prior strength on attempts/opp (opportunity-equivalent units).
        Tuned to 50 via experiment_sb_model.py.
    sprint_coef : float
        Additive adjustment to projected attempts/opp per (sprint_speed -
        sprint_baseline). 0.005 means a 30 ft/s sprinter gets +0.015 to
        their attempts/opp over a 27 ft/s baseline.
    sprint_baseline : float
        ft/s value treated as "league average" for the sprint adjustment.

    Returns
    -------
    DataFrame with PlayerId, Pred_attempts_per_opp, n_eff_opp, sprint_speed_used.
    """
    prior, league_att_per_opp, _ = _prepare_history(rate_df, target_year, max_history_years)

    rows = []
    if prior.empty:
        return pd.DataFrame(columns=["PlayerId", "Pred_attempts_per_opp",
                                     "n_eff_opp", "sprint_speed_used"])

    for pid, g in prior.groupby("PlayerId"):
        g = g.sort_values("Season").copy()
        g["years_back"] = target_year - g["Season"].astype(int)
        w_decay = decay ** g["years_back"].values
        w_opp = g["steal_opp"].astype(float).values * w_decay

        if w_opp.sum() > 0:
            weighted_att_per_opp = (
                float(np.sum(g["attempts"].astype(float).values * w_decay))
                / float(w_opp.sum())
            )
        else:
            weighted_att_per_opp = league_att_per_opp
        n_eff_opp = float(w_opp.sum())
        pred = ((n_eff_opp * weighted_att_per_opp + k_attempts * league_att_per_opp)
                / (n_eff_opp + k_attempts))

        ss = _get_sprint_speed(int(pid), target_year - 1, sprint_df)
        if not np.isnan(ss):
            pred = max(0.0, pred + (ss - sprint_baseline) * sprint_coef)

        rows.append({
            "PlayerId": int(pid),
            "Pred_attempts_per_opp": float(pred),
            "n_eff_opp": n_eff_opp,
            "sprint_speed_used": float(ss) if not np.isnan(ss) else None,
        })

    return pd.DataFrame(rows)


def project_success_rate(rate_df: pd.DataFrame, target_year: int,
                          k_success: float = 200.0,
                          max_history_years: int = 5,
                          decay: float = 0.85) -> pd.DataFrame:
    """Per-player projection of SB success rate (SB / attempts).

    Per-player success rate has very low YoY stability (r ~ 0.12-0.20), so
    we shrink heavily to the league mean. The k_success prior is in
    "attempts" units — a player needs ~200 weighted attempts to half-pull
    from league. Most players never accumulate 200 lifetime attempts, so the
    bulk of the population sits very close to the league success rate.

    This is intentional: success rate is mostly situational (catcher arm,
    pitcher hold time, game context) rather than player skill, so projecting
    elite individuated success rates would be over-fitting.

    Returns
    -------
    DataFrame with PlayerId, Pred_success_rate, n_eff_attempts.
    """
    prior, _, league_success = _prepare_history(rate_df, target_year, max_history_years)

    rows = []
    if prior.empty:
        return pd.DataFrame(columns=["PlayerId", "Pred_success_rate",
                                     "n_eff_attempts"])

    for pid, g in prior.groupby("PlayerId"):
        g = g.sort_values("Season").copy()
        g["years_back"] = target_year - g["Season"].astype(int)
        w_decay = decay ** g["years_back"].values
        w_att = g["attempts"].astype(float).values * w_decay

        if w_att.sum() > 0:
            weighted_success = (
                float(np.sum(g["SB"].astype(float).values * w_decay))
                / float(w_att.sum())
            )
        else:
            weighted_success = league_success
        n_eff_att = float(w_att.sum())
        pred = ((n_eff_att * weighted_success + k_success * league_success)
                / (n_eff_att + k_success))
        # Clip to safe range
        pred = float(np.clip(pred, 0.0, 1.0))

        rows.append({
            "PlayerId": int(pid),
            "Pred_success_rate": pred,
            "n_eff_attempts": n_eff_att,
        })

    return pd.DataFrame(rows)


def combine_sb_components(rate_projections: pd.DataFrame,
                          attempts_projections: pd.DataFrame,
                          success_projections: pd.DataFrame) -> pd.DataFrame:
    """Merge stage-2 projections into the rate-projection frame.

    Adds columns: Pred_attempts_per_opp, Pred_success_rate, n_eff_opp,
    n_eff_attempts, sprint_speed_used. Default values when a player has no
    SB history: zero attempts, league-average success rate.
    """
    out = rate_projections.copy()
    out = out.merge(
        attempts_projections[["PlayerId", "Pred_attempts_per_opp",
                              "n_eff_opp", "sprint_speed_used"]],
        on="PlayerId", how="left",
    )
    out = out.merge(
        success_projections[["PlayerId", "Pred_success_rate", "n_eff_attempts"]],
        on="PlayerId", how="left",
    )
    out["Pred_attempts_per_opp"] = out["Pred_attempts_per_opp"].fillna(0.0)
    out["Pred_success_rate"] = out["Pred_success_rate"].fillna(_DEFAULT_LEAGUE_SUCCESS_RATE)
    out["n_eff_opp"] = out["n_eff_opp"].fillna(0.0)
    out["n_eff_attempts"] = out["n_eff_attempts"].fillna(0.0)
    return out


def derive_sb_per_pa(out_df: pd.DataFrame, P_1B_col: str = "P_1B",
                      P_BB_col: str = "P_BB", P_HBP_col: str = "P_HBP",
                      k_attempts: float = 50.0,
                      k_success: float = 200.0) -> pd.DataFrame:
    """Compute per-PA SB attempts, SB, and CS rates, plus SDs.

    Adds:
        Pred_steal_opp_per_PA  — opportunity rate (from rate model outputs)
        P_SB_ATTEMPT           — per-PA rate of attempting to steal
        P_SB                   — per-PA successful steal rate
        P_CS                   — per-PA caught stealing rate
        SD_SB_ATTEMPT, SD_SB   — binomial SEs propagated through the chain
    """
    out = out_df.copy()
    opp_per_pa = (out[P_1B_col].fillna(0) + out[P_BB_col].fillna(0)
                  + out[P_HBP_col].fillna(0))
    out["Pred_steal_opp_per_PA"] = opp_per_pa

    att_per_opp = out["Pred_attempts_per_opp"].fillna(0.0)
    succ = out["Pred_success_rate"].fillna(_DEFAULT_LEAGUE_SUCCESS_RATE)

    out["P_SB_ATTEMPT"] = opp_per_pa * att_per_opp
    out["P_SB"] = out["P_SB_ATTEMPT"] * succ
    out["P_CS"] = out["P_SB_ATTEMPT"] * (1 - succ)

    # Standard errors via binomial uncertainty on each component
    n_total_opp = out["n_eff_opp"].fillna(0) + k_attempts
    se_att = np.sqrt(np.maximum(att_per_opp.clip(0, 1) * (1 - att_per_opp.clip(0, 1)), 1e-6)
                     / np.maximum(n_total_opp, 10))
    out["SD_SB_ATTEMPT"] = opp_per_pa * se_att

    # SD on SB combines (attempts uncertainty × succ) + (succ uncertainty × attempts)
    # via delta method: var(att * succ) ≈ succ² var(att) + att² var(succ)
    n_total_att = out["n_eff_attempts"].fillna(0) + k_success
    se_succ = np.sqrt(np.maximum(succ.clip(0, 1) * (1 - succ.clip(0, 1)), 1e-6)
                      / np.maximum(n_total_att, 20))
    var_sb_per_opp = (succ ** 2) * (se_att ** 2) + (att_per_opp ** 2) * (se_succ ** 2)
    out["SD_SB"] = opp_per_pa * np.sqrt(var_sb_per_opp)

    return out

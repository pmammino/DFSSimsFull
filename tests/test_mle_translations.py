"""
Tests for the minor-league-translation (MLE) baseline path.

Run with:  python -m pytest tests/test_mle_translations.py -q
       or:  python tests/test_mle_translations.py   (falls back to a runner)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline_config as C
from mle_translations import (
    _parse_hitter, _parse_pitcher, translate_hitter, translate_pitcher,
    build_name_to_mlbam, build_synthetic_rows, apply_mle_bip_override,
)

# The two sample feed records the user provided.
HITTER = {
    "playerID": "17781", "player": "Ryan Fitzgerald",
    "firstname": "Ryan", "lastname": "Fitzgerald", "currentTeam": "LAD",
    "team": "Oklahoma C", "position": "3B", "games": "86", "ab": "352",
    "runs": "60", "hits": "108", "doubles": "20", "triples": "4", "hr": "7",
    "rbi": "56", "walks": "33", "strikes": "60", "steals": "8", "caught": "3",
    "avg": ".307", "obp": ".376", "slg": ".446", "ops": ".822",
}
PITCHER = {
    "playerID": "14590", "player": "Casey Lawrence",
    "firstname": "Casey", "lastname": "Lawrence", "team": "Tacoma", "pos": "P",
    "games": "19", "gs": "19", "ip": "108.2", "h": "119", "er": "48",
    "hr": "15", "k": "68", "bb": "16", "kpct": "15.3", "bbpct": "3.6",
    "era": "3.98", "whip": "1.25",
}
FEED = {"AAA": {"hitters": [HITTER], "pitchers": [PITCHER]}}


def _chadwick():
    return pd.DataFrame([
        {"key_mlbam": 600001, "name_first": "Ryan", "name_last": "Fitzgerald",
         "birth_year": 2000},
        {"key_mlbam": 500001, "name_first": "Casey", "name_last": "Lawrence",
         "birth_year": 1998},
    ])


def test_hitter_parse_pa_estimate():
    obs = _parse_hitter(HITTER)
    # PA ≈ (AB + BB) inflated for HBP/SF/SH (~2.5%)
    assert 380 < obs["PA"] < 410
    assert obs["1B"] == 108 - 20 - 4 - 7   # H - 2B - 3B - HR


def test_hitter_translation_factors():
    obs = _parse_hitter(HITTER)
    tr = translate_hitter(obs, "AAA")
    k_obs = obs["K"] / obs["PA"]
    bb_obs = obs["BB"] / obs["PA"]
    # K% inflated by the AAA factor, BB% deflated
    assert abs(tr["K%"] - k_obs * C.MLE_HITTER_FACTORS["AAA"]["K%"]) < 1e-9
    assert abs(tr["BB%"] - bb_obs * C.MLE_HITTER_FACTORS["AAA"]["BB%"]) < 1e-9
    # Per-BIP distribution is a valid probability vector
    bip = tr["bip"]
    assert abs(sum(bip.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in bip.values())


def test_pitcher_tbf_recovery_and_translation():
    obs = _parse_pitcher(PITCHER)
    # TBF recovered from K / kpct
    assert abs(obs["TBF"] - 68 / 0.153) < 1e-6
    tr = translate_pitcher(obs, "AAA")
    # K% allowed deflated, BB% allowed inflated in MLB
    assert tr["K%"] < 0.153
    assert tr["BB%"] > 0.036
    assert abs(sum(tr["bip"].values()) - 1.0) < 1e-9


def test_credibility_discount_deflates_sample():
    _, _, _ = build_synthetic_rows(FEED, "hitter", 2027, {}, None, set())  # smoke
    idx = build_name_to_mlbam(_chadwick())
    rows, bip, stats = build_synthetic_rows(
        FEED, "hitter", 2027, idx, _chadwick(), existing_ids=set())
    assert stats["used"] == 1
    obs = _parse_hitter(HITTER)
    pa_eff = rows.iloc[0]["PA"]
    assert abs(pa_eff - obs["PA"] * C.MLE_PA_CREDIBILITY["AAA"]) < 1e-6
    # Season is target - offset
    assert rows.iloc[0]["Season"] == 2027 - C.MLE_SEASON_OFFSET
    # Age derived from birth_year
    assert rows.iloc[0]["Age"] == (2027 - C.MLE_SEASON_OFFSET) - 2000


def test_synthetic_rows_match_statsapi_schema():
    """Synthetic rows must concat cleanly onto the statsapi history frame."""
    idx = build_name_to_mlbam(_chadwick())
    rows, _, _ = build_synthetic_rows(
        FEED, "hitter", 2027, idx, _chadwick(), existing_ids=set())
    required = {"Season", "PlayerId", "Name", "Team", "TeamId", "Age", "PA",
                "AB", "K", "BB", "HBP", "SF", "H", "HR", "2B", "3B", "SB", "CS",
                "K%", "BB%", "HBP%", "SF%"}
    assert required.issubset(set(rows.columns))
    # Derived rate columns are internally consistent with the counts
    r = rows.iloc[0]
    assert abs(r["K%"] - r["K"] / r["PA"]) < 1e-6
    assert abs(r["BB%"] - r["BB"] / r["PA"]) < 1e-6


def test_existing_mlb_players_are_skipped():
    idx = build_name_to_mlbam(_chadwick())
    rows, _, stats = build_synthetic_rows(
        FEED, "hitter", 2027, idx, _chadwick(), existing_ids={600001})
    assert stats["skipped_existing"] == 1
    assert stats["used"] == 0
    assert len(rows) == 0


def test_ambiguous_and_unmatched_names_skipped():
    # Two MLBAM ids for the same name → ambiguous → skip
    chad = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "Ryan", "name_last": "Fitzgerald"},
        {"key_mlbam": 2, "name_first": "Ryan", "name_last": "Fitzgerald"},
    ])
    idx = build_name_to_mlbam(chad)
    rows, _, stats = build_synthetic_rows(
        FEED, "hitter", 2027, idx, chad, existing_ids=set())
    assert stats["skipped_unresolved"] == 1
    assert stats["used"] == 0


def test_bip_override_preserves_sum_and_scope():
    idx = build_name_to_mlbam(_chadwick())
    _, bip, _ = build_synthetic_rows(
        FEED, "hitter", 2027, idx, _chadwick(), existing_ids=set())
    # A tiny 9-event frame: one MLE rookie + one normal player
    final = pd.DataFrame([
        {"PlayerId": 600001, "P_K": 0.20, "P_BB": 0.08, "P_HBP": 0.012,
         "P_SF": 0.01, "P_HR": 0.03, "P_1B": 0.15, "P_2B": 0.05, "P_3B": 0.005,
         "P_BIPOut": 0.463},
        {"PlayerId": 999999, "P_K": 0.25, "P_BB": 0.09, "P_HBP": 0.012,
         "P_SF": 0.01, "P_HR": 0.04, "P_1B": 0.14, "P_2B": 0.05, "P_3B": 0.006,
         "P_BIPOut": 0.402},
    ])
    before = final[final.PlayerId == 999999].iloc[0].to_dict()
    out = apply_mle_bip_override(final, bip)
    ev = ["P_K", "P_BB", "P_HBP", "P_SF", "P_HR", "P_1B", "P_2B", "P_3B", "P_BIPOut"]
    rookie = out[out.PlayerId == 600001].iloc[0]
    assert abs(sum(rookie[c] for c in ev) - 1.0) < 1e-9      # still sums to 1
    # Non-MLE player untouched
    normal = out[out.PlayerId == 999999].iloc[0]
    for c in ev:
        assert abs(normal[c] - before[c]) < 1e-12


def test_fit_df_isolation_in_rate_models():
    """A synthetic rookie must get a projection WITHOUT shifting the league mean
    that real players shrink toward."""
    from rate_models import (
        build_inference_panel, fit_and_predict_decay_rate, league_mean,
    )
    from pipeline_config import SHRINK_K, RATE_DECAY, RATE_SHRINK_K_HITTER

    rng = np.random.default_rng(0)
    real_rows = []
    for pid in range(1, 41):
        for season in (2024, 2025, 2026):
            pa = 500
            kpct = 0.22 + rng.normal(0, 0.02)
            bbpct = 0.08 + rng.normal(0, 0.01)
            real_rows.append({
                "Season": season, "PlayerId": pid, "Name": f"P{pid}",
                "Team": "AAA", "TeamId": 100, "Age": 27, "PA": pa, "AB": pa * 0.9,
                "K": kpct * pa, "BB": bbpct * pa, "HBP": 0.01 * pa, "SF": 0.01 * pa,
                "H": 130, "HR": 20, "2B": 25, "3B": 2, "SB": 5, "CS": 2,
                "K%": kpct, "BB%": bbpct, "HBP%": 0.01, "SF%": 0.01,
            })
    real = pd.DataFrame(real_rows)

    # A rookie synthetic row at 2026 only, with a distinctive low K%
    rookie = real.iloc[0].to_dict()
    rookie.update({"PlayerId": 99999, "Name": "Rookie", "Season": 2026,
                   "PA": 250, "K%": 0.12, "BB%": 0.11, "HBP%": 0.012, "SF%": 0.01,
                   "K": 0.12 * 250, "BB": 0.11 * 250})
    combined = pd.concat([real, pd.DataFrame([rookie])], ignore_index=True)

    mu_real = league_mean(real, 2027, "K%", "PA")
    mu_comb = league_mean(combined, 2027, "K%", "PA")
    # The synthetic row DOES nudge a naive combined mean...
    # ...but the pipeline must shrink toward the REAL mean (fit_df=real).
    infer = build_inference_panel(combined, 2027, "PA", SHRINK_K)
    assert 99999 in set(infer["PlayerId"])          # rookie is projected

    out, _, _ = fit_and_predict_decay_rate(
        combined, 2027, "PA", "K%", prior_k=RATE_SHRINK_K_HITTER,
        decay=RATE_DECAY, infer=infer, max_history_years=5, fit_df=real)
    rk = out[out.PlayerId == 99999].iloc[0]
    # Rookie K% pulled from 0.12 toward the REAL league mean (~0.22), landing
    # between the two — i.e. shrinkage used the clean mean, not the combined one.
    assert 0.12 < rk["Pred_K%"] < mu_real + 1e-6
    assert rk["Pred_K%"] > 0.12


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

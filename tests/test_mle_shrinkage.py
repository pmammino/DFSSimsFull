"""
Tests for MLE credibility shrinkage and parent-org resolution.

These pin the two defects that the first refresh run with a POPULATED
minor-league feed exposed. Both were latent for as long as the feed returned a
single player, which is the reason they need permanent tests rather than a
one-off fix: nothing about the code changed to break them, only the volume of
data flowing through it.

    1. Translated rates were never shrunk toward the league prior. The
       credibility discount deflated the PA the synthetic row CARRIED, but the
       per-BIP profile went downstream as a finished distribution, so a
       prospect with a handful of plate appearances kept his sample rates
       verbatim:

           Carter Garate, Round Rock, 2.3 effective PA -> P_HR 0.283
           Ben Hansen,    Midland,    tiny TBF         -> RA9 -0.815

       0.283 per PA is 170 home runs per 600. A negative ERA is not a physical
       quantity. Both came from the same missing step.

    2. statsapi's minor-league /stats response names only the AFFILIATE, and
       `_parent_org` read `currentTeam`, which that feed does not supply. Every
       translated player therefore landed with an affiliate name and no team
       id: no organization, no park factor, neutral team context, and 150
       distinct team labels in the output.

Run with:  python -m pytest tests/test_mle_shrinkage.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline_config as C
from mle_translations import (
    _parent_org, credibility_weight, translate_hitter, translate_pitcher,
)
from pitcher_outputs import (
    MIN_RUNS_PER_PA, compute_ra9, compute_runs_allowed_per_pa,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hitter_obs(pa: float, hr: float) -> dict:
    """A plausible hitter line at `pa` plate appearances with `hr` home runs."""
    return {"PA": pa, "K": 0.22 * pa, "BB": 0.08 * pa, "HR": hr,
            "2B": 0.05 * pa, "3B": 0.005 * pa, "1B": 0.15 * pa,
            "SB": 0.02 * pa, "CS": 0.005 * pa}


def _pitcher_obs(tbf: float, k: float, h: float, hr: float) -> dict:
    return {"TBF": tbf, "K": k, "BB": 0.08 * tbf, "H": h, "HR": hr}


def _pa_eff(sample: float, level: str) -> float:
    return max(1.0, sample * C.MLE_PA_CREDIBILITY[level])


def _probs_frame(tr: dict) -> pd.DataFrame:
    """Turn a translation's rates + BIP profile into a per-PA event frame."""
    b = tr["bip"]
    mass = 1.0 - tr["K%"] - tr["BB%"] - tr["HBP%"] - tr["SF%"]
    return pd.DataFrame([{
        "P_K": tr["K%"], "P_BB": tr["BB%"], "P_HBP": tr["HBP%"],
        "P_SF": tr["SF%"],
        "P_HR": b["home_run"] * mass, "P_1B": b["single"] * mass,
        "P_2B": b["double"] * mass, "P_3B": b["triple"] * mass,
        "P_BIPOut": b["out"] * mass,
    }])


def _league_hr_per_bip() -> float:
    tot = sum(C.LEAGUE_BIP_PROFILE.values())
    return C.LEAGUE_BIP_PROFILE["home_run"] / tot


# ─────────────────────────────────────────────────────────────────────────────
# credibility_weight
# ─────────────────────────────────────────────────────────────────────────────

def test_credibility_weight_is_bounded_and_monotone():
    w = [credibility_weight(pa, C.MLE_SHRINK_PA)
         for pa in (0, 1, 10, 100, 1000, 100000)]
    assert all(0.0 <= x <= 1.0 for x in w)
    assert w == sorted(w), f"weight must rise with sample size: {w}"
    assert w[0] == 0.0, "a zero sample keeps none of its own signal"
    assert w[-1] > 0.99, "a huge sample keeps essentially all of it"


def test_credibility_weight_is_half_at_the_shrink_constant():
    """The constant's meaning: the sample size at which own signal = prior."""
    assert abs(credibility_weight(C.MLE_SHRINK_PA, C.MLE_SHRINK_PA) - 0.5) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# hitters
# ─────────────────────────────────────────────────────────────────────────────

def test_tiny_sample_home_run_rate_collapses_to_the_league_prior():
    """THE regression. 1 HR in 4 AAA PA must not survive as a 0.28 HR rate."""
    obs = _hitter_obs(4, 1)
    unshrunk = translate_hitter(obs, "AAA")
    shrunk = translate_hitter(obs, "AAA", _pa_eff(4, "AAA"))

    league = _league_hr_per_bip()
    # The fixture must still be badly wrong WITHOUT shrinkage, or this test
    # proves nothing. It no longer reaches the 0.283 we shipped only because
    # _HRPA_CLIP now caps the translated rate as a second line of defense;
    # 4x league is still a projection nobody would ship.
    assert unshrunk["bip"]["home_run"] > league * 4, (
        f"fixture stopped reproducing the defect: "
        f"{unshrunk['bip']['home_run']:.4f} vs league {league:.4f}")
    assert abs(shrunk["bip"]["home_run"] - league) < 0.01, (
        f"a 4-PA line must project ~league {league:.4f}, "
        f"got {shrunk['bip']['home_run']:.4f}")


def test_observed_defect_values_are_now_impossible():
    """No AAA/AA/A+/A line of any size may produce the values we shipped.

    0.12 per PA is above the all-time single-season record (~0.11, Bonds
    2001), so this is a physical bound and not a taste judgement.
    """
    worst = 0.0
    for level in C.MLE_LEVELS:
        for pa in (1, 2, 4, 10, 30, 100, 250, 500, 700):
            for hr_share in (0.1, 0.25, 0.5):
                obs = _hitter_obs(pa, pa * hr_share)
                tr = translate_hitter(obs, level, _pa_eff(pa, level))
                mass = 1.0 - tr["K%"] - tr["BB%"] - tr["HBP%"] - tr["SF%"]
                worst = max(worst, tr["bip"]["home_run"] * mass)
    assert worst < 0.12, f"projected {worst:.4f} HR/PA, above the MLB record"


def test_shrinkage_is_monotone_in_sample_size():
    """More evidence -> more of the player's own (high) rate survives."""
    seen = [translate_hitter(_hitter_obs(pa, pa * 0.06), "AAA",
                             _pa_eff(pa, "AAA"))["bip"]["home_run"]
            for pa in (4, 20, 100, 300, 550)]
    assert seen == sorted(seen), f"must rise with sample size: {seen}"


def test_a_full_season_keeps_most_of_its_own_signal():
    """Shrinkage must not flatten everyone: a real AAA season still differs."""
    pa = 550
    good = translate_hitter(_hitter_obs(pa, 30), "AAA", _pa_eff(pa, "AAA"))
    weak = translate_hitter(_hitter_obs(pa, 3), "AAA", _pa_eff(pa, "AAA"))
    league = _league_hr_per_bip()
    assert good["bip"]["home_run"] > league * 1.15
    assert weak["bip"]["home_run"] < league * 0.85


def test_lower_levels_shrink_harder_at_equal_sample():
    """Credibility ladder must still bite after shrinkage."""
    pa = 500
    hr = [translate_hitter(_hitter_obs(pa, 25), lvl, _pa_eff(pa, lvl))
          ["bip"]["home_run"] for lvl in ("AAA", "AA", "A+", "A")]
    assert hr == sorted(hr, reverse=True), (
        f"a Single-A line must move the needle least: {hr}")


def test_shrunk_bip_profile_stays_a_distribution():
    for level in C.MLE_LEVELS:
        for pa in (1, 7, 60, 600):
            tr = translate_hitter(_hitter_obs(pa, pa * 0.3), level,
                                  _pa_eff(pa, level))
            assert abs(sum(tr["bip"].values()) - 1.0) < 1e-9
            assert all(v >= 0.0 for v in tr["bip"].values())


def test_steal_rate_is_shrunk_too():
    """One steal in 3 PA is not a 0.33 SB rate."""
    obs = _hitter_obs(3, 0)
    obs["SB"] = 1
    tr = translate_hitter(obs, "AAA", _pa_eff(3, "AAA"))
    assert tr["SB_rate"] < 0.05, f"got {tr['SB_rate']:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# pitchers
# ─────────────────────────────────────────────────────────────────────────────

def test_tiny_sample_pitcher_never_produces_negative_runs():
    """THE other regression: RA9 -0.815 for a translated prospect."""
    obs = _pitcher_obs(12, 9, 0, 0)          # the Ben Hansen shape
    tr = translate_pitcher(obs, "AA", _pa_eff(12, "AA"))
    ra9, era, rpa, _ = compute_ra9(_probs_frame(tr))
    assert float(rpa.iloc[0]) > 0
    assert float(ra9.iloc[0]) > 0
    assert float(era.iloc[0]) > 0


def test_tiny_sample_pitcher_lands_in_a_believable_band():
    """Not merely positive — a no-information pitcher is a league-average one."""
    obs = _pitcher_obs(12, 9, 0, 0)
    tr = translate_pitcher(obs, "AA", _pa_eff(12, "AA"))
    ra9, _, _, _ = compute_ra9(_probs_frame(tr))
    assert 3.0 <= float(ra9.iloc[0]) <= 6.0, f"RA9 {float(ra9.iloc[0]):.2f}"


def test_no_translated_pitcher_of_any_size_goes_negative():
    worst = np.inf
    for level in C.MLE_LEVELS:
        for tbf in (1, 3, 12, 40, 150, 400, 700):
            for k_share in (0.0, 0.25, 0.5, 0.75, 1.0):
                obs = _pitcher_obs(tbf, tbf * k_share, 0, 0)
                tr = translate_pitcher(obs, level, _pa_eff(tbf, level))
                ra9, _, rpa, _ = compute_ra9(_probs_frame(tr))
                worst = min(worst, float(ra9.iloc[0]))
    assert worst > 0, f"worst projected RA9 {worst:.3f}"


def test_pitcher_shrinkage_preserves_ordering_at_a_real_sample():
    """A genuinely good AAA arm must still project better than a bad one."""
    tbf = 600
    good = translate_pitcher(_pitcher_obs(tbf, 200, 100, 8), "AAA",
                             _pa_eff(tbf, "AAA"))
    bad = translate_pitcher(_pitcher_obs(tbf, 80, 190, 30), "AAA",
                            _pa_eff(tbf, "AAA"))
    ra9_good = float(compute_ra9(_probs_frame(good))[0].iloc[0])
    ra9_bad = float(compute_ra9(_probs_frame(bad))[0].iloc[0])
    assert ra9_good < ra9_bad, f"good {ra9_good:.2f} vs bad {ra9_bad:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# the runs-mapping floor (the net, not the fix)
# ─────────────────────────────────────────────────────────────────────────────

def test_runs_mapping_is_floored_not_negative():
    """Feed the mapping a vector outside its domain directly."""
    df = pd.DataFrame([{"P_K": 0.90, "P_BB": 0.0, "P_HBP": 0.0, "P_SF": 0.0,
                        "P_HR": 0.0, "P_1B": 0.0, "P_2B": 0.0, "P_3B": 0.0,
                        "P_BIPOut": 0.10}])
    out = compute_runs_allowed_per_pa(df)
    assert float(out.iloc[0]) == pytest.approx(MIN_RUNS_PER_PA)


def test_runs_mapping_floor_is_below_every_real_pitcher():
    """The floor must never touch a legitimate projection."""
    df = pd.DataFrame([{"P_K": 0.35, "P_BB": 0.05, "P_HBP": 0.008,
                        "P_SF": 0.008, "P_HR": 0.015, "P_1B": 0.13,
                        "P_2B": 0.035, "P_3B": 0.003, "P_BIPOut": 0.40}])
    assert float(compute_runs_allowed_per_pa(df).iloc[0]) > MIN_RUNS_PER_PA * 5


def test_runs_mapping_unchanged_for_normal_input():
    """The floor is a clip, not a shift: ordinary rows must be untouched."""
    df = pd.DataFrame([
        {"P_K": 0.20, "P_BB": 0.09, "P_HBP": 0.01, "P_SF": 0.01,
         "P_HR": 0.035, "P_1B": 0.15, "P_2B": 0.05, "P_3B": 0.005,
         "P_BIPOut": 0.44},
        {"P_K": 0.30, "P_BB": 0.06, "P_HBP": 0.01, "P_SF": 0.01,
         "P_HR": 0.020, "P_1B": 0.13, "P_2B": 0.04, "P_3B": 0.004,
         "P_BIPOut": 0.43},
    ])
    manual = sum(w * df[c] for c, w in
                 __import__("pitcher_outputs").LINEAR_WEIGHTS_RUNS.items()
                 if c in df.columns) - 0.047
    got = compute_runs_allowed_per_pa(df)
    assert np.allclose(got.to_numpy(), manual.to_numpy())


# ─────────────────────────────────────────────────────────────────────────────
# parent-org resolution
# ─────────────────────────────────────────────────────────────────────────────

def test_parent_org_prefers_the_native_org_id():
    """A native MLBAM org id beats any name, and yields a real team id."""
    abbr, tid = _parent_org({"team": "Round Rock Express",
                             "parent_org_id": 140})        # TEX
    assert abbr == "TEX"
    assert tid == 140.0


def test_parent_org_id_wins_over_a_conflicting_name():
    abbr, tid = _parent_org({"team": "Round Rock Express",
                             "currentTeam": "Oklahoma C",
                             "parent_org_id": 140})
    assert (abbr, tid) == ("TEX", 140.0)


def test_parent_org_still_degrades_without_an_id():
    """No guessing: an affiliate with no parent stays unresolved."""
    abbr, tid = _parent_org({"team": "Round Rock Express"})
    assert abbr == "Round Rock Express"
    assert np.isnan(tid)


def test_parent_org_ignores_a_junk_id():
    for bad in (None, np.nan, "", "not-an-id", 99999):
        abbr, tid = _parent_org({"team": "Somewhere", "parent_org_id": bad})
        assert abbr == "Somewhere", f"{bad!r} should not resolve"
        assert np.isnan(tid), f"{bad!r} should not resolve"


def test_parent_org_honours_the_free_agent_sentinel():
    """-1 is FREE_AGENT_TEAM_ID, not junk — it must keep resolving to FA."""
    from team_context import FREE_AGENT_ABBR, FREE_AGENT_TEAM_ID
    abbr, tid = _parent_org({"team": "Somewhere",
                             "parent_org_id": FREE_AGENT_TEAM_ID})
    assert abbr == FREE_AGENT_ABBR
    assert tid == float(FREE_AGENT_TEAM_ID)


def test_every_mlb_org_id_resolves():
    """The map must cover all 30, or some org silently loses its farmhands."""
    from team_context import TEAM_ABBR_BY_ID
    for tid, abbr in TEAM_ABBR_BY_ID.items():
        got_abbr, got_id = _parent_org({"team": "x", "parent_org_id": tid})
        assert got_abbr == abbr
        assert got_id == float(tid)


# ─────────────────────────────────────────────────────────────────────────────
# feed cache schema guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("feed,expected", [
    ({"AAA": {"hitters": [{"mlbam_id": 1, "parent_org_id": 140}],
              "pitchers": []}}, True),
    ({"AAA": {"hitters": [{"mlbam_id": 1}], "pitchers": []}}, False),
    # Key present but unresolved is still a CURRENT cache: statsapi can
    # legitimately have no parent for an affiliate, and refetching won't help.
    ({"AAA": {"hitters": [{"mlbam_id": 1, "parent_org_id": None}]}}, True),
    ({"AAA": {"hitters": [], "pitchers": [{"parent_org_id": 111}]}}, True),
    ({"AAA": {"hitters": [], "pitchers": []}}, False),
    ({}, False),
    (None, False),
    ("garbage", False),
])
def test_stale_minors_cache_is_detected(feed, expected):
    """A pre-parent-org cache is valid JSON, so only this notices it."""
    from data_acquisition import _feed_has_parent_orgs
    assert _feed_has_parent_orgs(feed) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

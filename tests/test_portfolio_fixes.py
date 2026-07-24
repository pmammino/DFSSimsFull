"""Tests for the portfolio de-biasing fixes:
  * held-out sim split (portfolio_ev.sim_split / field_cut_scores)
  * out-of-sample EV reporting (select_portfolio_ev pay_report)
  * near-twin per-member exposure balancing (value_group_member_caps)
  * tie-banded ranking (select_portfolio tie_sims/tie_seed)
  * projection-uncertainty mean shrinkage (shrink_value_group_means)

These need only numpy/pandas + the two pure portfolio modules (no Streamlit).
"""
import numpy as np
import pandas as pd
import pytest

import portfolio_ev as pev
from portfolio import (select_portfolio, select_portfolio_ev,
                       value_group_member_caps, shrink_value_group_means,
                       detect_value_groups)

COLS = ['P1', 'P2', 'C', '1B', '2B', '3B', 'SS', 'OF1', 'OF2', 'OF3']
HITC = COLS[2:]


def _mk(sb_name, cid, wins):
    """A candidate row that puts `sb_name` in the 2B slot (the twin slot)."""
    names = {'P1': 'pa (AAA)', 'P2': 'pb (BBB)', 'C': 'c1 (CLE)',
             '1B': 'b1 (CLE)', '2B': f'{sb_name} (CLE)', '3B': 'b3 (CLE)',
             'SS': 's1 (CLE)', 'OF1': 'o1 (KC)', 'OF2': 'o2 (KC)',
             'OF3': 'o3 (NYY)'}
    return {'Candidate': cid, **names,
            'Wins': wins, 'Top10': wins, 'Top100': wins}


# --------------------------------------------------------------------------- #
# Held-out split
# --------------------------------------------------------------------------- #
def test_sim_split_is_a_reproducible_partition():
    parts = pev.sim_split(1000, fractions=(0.4, 0.4, 0.2), seed=7)
    idx = np.concatenate(parts)
    assert len(idx) == 1000 and len(np.unique(idx)) == 1000      # disjoint + full
    assert all(np.all(np.diff(p) > 0) for p in parts)            # ascending
    again = pev.sim_split(1000, fractions=(0.4, 0.4, 0.2), seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(parts, again))


def test_sim_split_degrades_when_too_few_sims():
    parts = pev.sim_split(4, fractions=(0.4, 0.4, 0.2), seed=1)
    assert all(np.array_equal(p, np.arange(4)) for p in parts)


def test_field_cut_scores_matches_ladder_definition():
    fmat = np.array([[10., 20., 30., 40.], [1., 2., 3., 4.]], dtype=np.float32)
    fcs = pev.field_cut_scores(fmat, np.array([1, 2, 4]))
    assert fcs[0].tolist() == [40., 30., 10.]
    assert fcs[1].tolist() == [4., 3., 1.]


# --------------------------------------------------------------------------- #
# Out-of-sample EV reporting
# --------------------------------------------------------------------------- #
def test_pay_report_makes_reported_ev_out_of_sample():
    rows = [_mk('sA', 1, 5), _mk('sB', 2, 4)]
    df = pd.DataFrame(rows)
    pay = np.array([[100., 0.], [100., 0.]], dtype=np.float32)   # col0 always cashes
    report = np.zeros_like(pay)                                  # held-out: nobody cashes
    _, info, W = select_portfolio_ev(df, 1, pay, pev.utility("Aggressive (max ceiling)"),
                                     cols=COLS, hitc=HITC, pay_report=report)
    assert info["exp_return"] == 0.0 and float(np.sum(W)) == 0.0
    # without the held-out report the same pick looks profitable (in-sample bias)
    _, info_is, _ = select_portfolio_ev(df, 1, pay, pev.utility("Aggressive (max ceiling)"),
                                        cols=COLS, hitc=HITC)
    assert info_is["exp_return"] > info["exp_return"]


def test_pay_report_shape_mismatch_raises():
    df = pd.DataFrame([_mk('sA', 1, 5), _mk('sB', 2, 4)])
    pay = np.array([[100., 0.], [100., 0.]], dtype=np.float32)
    with pytest.raises(ValueError):
        select_portfolio_ev(df, 1, pay, pev.utility("Balanced"), cols=COLS,
                            hitc=HITC, pay_report=np.zeros((2, 3), np.float32))


# --------------------------------------------------------------------------- #
# Near-twin member caps
# --------------------------------------------------------------------------- #
def test_value_group_member_caps_split_a_pair():
    caps = value_group_member_caps([{"players": ["sA", "sB"], "pos": "2B"}],
                                   slack=0.25)
    assert caps == pytest.approx({"sA": 0.625, "sB": 0.625})


def test_member_caps_balance_twin_exposure_end_to_end():
    # 30 lineups per twin; twin A ranks higher (noise edge) so rank alone takes A.
    rows = ([_mk('sA', i, 100 - i) for i in range(30)]
            + [_mk('sB', 100 + i, 60 - i) for i in range(30)])
    df = pd.DataFrame(rows)
    keys = ['Wins', 'Top10', 'Top100']
    _, i_none = select_portfolio(df, 20, keys, cols=COLS, hitc=HITC)
    assert i_none["player_expo"].get("sA", 0) == 20            # A eats it all
    assert i_none["player_expo"].get("sB", 0) == 0
    caps = value_group_member_caps([{"players": ["sA", "sB"], "pos": "2B"}],
                                   slack=0.25)
    _, i_bal = select_portfolio(df, 20, keys, cols=COLS, hitc=HITC,
                                player_caps=caps)
    a, b = i_bal["player_expo"]["sA"], i_bal["player_expo"]["sB"]
    assert a <= 13 and b >= 7, (a, b)                          # near-even split


# --------------------------------------------------------------------------- #
# Tie-banded ranking
# --------------------------------------------------------------------------- #
def test_tie_band_is_seed_stable_and_shuffles_within_noise():
    rows = [_mk('sA', i, 100) for i in range(6)] + [_mk('sB', 6 + i, 100) for i in range(6)]
    df = pd.DataFrame(rows)
    keys = ['Wins', 'Top10', 'Top100']
    c1, _ = select_portfolio(df, 4, keys, cols=COLS, hitc=HITC,
                             tie_sims=1000, tie_seed=3)
    c2, _ = select_portfolio(df, 4, keys, cols=COLS, hitc=HITC,
                             tie_sims=1000, tie_seed=3)
    assert [r['Candidate'] for r in c1] == [r['Candidate'] for r in c2]


def test_tie_band_preserves_order_for_well_separated_lineups():
    # a clearly-best lineup (many wins) must still come first despite the band
    rows = [_mk('sA', 1, 900)] + [_mk('sB', 2 + i, 5) for i in range(10)]
    df = pd.DataFrame(rows)
    chosen, _ = select_portfolio(df, 1, ['Wins', 'Top10', 'Top100'],
                                 cols=COLS, hitc=HITC, tie_sims=1000, tie_seed=0)
    assert chosen[0]['Candidate'] == 1


# --------------------------------------------------------------------------- #
# Projection-uncertainty mean shrinkage
# --------------------------------------------------------------------------- #
def test_detect_then_shrink_and_caps_use_the_groups_list():
    # detect_value_groups returns (group_of, groups); the group-consuming helpers
    # take the SECOND element (the list of group dicts). This guards the wiring
    # that previously passed the group_of dict by mistake.
    meta = {"sA": {"pos": "2B", "salary": 4500, "proj": 8.10, "team": "ATL"},
            "sB": {"pos": "2B", "salary": 4500, "proj": 8.05, "team": "ATL"},
            "solo": {"pos": "1B", "salary": 3000, "proj": 6.0, "team": "ATL"}}
    group_of, groups = detect_value_groups(meta)
    assert isinstance(groups, list) and groups and "players" in groups[0]
    caps = value_group_member_caps(groups, slack=0.25)
    assert set(caps) == {"sA", "sB"}
    score = {"sA": np.full(50, 10.0, np.float32), "sB": np.full(50, 6.0, np.float32)}
    out = shrink_value_group_means(score, groups, strength=1.0)
    assert abs(out["sA"].mean() - out["sB"].mean()) < 1e-3
    # passing the group_of dict by mistake fails loudly with a clear message
    with pytest.raises(TypeError):
        shrink_value_group_means(score, group_of, strength=1.0)
    with pytest.raises(TypeError):
        value_group_member_caps(group_of, slack=0.25)


def test_shrink_collapses_means_but_keeps_correlation():
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(400)
    score = {"sA": 10.0 + shared + rng.standard_normal(400) * 0.1,
             "sB": 6.0 + shared + rng.standard_normal(400) * 0.1}
    corr0 = np.corrcoef(score["sA"], score["sB"])[0, 1]
    out = shrink_value_group_means(score, [{"players": ["sA", "sB"]}], strength=1.0)
    assert abs(out["sA"].mean() - out["sB"].mean()) < 1e-3
    corr1 = np.corrcoef(out["sA"], out["sB"])[0, 1]
    assert abs(corr0 - corr1) < 1e-6
    assert shrink_value_group_means(score, [], strength=0.5) is not score

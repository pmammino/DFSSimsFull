"""Showdown counterparts of the portfolio de-biasing fixes:
  * out-of-sample EV reporting (select_showdown_portfolio_ev pay_report)
  * near-twin per-member exposure balancing (value_group_member_caps)
  * tie-banded ranking (select_showdown_portfolio tie_sims/tie_seed)

Held-out split and mean-shrink live in portfolio_ev / portfolio and are covered
by test_portfolio_fixes.py; here we check the showdown selector wiring.
"""
import numpy as np
import pandas as pd
import pytest

from portfolio_ev import utility, make_payout_curve
from portfolio import value_group_member_caps
from showdown_portfolio import (select_showdown_portfolio,
                                 select_showdown_portfolio_ev, SD_COLS)


def _mk(cpt, cpt_tm, utils, **ranks):
    row = {'CPT': f"{cpt} ({cpt_tm})"}
    for i, (nm, tm) in enumerate(utils, 1):
        row[f'UTIL{i}'] = f"{nm} ({tm})"
    row.update(ranks)
    return row


# --------------------------------------------------------------------------- #
# Out-of-sample EV reporting
# --------------------------------------------------------------------------- #
def test_pay_report_makes_showdown_ev_out_of_sample():
    rows = [
        _mk('A', 'AAA', [('a1', 'AAA'), ('x1', 'BBB'), ('x2', 'BBB'),
                         ('x3', 'BBB'), ('x4', 'BBB')]),
        _mk('C', 'BBB', [('c1', 'BBB'), ('z1', 'AAA'), ('z2', 'AAA'),
                         ('z3', 'AAA'), ('z4', 'AAA')]),
    ]
    df = pd.DataFrame(rows)
    pay = np.array([[100., 0.], [100., 0.]], dtype=np.float32)   # col0 always cashes
    report = np.zeros_like(pay)                                  # held-out: nobody cashes
    _, info, W = select_showdown_portfolio_ev(df, 1, pay, utility("Aggressive (max ceiling)"),
                                              pay_report=report)
    assert info["exp_return"] == 0.0 and float(np.sum(W)) == 0.0
    _, info_is, _ = select_showdown_portfolio_ev(df, 1, pay, utility("Aggressive (max ceiling)"))
    assert info_is["exp_return"] > info["exp_return"]


def test_showdown_pay_report_shape_mismatch_raises():
    df = pd.DataFrame([
        _mk('A', 'AAA', [('a1', 'AAA'), ('x1', 'BBB'), ('x2', 'BBB'),
                         ('x3', 'BBB'), ('x4', 'BBB')]),
        _mk('C', 'BBB', [('c1', 'BBB'), ('z1', 'AAA'), ('z2', 'AAA'),
                         ('z3', 'AAA'), ('z4', 'AAA')]),
    ])
    pay = np.array([[100., 0.], [100., 0.]], dtype=np.float32)
    with pytest.raises(ValueError):
        select_showdown_portfolio_ev(df, 1, pay, utility("Balanced"),
                                     pay_report=np.zeros((2, 3), np.float32))


# --------------------------------------------------------------------------- #
# Near-twin member caps (reuse the shared helper in a showdown selection)
# --------------------------------------------------------------------------- #
def test_member_caps_balance_showdown_twin_utils():
    # 20 lineups per twin UTIL; twin uA ranks higher, so rank alone takes all uA.
    rows = []
    for i in range(20):
        rows.append(_mk('Stud', 'AAA', [('uA', 'AAA'), ('f1', 'BBB'), ('f2', 'BBB'),
                                        ('f3', 'BBB'), ('f4', 'BBB')],
                        Wins=100 - i, Top10=0, Top100=0))
    for i in range(20):
        rows.append(_mk('Stud', 'AAA', [('uB', 'AAA'), ('f1', 'BBB'), ('f2', 'BBB'),
                                        ('f3', 'BBB'), ('f4', 'BBB')],
                        Wins=60 - i, Top10=0, Top100=0))
    df = pd.DataFrame(rows)
    keys = ['Wins', 'Top10', 'Top100']
    _, i_none = select_showdown_portfolio(df, 20, keys)
    assert i_none["player_expo"].get("uA", 0) == 20
    assert i_none["player_expo"].get("uB", 0) == 0
    caps = value_group_member_caps([{"players": ["uA", "uB"], "pos": "UTIL"}],
                                   slack=0.25)
    _, i_bal = select_showdown_portfolio(df, 20, keys, player_caps=caps)
    a, b = i_bal["player_expo"]["uA"], i_bal["player_expo"]["uB"]
    assert a <= 13 and b >= 7, (a, b)


# --------------------------------------------------------------------------- #
# Tie-banded ranking
# --------------------------------------------------------------------------- #
def test_showdown_tie_band_seed_stable():
    rows = [_mk('Stud', 'AAA', [('u1', 'AAA'), ('u2', 'AAA'), ('u3', 'BBB'),
                                ('u4', 'BBB'), ('u5', 'BBB')], Wins=100, Top10=0)
            for _ in range(6)]
    df = pd.DataFrame(rows)
    c1, _ = select_showdown_portfolio(df, 3, ['Wins', 'Top10'],
                                      tie_sims=1000, tie_seed=5)
    c2, _ = select_showdown_portfolio(df, 3, ['Wins', 'Top10'],
                                      tie_sims=1000, tie_seed=5)
    assert [r['CPT'] for r in c1] == [r['CPT'] for r in c2]


def test_showdown_tie_band_keeps_clear_winner_first():
    rows = [_mk('Ace', 'AAA', [('u1', 'AAA'), ('u2', 'AAA'), ('u3', 'BBB'),
                               ('u4', 'BBB'), ('u5', 'BBB')], Wins=900, Top10=0)]
    rows += [_mk('Sub', 'AAA', [('u1', 'AAA'), ('u2', 'AAA'), ('u3', 'BBB'),
                                ('u4', 'BBB'), ('u5', 'BBB')], Wins=5, Top10=0)
             for _ in range(10)]
    df = pd.DataFrame(rows)
    chosen, _ = select_showdown_portfolio(df, 1, ['Wins', 'Top10'],
                                          tie_sims=1000, tie_seed=0)
    assert chosen[0]['CPT'].startswith('Ace')

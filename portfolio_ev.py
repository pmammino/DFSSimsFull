#!/usr/bin/env python3
"""
portfolio_ev.py
===============
Payout-aware portfolio math: turn the correlated per-simulation lineup scores
into DOLLAR outcomes, so the export step can maximize the *portfolio's* expected
result instead of ranking each lineup in a vacuum.

The problem this solves: the top lineups by standalone Win% almost all win in the
SAME simulations (the ones where the same chalk stack booms), so an exported set
that looks diverse on paper is concentrated in outcome-space and booms/busts
together. Optimizing a *concave* utility of the portfolio's per-sim dollar return
rewards sets whose winning sims are spread across different slate outcomes.

Three pure pieces live here (no lineup knowledge — that's `portfolio.py`):

  * make_payout_curve  - a parametric top-heavy GPP prize table
  * utility            - the risk-posture knob (linear / sqrt / log)
  * candidate_payout_matrix - per-lineup x per-sim payouts from finishing place

Everything is plain numpy so it is trivially unit-testable (see __main__).
"""
import numpy as np


# --------------------------------------------------------------------------- #
# Parametric payout curve
# --------------------------------------------------------------------------- #
def make_payout_curve(field_size, entry_fee, *, top_heaviness=0.9,
                      pct_paid=0.20, rake=0.15, prize_pool=None,
                      min_cash_mult=1.5):
    """A realistic top-heavy GPP prize table.

    Returns an int-indexed float array `prize` of length ``field_size + 1`` where
    ``prize[p]`` is the dollars paid for finishing in place ``p`` (place 1 = win);
    ``prize[0]`` is unused and places past the paid cutoff are 0.

    Parameters
    ----------
    field_size    : number of entries in the contest.
    entry_fee     : dollars per entry (sets the prize pool and the min-cash floor).
    top_heaviness : power-law exponent for the prize decay. ~0.3 is nearly flat
                    (double-up-ish), ~0.9 is a typical GPP, ~1.5 is very top-heavy
                    (winner-take-most). Higher => more concentrated at the top.
    pct_paid      : fraction of the field that cashes (GPPs are usually ~0.20).
    rake          : operator's cut of entry fees; prize_pool defaults to
                    ``field_size * entry_fee * (1 - rake)``.
    prize_pool    : override the total prize pool directly (ignores rake).
    min_cash_mult : the min-cash prize is ``min_cash_mult * entry_fee`` (a real
                    GPP's smallest cash is a bit above the entry fee); if that
                    floor would exceed the pool it is flattened to fit.

    The prizes are guaranteed non-increasing in place and to sum to the pool.
    """
    field_size = int(field_size)
    if field_size < 1:
        raise ValueError("field_size must be >= 1")
    if prize_pool is None:
        prize_pool = field_size * float(entry_fee) * (1.0 - float(rake))
    prize_pool = float(prize_pool)

    places_paid = max(1, int(round(float(pct_paid) * field_size)))
    places_paid = min(places_paid, field_size)

    prize = np.zeros(field_size + 1, dtype=np.float64)
    if prize_pool <= 0:
        return prize

    min_cash = float(min_cash_mult) * float(entry_fee)
    reserved = min_cash * places_paid
    if reserved >= prize_pool:
        # Pool can't even fund a flat min-cash to everyone paid: pay a flat share.
        prize[1:places_paid + 1] = prize_pool / places_paid
        return prize

    p = np.arange(1, places_paid + 1, dtype=np.float64)
    w = p ** (-float(top_heaviness))          # top-heavy, strictly decreasing
    w = w / w.sum()
    extra = (prize_pool - reserved) * w        # sums to (pool - reserved)
    prize[1:places_paid + 1] = min_cash + extra
    return prize


def payout_curve_summary(prize, entry_fee):
    """Human-readable headline stats for a prize array (from make_payout_curve)."""
    paid = int((prize > 0).sum())
    total = float(prize.sum())
    top = float(prize[1]) if len(prize) > 1 else 0.0
    min_cash = float(prize[prize > 0].min()) if paid else 0.0
    return {
        "places_paid": paid,
        "prize_pool": total,
        "first_place": top,
        "min_cash": min_cash,
        "min_cash_mult": (min_cash / entry_fee) if entry_fee else 0.0,
    }


# --------------------------------------------------------------------------- #
# Risk posture: the concave-utility knob
# --------------------------------------------------------------------------- #
# Concavity is what makes decorrelation pay: two slates that each cash $X are
# worth more than one slate that cashes $2X, so the optimizer spreads the
# portfolio's winning sims across different slate outcomes.
UTILITIES = {
    # label -> (fn over winnings W >= 0, one-line description)
    "Aggressive (max ceiling)": (lambda w: w,
        "Linear utility ~ pure expected dollars. Barely diversifies; chases the "
        "single highest-EV builds (best for large-field GPP ceiling)."),
    "Balanced": (lambda w: np.sqrt(w),
        "Square-root utility. Rewards spreading winning sims across slate "
        "outcomes without giving up much ceiling."),
    "Conservative (consistent cashing)": (lambda w: np.log1p(w),
        "Log (Kelly-style) utility. Strong boom/bust aversion; prioritizes "
        "cashing across as many slate states as possible."),
}


def utility(kind):
    """Return the (vectorized) concave utility function for a risk-posture label.

    Falls back to the balanced utility for an unknown label."""
    fn, _ = UTILITIES.get(kind, UTILITIES["Balanced"])
    return fn


# --------------------------------------------------------------------------- #
# Per-lineup x per-sim payouts
# --------------------------------------------------------------------------- #
def field_place_cutpoints(n_field, fine=300, coarse=60):
    """Place cutoffs at which to sample the sorted field score per sim.

    Exact for the top `fine` places (where prizes vary fastest), then
    geometrically spaced out to `n_field` (prizes are smooth that deep, so
    bucketing barely moves the payout). Returns a sorted int array of places,
    each in ``1..n_field``."""
    n_field = int(n_field)
    fine = min(int(fine), n_field)
    cuts = set(range(1, fine + 1))
    if n_field > fine:
        geo = np.geomspace(fine + 1, n_field, num=int(coarse))
        cuts.update(int(round(x)) for x in geo)
    return np.array(sorted(c for c in cuts if 1 <= c <= n_field), dtype=np.int64)


def candidate_payout_matrix(cand_scores, field_cut_scores, cut_places, prize):
    """Dollars each candidate wins in each simulation.

    Parameters
    ----------
    cand_scores      : (n_sim, M) candidate fantasy-point totals per sim.
    field_cut_scores : (n_sim, n_cut) the field score needed to reach each place
                       in `cut_places`, per sim (place p's score = the p-th
                       highest field total in that sim).
    cut_places       : (n_cut,) the places `field_cut_scores` samples, ascending.
    prize            : payout array from make_payout_curve (index = place).

    Returns (n_sim, M) float32 payouts. A candidate wins the prize of the best
    (fewest-place) cutoff whose score it meets; below the deepest paid cutoff it
    wins 0.
    """
    cand_scores = np.asarray(cand_scores, dtype=np.float32)
    field_cut_scores = np.asarray(field_cut_scores, dtype=np.float32)
    cut_places = np.asarray(cut_places, dtype=np.int64)
    n_sim, M = cand_scores.shape

    # prize for achieving each cutpoint's place; ascending place => non-increasing $
    prize_at_cut = prize[cut_places]
    pay = np.zeros((n_sim, M), dtype=np.float32)
    for s in range(n_sim):
        # thresholds descending in score as place deepens -> reverse to ascending
        thr_asc = field_cut_scores[s][::-1]
        prize_asc = prize_at_cut[::-1]          # ascending $ (deep place -> shallow)
        # k = how many thresholds the candidate's score meets/exceeds; the best
        # (largest, i.e. last) of those qualifying thresholds carries the top $.
        k = np.searchsorted(thr_asc, cand_scores[s], side="right")
        hit = k > 0
        pay[s, hit] = prize_asc[k[hit] - 1]
    return pay


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ---- payout curve ----
    prize = make_payout_curve(10000, 20, top_heaviness=0.9, pct_paid=0.20, rake=0.15)
    s = payout_curve_summary(prize, 20)
    assert abs(prize.sum() - 10000 * 20 * 0.85) < 1.0, s
    assert s["places_paid"] == 2000, s
    assert prize[1] == prize.max() and prize[1] > prize[2] > prize[100], s
    # monotone non-increasing over paid places
    paid = prize[1:s["places_paid"] + 1]
    assert np.all(np.diff(paid) <= 1e-6), "prizes must not increase with place"
    assert s["min_cash"] >= 20 * 1.5 - 1e-6, s
    # top-heaviness concentrates the top prize
    flat = make_payout_curve(10000, 20, top_heaviness=0.3)
    steep = make_payout_curve(10000, 20, top_heaviness=1.5)
    assert steep[1] > flat[1], (steep[1], flat[1])

    # ---- utility ----
    assert utility("Aggressive (max ceiling)")(4.0) == 4.0
    assert abs(utility("Balanced")(4.0) - 2.0) < 1e-9
    assert abs(utility("Conservative (consistent cashing)")(np.e - 1) - 1.0) < 1e-9

    # ---- payout matrix: 2 sims, tiny field, hand-checkable ----
    # cut places 1..3, field thresholds so we know exact places
    cut_places = np.array([1, 2, 3])
    # sim0 thresholds: place1 needs >=100, place2 >=90, place3 >=80
    # sim1 thresholds: place1 needs >=50, place2 >=40, place3 >=30
    field_cut = np.array([[100, 90, 80],
                          [50, 40, 30]], dtype=np.float32)
    pr = np.array([0, 1000, 100, 10])   # prize[1]=1000, [2]=100, [3]=10
    cand = np.array([[95, 79],      # sim0: c0 makes place2 ($100), c1 misses ($0)
                     [55, 35]], dtype=np.float32)  # sim1: c0 place1 ($1000), c1 place3 ($10)
    pay = candidate_payout_matrix(cand, field_cut, cut_places, pr)
    assert pay[0, 0] == 100 and pay[0, 1] == 0, pay
    assert pay[1, 0] == 1000 and pay[1, 1] == 10, pay

    # ---- cutpoints ----
    cp = field_place_cutpoints(20000)
    assert cp[0] == 1 and cp[-1] == 20000 and np.all(np.diff(cp) > 0)
    assert cp[299] == 300, cp[295:305]

    print("portfolio_ev.py self-test passed:", s)

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
# Held-out simulation split (de-biases selection / reported EV)
# --------------------------------------------------------------------------- #
def sim_split(n_sim, fractions=(0.4, 0.4, 0.2), seed=1234):
    """Partition the simulation axis into disjoint index sets.

    The draws are i.i.d. along the sim axis, so a plain index split is a valid
    train/eval partition. Ranking, selecting, and REPORTING a portfolio on one
    shared sim set inflates the reported EV (you grade the set on the same sims
    you optimized against — the winner's curse). Splitting lets the caller rank
    on one slice, select on a second, and report on a third, so the headline EV
    is genuinely out-of-sample.

    Returns a tuple of ``len(fractions)`` ascending int index arrays. If there
    are too few sims to give every part at least one index, every part is the
    full range (splitting would be worse than not splitting)."""
    n_sim = int(n_sim)
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    fr = np.asarray(fractions, dtype=np.float64)
    if len(fr) < 1 or (fr <= 0).any() or not np.isclose(fr.sum(), 1.0):
        raise ValueError("fractions must be positive and sum to 1")
    k = len(fr)
    if n_sim < 2 * k:
        full = np.arange(n_sim)
        return tuple(full.copy() for _ in range(k))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_sim)
    cuts = np.floor(np.cumsum(fr)[:-1] * n_sim).astype(int)
    parts = np.split(perm, cuts)
    return tuple(np.sort(p) for p in parts)


def field_cut_scores(field_mat, cut_places):
    """The field score needed to reach each place in `cut_places`, per sim.

    ``(n_sim, len(cut_places))`` where entry ``[s, i]`` is the ``cut_places[i]``-th
    highest field total in sim ``s`` — the same placement ladder
    ``run_contest_dist`` captures, but as a standalone pass so a payout matrix can
    be built on any sim slice. `field_mat` is ``(n_sim, n_field)``."""
    field_mat = np.asarray(field_mat, dtype=np.float32)
    cut_places = np.asarray(cut_places, dtype=np.int64)
    n_sim, n_field = field_mat.shape
    take = n_field - cut_places          # index into each ascending-sorted row
    out = np.empty((n_sim, len(cut_places)), dtype=np.float32)
    for s in range(n_sim):
        out[s] = np.sort(field_mat[s])[take]
    return out


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


def portfolio_payout(cand_scores, field_cut_scores, cut_places, prize):
    """Per-sim TOTAL winnings for a set of lineups entered into ONE contest.

    ``candidate_payout_matrix`` scores every lineup as if it were your only entry,
    so two lineups that both clear 1st place in the same simulation each collect
    the 1st-place prize. In a real contest only one entry finishes 1st — the next
    takes 2nd, and so on — so summing the independent columns double-counts the
    prize exactly when your lineups are correlated (they boom in the SAME sims,
    which is precisely the case top-Win% builds fall into). That inflated sum is
    what makes a correlated portfolio's reported expected return / ROI overshoot.

    This charges each lineup its true finishing place among BOTH the field and
    your other lineups: a lineup's place is its field place (from the sampled cut
    thresholds, exactly as ``candidate_payout_matrix`` reads them) plus the number
    of your own lineups that outscore it in that sim. Distinct own-lineup ranks
    break score ties so two entries never share a place, so a shared boom is paid
    once (1st + 2nd + …) instead of N times.

    Parameters
    ----------
    cand_scores      : (n_sim, k) fantasy-point totals for the k CHOSEN lineups.
    field_cut_scores : (n_sim, n_cut) field score needed to reach each cut place.
    cut_places       : (n_cut,) the places ``field_cut_scores`` samples, ascending.
    prize            : payout array from ``make_payout_curve`` (index = place).

    Returns (n_sim,) float64 total portfolio winnings per simulation.
    """
    cand_scores = np.asarray(cand_scores, dtype=np.float32)
    field_cut_scores = np.asarray(field_cut_scores, dtype=np.float32)
    cut_places = np.asarray(cut_places, dtype=np.int64)
    prize = np.asarray(prize, dtype=np.float64)
    n_sim, k = cand_scores.shape
    places_desc = cut_places[::-1]              # deepest (largest) place first
    max_place = len(prize) - 1
    W = np.zeros(n_sim, dtype=np.float64)
    if k == 0:
        return W
    for s in range(n_sim):
        c = cand_scores[s]
        thr_asc = field_cut_scores[s][::-1]     # field thresholds ascending in $
        # best field place each lineup qualifies for (kk == 0 => below the deepest
        # paid cutoff, i.e. out of the money regardless of own placement).
        kk = np.searchsorted(thr_asc, c, side="right")
        qualified = kk > 0
        field_place = places_desc[np.clip(kk - 1, 0, len(places_desc) - 1)]
        # your own lineups strictly ahead of each; distinct ranks (stable argsort)
        # keep two entries from claiming the same place on a score tie.
        rank = np.empty(k, dtype=np.int64)
        rank[np.argsort(-c, kind="stable")] = np.arange(k)
        place = np.clip(field_place + rank, 0, max_place)
        W[s] = np.where(qualified, prize[place], 0.0).sum()
    return W


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

    # ---- collision-aware portfolio payout ----
    # sim0: both candidates clear 1st (>=100); summing independent payouts would
    # pay 1000+1000, but only one can win 1st -> 1000 (1st) + 100 (2nd).
    Wp = portfolio_payout(np.array([[105, 101]], dtype=np.float32),
                          field_cut[:1], cut_places, pr)
    assert Wp[0] == 1000 + 100, Wp
    # independent sum over the same scores double-counts the top prize
    indep = candidate_payout_matrix(np.array([[105, 101]], dtype=np.float32),
                                    field_cut[:1], cut_places, pr)
    assert indep.sum() == 2000 and Wp[0] < indep.sum(), (indep, Wp)
    # the best lineup's own payout is unchanged (nothing of yours is above it):
    # here c0 makes 2nd ($100) alone; add a lower c1 that misses -> still $100.
    Wp2 = portfolio_payout(np.array([[95, 50]], dtype=np.float32),
                           field_cut[:1], cut_places, pr)
    assert Wp2[0] == 100, Wp2

    # ---- cutpoints ----
    cp = field_place_cutpoints(20000)
    assert cp[0] == 1 and cp[-1] == 20000 and np.all(np.diff(cp) > 0)
    assert cp[299] == 300, cp[295:305]

    # ---- sim split: disjoint, exhaustive, reproducible ----
    parts = sim_split(1000, fractions=(0.4, 0.4, 0.2), seed=7)
    assert len(parts) == 3
    allidx = np.concatenate(parts)
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000   # a partition
    assert all(np.all(np.diff(p) > 0) for p in parts)               # each ascending
    assert abs(len(parts[0]) - 400) <= 1 and abs(len(parts[2]) - 200) <= 1
    parts2 = sim_split(1000, fractions=(0.4, 0.4, 0.2), seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(parts, parts2))  # seed-stable
    # too few sims -> everyone shares the full set instead of starving a part
    tiny = sim_split(4, fractions=(0.4, 0.4, 0.2), seed=7)
    assert all(np.array_equal(t, np.arange(4)) for t in tiny), tiny

    # ---- field_cut_scores matches run_contest_dist's ladder definition ----
    fmat = np.array([[10., 20., 30., 40.], [1., 2., 3., 4.]], dtype=np.float32)
    cplaces = np.array([1, 2, 4])   # 1st, 2nd, 4th highest
    fcs = field_cut_scores(fmat, cplaces)
    assert fcs[0].tolist() == [40., 30., 10.], fcs
    assert fcs[1].tolist() == [4., 3., 1.], fcs

    print("portfolio_ev.py self-test passed:", s)

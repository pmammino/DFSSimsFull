"""
mle_translations.py
===================
Major-League Equivalency (MLE) translations for players with no MLB history.

The rate/BIP projection models only produce a projection for players who have
prior MLB data (see rate_models.build_inference_panel — the `len(prior) == 0`
gate drops everyone else). A rookie making his debut therefore gets no baseline
at all. This module closes that gap: it takes a player's most-recent
minor-league line, translates it into a Major-League-Equivalent (MLE) season,
and emits a **synthetic "prior season" row** in exactly the schema the statsapi
history uses. Injected into the history frame, that row flows through the same
shrinkage / recency-decay machinery as everyone else, so a debut rookie comes
out with a heavily-regressed but non-empty, skill-differentiated baseline.

Two translation surfaces are produced per player:

  1. RATE row  — K%, BB%, HBP%, SF% (+ the underlying counts, PA/TBF, SB/CS,
     R/RBI, team) so the row is a drop-in for hit_df / pit_df. The credibility
     discount (MLE_PA_CREDIBILITY) deflates the PA/TBF the row carries so the
     shrinkage pulls these players appropriately hard toward league mean.

  2. BIP profile — a per-batted-ball {out, single, double, triple, home_run}
     distribution translated from the box-score power line. Applied downstream
     (apply_mle_bip_override) so a slugging prospect keeps his power signal
     instead of collapsing to the league-average BIP fallback.

Linkage: the RotoWire minors feed keys players by a RotoWire id, not MLBAM.
We resolve to MLBAM via the Chadwick name lookup. Any player we cannot resolve
unambiguously is skipped (logged), and any player already present in the MLB
history is skipped too — we only ever ADD no-MLB-history players.

The feed shape (one dict per player) is documented in
minors_inputs/sample_minors_2026.json. Hitter and pitcher records have
different keys; see _parse_hitter / _parse_pitcher.
"""

from __future__ import annotations

import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_config import (
    DEFAULT_LEAGUE_RATES,
    MLE_HITTER_FACTORS, MLE_PITCHER_FACTORS,
    MLE_PA_CREDIBILITY, MLE_DEFAULT_AGE,
)

# League split of NON-home-run hits into 1B / 2B / 3B, used when a feed gives
# only total hits + HR (pitcher records). Roughly the modern MLB average.
_NONHR_HIT_SPLIT = {"1B": 0.760, "2B": 0.218, "3B": 0.022}

# Physical / sanity clips on translated per-PA rates.
_KPCT_CLIP  = (0.05, 0.55)
_BBPCT_CLIP = (0.01, 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_name(n: str) -> str:
    """Lightweight accent/punctuation-insensitive name key (mirrors slate_ingest)."""
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = n.lower().replace(".", "").replace(",", "").replace("'", "")
    for s in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(s):
            n = n[:-len(s)]
    return " ".join(n.split())


def _f(x, default: float = 0.0) -> float:
    """Parse a feed value (often a string like '.307' or '108.2') to float."""
    if x is None:
        return default
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return default


def _ip_to_float(s) -> float:
    """MLB 'X.Y' innings notation (Y = outs) → real innings. '108.2' → 108.667."""
    txt = str(s).strip()
    if not txt:
        return 0.0
    try:
        if "." in txt:
            whole, frac = txt.split(".", 1)
            outs = int(frac[:1]) if frac else 0
            return float(whole) + outs / 3.0
        return float(txt)
    except ValueError:
        return 0.0


def _clip(v: float, lo_hi: tuple[float, float]) -> float:
    return float(min(max(v, lo_hi[0]), lo_hi[1]))


# ─────────────────────────────────────────────────────────────────────────────
# MLBAM linkage via the Chadwick lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_name_to_mlbam(chadwick: pd.DataFrame) -> dict[str, list[int]]:
    """{norm(name): [mlbam_id, ...]} from a Chadwick lookup frame.

    Chadwick columns: key_mlbam, name_first, name_last (see data_acquisition
    .fetch_chadwick_lookup). Names with more than one MLBAM id are kept as a
    list so the caller can detect (and skip) ambiguous matches.
    """
    idx: dict[str, list[int]] = {}
    if chadwick is None or chadwick.empty:
        return idx
    for _, r in chadwick.iterrows():
        mlbam = r.get("key_mlbam")
        if pd.isna(mlbam):
            continue
        full = f"{r.get('name_first', '')} {r.get('name_last', '')}"
        key = _norm_name(full)
        if not key:
            continue
        idx.setdefault(key, [])
        mid = int(mlbam)
        if mid not in idx[key]:
            idx[key].append(mid)
    return idx


def _resolve_mlbam(rec: dict, name_idx: dict[str, list[int]]) -> Optional[int]:
    """Resolve one feed record to a single MLBAM id, or None if unresolved."""
    name = rec.get("player") or f"{rec.get('firstname', '')} {rec.get('lastname', '')}"
    cands = name_idx.get(_norm_name(name), [])
    if len(cands) == 1:
        return cands[0]
    return None  # 0 (unmatched) or >1 (ambiguous) → skip, logged by caller


# ─────────────────────────────────────────────────────────────────────────────
# Per-record parsing → observed box-score line
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hitter(rec: dict) -> dict:
    """Feed hitter record → observed counting line with a PA estimate.

    The RotoWire hitter feed omits PA, HBP, SF and SH. We estimate PA from
    AB + BB inflated for the missing non-AB events (league HBP+SF+SH ≈ 2.5% of
    PA), and back out league-rate HBP / SF counts from that PA.
    """
    ab  = _f(rec.get("ab"))
    bb  = _f(rec.get("walks"))
    k   = _f(rec.get("strikes"))
    h   = _f(rec.get("hits"))
    dbl = _f(rec.get("doubles"))
    trp = _f(rec.get("triples"))
    hr  = _f(rec.get("hr"))
    sb  = _f(rec.get("steals"))
    cs  = _f(rec.get("caught"))
    r   = _f(rec.get("runs"))
    rbi = _f(rec.get("rbi"))
    g   = _f(rec.get("games"))

    hbp_rate = DEFAULT_LEAGUE_RATES["HBP%"]
    sf_rate  = DEFAULT_LEAGUE_RATES["SF%"]
    other_rate = hbp_rate + sf_rate + 0.003   # + a small sac-bunt allowance
    pa = (ab + bb) / max(1e-9, (1.0 - other_rate)) if (ab + bb) > 0 else 0.0
    hbp = hbp_rate * pa
    sf  = sf_rate * pa
    singles = max(0.0, h - dbl - trp - hr)

    return {
        "role": "hitter", "PA": pa, "AB": ab, "BB": bb, "K": k, "H": h,
        "1B": singles, "2B": dbl, "3B": trp, "HR": hr, "HBP": hbp, "SF": sf,
        "SB": sb, "CS": cs, "R": r, "RBI": rbi, "G": g,
        "currentTeam": rec.get("currentTeam"),
    }


def _parse_pitcher(rec: dict) -> dict:
    """Feed pitcher record → observed line allowed, with a TBF estimate.

    kpct / bbpct are per-TBF, so TBF is recoverable as K / (kpct/100) (falling
    back to BB / (bbpct/100)). The feed gives total H and HR but not the 2B/3B
    split, so non-HR hits are split by league ratios downstream.
    """
    k   = _f(rec.get("k"))
    bb  = _f(rec.get("bb"))
    h   = _f(rec.get("h"))
    hr  = _f(rec.get("hr"))
    er  = _f(rec.get("er"))
    ip  = _ip_to_float(rec.get("ip"))
    kpct = _f(rec.get("kpct")) / 100.0
    bbpct = _f(rec.get("bbpct")) / 100.0
    gs  = _f(rec.get("gs"))
    g   = _f(rec.get("games"))

    tbf = 0.0
    if kpct > 0:
        tbf = k / kpct
    elif bbpct > 0:
        tbf = bb / bbpct
    elif ip > 0:
        tbf = ip * 4.3   # ~league batters-faced per inning fallback

    return {
        "role": "pitcher", "TBF": tbf, "K": k, "BB": bb, "H": h, "HR": hr,
        "ER": er, "IP": ip, "G": g, "GS": gs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Translation: observed line → MLB-equivalent rates + BIP profile
# ─────────────────────────────────────────────────────────────────────────────

def _bip_profile_from_rates(p_k, p_bb, p_hbp, p_sf, hr_pa, dbl_pa, trp_pa,
                            sng_pa) -> dict:
    """Turn MLB-equivalent per-PA event rates into a per-BATTED-BALL distribution
    {out, single, double, triple, home_run} that sums to 1.0.

    The five outcomes partition the PA mass left after K/BB/HBP/SF. `out` is the
    residual. A small floor keeps the distribution non-degenerate for extreme
    profiles.
    """
    bip_mass = max(1e-6, 1.0 - p_k - p_bb - p_hbp - p_sf)
    hr = hr_pa / bip_mass
    sg = sng_pa / bip_mass
    db = dbl_pa / bip_mass
    tp = trp_pa / bip_mass
    out = 1.0 - (hr + sg + db + tp)
    out = max(0.05, out)               # floor: nobody is a pure hitting machine
    vec = np.array([out, sg, db, tp, hr], dtype=float)
    vec = np.clip(vec, 0.0, None)
    vec = vec / vec.sum()
    return {"out": vec[0], "single": vec[1], "double": vec[2],
            "triple": vec[3], "home_run": vec[4]}


def translate_hitter(obs: dict, level: str) -> dict:
    """Observed hitter line → MLB-equivalent rates + per-BIP profile."""
    fac = MLE_HITTER_FACTORS[level]
    pa = obs["PA"]
    if pa <= 0:
        return {}
    p_k   = _clip((obs["K"]  / pa) * fac["K%"],  _KPCT_CLIP)
    p_bb  = _clip((obs["BB"] / pa) * fac["BB%"], _BBPCT_CLIP)
    p_hbp = DEFAULT_LEAGUE_RATES["HBP%"]
    p_sf  = DEFAULT_LEAGUE_RATES["SF%"]
    hr_pa  = (obs["HR"] / pa) * fac["HR"]
    dbl_pa = (obs["2B"] / pa) * fac["2B"]
    trp_pa = (obs["3B"] / pa) * fac["3B"]
    sng_pa = (obs["1B"] / pa) * fac["BABIP"]
    bip = _bip_profile_from_rates(p_k, p_bb, p_hbp, p_sf, hr_pa, dbl_pa, trp_pa, sng_pa)
    return {
        "K%": p_k, "BB%": p_bb, "HBP%": p_hbp, "SF%": p_sf,
        "SB_rate": (obs["SB"] / pa) * fac["SB"],
        "CS_rate": (obs["CS"] / pa) * fac["SB"],
        "bip": bip,
    }


def translate_pitcher(obs: dict, level: str) -> dict:
    """Observed pitcher line allowed → MLB-equivalent rates + per-BIP profile."""
    fac = MLE_PITCHER_FACTORS[level]
    tbf = obs["TBF"]
    if tbf <= 0:
        return {}
    p_k   = _clip((obs["K"]  / tbf) * fac["K%"],  _KPCT_CLIP)
    p_bb  = _clip((obs["BB"] / tbf) * fac["BB%"], _BBPCT_CLIP)
    p_hbp = DEFAULT_LEAGUE_RATES["HBP%"]
    p_sf  = DEFAULT_LEAGUE_RATES["SF%"]
    hr_pa = (obs["HR"] / tbf) * fac["HR"]
    nonhr_hits = max(0.0, obs["H"] - obs["HR"])
    nonhr_pa = (nonhr_hits / tbf) * fac["BABIP"]
    sng_pa = nonhr_pa * _NONHR_HIT_SPLIT["1B"]
    dbl_pa = nonhr_pa * _NONHR_HIT_SPLIT["2B"]
    trp_pa = nonhr_pa * _NONHR_HIT_SPLIT["3B"]
    bip = _bip_profile_from_rates(p_k, p_bb, p_hbp, p_sf, hr_pa, dbl_pa, trp_pa, sng_pa)
    return {"K%": p_k, "BB%": p_bb, "HBP%": p_hbp, "SF%": p_sf, "bip": bip}


# ─────────────────────────────────────────────────────────────────────────────
# Build synthetic history rows + BIP profiles for a whole feed
# ─────────────────────────────────────────────────────────────────────────────

def _age_from_chadwick(chadwick: pd.DataFrame, mlbam: int, season: int) -> int:
    if chadwick is None or chadwick.empty or "birth_year" not in chadwick.columns:
        return MLE_DEFAULT_AGE
    row = chadwick[chadwick["key_mlbam"] == mlbam]
    if row.empty or pd.isna(row.iloc[0]["birth_year"]):
        return MLE_DEFAULT_AGE
    try:
        return int(season - int(row.iloc[0]["birth_year"]))
    except (TypeError, ValueError):
        return MLE_DEFAULT_AGE


def build_synthetic_rows(feed: dict, role: str, target_year: int,
                         name_idx: dict[str, list[int]],
                         chadwick: pd.DataFrame,
                         existing_ids: set[int],
                         season_offset: int = 1,
                         levels: tuple[str, ...] = ("AAA", "AA"),
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Translate every record for one role into synthetic history rows.

    Parameters
    ----------
    feed : {level: {"hitters": [...], "pitchers": [...]}}
    role : 'hitter' or 'pitcher'
    existing_ids : MLBAM ids already present in the MLB history (skip these —
        we only add players with NO MLB history).
    levels : which levels to use, in PRIORITY order. A player appearing at more
        than one level is taken from the highest-priority level only.

    Returns (rows_df, bip_df, stats) where
      rows_df : synthetic season rows matching the statsapi schema
      bip_df  : PlayerId + {out, single, double, triple, home_run} per-BIP dist
      stats   : {'matched','skipped_unresolved','skipped_existing','used'}
    """
    season = target_year - season_offset
    key = "hitters" if role == "hitter" else "pitchers"
    parse = _parse_hitter if role == "hitter" else _parse_pitcher
    translate = translate_hitter if role == "hitter" else translate_pitcher

    seen_ids: set[int] = set()
    rows, bip_rows = [], []
    stats = {"matched": 0, "skipped_unresolved": 0,
             "skipped_existing": 0, "used": 0}

    for level in levels:
        for rec in (feed.get(level, {}) or {}).get(key, []) or []:
            mlbam = _resolve_mlbam(rec, name_idx)
            if mlbam is None:
                stats["skipped_unresolved"] += 1
                continue
            stats["matched"] += 1
            if mlbam in existing_ids:
                stats["skipped_existing"] += 1
                continue
            if mlbam in seen_ids:
                continue  # already taken from a higher-priority level
            obs = parse(rec)
            tr = translate(obs, level)
            if not tr:
                continue
            seen_ids.add(mlbam)
            stats["used"] += 1

            cred = MLE_PA_CREDIBILITY.get(level, 0.4)
            name = rec.get("player") or f"{rec.get('firstname','')} {rec.get('lastname','')}".strip()
            age = _age_from_chadwick(chadwick, mlbam, season)

            if role == "hitter":
                pa_eff = max(1.0, obs["PA"] * cred)
                row = _hitter_row(mlbam, name, rec, season, age, pa_eff, tr)
            else:
                pa_eff = max(1.0, obs["TBF"] * cred)
                row = _pitcher_row(mlbam, name, rec, season, age, pa_eff, tr, obs)
            rows.append(row)
            b = {"PlayerId": mlbam}
            b.update(tr["bip"])
            bip_rows.append(b)

    rows_df = pd.DataFrame(rows)
    bip_df = pd.DataFrame(bip_rows)
    return rows_df, bip_df, stats


def _hitter_row(mlbam, name, rec, season, age, pa_eff, tr) -> dict:
    """Assemble a synthetic hitter history row (statsapi schema + rates)."""
    k   = tr["K%"]  * pa_eff
    bb  = tr["BB%"] * pa_eff
    hbp = tr["HBP%"] * pa_eff
    sf  = tr["SF%"] * pa_eff
    bip = tr["bip"]
    bip_mass = max(0.0, 1.0 - tr["K%"] - tr["BB%"] - tr["HBP%"] - tr["SF%"]) * pa_eff
    hr  = bip["home_run"] * bip_mass
    sg  = bip["single"]   * bip_mass
    db  = bip["double"]   * bip_mass
    tp  = bip["triple"]   * bip_mass
    h   = sg + db + tp + hr
    ab  = pa_eff - bb - hbp - sf
    return {
        "Season": season, "PlayerId": mlbam, "Name": name,
        "Team": rec.get("currentTeam") or rec.get("team"), "TeamId": np.nan,
        "Age": age, "PA": pa_eff, "AB": ab,
        "K": k, "BB": bb, "IBB": 0.0, "HBP": hbp, "SF": sf, "SH": 0.0,
        "H": h, "HR": hr, "2B": db, "3B": tp,
        "SB": tr["SB_rate"] * pa_eff, "CS": tr["CS_rate"] * pa_eff,
        "R": 0.0, "RBI": 0.0, "G": _f(rec.get("games")),
        "TBF": 0.0, "WP": 0.0, "BK": 0.0, "ER": 0.0, "RA": 0.0, "IP": 0.0,
        "K%": tr["K%"], "BB%": tr["BB%"], "HBP%": tr["HBP%"], "SF%": tr["SF%"],
        "mle_source": "MiLB",
    }


def _pitcher_row(mlbam, name, rec, season, age, tbf_eff, tr, obs) -> dict:
    """Assemble a synthetic pitcher history row (statsapi schema + rates)."""
    k   = tr["K%"]  * tbf_eff
    bb  = tr["BB%"] * tbf_eff
    hbp = tr["HBP%"] * tbf_eff
    sf  = tr["SF%"] * tbf_eff
    bip = tr["bip"]
    bip_mass = max(0.0, 1.0 - tr["K%"] - tr["BB%"] - tr["HBP%"] - tr["SF%"]) * tbf_eff
    hr  = bip["home_run"] * bip_mass
    h   = (bip["single"] + bip["double"] + bip["triple"]) * bip_mass + hr
    return {
        "Season": season, "PlayerId": mlbam, "Name": name,
        "Team": rec.get("team"), "TeamId": np.nan, "Age": age,
        "PA": 0.0, "AB": 0.0,
        "K": k, "BB": bb, "IBB": 0.0, "HBP": hbp, "SF": sf, "SH": 0.0,
        "H": h, "HR": hr, "2B": bip["double"] * bip_mass,
        "3B": bip["triple"] * bip_mass,
        "SB": 0.0, "CS": 0.0, "R": 0.0, "RBI": 0.0, "G": _f(rec.get("games")),
        "TBF": tbf_eff, "WP": 0.0, "BK": 0.0,
        "ER": obs["ER"], "RA": obs["ER"], "IP": obs["IP"],
        "K%": tr["K%"], "BB%": tr["BB%"], "HBP%": tr["HBP%"], "SF%": tr["SF%"],
        "mle_source": "MiLB",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Downstream BIP override
# ─────────────────────────────────────────────────────────────────────────────

def apply_mle_bip_override(final_df: pd.DataFrame, bip_df: pd.DataFrame,
                           id_col: str = "PlayerId") -> pd.DataFrame:
    """Replace the (league-average fallback) BIP portion of MLE players' 9-event
    PA distribution with their MLE-translated per-BIP profile.

    The rate side (P_K, P_BB, P_HBP, P_SF) is kept as-is — it already came from
    the shrunk projection of the synthetic row. Only the batted-ball mass
    (1 - those four) is re-apportioned across {BIPOut, 1B, 2B, 3B, HR} using the
    MLE profile, so the 9 event probabilities still sum to 1.0.
    """
    if bip_df is None or bip_df.empty or final_df is None or final_df.empty:
        return final_df
    prof = bip_df.set_index("PlayerId")
    df = final_df.copy()
    need = ["P_K", "P_BB", "P_HBP", "P_SF"]
    if not all(c in df.columns for c in need):
        return df
    for i, row in df.iterrows():
        pid = row.get(id_col)
        if pid not in prof.index:
            continue
        p = prof.loc[pid]
        bip_mass = max(0.0, 1.0 - row["P_K"] - row["P_BB"]
                       - row["P_HBP"] - row["P_SF"])
        df.at[i, "P_BIPOut"] = bip_mass * p["out"]
        df.at[i, "P_1B"]     = bip_mass * p["single"]
        df.at[i, "P_2B"]     = bip_mass * p["double"]
        df.at[i, "P_3B"]     = bip_mass * p["triple"]
        df.at[i, "P_HR"]     = bip_mass * p["home_run"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke test against the sample fixture
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path

    sample = Path(__file__).parent / "minors_inputs" / "sample_minors_2026.json"
    feed = json.loads(sample.read_text())

    # Fake a Chadwick lookup that resolves both sample players.
    chad = pd.DataFrame([
        {"key_mlbam": 600001, "name_first": "Ryan", "name_last": "Fitzgerald",
         "birth_year": 1994},
        {"key_mlbam": 500001, "name_first": "Casey", "name_last": "Lawrence",
         "birth_year": 1987},
    ])
    name_idx = build_name_to_mlbam(chad)

    print("── Hitter translation ──")
    hrows, hbip, hstats = build_synthetic_rows(
        feed, "hitter", 2027, name_idx, chad, existing_ids=set())
    print("stats:", hstats)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(hrows[["Name", "Season", "PlayerId", "Age", "PA",
                     "K%", "BB%", "HBP%", "SF%", "HR", "SB"]].to_string(index=False))
        print("per-BIP:", hbip.to_dict("records"))

    print("\n── Pitcher translation ──")
    prows, pbip, pstats = build_synthetic_rows(
        feed, "pitcher", 2027, name_idx, chad, existing_ids=set())
    print("stats:", pstats)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(prows[["Name", "Season", "PlayerId", "Age", "TBF",
                     "K%", "BB%", "HBP%", "SF%", "HR", "ER", "IP"]].to_string(index=False))
        print("per-BIP:", pbip.to_dict("records"))

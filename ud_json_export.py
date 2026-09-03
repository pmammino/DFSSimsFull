"""ud_json_export.py — export the Underdog-scored sims to the JSON "bank" the
Battle Royale browser extension reads out of R2 via its Cloudflare Worker.

One file per day, always named ``slate_UD.json`` (see shared_store.UD_JSON_KEY
for where it's uploaded) and always rebuilt fresh from that day's sims — no
history is kept for it, unlike the dated `.npy` archive.

ALIGNED BANKS (team logic)
--------------------------
Each player's bank is a subset of his sim array, but the SAME sim indices are
used for every player (hitters and pitchers alike). Because sim_proj.simulate()
draws every player's array from shared per-game latents (shared['away']/
shared['home'], the opposing pitcher's m_opp, etc. — see sim_proj.py), sim
index j is the SAME simulated slate across every player it was built from:
teammates who boom together in world j still boom together in the bank, and
the opposing pitcher who gets shelled in world j is still shelled in that same
bank slot. Sorting each player's bank independently (as an earlier version of
this module did) would destroy that alignment — every player's bank would be
independently ordered from worst to best, so no bank slot would correspond to
one shared simulated slate anymore, and the extension couldn't use the bank to
find "games where my stack peaks" or any other cross-player correlation.
"""
import json
import re
import unicodedata

import numpy as np

BANK_SIZE = 1000
DEFAULT_SAMPLE = 75
ALIGN_SEED = 17


def nkey(name):
    """Normalize a name for draft-board matching: lowercase, strip accents,
    turn '.'/"'"/'-' into spaces, drop Jr/Sr/II/III/IV/V suffixes, collapse
    whitespace. E.g. "José Ramírez" and "Jose Ramirez" both -> "jose ramirez"."""
    s = unicodedata.normalize('NFKD', name or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()
    s = re.sub(r"[.'\-]", ' ', s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", '', s)
    return re.sub(r"\s+", ' ', s).strip()


def _aligned_indices(n, bank_size=BANK_SIZE, seed=ALIGN_SEED):
    """bank_size sim-world indices into an array of length n, shared by every
    player so their banks stay cross-player correlated (see module docstring).
    Sorted only for a stable/reproducible ordering — the values behind each
    index are NOT sorted, since that's what would break the alignment."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, min(bank_size, n), replace=(n < bank_size)))


def _bank_values(sim_array, idx):
    arr = np.asarray(sim_array, dtype=float)[idx]
    return [int(x) if float(x) == int(x) else round(float(x), 2) for x in arr]


def build_ud_json(hitter_ud, pitcher_ud, bank_size=BANK_SIZE,
                  default_sample=DEFAULT_SAMPLE, sport="MLB", fmt="Battle Royale",
                  pos_map=None):
    """Assemble the {players, bank_size, default_sample, aligned, meta} payload
    from the {name: sim_array} dicts sim_proj.simulate() already produces for
    Underdog scoring — hitters and pitchers merged into one flat player list,
    every player's bank drawn from the SAME sim-world indices (see module
    docstring). `pos_map` (optional {nkey: "P"/"IF"/"OF"}) bakes in positions;
    left out, `pos` is null and the extension fills it in from the draft board
    (the preferred path — sims are keyed by name only)."""
    pos_map = pos_map or {}
    sims = list(hitter_ud.items()) + list(pitcher_ud.items())
    if not sims:
        idx = np.array([], dtype=int)
    else:
        n = len(np.asarray(sims[0][1]))
        idx = _aligned_indices(n, bank_size)

    players = []
    for name, arr in sims:
        key = nkey(name)
        players.append({
            "name": name,
            "pos": pos_map.get(key),
            "bank": _bank_values(arr, idx),          # ALIGNED across players
            "mean": round(float(np.mean(arr)), 2),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "nkey": key,
        })

    return {
        "players": players,
        "bank_size": bank_size,
        "default_sample": default_sample,
        "aligned": True,           # banks share sim-world indices — see module docstring
        "meta": {"sport": sport, "format": fmt, "n_players": len(players)},
    }


def write_ud_json(hitter_ud, pitcher_ud, path, **kw):
    payload = build_ud_json(hitter_ud, pitcher_ud, **kw)
    with open(path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    return path


def _load_npy(path):
    d = np.load(path, allow_pickle=True).item()
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}


def _load_positions_csv(path):
    import csv
    out = {}
    for row in csv.DictReader(open(path)):
        out[nkey(row['name'])] = row['pos'].strip().upper()
    return out


def _main():
    """Standalone CLI to (re)generate slate_UD.json from already-saved .npy
    files, e.g. deliverables/hitter_ud_sims.npy / pitcher_ud_sims.npy, without
    rerunning the sim. Mirrors run_slate.py's in-pipeline call."""
    import argparse
    ap = argparse.ArgumentParser(description=_main.__doc__)
    ap.add_argument('--hitters', help='hitter_ud_sims.npy path')
    ap.add_argument('--pitchers', help='pitcher_ud_sims.npy path')
    ap.add_argument('--positions', help='optional CSV name,pos to tag positions')
    ap.add_argument('--bank', type=int, default=BANK_SIZE)
    ap.add_argument('--default-sample', type=int, default=DEFAULT_SAMPLE)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    hitter_ud = _load_npy(a.hitters) if a.hitters else {}
    pitcher_ud = _load_npy(a.pitchers) if a.pitchers else {}
    if not hitter_ud and not pitcher_ud:
        raise SystemExit('no sims loaded — pass --hitters and/or --pitchers')
    pos_map = _load_positions_csv(a.positions) if a.positions else None

    write_ud_json(hitter_ud, pitcher_ud, a.out, bank_size=a.bank,
                 default_sample=a.default_sample, pos_map=pos_map)
    n = len(hitter_ud) + len(pitcher_ud)
    print(f'wrote {a.out}: {n} players, bank {a.bank}')


if __name__ == '__main__':
    _main()

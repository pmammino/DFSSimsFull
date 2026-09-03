"""ud_json_export.py — export the Underdog-scored sims to the JSON shape the
Battle Royale browser extension's Cloudflare Worker reads out of R2.

One file per day, always named ``slate_UD.json`` (see shared_store.UD_JSON_KEY
for where it's uploaded) and always rebuilt fresh from that day's sims — no
history is kept for it, unlike the dated `.npy` archive.
"""
import json
import re
import unicodedata

import numpy as np

BANK_SIZE = 500
DEFAULT_SAMPLE = 75


def nkey(name):
    """Normalize a name for draft-board matching: lowercase, strip accents,
    turn '.'/"'"/'-' into spaces, drop Jr/Sr/II/III/IV/V suffixes, collapse
    whitespace. E.g. "José Ramírez" and "Jose Ramirez" both -> "jose ramirez"."""
    s = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()
    s = re.sub(r"[.'\-]", ' ', s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", '', s)
    return re.sub(r"\s+", ' ', s).strip()


def _bank(sim_array, bank_size=BANK_SIZE):
    """~bank_size representative values spanning the sim distribution: sort,
    then take evenly-spaced values. Deterministic and shape-preserving, unlike
    a random draw."""
    s = np.sort(np.asarray(sim_array, dtype=float))
    idx = np.linspace(0, len(s) - 1, min(bank_size, len(s))).round().astype(int)
    return [int(x) if float(x) == int(x) else round(float(x), 2) for x in s[idx]]


def _player_entry(name, sim_array, bank_size=BANK_SIZE):
    arr = np.asarray(sim_array, dtype=float)
    return {
        "name": name,
        "pos": None,
        "bank": _bank(arr, bank_size),
        "mean": round(float(arr.mean()), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "nkey": nkey(name),
    }


def build_ud_json(hitter_ud, pitcher_ud, bank_size=BANK_SIZE,
                  default_sample=DEFAULT_SAMPLE, sport="MLB", fmt="Battle Royale"):
    """Assemble the {players, bank_size, default_sample, meta} payload from the
    {name: sim_array} dicts sim_proj.simulate() already produces for Underdog
    scoring — hitters and pitchers merged into one flat player list."""
    players = [_player_entry(nm, arr, bank_size)
               for nm, arr in list(hitter_ud.items()) + list(pitcher_ud.items())]
    return {
        "players": players,
        "bank_size": bank_size,
        "default_sample": default_sample,
        "meta": {"sport": sport, "format": fmt},
    }


def write_ud_json(hitter_ud, pitcher_ud, path, **kw):
    payload = build_ud_json(hitter_ud, pitcher_ud, **kw)
    with open(path, 'w') as f:
        json.dump(payload, f)
    return path

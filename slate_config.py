"""
config.py — shared constants for the MLB DFS projection + simulation pipeline.

Everything that is slate-independent lives here: park factors, team-code maps,
DraftKings scoring, model loadings, and file paths.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")     # intermediate cached JSON
OUTPUT_DIR = os.path.join(BASE_DIR, "output")   # deliverables (csv, npy, manifest)
for _d in (DATA_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Feed URLs ──────────────────────────────────────────────────────────────────
FEED_CONFIRMED = "https://rotowire-secrets-ebgmaeh8ecc4huhf.canadaeast-01.azurewebsites.net/api/proxy?feed=lineups"
FEED_EXPECTED  = "https://rotowire-secrets-ebgmaeh8ecc4huhf.canadaeast-01.azurewebsites.net/api/proxy?feed=explineups"
# FantasyLabs Vegas feed (date is substituted in at runtime: YYYY-MM-DD)
FEED_VEGAS_TMPL = "https://www.fantasylabs.com/api/sportevents/3/{date}/vegas"

# statsapi base
STATSAPI = "https://statsapi.mlb.com/api/v1"

# ── Simulation parameters ────────────────────────────────────────────────────
N_SIMS = 10_000
SEED   = 20260610

# Correlation loadings (validated: teammate ~+0.24, hitter-vs-SP ~-0.37, unrelated ~0)
SG          = 0.20   # game-environment loading (both teams + both pitchers)
ST          = 0.50   # team-offense loading (teammates share; opposing SP inverse)
SI          = 0.30   # idiosyncratic hitter loading
SG_HR_EXTRA = 0.12   # extra game loading on HR (wind/air)

# Opener workload (≈1 inning + traffic)
OPENER_BF_MEAN = 4.6
OPENER_BF_SD   = 1.3

# Recency window for role-change detection (appearances)
RECENCY_WINDOW = 4
ROLE_CHANGE_RECENT_WEIGHT = 0.80   # weight on recent BF/app when a role change is detected
NORMAL_RECENT_WEIGHT      = 0.50   # weight on recent BF/app otherwise

# ── Team code mapping: Rotowire code -> standard abbreviation ────────────────
TEAM_CODE_MAP = {
    'ANA':'LAA','AZ':'ARI','NY-A':'NYY','NY-N':'NYM','CHI-A':'CWS','CHI-N':'CHC','LA':'LAD',
    'ATH':'OAK','WSH':'WSH','SD':'SD','SF':'SF','TB':'TB','KC':'KC','ATL':'ATL','BAL':'BAL','BOS':'BOS',
    'CIN':'CIN','CLE':'CLE','COL':'COL','DET':'DET','HOU':'HOU','MIA':'MIA','MIL':'MIL','MIN':'MIN',
    'PHI':'PHI','PIT':'PIT','SEA':'SEA','STL':'STL','TEX':'TEX','TOR':'TOR',
}

# ── Park factors (HR / runs multipliers, handedness-neutral) ─────────────────
# Keyed by the home team's standard abbreviation so we don't depend on venue strings.
PARK_FACTORS = {
    'ARI':{'hr':1.05,'r':1.02}, 'ATL':{'hr':1.02,'r':1.01}, 'BAL':{'hr':1.07,'r':1.03},
    'BOS':{'hr':1.03,'r':1.04}, 'CHC':{'hr':1.05,'r':1.02}, 'CWS':{'hr':1.03,'r':1.01},
    'CIN':{'hr':1.12,'r':1.05}, 'CLE':{'hr':0.97,'r':0.98}, 'COL':{'hr':1.25,'r':1.28},
    'DET':{'hr':0.90,'r':0.96}, 'HOU':{'hr':1.02,'r':1.01}, 'KC':{'hr':0.92,'r':0.97},
    'LAA':{'hr':0.97,'r':0.97}, 'LAD':{'hr':0.95,'r':0.97}, 'MIA':{'hr':0.93,'r':0.96},
    'MIL':{'hr':1.02,'r':1.00}, 'MIN':{'hr':0.97,'r':0.98}, 'NYM':{'hr':0.95,'r':0.97},
    'NYY':{'hr':1.10,'r':1.04}, 'OAK':{'hr':1.05,'r':1.04}, 'PHI':{'hr':1.08,'r':1.04},
    'PIT':{'hr':0.90,'r':0.94}, 'SD':{'hr':0.93,'r':0.95},  'SF':{'hr':0.90,'r':0.93},
    'SEA':{'hr':0.95,'r':0.95}, 'STL':{'hr':0.97,'r':0.98}, 'TB':{'hr':0.95,'r':0.96},
    'TEX':{'hr':0.95,'r':0.97}, 'TOR':{'hr':1.00,'r':1.00}, 'WSH':{'hr':0.99,'r':0.99},
}
DEFAULT_PARK = {'hr':1.00,'r':1.00}

# ── League-average fallback rates ────────────────────────────────────────────
LG_HITTER = {'avg':0.248,'obp':0.316,'slg':0.402,'ops':0.718,
             'k_pct':0.228,'bb_pct':0.083,'hr_pct':0.033,'pa':100}
LG_PITCHER = {'era':4.50,'whip':1.30,'k_pct':0.220,'bb_pct':0.085,'gs':10,
              'ip_total':55.0,'bf':230,'hits_total':54,'hr_allowed_total':10,
              'bf_per_app':23.0,'outs_per_bf':0.70,'appearances':10}
LG_ERA   = 4.20
DEFAULT_TEAM_RUNS = 4.4

# ── DraftKings MLB scoring ───────────────────────────────────────────────────
DK = {
    # hitters
    'single':3, 'double':5, 'triple':8, 'hr':10, 'rbi':2, 'run':2, 'bb':2, 'hbp':2, 'sb':5,
    # pitchers
    'out':0.75,      # +2.25 / inning = +0.75 / out
    'k':2, 'win':4, 'er':-2, 'hit_a':-0.6, 'bb_a':-0.6, 'hbp_a':-0.6,
    'cg':2.5, 'cgs':2.5, 'no_hitter':5,
}

def dk_hitter(s, d, t, hr, rbi, r, bb, hbp, sb):
    return (s*DK['single'] + d*DK['double'] + t*DK['triple'] + hr*DK['hr']
            + rbi*DK['rbi'] + r*DK['run'] + bb*DK['bb'] + hbp*DK['hbp'] + sb*DK['sb'])

def dk_pitcher(outs, k, win, er, hits_a, bb_a, hbp_a, cg, cgs, nh):
    return (outs*DK['out'] + k*DK['k'] + win*DK['win'] + er*DK['er']
            + hits_a*DK['hit_a'] + bb_a*DK['bb_a'] + hbp_a*DK['hbp_a']
            + cg*DK['cg'] + cgs*DK['cgs'] + nh*DK['no_hitter'])

def park_for(home_abbr):
    return PARK_FACTORS.get(home_abbr, DEFAULT_PARK)

def std_code(rotowire_code):
    return TEAM_CODE_MAP.get(rotowire_code, rotowire_code)

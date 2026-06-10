import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import pybaseball as pb
from datetime import date, timedelta
pb.cache.enable()

def scrape_year(year, start_md=(3,20), chunk=7):
    start = date(year, *start_md)
    end = date.today() if year == date.today().year else date(year, 11, 1)
    frames=[]; cur=start
    while cur <= end:
        ce = min(cur+timedelta(days=chunk-1), end)
        try:
            df = pb.statcast(start_dt=cur.isoformat(), end_dt=ce.isoformat(), verbose=False)
            if df is not None and len(df):
                df = df[df["description"].isin(["hit_into_play","hit_into_play_no_out","hit_into_play_score"])]
                if len(df):
                    frames.append(df[["batter","pitcher","events","stand","launch_speed","launch_angle","hc_x","hc_y","home_team"]].copy())
        except Exception as e:
            print(f"  {cur}->{ce} FAIL {e}", flush=True)
        cur = ce+timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["Season"]=year
    df["x"]=df["hc_x"]-125.42; df["y"]=198.27-df["hc_y"]
    df["spray_angle"]=np.degrees(np.arctan2(df["x"],df["y"]))
    df["adjusted_angle"]=np.where(df["stand"]=="L",-df["spray_angle"],df["spray_angle"])
    df=df[["Season","batter","pitcher","events","stand","launch_speed","launch_angle","adjusted_angle","home_team"]].drop_duplicates().reset_index(drop=True)
    return df

year=int(sys.argv[1])
t=time.time()
df=scrape_year(year)
df.to_csv(f"bip_inputs/bip_{year}.csv", index=False)
print(f"{year}: {len(df)} BIP rows in {time.time()-t:.0f}s -> bip_inputs/bip_{year}.csv", flush=True)

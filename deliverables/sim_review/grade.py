import numpy as np, json, glob, os
from scipy import stats
import os
H=os.environ.get("HIST_DIR","./History")
dates=["2026-07-26","2026-07-27","2026-07-28","2026-07-29"]

def dk_hitter(s,d,t,hr,rbi,r,bb,hbp,sb): return s*3+d*5+t*8+hr*10+rbi*2+r*2+bb*2+hbp*2+sb*5
def dk_pitcher(outs,k,win,er,h,bb,hbp,cg,cgs,nh): return outs*0.75+k*2+win*4-er*2-h*0.6-bb*0.6-hbp*0.6+cg*2.5+cgs*2.5+nh*5
def ip_to_outs(ip):
    w=int(ip); freq=round((ip-w)*10); return w*3+int(round(freq))

def load(d,k): return np.load(f"{H}/history_{d}_{k}_dk_sims.npy",allow_pickle=True).item()

def pit(sim, actual):
    sim=np.asarray(sim,dtype=float)
    return (np.mean(sim<actual)+0.5*np.mean(sim==actual))

rows=[]  # (date,kind,name,proj_mean,actual,pit,p10,p90,p25,p75,p50)
for d in dates:
    f=f"actuals/{d}.json"
    if not os.path.exists(f): 
        print("MISSING",f); continue
    js=json.load(open(f))
    hsim=load(d,"hitter"); psim=load(d,"pitcher")
    for rec in js.get("pitchers",[]):
        nm=rec["name"]
        if rec.get("did_not_pitch") or rec.get("not_found"): continue
        if nm not in psim: 
            print("  no sim for pitcher",d,nm); continue
        outs=ip_to_outs(rec["IP"]); er=rec["ER"]; h=rec["H"]; bb=rec.get("BB",0); hbp=rec.get("HBP",0)
        k=rec["SO"]; win=1 if str(rec.get("decision","")).upper().startswith("W") else 0
        ip=rec["IP"]; cg=1 if ip>=9.0 else 0; cgs=1 if (cg and er==0) else 0; nh=1 if (cgs and h==0) else 0
        act=dk_pitcher(outs,k,win,er,h,bb,hbp,cg,cgs,nh)
        sim=psim[nm]
        rows.append([d,"P",nm,float(np.mean(sim)),act,pit(sim,act),
                     np.percentile(sim,10),np.percentile(sim,90),np.percentile(sim,25),np.percentile(sim,75),np.percentile(sim,50)])
    for rec in js.get("hitters",[]):
        nm=rec["name"]
        if rec.get("did_not_play") or rec.get("not_found"): continue
        if nm not in hsim: 
            print("  no sim for hitter",d,nm); continue
        hh=rec.get("H",0);d2=rec.get("2B",0);t=rec.get("3B",0);hr=rec.get("HR",0)
        s=hh-d2-t-hr
        act=dk_hitter(s,d2,t,hr,rec.get("RBI",0),rec.get("R",0),rec.get("BB",0),rec.get("HBP",0),rec.get("SB",0))
        sim=hsim[nm]
        rows.append([d,"H",nm,float(np.mean(sim)),act,pit(sim,act),
                     np.percentile(sim,10),np.percentile(sim,90),np.percentile(sim,25),np.percentile(sim,75),np.percentile(sim,50)])

import numpy as np
def report(kind):
    r=[x for x in rows if x[1]==kind]
    if not r: print(f"\n[{kind}] no data"); return
    proj=np.array([x[3] for x in r]); act=np.array([x[4] for x in r]); pits=np.array([x[5] for x in r])
    p10=np.array([x[6] for x in r]);p90=np.array([x[7] for x in r]);p25=np.array([x[8] for x in r]);p75=np.array([x[9] for x in r])
    err=act-proj
    print(f"\n===== {kind} (n={len(r)}) =====")
    print(f"  mean proj={proj.mean():.2f}  mean actual={act.mean():.2f}  BIAS(actual-proj)={err.mean():+.2f}")
    print(f"  MAE={np.abs(err).mean():.2f}  RMSE={np.sqrt((err**2).mean()):.2f}")
    sp=stats.spearmanr(proj,act).correlation; pr=stats.pearsonr(proj,act)[0]
    print(f"  rank corr (Spearman)={sp:.3f}  Pearson={pr:.3f}")
    print(f"  PIT: mean={pits.mean():.3f} (ideal .50)  std={pits.std():.3f} (ideal .289)")
    print(f"  interval coverage: inside[p10,p90]={((act>=p10)&(act<=p90)).mean():.3f} (ideal .80)  inside[p25,p75]={((act>=p25)&(act<=p75)).mean():.3f} (ideal .50)")
    print(f"  below p10={ (act<p10).mean():.3f} (.10)   above p90={(act>p90).mean():.3f} (.10)")
    # PIT histogram deciles
    hist,_=np.histogram(pits,bins=np.linspace(0,1,11))
    print(f"  PIT decile counts: {hist.tolist()}  (uniform≈{len(r)/10:.1f} each)")
report("P"); report("H")
# save
json.dump([{"date":x[0],"kind":x[1],"name":x[2],"proj":round(x[3],2),"actual":round(float(x[4]),2),"pit":round(x[5],3)} for x in rows], open("graded.json","w"), indent=1)
print("\nsaved graded.json with",len(rows),"matched players")

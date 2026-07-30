import numpy as np, json
from scipy import stats
import os
H=os.environ.get("HIST_DIR","./History"); dates=["2026-07-26","2026-07-27","2026-07-28","2026-07-29"]
def dk_p(o,k,w,er,h,bb,hbp,cg,cgs,nh): return o*0.75+k*2+w*4-er*2-h*0.6-bb*0.6-hbp*0.6+cg*2.5+cgs*2.5+nh*5
def outs(ip): w=int(ip);return w*3+int(round((ip-w)*10))
def load(d,k): return np.load(f"{H}/history_{d}_{k}_dk_sims.npy",allow_pickle=True).item()
P=[]
for d in dates:
    js=json.load(open(f"actuals/{d}.json")); ps=load(d,"pitcher")
    for r in js["pitchers"]:
        if r.get("did_not_pitch") or r.get("not_found") or r["name"] not in ps: continue
        ip=r["IP"];cg=1 if ip>=9 else 0;cgs=1 if(cg and r["ER"]==0)else 0;nh=1 if(cgs and r["H"]==0)else 0
        a=dk_p(outs(ip),r["SO"],1 if str(r.get("decision","")).upper().startswith("W")else 0,r["ER"],r["H"],r.get("BB",0),r.get("HBP",0),cg,cgs,nh)
        sim=np.asarray(ps[r["name"]],float); P.append((sim,a))
act=np.array([a for _,a in P])
for thr in [0,3,5,10]:
    predicted=np.mean([np.mean(s<thr) for s,_ in P]); observed=np.mean(act<thr)
    print(f"  P(pitcher DK < {thr:2d}): sim-predicted={predicted:.3f}  observed={observed:.3f}")
print(f"  actual pitcher DK: mean={act.mean():.1f} std={act.std():.1f}  |  sim avg per-player std={np.mean([s.std() for s,_ in P]):.1f}")
print(f"  cross-sectional actual std (incl. player-to-player)={act.std():.1f} vs sim within-player std {np.mean([s.std() for s,_ in P]):.1f}")
# hitter ceiling check
Hh=[]
def dk_h(s,d,t,hr,rbi,r,bb,hbp,sb): return s*3+d*5+t*8+hr*10+rbi*2+r*2+bb*2+hbp*2+sb*5
for d in dates:
    js=json.load(open(f"actuals/{d}.json")); hs=load(d,"hitter")
    for r in js["hitters"]:
        if r.get("did_not_play") or r.get("not_found") or r["name"] not in hs: continue
        s=r.get("H",0)-r.get("2B",0)-r.get("3B",0)-r.get("HR",0)
        a=dk_h(s,r.get("2B",0),r.get("3B",0),r.get("HR",0),r.get("RBI",0),r.get("R",0),r.get("BB",0),r.get("HBP",0),r.get("SB",0))
        Hh.append((np.asarray(hs[r["name"]],float),a))
acth=np.array([a for _,a in Hh])
for thr in [0,20,30]:
    pr=np.mean([np.mean(s>=thr) for s,_ in Hh]) if thr>0 else np.mean([np.mean(s<=0) for s,_ in Hh])
    ob=np.mean(acth>=thr) if thr>0 else np.mean(acth<=0)
    lbl=f">= {thr}" if thr>0 else "== 0"
    print(f"  P(hitter DK {lbl}): sim={pr:.3f} observed={ob:.3f}")

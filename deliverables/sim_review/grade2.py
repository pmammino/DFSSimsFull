import numpy as np, json, os
from scipy import stats
import os
H=os.environ.get("HIST_DIR","./History")
dates=["2026-07-26","2026-07-27","2026-07-28","2026-07-29"]
def dk_h(s,d,t,hr,rbi,r,bb,hbp,sb): return s*3+d*5+t*8+hr*10+rbi*2+r*2+bb*2+hbp*2+sb*5
def dk_p(o,k,w,er,h,bb,hbp,cg,cgs,nh): return o*0.75+k*2+w*4-er*2-h*0.6-bb*0.6-hbp*0.6+cg*2.5+cgs*2.5+nh*5
def outs(ip): w=int(ip);return w*3+int(round((ip-w)*10))
def load(d,k): return np.load(f"{H}/history_{d}_{k}_dk_sims.npy",allow_pickle=True).item()
def pit(sim,a): sim=np.asarray(sim,float);return np.mean(sim<a)+0.5*np.mean(sim==a)

recs=[]
for d in dates:
    js=json.load(open(f"actuals/{d}.json")); hs=load(d,"hitter"); ps=load(d,"pitcher")
    for r in js["pitchers"]:
        if r.get("did_not_pitch") or r.get("not_found") or r["name"] not in ps: continue
        ip=r["IP"];cg=1 if ip>=9 else 0;cgs=1 if(cg and r["ER"]==0)else 0;nh=1 if(cgs and r["H"]==0)else 0
        a=dk_p(outs(ip),r["SO"],1 if str(r.get("decision","")).upper().startswith("W")else 0,r["ER"],r["H"],r.get("BB",0),r.get("HBP",0),cg,cgs,nh)
        sim=np.asarray(ps[r["name"]],float)
        recs.append(dict(d=d,k="P",n=r["name"],proj=sim.mean(),act=a,pit=pit(sim,a),ip=ip,
                         p90=np.percentile(sim,90),p99=np.percentile(sim,99),p10=np.percentile(sim,10),std=sim.std()))
    for r in js["hitters"]:
        if r.get("did_not_play") or r.get("not_found") or r["name"] not in hs: continue
        s=r.get("H",0)-r.get("2B",0)-r.get("3B",0)-r.get("HR",0)
        a=dk_h(s,r.get("2B",0),r.get("3B",0),r.get("HR",0),r.get("RBI",0),r.get("R",0),r.get("BB",0),r.get("HBP",0),r.get("SB",0))
        sim=np.asarray(hs[r["name"]],float)
        recs.append(dict(d=d,k="H",n=r["name"],proj=sim.mean(),act=a,pit=pit(sim,a),
                         p90=np.percentile(sim,90),p99=np.percentile(sim,99),p10=np.percentile(sim,10),std=sim.std()))

def stat(rs,lbl):
    if not rs: return
    proj=np.array([r["proj"] for r in rs]);act=np.array([r["act"] for r in rs])
    print(f"{lbl:28s} n={len(rs):3d} biasProjActual={act.mean()-proj.mean():+5.2f} MAE={np.abs(act-proj).mean():4.1f} RMSE={np.sqrt(((act-proj)**2).mean()):4.1f} Spearman={stats.spearmanr(proj,act).correlation:+.3f}")

P=[r for r in recs if r["k"]=="P"]; Hh=[r for r in recs if r["k"]=="H"]
print("=== PER-DAY ===")
for d in dates:
    stat([r for r in P if r["d"]==d], f"P {d}")
    stat([r for r in Hh if r["d"]==d], f"H {d}")
print("=== POOLED ===")
stat(P,"PITCHERS all"); stat(Hh,"HITTERS all")
# pitchers who actually started & threw >=4 IP (remove openers/quick hooks that are unmodelable)
Ps=[r for r in P if r["ip"]>=4.0]; stat(Ps,"PITCHERS IP>=4 (true starts)")

print("\n=== BOOM COVERAGE (did sim ceiling reach reality?) ===")
for k,rs in [("P",P),("H",Hh)]:
    booms=[r for r in rs if r["act"]>=30]
    cov=sum(1 for r in booms if r["act"]<=r["p99"])
    print(f"  {k}: actual 30+ games={len(booms)}, of those within sim p99={cov}  ({[ (r['n'],round(r['act']),round(r['p99'])) for r in booms]})")

print("\n=== BIGGEST MISSES (|actual-proj|) ===")
for r in sorted(recs,key=lambda r:-abs(r['act']-r['proj']))[:12]:
    print(f"  {r['k']} {r['d']} {r['n']:20s} proj={r['proj']:5.1f} actual={r['act']:6.1f} pit={r['pit']:.2f} p10={r['p10']:.0f} p90={r['p90']:.0f} p99={r['p99']:.0f}")

print("\n=== CALIBRATION BY PROJECTION TIER (PIT mean should ~.50 each) ===")
for k,rs in [("P",P),("H",Hh)]:
    rs2=sorted(rs,key=lambda r:r["proj"]); ter=np.array_split(rs2,3)
    for i,t in enumerate(ter):
        pits=np.array([r["pit"] for r in t]); proj=np.array([r["proj"] for r in t])
        print(f"  {k} tier{i+1} projRange[{proj.min():.1f},{proj.max():.1f}] PITmean={pits.mean():.3f} (bias {'high proj/low real' if pits.mean()<.45 else 'low proj/high real' if pits.mean()>.55 else 'ok'})")

# overall PIT uniformity pooled
allpit=np.array([r["pit"] for r in recs])
print(f"\nPOOLED PIT (n={len(recs)}): mean={allpit.mean():.3f} std={allpit.std():.3f} | frac in tails (<.1 or >.9)={((allpit<.1)|(allpit>.9)).mean():.3f} (ideal .20)")

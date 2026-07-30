import numpy as np, json, sys, os
SCR=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCR, "..", "..")))  # repo root (for sim_proj)
import sim_proj
from scipy import stats
H=os.environ.get("HIST_DIR","./History"); dates=["2026-07-26","2026-07-27","2026-07-28","2026-07-29"]
def dk_p(o,k,w,er,h,bb,hbp,cg,cgs,nh): return o*0.75+k*2+w*4-er*2-h*0.6-bb*0.6-hbp*0.6+cg*2.5+cgs*2.5+nh*5
def outs(ip): w=int(ip);return w*3+int(round((ip-w)*10))

# OLD engine (verbatim reproduction of pre-change _sim_pitcher) for faithfulness reconstruction
def old_sim(vec,bf_sim,m_opp,z,can_win,win_base,outs_cap,rng,n):
    bf_sim=np.clip(bf_sim,0,40).astype(int); opb=max(0.55,min(0.80,3.0/max(vec['tbf_per_ip'],3.2)))
    o=np.clip(np.round(bf_sim*opb).astype(int),0,outs_cap); ip=o/3.0
    k=rng.binomial(bf_sim,min(0.6,vec['k_pct']))
    h_r=np.clip(vec['h_per_bf']*m_opp,0.05,0.60);hr_r=np.clip(vec['hr_per_bf']*m_opp,0.003,0.12);bb_r=np.clip(vec['bb_pct']*np.sqrt(m_opp),0.02,0.25)
    h=rng.binomial(bf_sim,h_r);hr=rng.binomial(bf_sim,hr_r);bb=rng.binomial(bf_sim,bb_r);hbp=rng.binomial(bf_sim,min(0.05,vec['hbp_per_bf']))
    er_mean=vec.get('ra9',4.6)*ip/9.0
    er=np.round(np.clip(er_mean+(h*0.10+hr*0.55+bb*0.07-0.9)+z*0.5+rng.normal(0,0.6,n),0,15)).astype(int)
    wp=np.clip(win_base-er*0.04-z*0.03,0.02,0.85) if can_win else 0
    win=((ip>=5.0)&(rng.uniform(0,1,n)<wp)).astype(int) if can_win else np.zeros(n,int)
    cg=(ip>=9.0).astype(int);cgs=(cg&(er==0)).astype(int);nh=(cgs&(h==0)).astype(int)
    return dict(dk=dk_p(o,k,win,er,h,bb,hbp,cg,cgs,nh),ip=ip)
def vec_q(q):
    return dict(tbf_per_ip=4.30-0.06*q,k_pct=float(np.clip(0.215+0.045*q,0.10,0.38)),
        h_per_bf=float(np.clip(0.230-0.020*q,0.14,0.32)),hr_per_bf=float(np.clip(0.032-0.006*q,0.010,0.060)),
        bb_pct=float(np.clip(0.082-0.008*q,0.03,0.14)),hbp_per_bf=0.010,era=4.20-0.55*q,ra9=4.55-0.60*q)
def workload(vec,rng,n):
    Lg=rng.standard_normal(n);Lt=rng.standard_normal(n);sho=0.20*Lg+0.50*Lt;shvar=0.20**2+0.50**2
    m_opp=np.exp(sho-0.5*shvar); z=(sho-sho.mean())/(sho.std()+1e-9)
    bf=np.clip(rng.normal(vec['tbf_per_ip']*5.6,3.0,n)-z*3.0-(vec['era']-4.20)*0.8,8,34)
    wb=max(0.20,min(0.75,0.500+(4.20-vec['era'])*0.03))
    return bf,m_opp,z,wb

# build q -> old_mean grid
grid_q=np.linspace(-3.0,3.5,60); rng=np.random.default_rng(1)
gm=[]
for q in grid_q:
    v=vec_q(q); bf,mo,z,wb=workload(v,rng,4000); gm.append(old_sim(v,bf,mo,z,True,wb,27,rng,4000)['dk'].mean())
gm=np.array(gm)
def q_for_mean(mu): return float(np.interp(mu,gm,grid_q))

# reconstruct panel from matched pitchers + re-grade
oldrows=[]; newrows=[]
N=10000
for d in dates:
    js=json.load(open(f"{SCR}/actuals/{d}.json")); ps=np.load(f"{H}/history_{d}_pitcher_dk_sims.npy",allow_pickle=True).item()
    for r in js["pitchers"]:
        if r.get("did_not_pitch") or r.get("not_found") or r["name"] not in ps: continue
        rec=np.asarray(ps[r["name"]],float)           # RECORDED old distribution (real history)
        ip=r["IP"];cg=1 if ip>=9 else 0;cgs=1 if(cg and r["ER"]==0)else 0;nh=1 if(cgs and r["H"]==0)else 0
        act=dk_p(outs(ip),r["SO"],1 if str(r.get("decision","")).upper().startswith("W")else 0,r["ER"],r["H"],r.get("BB",0),r.get("HBP",0),cg,cgs,nh)
        q=q_for_mean(rec.mean()); v=vec_q(q)
        rng=np.random.default_rng(hash(r["name"])%2**32)
        bf,mo,z,wb=workload(v,rng,N)
        # reconstructed OLD (faithfulness) and NEW (engine code) on the SAME inputs
        recon_old=old_sim(v,bf,mo,z,True,wb,27,rng,N)['dk']
        rng2=np.random.default_rng((hash(r["name"])%2**32)^0xABCD)
        newd=sim_proj._sim_pitcher(v,bf,mo,z,True,wb,27,rng2,N)['dk']
        def pit(sim): sim=np.asarray(sim,float);return np.mean(sim<act)+0.5*np.mean(sim==act)
        oldrows.append(dict(name=r["name"],proj=rec.mean(),act=act,dist=rec,pit=pit(rec)))
        newrows.append(dict(name=r["name"],proj=newd.mean(),act=act,dist=newd,pit=pit(newd),reconold=recon_old))

def summarize(rows,key="dist",lbl=""):
    proj=np.array([r["proj"] for r in rows]);act=np.array([r["act"] for r in rows]);pits=np.array([r["pit"] for r in rows])
    p10=np.array([np.percentile(r[key],10) for r in rows]);p90=np.array([np.percentile(r[key],90) for r in rows])
    stds=np.mean([r[key].std() for r in rows])
    pred0=np.mean([np.mean(r[key]<0) for r in rows]); pred3=np.mean([np.mean(r[key]<3) for r in rows])
    pred10=np.mean([np.mean(r[key]<10) for r in rows])
    print(f"{lbl}: n={len(rows)} meanProj={proj.mean():.2f} meanAct={act.mean():.2f} bias={act.mean()-proj.mean():+.2f} "
          f"avgStd={stds:.1f} Spear={stats.spearmanr(proj,act).correlation:+.3f}")
    print(f"     PITmean={pits.mean():.3f} PITstd={pits.std():.3f} cover[p10,p90]={((act>=p10)&(act<=p90)).mean():.3f}(.80) "
          f"P(<0) sim={pred0:.3f} P(<3) sim={pred3:.3f} P(<10) sim={pred10:.3f}")

act=np.array([r["act"] for r in oldrows])
print("OBSERVED actuals: P(<0)={:.3f} P(<3)={:.3f} P(<10)={:.3f} std(cross)={:.1f}\n".format((act<0).mean(),(act<3).mean(),(act<10).mean(),act.std()))
# faithfulness: recorded old vs reconstructed old aggregate
rec_all=np.concatenate([r["dist"] for r in oldrows]); ro_all=np.concatenate([r["reconold"] for r in newrows])
print("FAITHFULNESS (panel reproduces real history):")
print(f"  recorded-old   pooled: mean={rec_all.mean():.2f} std={rec_all.std():.2f} P(<0)={(rec_all<0).mean():.3f}")
print(f"  reconstruct-old pooled: mean={ro_all.mean():.2f} std={ro_all.std():.2f} P(<0)={(ro_all<0).mean():.3f}\n")
summarize(oldrows,"dist","OLD (recorded history)")
summarize(newrows,"dist","NEW (engine code)     ")

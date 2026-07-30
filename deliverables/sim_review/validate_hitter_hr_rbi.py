"""
P2 validation — HR->R/RBI coupling in the hitter run-conservation step.

Self-contained (synthetic 9-man lineup, no history needed). Runs the OLD and
NEW allocation side by side and reports mean preservation, exact team
conservation, the HR self-run/RBI guarantee, and the boom-tail lift.
Run: python3 validate_hitter_hr_rbi.py
"""
import numpy as np
SG,ST,SI,SG_HR_EXTRA=0.20,0.50,0.30,0.12
def dk_h(s,d,t,hr,rbi,r,bb,hbp,sb): return s*3+d*5+t*8+hr*10+rbi*2+r*2+bb*2+hbp*2+sb*5
def _pa(slot): return 4.2-(slot-1)*0.055

def lineup():
    L=[]
    grad=[1.2,1.0,1.4,0.8,0.6,0.2,-0.2,-0.6,-1.0]  # quality by slot
    for i,q in enumerate(grad):
        L.append(dict(name=f"H{i+1}",slot=i+1,vec=dict(
            p_hr=float(np.clip(0.032+0.012*q,0.008,0.06)),p_3b=0.004,
            p_2b=float(np.clip(0.045+0.008*q,0.02,0.07)),p_1b=float(np.clip(0.150+0.010*q,0.10,0.20)),
            p_bb=float(np.clip(0.082+0.010*q,0.03,0.13)),p_hbp=0.010,p_k=float(np.clip(0.220-0.010*q,0.12,0.32)),
            p_sb=0.02,r_pa=float(np.clip(0.130+0.020*q,0.08,0.19)),rbi_pa=float(np.clip(0.125+0.020*q,0.07,0.18)),proj_slot=i+1)))
    return L

def sim_team(mode,seed=7,n=40000):
    rng=np.random.default_rng(seed)
    Lg=rng.standard_normal(n);Lt=rng.standard_normal(n)
    sh=SG*Lg+ST*Lt; sh_var=SG**2+ST**2
    plist=[]
    for p in lineup():
        vec=p['vec']; slot=p['slot']; pa=np.clip(rng.poisson(_pa(slot),n),1,7)
        idio=SI*rng.standard_normal(n)
        m_off=np.exp(sh+idio-0.5*(sh_var+SI**2))
        m_hr=np.exp(sh+idio+SG_HR_EXTRA*Lg-0.5*(sh_var+SI**2+SG_HR_EXTRA**2))
        p_hr=np.clip(vec['p_hr']*m_hr,0.001,0.20);p_3b=np.clip(vec['p_3b']*m_off,0,0.05)
        p_2b=np.clip(vec['p_2b']*m_off,0,0.20);p_1b=np.clip(vec['p_1b']*m_off,0,0.55)
        p_bb=np.clip(vec['p_bb']*np.sqrt(m_off),0,0.30);p_hbp=np.full(n,vec['p_hbp']);p_k=np.full(n,vec['p_k'])
        c_hr=p_hr;c3=c_hr+p_3b;c2=c3+p_2b;c1=c2+p_1b;cb=c1+p_bb;chb=cb+p_hbp;ck=chb+p_k
        sgl=np.zeros(n,int);dbl=np.zeros(n,int);trp=np.zeros(n,int);hr=np.zeros(n,int);bb=np.zeros(n,int);hbp=np.zeros(n,int);ks=np.zeros(n,int)
        for i in range(int(pa.max())):
            a=pa>i
            if not a.any(): break
            u=rng.uniform(0,1,a.sum())
            chr_=c_hr[a];c3a=c3[a];c2a=c2[a];c1a=c1[a];cba=cb[a];chba=chb[a];cka=ck[a]
            hr[a]+=(u<chr_).astype(int);trp[a]+=((u>=chr_)&(u<c3a)).astype(int)
            dbl[a]+=((u>=c3a)&(u<c2a)).astype(int);sgl[a]+=((u>=c2a)&(u<c1a)).astype(int)
            bb[a]+=((u>=c1a)&(u<cba)).astype(int);hbp[a]+=((u>=cba)&(u<chba)).astype(int);ks[a]+=((u>=chba)&(u<cka)).astype(int)
        sb_s=rng.poisson(np.clip(vec['p_sb']*pa,0,3))
        plist.append(dict(vec=vec,pa=pa,m_off=m_off,sgl=sgl,dbl=dbl,trp=trp,hr=hr,bb=bb,hbp=hbp,ks=ks,sb=sb_s))
    team_raw=np.zeros(n);team_hr=np.zeros(n,int);r_wsum=np.zeros(n);rbi_wsum=np.zeros(n);r_exp=np.zeros(n)
    for d in plist:
        team_raw+=d['hr']+0.6*(d['dbl']+d['trp'])+0.3*d['sgl']+0.2*(d['bb']+d['hbp'])
        team_hr+=d['hr'];rw=d['vec']['r_pa']*d['pa'];bw=d['vec']['rbi_pa']*d['pa']
        r_wsum+=rw;rbi_wsum+=bw;r_exp+=rw*d['m_off']
    c_scale=(r_exp.mean()/team_raw.mean()) if team_raw.mean()>0 else 1.0
    tri=np.clip(np.round(team_raw*c_scale),0,None).astype(int)
    r_wsum=np.where(r_wsum>0,r_wsum,1.0);rbi_wsum=np.where(rbi_wsum>0,rbi_wsum,1.0)
    if mode=="new": tri=np.maximum(tri,team_hr); residual=tri-team_hr
    else: residual=tri
    def alloc(rate,wsum,key):
        rem=residual.copy();rs=np.ones(n);shares=[(d['vec'][rate]*d['pa'])/wsum for d in plist]
        for k,d in enumerate(plist):
            take=rem.copy() if k==len(plist)-1 else rng.binomial(rem,np.clip(shares[k]/np.maximum(rs,1e-9),0,1))
            d[key]=(d['hr']+take) if mode=="new" else take
            rem=rem-take;rs=rs-shares[k]
    alloc('r_pa',r_wsum,'R');alloc('rbi_pa',rbi_wsum,'RBI')
    dks=[];sumR=np.zeros(n,int);sumRBI=np.zeros(n,int)
    guar_viol=0; hr_events=0
    for d in plist:
        r_s=np.clip(d['R'],0,6);rbi_s=np.clip(d['RBI'],0,8)
        sumR+=d['R'];sumRBI+=d['RBI']
        # guaranteed check (pre-clip): if hr>0 then R>=hr and RBI>=hr
        m=d['hr']>0; hr_events+=int(m.sum())
        guar_viol+=int(((d['R']<d['hr'])|(d['RBI']<d['hr']))[m].sum())
        dk=dk_h(d['sgl'],d['dbl'],d['trp'],d['hr'],rbi_s,r_s,d['bb'],d['hbp'],d['sb'])
        dks.append(dk)
    return dict(dks=np.array(dks),tri=tri,sumR=sumR,sumRBI=sumRBI,guar_viol=guar_viol,hr_events=hr_events,plist=plist)

for mode in ("old","new"):
    r=sim_team(mode)
    dks=r['dks']; pooled=dks.reshape(-1)
    consR=(r['sumR']==r['tri']).mean(); consRBI=(r['sumRBI']==r['tri']).mean()
    print(f"\n{mode.upper()}: teamMeanDK={dks.mean(axis=1).sum():.2f}  poolMean={pooled.mean():.2f} poolStd={pooled.std():.2f}")
    print(f"   pooled P(>=20)={(pooled>=20).mean():.4f} P(>=30)={(pooled>=30).mean():.4f} P(>=40)={(pooled>=40).mean():.5f} max={pooled.max()}")
    print(f"   conservation ΣR==team={consR:.3f} ΣRBI==team={consRBI:.3f}  | HR-events={r['hr_events']} guarantee_violations={r['guar_viol']}")

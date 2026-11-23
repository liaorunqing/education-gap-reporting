"""Post-pilot, fixed-design interval sensitivity. No new holdout or model fitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import norm,t,beta
from acs221_sipp_pilot import setup,METHODS
from acs184_reporting_methods import integer_allocation

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'outputs/acs_research215_2026-11-22'
OUT=ROOT/'outputs/acs_boundaries235_2026-11-22'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plan():
    OUT.mkdir(exist_ok=True)
    p=OUT/'analysis_plan.json'
    assert not p.exists(),'Keep the original plan; run analysis separately.'
    obj=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Post-pilot sensitivity on already accessed SIPP; not preregistration or new confirmation',
        scopes=['all_released','reported_only'],budgets=[300,1000],methods=METHODS,R=2000,
        intervals=['normal','t_Satterthwaite','Serfling_range'],levels=[.90,.95],report_resolutions=[.01,.02,.05],
        fixed_draws='Exactly reconstruct old seeds/designs and compare points/variances before interpreting',
        t='For stratum variance contributions v_j, df=(sum v_j)^2/sum[v_j^2/(n_j-1)]; zero variance remains degenerate, not repaired',
        range='Use binary-outcome household score bounds from prior demographics only, not observed target min/max. Stratum Serfling radius with alpha/G union bound; census strata radius zero; intersect with known [-1,1]',
        outcomes=['coverage and exact binomial Monte Carlo limits','mean width','interval score','all-attempt correct/wrong/unresolved','zero residual sampled diagnostic'],
        preserved={str(p.resolve()):sha(p) for p in OLD.rglob('*') if p.is_file() and p.parts[-2] in ['sipp_pilot','deliverable'] and p.suffix in ['.npz','.csv','.pdf','.tex','.html']},
        claim='Application and conditions of use; no novel interval, new model, attrition correction, monetary cost or resident-population guarantee')
    p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8');print('PLAN_FROZEN',sha(p),flush=True)

def known_bounds(d,q):
    age=d.TAGE_EHC_prior.to_numpy();sex=d.ESEX_prior.to_numpy(int);nat=d.nativity_prior.to_numpy(int);w=d.WPFINWGT_prior.to_numpy()
    cell=2*np.where(age<=39,0,np.where(age<=59,1,2))+sex-1;g=2*cell+nat
    W=np.bincount(g,weights=w,minlength=12);units,h=np.unique(d.cluster.astype(str),return_inverse=True);H=len(units)
    b=H*q[cell]*(2*nat-1)*w/W[g]
    return np.bincount(h,weights=np.minimum(b,0),minlength=H),np.bincount(h,weights=np.maximum(b,0),minlength=H)

def run():
    config=json.loads((OUT/'analysis_plan.json').read_text());assert not (OUT/'interval_results.csv').exists()
    d=pd.read_csv(OLD/'private_sipp/linked_cohort_private.csv.gz',dtype={'SSUID':'string','SHHADID_prior':'string','SHHADID_current':'string'})
    q=np.array(json.loads((OLD/'sipp_prior_background_freeze.json').read_text())['q'])
    eligible=(d.AEDUC_prior.eq(1)&d.AEDUC_current.eq(1)&d.observed_for_pilot).groupby(d.cluster).all()
    scopes={'all_released':d.loc[d.complete_cluster].copy(),'reported_only':d.loc[d.cluster.isin(eligible.index[eligible])].copy()}
    rows=[];reports=[];diagnostics=[];maxdiff=0.
    for scope,frame in scopes.items():
        z=setup(frame,d,q);H=z['H'];lo,hi=known_bounds(frame,q)
        assert np.all(z['Y']>=lo-1e-12) and np.all(z['Y']<=hi+1e-12)
        changed=frame.copy();changed.EEDUC_current=31
        lo2,hi2=known_bounds(changed,q);assert np.array_equal(lo,lo2) and np.array_equal(hi,hi2)
        for m in config['budgets']:
            old=np.load(OLD/'sipp_pilot'/f'{scope}_{m}_simulation.npz',allow_pickle=False)
            for j,method in enumerate(METHODS):
                F=np.zeros(H) if j==0 else z['Fcell'] if j==1 else z['Flag'];e=z['Y']-F
                groups=[np.arange(H)] if j<3 else z['groups'] if j==3 else z['fine_groups']
                N=np.array([len(ix) for ix in groups]);G=len(groups)
                n=np.array([m]) if j<3 else integer_allocation(N,np.ones(G) if j==3 else np.sqrt([z['working'][ix].mean() for ix in groups]),m)
                P=[];V=[];DF=[];ZERO=[]
                for r in range(2000):
                    rng=np.random.default_rng(739+m*10000+(0 if j<3 else 100000000)+r)
                    p=float(F.mean());v=[];ev=0
                    for ix,nn,NN in zip(groups,n,N):
                        s=ix if nn==NN else ix[rng.choice(NN,int(nn),replace=False)]
                        p+=NN/H*e[s].mean();ev+=int(np.count_nonzero(abs(e[s])>1e-12))
                        v.append(0. if nn==NN else (NN/H)**2*(1-nn/NN)*e[s].var(ddof=1)/nn)
                    v=np.array(v);vv=float(v.sum());den=float(np.sum(v*v/np.maximum(n-1,1)))
                    P.append(p);V.append(vv);DF.append(vv*vv/den if den>0 else np.inf);ZERO.append(ev==0)
                P=np.array(P);V=np.array(V);DF=np.array(DF);SE=np.sqrt(V)
                diff=max(float(abs(P-old['point'][:,j]).max()),float(abs(V-old['variance'][:,j]).max()));maxdiff=max(maxdiff,diff)
                assert diff<2e-12,(scope,m,method,diff)
                diagnostics.append(dict(scope=scope,m=m,method=method,zero_variance=int((V==0).sum()),zero_sampled_residual=int(np.sum(ZERO)),
                    median_effective_df=float(np.median(DF)),minimum_effective_df=float(DF.min()),point_variance_max_difference=diff))
                for level in config['levels']:
                    alpha=1-level
                    radius=0.
                    for ix,nn,NN in zip(groups,n,N):
                        if nn<NN:
                            width=float(np.max(hi[ix]-F[ix])-np.min(lo[ix]-F[ix]))
                            radius+=NN/H*width*np.sqrt((1-(nn-1)/NN)*np.log(2*G/alpha)/(2*nn))
                    for interval,w in [('normal',norm.ppf(1-alpha/2)*SE),('t_Satterthwaite',t.ppf(1-alpha/2,DF)*SE),('Serfling_range',np.full(2000,radius))]:
                        L=P-w;U=P+w
                        if interval=='Serfling_range':L=np.maximum(L,-1);U=np.minimum(U,1)
                        covered=(L<=z['truth'])&(U>=z['truth']);k=int(covered.sum())
                        cplo=0 if k==0 else beta.ppf(.025,k,2001-k);cphi=1 if k==2000 else beta.ppf(.975,k+1,2000-k)
                        score=U-L+2/alpha*((L-z['truth'])*(z['truth']<L)+(z['truth']-U)*(z['truth']>U))
                        rows.append(dict(scope=scope,m=m,method=method,interval=interval,level=level,R=2000,truth=z['truth'],coverage=float(covered.mean()),
                            coverage_MC95_low=cplo,coverage_MC95_high=cphi,width=float((U-L).mean()),interval_score=float(score.mean()),
                            width_MCSE=float((U-L).std(ddof=1)/np.sqrt(2000)),degenerate=int((U==L).sum())))
                        if level==.95:
                            for tau in config['report_resolutions']:
                                c=np.full(2000,3);c[L>tau]=0;c[U< -tau]=1;c[(L>=-tau)&(U<=tau)]=2
                                tc=0 if z['truth']>tau else 1 if z['truth']< -tau else 2
                                correct=int((c==tc).sum());wrong=int(((c!=3)&(c!=tc)).sum());un=int((c==3).sum())
                                reports.append(dict(scope=scope,m=m,method=method,interval=interval,tau=tau,R=2000,correct=correct,wrong=wrong,unresolved=un,
                                    correct_yield=correct/2000,wrong_per_attempt=wrong/2000,report_rate=(correct+wrong)/2000,
                                    wrong_given_report=wrong/(correct+wrong) if correct+wrong else np.nan))
                print('INTERVALS_DONE',scope,m,method,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'interval_results.csv',index=False);pd.DataFrame(reports).to_csv(OUT/'reporting_results.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'event_support_diagnostics.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in config['preserved'].items())
    result=dict(post_pilot_not_new_confirmation=True,all_20_cases_retained=True,interval_rows=len(rows),report_rows=len(reports),
        point_and_variance_max_difference=maxdiff,range_bounds_invariant_to_current_labels=True,all_original_files_unchanged=True,
        normal_and_t_are_approximations=True,Serfling_guarantee_scope='Fixed complete-case finite frame and compatible fixed probability design only; not residents, missing outcomes or original survey uncertainty',
        bootstrap_not_implemented=True)
    (OUT/'QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',action='store_true')
    parser.add_argument('--pilot-dir',type=Path,default=OLD);parser.add_argument('--out-dir',type=Path,default=OUT)
    args=parser.parse_args();OLD=args.pilot_dir.resolve();OUT=args.out_dir.resolve()
    plan() if args.plan else run()

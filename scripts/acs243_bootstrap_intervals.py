"""Design-rescaled bootstraps and empirical Bernstein-Serfling controls."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy.stats import beta
from acs221_sipp_pilot import setup,METHODS
from acs235_interval_boundaries import known_bounds
from acs184_reporting_methods import integer_allocation
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/acs_research215_2026-11-22';O=R/'outputs/acs_followup242_2026-11-22';ONLY_CASE=None

def metrics(P,L,U,truth,scope,m,method,interval,level):
    alpha=1-level;ok=(L<=truth)&(U>=truth);k=int(ok.sum());n=len(P)
    score=U-L+2/alpha*((L-truth)*(truth<L)+(truth-U)*(truth>U))
    row=dict(scope=scope,m=m,method=method,interval=interval,level=level,R=n,truth=truth,coverage=float(ok.mean()),
      coverage_MC95_low=0 if k==0 else beta.ppf(.025,k,n-k+1),coverage_MC95_high=1 if k==n else beta.ppf(.975,k+1,n-k),
      width=float((U-L).mean()),interval_score=float(score.mean()),degenerate=int((U-L<=1e-14).sum()))
    reports=[]
    if level==.95:
        for tau in [.01,.02,.05]:
            c=np.full(n,3);c[L>tau]=0;c[U< -tau]=1;c[(L>=-tau)&(U<=tau)]=2
            tc=0 if truth>tau else 1 if truth< -tau else 2
            a=int((c==tc).sum());b=int(((c!=3)&(c!=tc)).sum());un=int((c==3).sum())
            reports.append(dict(scope=scope,m=m,method=method,interval=interval,tau=tau,R=n,correct=a,wrong=b,unresolved=un,
                correct_yield=a/n,wrong_per_attempt=b/n,wrong_given_report=b/(a+b) if a+b else np.nan))
    return row,reports

def main():
    plan=json.loads((O/'analysis_plan.json').read_text());C=plan['phase1'];dest=O/'intervals';dest.mkdir(exist_ok=True)
    d=pd.read_csv(OLD/'private_sipp/linked_cohort_private.csv.gz',dtype={'SSUID':'string','SHHADID_prior':'string','SHHADID_current':'string'})
    q=np.array(json.loads((OLD/'sipp_prior_background_freeze.json').read_text())['q'])
    elig=(d.AEDUC_prior.eq(1)&d.AEDUC_current.eq(1)&d.observed_for_pilot).groupby(d.cluster).all()
    scopes={'all_released':d[d.complete_cluster],'reported_only':d[d.cluster.isin(elig.index[elig])]}
    allrows=[];allreports=[];precision=[];start=time.time()
    for si,(scope,frame) in enumerate(scopes.items()):
        z=setup(frame,d,q);H=z['H'];lo,hi=known_bounds(frame,q)
        for m in C['budgets']:
            reference=np.load(OLD/'sipp_pilot'/f'{scope}_{m}_simulation.npz',allow_pickle=False)
            for j,method in enumerate(METHODS):
                if ONLY_CASE is not None and (scope,str(m),method)!=tuple(ONLY_CASE):continue
                checkpoint=dest/f'{scope}_{m}_{method}.json'
                if checkpoint.exists():
                    saved=json.loads(checkpoint.read_text());allrows+=saved['rows'];allreports+=saved['reports'];precision+=saved['precision'];continue
                F=np.zeros(H) if j==0 else z['Fcell'] if j==1 else z['Flag'];e=z['Y']-F
                groups=[np.arange(H)] if j<3 else z['groups'] if j==3 else z['fine_groups'];N=np.array([len(x) for x in groups]);G=len(groups)
                n=np.array([m]) if j<3 else integer_allocation(N,np.ones(G) if j==3 else np.sqrt([z['working'][ix].mean() for ix in groups]),m)
                P=np.empty(2000);ends={(key,lv):[np.empty(2000),np.empty(2000)] for key in ['Preston_basic','RW_basic','empirical_Bernstein_Serfling'] for lv in C['levels']}
                sens=[];diffmax=0.
                for r in range(2000):
                    sample_rng=np.random.default_rng(739+m*10000+(0 if j<3 else 100000000)+r)
                    brng=np.random.default_rng(739+si*10**10+m*10**6+j*10**5+r)
                    B=C['precision_B'] if r<100 else C['B'];dp=np.zeros(B);dw=np.zeros(B);p=float(F.mean());v=0.;terms=[]
                    for ix,nn,NN in zip(groups,n,N):
                        s=ix if nn==NN else ix[sample_rng.choice(NN,int(nn),replace=False)];values=e[s];mean=float(values.mean());p+=NN/H*mean
                        if nn==NN:continue
                        f=nn/NN;v+=(NN/H)**2*(1-f)*values.var(ddof=1)/nn
                        u,counts=np.unique(values,return_counts=True);k=int(nn//2)
                        if len(u)>1:
                            c=brng.multivariate_hypergeometric(counts,k,size=B)
                            dp+=NN/H*np.sqrt(k*(1-f)/(nn-k))*(c@u/k-mean)
                            c=brng.multinomial(int(nn-1),counts/nn,size=B)
                            dw+=NN/H*np.sqrt(1-f)*(c@u/(nn-1)-mean)
                        width=float(np.max(hi[ix]-F[ix])-np.min(lo[ix]-F[ix]));rho=1-(nn-1)/NN if nn<=NN/2 else (1-nn/NN)*(1+1/nn)
                        terms.append((NN/H,values.std(ddof=0),rho,width,nn))
                    P[r]=p;diffmax=max(diffmax,abs(p-reference['point'][r,j]),abs(v-reference['variance'][r,j]))
                    for lv in C['levels']:
                        a=1-lv
                        for name,dev in [('Preston_basic',dp),('RW_basic',dw)]:
                            quant=np.quantile(dev[:999],[a/2,1-a/2]);L=p-quant[1];U=p-quant[0]
                            ends[name,lv][0][r]=L;ends[name,lv][1][r]=U
                            if r<100:
                                qq=np.quantile(dev,[a/2,1-a/2]);sens.append(dict(scope=scope,m=m,method=method,interval=name,level=lv,draw=r,
                                    lower_delta=(p-qq[1])-L,upper_delta=(p-qq[0])-U,
                                    coverage_changed=bool((L<=z['truth']<=U)!=((p-qq[1])<=z['truth']<=(p-qq[0])))))
                        log=np.log(10*G/a);radius=sum(w*(sd*np.sqrt(2*rho*log/nn)+(7/3+3/np.sqrt(2))*width*log/nn) for w,sd,rho,width,nn in terms)
                        ends['empirical_Bernstein_Serfling',lv][0][r]=max(-1,p-radius);ends['empirical_Bernstein_Serfling',lv][1][r]=min(1,p+radius)
                    if (r+1)%500==0:print('BOOT_PROGRESS',scope,m,method,r+1,round(time.time()-start),flush=True)
                assert diffmax<2e-12,(scope,m,method,diffmax)
                rows=[];reports=[]
                for (name,level),(L,U) in ends.items():
                    row,reps=metrics(P,L,U,z['truth'],scope,m,method,name,level);rows.append(row);reports+=reps
                checkpoint.write_text(json.dumps(dict(rows=rows,reports=reports,precision=sens,reference_max_difference=diffmax),indent=2)+'\n',encoding='utf8')
                allrows+=rows;allreports+=reports;precision+=sens
                print('BOOT_CASE_DONE',scope,m,method,round(time.time()-start),flush=True)
    if ONLY_CASE is not None:
        print('SINGLE_CASE_WORKER_COMPLETE',ONLY_CASE,flush=True);return
    pd.DataFrame(allrows).to_csv(O/'interval_results.csv',index=False);pd.DataFrame(allreports).to_csv(O/'reporting_results.csv',index=False)
    pd.DataFrame(precision).to_csv(O/'bootstrap_precision.csv',index=False)
    assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in plan['preserved'].items())
    (O/'phase1_QA.json').write_text(json.dumps(dict(all_20_cases_complete=True,interval_rows=len(allrows),report_rows=len(allreports),B=999,precision_B=3999,
        prior_results_unchanged=True,design_points_and_variances_reproduced=True,bootstrap_is_approximate=True,
        no_new_original_SIPP_PSU_claim=True),indent=2)+'\n',encoding='utf8')
    print('PHASE1_COMPLETE',len(allrows),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-dir',type=Path,default=OLD);parser.add_argument('--out-dir',type=Path,default=O);parser.add_argument('--case',nargs=3)
    args=parser.parse_args();OLD=args.pilot_dir.resolve();O=args.out_dir.resolve();ONLY_CASE=args.case;main()

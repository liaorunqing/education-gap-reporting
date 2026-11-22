"""Prespecified mature controls in a retrospective, fully observed SIPP cohort.

This is not field performance: complete-cluster membership is learned after follow-up.
Current weights never enter predictions, standardization or sampling allocations.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import norm,beta
from acs184_reporting_methods import group_indices,integer_allocation,strat_estimate
from acs186_reporting_run import risk_bins

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acs_research215_2026-11-22'
PRIVATE=OUT/'private_sipp'
METHODS=['SRS_HT','SRS_prior_cells','SRS_prior_BA','proportional_prior_BA','working_Neyman_prior_BA']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(k,n):return (0. if not k else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k)))


def setup(d,full_prior,q):
    # Compute demographic coefficients only from the fixed retrospective frame's
    # historical background; evaluation membership is still a future-known condition.
    nat=d.nativity_prior.to_numpy(int);sex=d.ESEX_prior.to_numpy(int)
    age=d.TAGE_EHC_prior.to_numpy(float);w=d.WPFINWGT_prior.to_numpy(float)
    cell=2*np.where(age<=39,0,np.where(age<=59,1,2))+sex-1;g=2*cell+nat
    W=np.bincount(g,weights=w,minlength=12);assert (W>0).all()
    units,h=np.unique(d.cluster.astype(str),return_inverse=True);H=len(units)
    b=H*q[cell]*(2*nat-1)*w/W[g]
    prior=full_prior.copy()
    p_age=prior.TAGE_EHC_prior.to_numpy(float);p_sex=prior.ESEX_prior.to_numpy(int)
    pg=2*(2*np.where(p_age<=39,0,np.where(p_age<=59,1,2))+p_sex-1)+prior.nativity_prior.to_numpy(int)
    pw=prior.WPFINWGT_prior.to_numpy(float);py=prior.EEDUC_prior.ge(43).to_numpy(float)
    totals=np.bincount(pg,weights=pw,minlength=12);assert (totals>0).all()
    pc=np.bincount(pg,weights=pw*py,minlength=12)/totals
    agg=lambda x:np.bincount(h,weights=x,minlength=H)
    Fcell=agg(b*pc[g]);Flag=agg(b*d.EEDUC_prior.ge(43).to_numpy(float))
    Y=agg(b*d.EEDUC_current.ge(43).to_numpy(float))
    adult=np.bincount(h,minlength=H);any_foreign=agg(nat)>0;any_older=agg(age>=60)>0
    stratum=4*any_foreign.astype(int)+2*any_older.astype(int)+(adult>1).astype(int)
    working=agg(b*b*pc[g]*(1-pc[g]));fine=risk_bins(stratum,working)
    return dict(Y=Y,Fcell=Fcell,Flag=Flag,H=H,truth=float(Y.mean()),groups=group_indices(stratum),
                fine_groups=group_indices(fine),working=working,adult=adult,foreign=agg(nat),
                source_prior_cells=pc.tolist(),current_weights_used=False,prior_cell_weights=W.tolist())


def simulate(z,m):
    H=z['H'];R=2000;assert H>m
    groups=z['groups'];N=np.array([len(ix) for ix in groups]);pn=integer_allocation(N,np.ones(len(N)),m)
    fine=z['fine_groups'];FN=np.array([len(ix) for ix in fine]);risk=np.array([z['working'][ix].mean() for ix in fine])
    nn=integer_allocation(FN,np.sqrt(risk),m)
    Y=z['Y'];Fc=z['Fcell'];Fl=z['Flag'];residual=Y-Fl
    points=np.empty((R,len(METHODS)));var=points.copy();resources=points.copy();count=np.full_like(points,m,dtype=int)
    zero_var=np.zeros(len(METHODS),int)
    for r in range(R):
        rng=np.random.default_rng(739+m*10000+r);si=rng.choice(H,m,replace=False)
        for j,F in [(0,np.zeros(H)),(1,Fc),(2,Fl)]:
            residual_j=Y-F;points[r,j]=F.mean()+residual_j[si].mean()
            var[r,j]=(1-m/H)*residual_j[si].var(ddof=1)/m;resources[r,j]=z['adult'][si].sum()
        for j,gg,n in [(3,groups,pn),(4,fine,nn)]:
            rng=np.random.default_rng(739+m*10000+100000000+r)
            p,v,sel=strat_estimate(gg,n,residual[:,None],rng)
            assert len(sel)==m and len(np.unique(sel))==m
            points[r,j]=Fl.mean()+p[0];var[r,j]=v[0];resources[r,j]=z['adult'][sel].sum()
    assert np.isfinite(points).all() and np.isfinite(var).all() and (var>=0).all()
    return dict(point=points,variance=var,resources=resources,count=count,methods=np.array(METHODS),truth=z['truth'],m=m,R=R,H=H)


def main():
    plan=json.loads((OUT/'sipp_pilot_plan.json').read_text(encoding='utf8'))
    assert plan['conditional_pilot']['repetitions']==2000
    gate=json.loads((OUT/'sipp_audit/feasibility_QA.json').read_text());assert gate['conditional_pilot_gate_passed']
    dest=OUT/'sipp_pilot';assert not dest.exists();dest.mkdir()
    d=pd.read_csv(PRIVATE/'linked_cohort_private.csv.gz',dtype={'SSUID':'string','SHHADID_prior':'string','SHHADID_current':'string'})
    full_prior=d.copy() # Exactly one row for each eligible baseline person, including later nonmatches.
    q=np.array(json.loads((OUT/'sipp_prior_background_freeze.json').read_text())['q'])
    scopes={'all_released':d.loc[d.complete_cluster].copy()}
    both_reported=d.AEDUC_prior.eq(1)&d.AEDUC_current.eq(1)&d.observed_for_pilot
    eligible_cluster=both_reported.groupby(d.cluster).all()
    scopes['reported_only']=d.loc[d.cluster.isin(eligible_cluster.index[eligible_cluster])].copy()
    pointrows=[];reportrows=[];paired=[];scopeinfo=[]
    for scope,frame in scopes.items():
        z=setup(frame,full_prior,q);assert z['H']>=1000
        scopeinfo.append(dict(scope=scope,clusters=z['H'],adults=len(frame),foreign_adults=int(frame.nativity_prior.eq(1).sum()),
            fixed_prior_weight_sum=float(frame.WPFINWGT_prior.sum()),target_gap=z['truth'],
            target='Retrospective fully linked baseline clusters; prior weights and prior-age/sex reference, not current residents',
            source_prior_cells=z['source_prior_cells'],current_weights_used=False))
        # Store private coefficients only outside the release tree, to permit audit.
        for m in [300,1000]:
            result=simulate(z,m);np.savez_compressed(dest/f'{scope}_{m}_simulation.npz',**result)
            error=result['point']-z['truth'];se=np.sqrt(result['variance'])
            for j,method in enumerate(METHODS):
                e=error[:,j];cov=(abs(e)<=norm.ppf(.975)*se[:,j]);lo,hi=cp(int(cov.sum()),len(e))
                pointrows.append(dict(scope=scope,m=m,method=method,R=len(e),H=z['H'],adults=len(frame),truth=z['truth'],
                    MSE=float(np.mean(e**2)),MSE_MCSE=float((e**2).std(ddof=1)/np.sqrt(len(e))),bias=float(e.mean()),MAE=float(abs(e).mean()),
                    coverage95=float(cov.mean()),coverage95_MC95_low=lo,coverage95_MC95_high=hi,
                    coverage90=float((abs(e)<=norm.ppf(.95)*se[:,j]).mean()),width95=float((2*norm.ppf(.975)*se[:,j]).mean()),
                    mean_adult_labels=float(result['resources'][:,j].mean()),zero_variance_draws=int((result['variance'][:,j]==0).sum())))
                for tau in [.01,.02,.05]:
                    L=result['point'][:,j]-norm.ppf(.975)*se[:,j];U=result['point'][:,j]+norm.ppf(.975)*se[:,j]
                    c=np.full(len(e),3);c[L>tau]=0;c[U< -tau]=1;c[(L>=-tau)&(U<=tau)]=2
                    truthclass=0 if z['truth']>tau else 1 if z['truth']< -tau else 2
                    correct=int((c==truthclass).sum());wrong=int(((c!=3)&(c!=truthclass)).sum());un=int((c==3).sum())
                    reportrows.append(dict(scope=scope,m=m,method=method,tau=tau,R=len(e),truth=z['truth'],correct=correct,wrong=wrong,unresolved=un,
                        correct_yield=correct/len(e),wrong_per_attempt=wrong/len(e),wrong_given_report=wrong/(correct+wrong) if correct+wrong else np.nan))
            for name,j0,j1 in [('prior_BA_vs_cells',1,2),('prior_BA_vs_direct',0,2),('allocation_vs_same_prior_BA',2,4)]:
                diff=error[:,j1]**2-error[:,j0]**2
                paired.append(dict(scope=scope,m=m,contrast=name,MSE_difference=float(diff.mean()),paired_MSE_difference_MCSE=float(diff.std(ddof=1)/np.sqrt(len(diff))),
                    relative_MSE_reduction=float(1-np.mean(error[:,j1]**2)/np.mean(error[:,j0]**2))))
            print('SIPP_CONDITIONAL_PILOT',scope,m,'DONE',flush=True)
    pd.DataFrame(pointrows).to_csv(dest/'point_interval_results.csv',index=False)
    pd.DataFrame(reportrows).to_csv(dest/'reporting_results.csv',index=False)
    pd.DataFrame(paired).to_csv(dest/'paired_comparisons.csv',index=False)
    (dest/'scope_and_QA.json').write_text(json.dumps(dict(scopes=scopeinfo,plan_sha256=sha(OUT/'sipp_pilot_plan.json'),
         current_weights_never_used=True,all_household_counts_exact=True,all_samples_unique=True,
         future_complete_case_membership_not_available_preselection=True,design_coverage_is_conditional=True,
         field_cost_saving_claimed=False,new_verified_degree_labels_claimed=False,independent_new_ACS_holdout_claimed=False),indent=2)+'\n',encoding='utf8')
    print(pd.DataFrame(pointrows)[['scope','m','method','MSE','coverage95','width95']].to_string(index=False))


if __name__=='__main__':main()

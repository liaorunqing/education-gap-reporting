"""Frozen, count-explicit reporting evaluation; conditional Monte Carlo intervals."""
import argparse,json,pickle
import numpy as np
import pandas as pd
from scipy.stats import norm,beta,spearmanr
from acs183_reporting_data import ROOT,OUT,FINAL,SEEN,loadjson,dump,sha,utc
from acs186_reporting_run import METHODS
from acs184_reporting_methods import integer_allocation

def cp(k,n,alpha=.05):
    if n==0:return np.nan,np.nan
    return (0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k)))

def categories(point,se,tau,ci=.95):
    h=norm.ppf((1+ci)/2)*se;L=point-h;U=point+h
    c=np.full(point.shape,3,int);c[L>tau]=0;c[U< -tau]=1;c[(L>=-tau)&(U<=tau)]=2
    return c

def report_counts(point,se,truth,tau,ci=.95):
    c=categories(point,se,tau,ci);truth_class=0 if truth>tau else 1 if truth< -tau else 2
    claims=c!=3;wrong=claims&(c!=truth_class);n=len(c);K=int(claims.sum());E=int(wrong.sum());lo,hi=cp(E,K)
    return dict(claim_count=K,wrong_count=E,unresolved_count=n-K,claim_rate=K/n,wrong_per_attempt=E/n,
      wrong_given_claim=E/K if K else np.nan,wrong_given_claim_MC95_low=lo,wrong_given_claim_MC95_high=hi,
      advantage_count=int((c==0).sum()),disadvantage_count=int((c==1).sum()),within_band_count=int((c==2).sum()))

def gaussian_reference(truth,bias,sd,half,tau):
    if sd<=0:
        c=int(categories(np.array([truth+bias]),np.array([half/norm.ppf(.975)]),tau)[0]);tc=0 if truth>tau else 1 if truth< -tau else 2
        return float(c!=3),float(c!=3 and c!=tc)
    loc=truth+bias
    probs=np.array([norm.sf((tau+half-loc)/sd),norm.cdf((-tau-half-loc)/sd),
      max(0.,norm.cdf((tau-half-loc)/sd)-norm.cdf((-tau+half-loc)/sd)) if half<=tau else 0.])
    true_class=0 if truth>tau else 1 if truth< -tau else 2
    return float(probs.sum()),float(probs.sum()-probs[true_class])

def contract():
    path=OUT/'analysis_and_figure_contract.json';assert not path.exists()
    c=report_counts(np.array([0.,0.]),np.array([1.,1.]),0,.02);assert c['claim_count']==0 and np.isnan(c['wrong_given_claim'])
    assert cp(0,100)[1]>0
    dump('analysis_and_figure_contract.json',dict(utc=utc(),primary_endpoint='standardized_gap',secondary='raw_gap',primary_budget=1000,
      CI=[.90,.95],reporting_tau=[.01,.02,.05],primary_reporting_tau=.02,
      claims='95%CI wholly above+tau/below-tau/inside[-tau,tau], otherwise unresolved; wrong if finite-frame target not in claimed category',
      denominators='All attempts, all claims and unresolved shown explicitly. Undefined conditional fractions remain missing, never0.',
      normal_reference='Within-case same-draw bias/empiricalSD and mean reported variance. Descriptive explanation, not out-of-sample calibration.',
      matched_diagnostics='Gaussian reference with common true gap0.05; signal/noise and MSE-equivalent residual sample size; associations noncausal',
      selection='At primary budget development only, minimum equal-state MSE among methods with average95coverage>=.94 and minimumstate>=.90. Report if none; final comparisons all methods retained.',
      figure_templates=['F1 raw/standardized gaps with separate original-ACS replicate intervals; foreign share','F2 budget curves for reporting yield/conditional error/coverage/width','F3 gap-to-band noise relation and Gaussian matched reference, plus residual-risk learnability','F4 final-state method heatmaps vs frozen development-selected mature comparator'],
      fixed_display='No outcome-selected states, automatic per-panel exaggerating axis crops or hidden failed methods. All16methods in source tables; compact main panels use frozen development selection.',
      benchmarks='Compare same-method same-state paired draw squared errors. MC intervals are conditional simulation precision, not population confidence intervals.',
      tests=dict(zero_claim_missing=True,zero_errors_positive_upper_bound=True)))

def analyze(phase):
    folder=OUT/f'draws/{"dev16" if phase=="dev" else phase}';paths=sorted(folder.glob('*.npz'));assert paths
    rows=[];reports=[];reference=[];paired=[];description=[];risk_validation=[];oracles=[]
    state_arrays={}
    for p in paths:
        with np.load(p) as z:point=z['point'];variance=z['variance'];truth=z['truth'];year=int(z['year']);s=str(z['state']);m=int(z['m']);resources=z['resources']
        R=len(point);se=np.sqrt(variance);desc=loadjson(OUT/f'scores/{year}_{s}_description.json')
        with np.load(OUT/f'scores/{year}_{s}.npz') as zs:resid=zs['T'][:,0]
        design=pickle.loads((OUT/f'designs/{year}_{s}_{m}.pkl').read_bytes());H=len(resid)
        for group_name,groups in [('coarse8',design['groups']),('fine_working',design['working_groups'])]:
            N=np.array([len(ix) for ix in groups]);sv=np.array([resid[ix].var(ddof=1) if len(ix)>1 else 0. for ix in groups])
            allocation=integer_allocation(N,np.sqrt(np.maximum(sv,1e-20)),m)
            vv=float(np.sum(np.where(N>0,(N/H)**2*(1-allocation/np.maximum(N,1))*sv/np.maximum(allocation,1),0)))
            oracles.append(dict(phase=phase,year=year,state=s,m=m,grouping=group_name,oracle_stratified_variance=vv,scope='Full-target-label diagnostic within fixed grouping, not deployable or a universal lower bound'))
        if m==1000:description.append(desc)
        for ei,endpoint in enumerate(['standardized','raw']):
            error=point[:,:,ei]-truth[ei];sq=error**2;state_arrays[year,s,m,endpoint]=sq
            for j,name in enumerate(METHODS):
                e=error[:,j];v=variance[:,j,ei];sd=float(e.std(ddof=1));bias=float(e.mean());mse=float(np.mean(e*e))
                c95=np.abs(e)<=norm.ppf(.975)*np.sqrt(v);c90=np.abs(e)<=norm.ppf(.95)*np.sqrt(v)
                lo,hi=cp(int(c95.sum()),R)
                row=dict(phase=phase,year=year,state=s,m=m,endpoint=endpoint,method=name,R=R,truth=float(truth[ei]),MSE=mse,
                  MSE_MCSE=float((e*e).std(ddof=1)/np.sqrt(R)),bias=bias,MAE=float(abs(e).mean()),empirical_SD=sd,
                  coverage95=float(c95.mean()),coverage95_MC95_low=lo,coverage95_MC95_high=hi,coverage90=float(c90.mean()),
                  width95=float(2*norm.ppf(.975)*np.sqrt(v).mean()),width90=float(2*norm.ppf(.95)*np.sqrt(v).mean()),
                  mean_estimated_variance=float(v.mean()),mean_adult_labels=float(resources[:,j,0].mean()),mean_foreign_labels=float(resources[:,j,1].mean()),
                  MSE_equivalent_residual_n=float(1/(mse/desc['population_residual_variance']+1/desc['H'])) if endpoint=='standardized' else np.nan,
                  distance_to_reporting_boundary=float(min(abs(truth[ei]-.02),abs(truth[ei]+.02))),foreign_weight_share=desc['foreign_weight_share'])
                rows.append(row)
                for tau in [.01,.02,.05]:
                    rc=report_counts(point[:,j,ei],se[:,j,ei],truth[ei],tau)
                    reports.append(dict(phase=phase,year=year,state=s,m=m,endpoint=endpoint,method=name,tau=tau,R=R,truth=float(truth[ei]),**rc))
                    if tau==.02:
                        predclaim,predwrong=gaussian_reference(truth[ei],bias,sd,np.sqrt(v.mean())*norm.ppf(.975),tau)
                        matchclaim,matchwrong=gaussian_reference(.05,0,sd,np.sqrt(v.mean())*norm.ppf(.975),tau)
                        reference.append(dict(phase=phase,year=year,state=s,m=m,endpoint=endpoint,method=name,observed_claim_rate=rc['claim_rate'],observed_wrong_per_attempt=rc['wrong_per_attempt'],normal_claim_rate=predclaim,normal_wrong_per_attempt=predwrong,matched_gap05_claim_rate=matchclaim,matched_gap05_wrong_per_attempt=matchwrong,
                          standardized_error_mean=bias/np.sqrt(v.mean()),standardized_error_SD=sd/np.sqrt(v.mean())))
        if m==1000:
            with np.load(OUT/f'scores/{year}_{s}.npz') as z:ry=z['T'][:,0]**2
            with np.load(OUT/f'risk_predictions/{year}_{s}.npz') as z:pred=z['risk']
            for j,name in enumerate(['coarse','RF','HGB']):
                risk_validation.append(dict(phase=phase,year=year,state=s,model=name,risk_prediction_MSE=float(np.mean((ry-pred[:,j])**2)),constant_MSE=float(np.mean((ry-ry.mean())**2)),mean_predicted_risk=float(pred[:,j].mean()),mean_actual_risk=float(ry.mean()),spearman=float(spearmanr(ry,pred[:,j]).statistic)))
    df=pd.DataFrame(rows);rp=pd.DataFrame(reports);rf=pd.DataFrame(reference)
    dest=OUT/f'analysis/{phase}';dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'point_interval_results.csv',index=False);rp.to_csv(dest/'reporting_results.csv',index=False);rf.to_csv(dest/'normal_and_matched_reference.csv',index=False)
    pd.DataFrame(description).to_csv(dest/'population_context.csv',index=False);pd.DataFrame(risk_validation).to_csv(dest/'risk_transport.csv',index=False)
    pd.DataFrame(oracles).to_csv(dest/'oracle_grouping_diagnostic.csv',index=False)
    decomposition=[]
    for s in FINAL:
        years=[2022,2023] if phase=='dev' else [2023,2024]
        p0=OUT/f'scores/{years[0]}_{s}_description.json';p1=OUT/f'scores/{years[1]}_{s}_description.json'
        if not p0.exists() or not p1.exists():continue
        a0=loadjson(p0);a1=loadjson(p1)
        if 'cell_weight_totals' not in a0 or 'cell_weight_totals' not in a1:continue
        w0=np.array(a0['cell_weight_totals']).reshape(6,2);w1=np.array(a1['cell_weight_totals']).reshape(6,2)
        mu0=np.array(a0['cell_outcome_means']).reshape(6,2);mu1=np.array(a1['cell_outcome_means']).reshape(6,2)
        p0=w0/w0.sum(0);p1=w1/w1.sum(0)
        composition=(.5*(mu0+mu1)*(p1-p0)).sum(0);within=(.5*(p0+p1)*(mu1-mu0)).sum(0)
        change=a1['raw_gap']-a0['raw_gap'];between=float(composition[1]-composition[0]);inside=float(within[1]-within[0])
        assert abs(change-between-inside)<1e-10
        decomposition.append(dict(state=s,year0=years[0],year1=years[1],raw_gap_change=change,age_sex_composition_component=between,within_age_sex_component=inside,scope='Descriptive symmetric product decomposition, not a causal mechanism or linked individual change'))
    pd.DataFrame(decomposition).to_csv(dest/'observed_change_decomposition.csv',index=False)
    # Matched observational diagnostic: all pairs meeting fixed calipers, no relaxed fallback.
    matched=[];ctx=pd.DataFrame(description)
    ctx['expected_SRS_foreign_ESS']=1000/ctx.H*ctx.foreign_weight_ESS
    for ia in range(len(ctx)):
        for ib in range(ia+1,len(ctx)):
            a=ctx.iloc[ia];b=ctx.iloc[ib]
            ratio=a.expected_SRS_foreign_ESS/b.expected_SRS_foreign_ESS
            if a.state==b.state or abs(a.standardized_gap-b.standardized_gap)>.02 or not .8<=ratio<=1.25:continue
            for method in METHODS:
                x=df[(df.m==1000)&(df.endpoint=='standardized')&(df.method==method)]
                xa=x[(x.year==a.year)&(x.state==a.state)].iloc[0];xb=x[(x.year==b.year)&(x.state==b.state)].iloc[0]
                matched.append(dict(state_a=a.state,year_a=a.year,state_b=b.state,year_b=b.year,method=method,gap_difference=float(a.standardized_gap-b.standardized_gap),expected_SRS_foreign_ESS_ratio=float(ratio),MSE_a=xa.MSE,MSE_b=xb.MSE,coverage_a=xa.coverage95,coverage_b=xb.coverage95,scope='Descriptive gap/expected-SRS-reference-ESS matching; pairs overlap; no causal or independent-pair inference'))
    pd.DataFrame(matched,columns=['state_a','year_a','state_b','year_b','method','gap_difference','expected_SRS_foreign_ESS_ratio','MSE_a','MSE_b','coverage_a','coverage_b','scope']).to_csv(dest/'matched_gap_and_reference_ESS.csv',index=False)
    summary=[]
    for (m,endpoint),part in df.groupby(['m','endpoint']):
        baseparts=[v for (y,s,mm,ep),v in state_arrays.items() if mm==m and ep==endpoint]
        base=np.mean([a[:,2].mean() for a in baseparts])
        for j,name in enumerate(METHODS):
            x=part[part.method==name];mse=float(x.MSE.mean());ifvar=sum(np.var(mse/base**2*a[:,2]-a[:,j]/base,ddof=1)/len(a) for a in baseparts)/len(baseparts)**2
            a=rp[(rp.m==m)&(rp.endpoint==endpoint)&(rp.method==name)&(rp.tau==.02)]
            summary.append(dict(phase=phase,m=m,endpoint=endpoint,method=name,state_years=len(x),equal_state_MSE=mse,
              reduction_vs_SRS_HGB_percent=100*(1-mse/base),paired_MCSE_pp=100*np.sqrt(ifvar),states_improved_vs_SRS_HGB=sum(v[:,j].mean()<v[:,2].mean() for v in baseparts),
              equal_state_coverage95=x.coverage95.mean(),minimum_state_coverage95=x.coverage95.min(),equal_state_width95=x.width95.mean(),
              equal_state_claim_rate=a.claim_rate.mean(),equal_state_wrong_per_attempt=a.wrong_per_attempt.mean(),
              total_claims=int(a.claim_count.sum()),total_wrong_claims=int(a.wrong_count.sum()),pooled_wrong_given_claim=a.wrong_count.sum()/a.claim_count.sum() if a.claim_count.sum() else np.nan,
              equal_state_adult_labels=x.mean_adult_labels.mean(),equal_state_foreign_labels=x.mean_foreign_labels.mean()))
    summary=pd.DataFrame(summary);summary.to_csv(dest/'equal_state_summary.csv',index=False)
    if phase=='dev':
        primary=summary[(summary.m==1000)&(summary.endpoint=='standardized')]
        ok=primary[(primary.equal_state_coverage95>=.94)&(primary.minimum_state_coverage95>=.90)]
        selected=None if ok.empty else str(ok.sort_values('equal_state_MSE').iloc[0].method)
        mature=ok[ok.method.isin(['SRS_HT','SRS_historical_cell','SRS_HGB','SRS_GREG','SRS_GREG_cells','proportional_HGB','Neyman_HGB','two_wave_HGB','Neyman_working_fine','CPS_working','CPS_working_GREG'])]
        comparator=None if mature.empty else str(mature.sort_values('equal_state_MSE').iloc[0].method)
        learned=ok[ok.method.isin(['CPS_RF','CPS_HGB','CPS_robust_HGB','CPS_RF_GREG'])]
        selected_learned=None if learned.empty else str(learned.sort_values('equal_state_MSE').iloc[0].method)
        dump('development_selection.json',dict(utc=utc(),selected_method=selected,selected_learned_method=selected_learned,mature_comparator=comparator,coverage_screen='.94equal-state,.90minimumstate',phase='development_only',all_final_methods_retained=True,primary_contrasts=['mature comparator versus SRS_HGB','development-selected learned-risk bundle versus mature comparator'],allocation_only_contrasts=['CPS_RF versus CPS_working','CPS_RF_GREG versus CPS_working_GREG']))
    else:
        selection=loadjson(OUT/'development_selection.json');comparator=selection['mature_comparator'] or 'SRS_HGB';k=METHODS.index(comparator)
        for endpoint in ['standardized','raw']:
            arrays=[v for (y,s,m,ep),v in state_arrays.items() if m==1000 and ep==endpoint];base=np.mean([a[:,k].mean() for a in arrays])
            for j,name in enumerate(METHODS):
                value=np.mean([a[:,j].mean() for a in arrays]);iv=sum(np.var(value/base**2*a[:,k]-a[:,j]/base,ddof=1)/len(a) for a in arrays)/len(arrays)**2
                paired.append(dict(endpoint=endpoint,method=name,comparator=comparator,reduction_percent=100*(1-value/base),paired_MCSE_pp=100*np.sqrt(iv),states_improved=sum(a[:,j].mean()<a[:,k].mean() for a in arrays)))
        pd.DataFrame(paired).to_csv(dest/'frozen_comparator_comparison.csv',index=False)
    assert (rp.wrong_count<=rp.claim_count).all() and (rp.claim_count+rp.unresolved_count==rp.R).all()
    dump(f'analysis/{phase}/analysis_QA.json',dict(utc=utc(),draw_files=len(paths),state_years=len(description),result_rows=len(df),report_rows=len(rp),all_counts_consistent=True,zero_claim_conditional_missing=bool(rp.loc[rp.claim_count.eq(0),'wrong_given_claim'].isna().all()),no_new_sampling=True))
    print('ANALYSIS_COMPLETE',phase,len(paths),len(df),flush=True)
    print(summary[(summary.m==1000)&(summary.endpoint=='standardized')].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','dev','final']);a=p.parse_args()
    if a.mode=='contract':contract()
    else:analyze(a.mode)

"""Known-prior-outcome masking diagnostic; never estimates missing later education."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'outputs/acs_research215_2026-11-22';O=ROOT/'outputs/acs_followup242_2026-11-22';STRONG=False

def features(d):
    # No education, education-allocation flag, observation indicator, follow-up input or ID predictor.
    units,h=np.unique(d.cluster.astype(str),return_inverse=True);H=len(units)
    age=d.TAGE_EHC_prior.to_numpy(float);sex=d.ESEX_prior.to_numpy(int);nat=d.nativity_prior.to_numpy(int);w=d.WPFINWGT_prior.to_numpy(float)
    n=np.bincount(h,minlength=H);agg=lambda x:np.bincount(h,weights=x,minlength=H)/n
    Xh=np.column_stack([n,agg(age),agg(sex==2),agg(nat),agg(age>=60),agg(np.log(w))])
    X=np.column_stack([age,sex==2,nat,d.ECITIZEN_prior.to_numpy(float),np.log(w),Xh[h]])
    assert np.isfinite(X).all() and np.isfinite(Xh).all()
    return X,Xh,h

def main():
    assert (O/'phase1_QA.json').exists(),'Finish and verify interval phase before correction comparisons.'
    assert not (O/('membership_strong_sensitivity.csv' if STRONG else 'membership_results.csv')).exists()
    cfg=json.loads((O/'analysis_plan.json').read_text());d=pd.read_csv(OLD/'private_sipp/linked_cohort_private.csv.gz',dtype={'SSUID':'string','SHHADID_prior':'string','SHHADID_current':'string'})
    X,Xh,h=features(d);H=len(Xh);y=d.EEDUC_prior.ge(43).to_numpy(float);obs=d.complete_cluster.to_numpy(bool)
    Rh=np.bincount(h,weights=obs)/np.bincount(h);assert np.isin(Rh,[0,1]).all()
    hidden=d.copy();hidden.loc[~obs,'EEDUC_prior']=np.where(y[~obs]>0,31,43)
    XX,HH,hh=features(hidden);assert np.array_equal(X,XX) and np.array_equal(Xh,HH)
    assert np.array_equal(y[obs],hidden.EEDUC_prior.ge(43).to_numpy(float)[obs])
    q=np.array(json.loads((OLD/'sipp_prior_background_freeze.json').read_text())['q'])
    age=d.TAGE_EHC_prior.to_numpy();sex=d.ESEX_prior.to_numpy(int);nat=d.nativity_prior.to_numpy(int);w=d.WPFINWGT_prior.to_numpy(float)
    cell=2*np.where(age<=39,0,np.where(age<=59,1,2))+sex-1;g=2*cell+nat
    W=np.bincount(g,weights=w,minlength=12);a=q[cell]*(2*nat-1)*w/W[g];truth=float(a@y)
    WR=np.bincount(g,weights=w*obs,minlength=12);assert (WR>0).all()
    cc=float(np.sum(q*((np.bincount(g,weights=w*y*obs,minlength=12)/WR)[1::2]-(np.bincount(g,weights=w*y*obs,minlength=12)/WR)[::2])))
    calibrated=float(np.sum(a*obs*y*W[g]/WR[g]));assert abs(calibrated-cc)<1e-14
    rows=[];weights=[]
    for seed in range(739,744):
        ph=np.empty(H);mu=np.empty(len(d));foldcount=0
        for tr,te in KFold(3,shuffle=True,random_state=seed).split(Xh):
            lr=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=739))
            lr.fit(Xh[tr],Rh[tr]);assert lr[-1].n_iter_.max()<1000
            ph[te]=lr.predict_proba(Xh[te])[:,1]
            train=np.isin(h,tr)&obs;test=np.isin(h,te)
            assert not np.intersect1d(h[train],h[test]).size
            model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=739)
            model.fit(X[train],y[train],sample_weight=w[train]/w[train].mean());mu[test]=np.clip(model.predict(X[test]),0,1)
            foldcount+=1
        for lower,upper in [(.05,.95),(.02,.98)]:
            raw=ph[h];p=np.clip(raw,lower,upper)
            ipw=float(np.sum(a*obs*y/p));aipw=float(np.sum(a*(mu+obs*(y-mu)/p)))
            method_values=[('complete_case',cc),('12_cell_calibration',calibrated),('IPW_logit',ipw),('AIPW_logit_HGB',aipw)]
            if STRONG:
                response_totals=np.bincount(g,weights=w*obs/p,minlength=12);assert (response_totals>0).all()
                scale=W[g]/response_totals[g]
                method_values += [('IPW_cell_normalized',float(np.sum(a*obs*y/p*scale))),
                    ('AIPW_cell_calibrated',float(np.sum(a*(mu+obs*(y-mu)/p*scale)))),('OR_HGB',float(a@mu))]
            for method,est in method_values:
                rows.append(dict(seed=seed,clip_lower=lower,clip_upper=upper,method=method,estimate=est,reference=truth,error=est-truth,
                    absolute_error=abs(est-truth),squared_error=(est-truth)**2,interpretation='Cross-fit splitting variation on one known-prior-outcome mask; not independent surveys'))
            for nativity in [0,1]:
                sel=obs&(nat==nativity);ww=w[sel]/p[sel]
                weights.append(dict(seed=seed,clip_lower=lower,nativity=nativity,min_response_probability=float(raw.min()),max_response_probability=float(raw.max()),
                    fraction_clipped=float(((raw<lower)|(raw>upper)).mean()),weighted_ESS=float(ww.sum()**2/(ww@ww)),observed_adults=int(sel.sum())))
        print('MEMBERSHIP_CROSSFIT_DONE',seed,flush=True)
    if STRONG:
        new=pd.DataFrame(rows);old=pd.read_csv(O/'membership_results.csv')
        check=new[new.method.isin(old.method.unique())].reset_index(drop=True)
        pd.testing.assert_frame_equal(old,check,check_exact=False,rtol=2e-12,atol=2e-14)
        extra=new[~new.method.isin(old.method.unique())]
        extra.to_csv(O/'membership_strong_sensitivity.csv',index=False)
        (O/'membership_strong_QA.json').write_text(json.dumps(dict(original_40_rows_reproduced=True,original_results_not_overwritten=True,added_rows=len(extra),
            calibrated_totals='12 known age/sex/nativity prior-weight margins; normalization uses response indicators and background, no masked education'),indent=2)+'\n',encoding='utf8')
        print(extra.query('clip_lower==.05').groupby('method')[['estimate','absolute_error']].agg(['mean','min','max']).to_string());return
    pd.DataFrame(rows).to_csv(O/'membership_results.csv',index=False);pd.DataFrame(weights).to_csv(O/'membership_weight_diagnostics.csv',index=False)
    (O/'phase2_QA.json').write_text(json.dumps(dict(known_prior_target=truth,standardized_complete_case=cc,cell_calibration_identical=calibrated,
        algebraic_duplicate_not_counted_as_improvement=True,masked_target_perturbation_leaves_all_features_and_training_labels_unchanged=True,
        folds_disjoint_by_household=True,all_logistic_fits_converged=True,followup_education_and_weights_never_used=True,
        actual_missing_followup_education_estimated=False,missing_at_random_verified=False,new_population_validation=False,
        five_splits_are_fitting_variation_not_population_replication=True),indent=2)+'\n',encoding='utf8')
    print(pd.DataFrame(rows).query('clip_lower==.05').groupby('method')[['estimate','absolute_error']].agg(['mean','min','max']).to_string())

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-dir',type=Path,default=OLD);parser.add_argument('--out-dir',type=Path,default=O);parser.add_argument('--strong-sensitivity',action='store_true')
    args=parser.parse_args();OLD=args.pilot_dir.resolve();O=args.out_dir.resolve();STRONG=args.strong_sensitivity;main()

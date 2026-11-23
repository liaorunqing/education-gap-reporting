"""Supplementary sensitivity bounds for unobserved later labels, not CIs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/acs_research215_2026-11-22';O=R/'outputs/acs_followup242_2026-11-22'

def gain_bound(a,w,eligible,budget):
    benefit=abs(a[eligible]);cost=w[eligible]
    ix=np.argsort(-(benefit/cost),kind='stable');benefit=benefit[ix];cost=cost[ix]
    take=np.minimum(cost,np.maximum(0,budget-np.r_[0,np.cumsum(cost)[:-1]]))
    return float(np.sum(benefit*take/cost))

def main(plan_only):
    path=O/'missing_change_sensitivity_plan.json'
    if plan_only:
        assert not path.exists()
        cfg=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Supplementary exploratory stress test added after initial interval cases, before this calculation',
            grid=[0,.01,.02,.05,.10,1],constraint='Within each baseline nativity group, at most r of the missing-record PRIOR WEIGHT mass may change binary BA status',
            target='Hypothetical later binary-label contrast of all baseline records with fixed prior weights and standard; current eligibility unresolved',
            bounds='Fractional-knapsack linear-relaxation outer bounds; direction-specific flips; not confidence intervals',
            assumptions='r not estimated or validated by respondents; observed current BA and missing prior BA used; no imputed reference truth or current-resident claim')
        path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8');print('CHANGE_SENSITIVITY_PLAN_FIXED');return
    assert (O/'phase2_QA.json').exists()
    cfg=json.loads(path.read_text());d=pd.read_csv(OLD/'private_sipp/linked_cohort_private.csv.gz')
    q=np.array(json.loads((OLD/'sipp_prior_background_freeze.json').read_text())['q']);nat=d.nativity_prior.to_numpy(int)
    age=d.TAGE_EHC_prior.to_numpy();sex=d.ESEX_prior.to_numpy(int);w=d.WPFINWGT_prior.to_numpy()
    cell=2*np.where(age<=39,0,np.where(age<=59,1,2))+sex-1;g=2*cell+nat;W=np.bincount(g,weights=w,minlength=12)
    a=q[cell]*(2*nat-1)*w/W[g];observed=d.observed_for_pilot.to_numpy(bool);missing=~observed
    prior=d.EEDUC_prior.ge(43).to_numpy(float);current=d.EEDUC_current.ge(43).to_numpy(float)
    carry=float(np.sum(a*np.where(observed,current,prior)));rows=[]
    for rr in cfg['grid']:
        lower=carry;upper=carry
        for group in [0,1]:
            groupmiss=missing&(nat==group);budget=float(rr*w[groupmiss].sum())
            positive=groupmiss&(((a>0)&(prior==0))|((a<0)&(prior==1)))
            negative=groupmiss&(((a>0)&(prior==1))|((a<0)&(prior==0)))
            upper+=gain_bound(a,w,positive,budget);lower-=gain_bound(a,w,negative,budget)
        rows.append(dict(max_missing_prior_weight_change_fraction=rr,lower=lower,upper=upper,carry_forward=carry,
            all_above_zero=lower>0,all_above_2pp=lower>.02,interpretation='Assumption-dependent outer bounds; not confidence limits or verified resident effects'))
    unrestricted_lo=float(np.sum(a[observed]*current[observed])+np.minimum(a[missing],0).sum())
    unrestricted_hi=float(np.sum(a[observed]*current[observed])+np.maximum(a[missing],0).sum())
    assert abs(rows[-1]['lower']-unrestricted_lo)<1e-12 and abs(rows[-1]['upper']-unrestricted_hi)<1e-12
    assert all(rows[k]['lower']>=rows[k+1]['lower']-1e-12 and rows[k]['upper']<=rows[k+1]['upper']+1e-12 for k in range(len(rows)-1))
    pd.DataFrame(rows).to_csv(O/'missing_change_bounds.csv',index=False)
    (O/'missing_change_QA.json').write_text(json.dumps(dict(missing_records=int(missing.sum()),carry_forward=carry,
        unrestricted_lower=unrestricted_lo,unrestricted_upper=unrestricted_hi,unrestricted_exact_binary_bounds_reproduced=True,
        nested_bounds=True,no_sampling_confidence_level_claim=True,missing_change_fraction_not_validated=True,current_eligibility_not_resolved=True),indent=2)+'\n',encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--pilot-dir',type=Path,default=OLD);p.add_argument('--out-dir',type=Path,default=O)
    args=p.parse_args();OLD=args.pilot_dir.resolve();O=args.out_dir.resolve();main(args.plan)

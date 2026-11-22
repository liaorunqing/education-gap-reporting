"""Transparent post-pilot diagnostics; no reranking or outcome-selected exclusions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acs_research215_2026-11-22'


def main():
    d=pd.read_csv(OUT/'private_sipp/linked_cohort_private.csv.gz',dtype={'SSUID':'string','SHHADID_prior':'string','SHHADID_current':'string'})
    q=np.array(json.loads((OUT/'sipp_prior_background_freeze.json').read_text())['q'])
    age=d.TAGE_EHC_prior.to_numpy(float);sex=d.ESEX_prior.to_numpy(int);nat=d.nativity_prior.to_numpy(int);w=d.WPFINWGT_prior.to_numpy(float)
    cell=2*np.where(age<=39,0,np.where(age<=59,1,2))+sex-1;g=2*cell+nat
    W=np.bincount(g,weights=w,minlength=12);a=q[cell]*(2*nat-1)*w/W[g]
    observed=d.observed_for_pilot.to_numpy(bool);miss=~observed
    y=np.where(observed,d.EEDUC_current.ge(43),0)
    known=float(np.dot(a,y));lower=known+float(np.minimum(a[miss],0).sum());upper=known+float(np.maximum(a[miss],0).sum())
    rows=[]
    common=d.loc[observed].copy()
    quality=common.nativity_prior.eq(common.nativity_current)&common.ESEX_prior.eq(common.ESEX_current)&(common.TAGE_EHC_current-common.TAGE_EHC_prior).between(0,2)
    for label,group in [('all_valid_linked',common),('consistent_background_records_posthoc',common.loc[quality]),
                         ('both_education_as_reported',common.loc[common.AEDUC_prior.eq(1)&common.AEDUC_current.eq(1)])]:
        rows.append(dict(diagnostic=label,records=len(group),BA_gains=int((group.EEDUC_prior.lt(43)&group.EEDUC_current.ge(43)).sum()),
            BA_reversals=int((group.EEDUC_prior.ge(43)&group.EEDUC_current.lt(43)).sum()),
            use='Post-pilot descriptive linkage/response quality diagnostic; no change to primary pilot inclusion'))
    pd.DataFrame(rows).to_csv(OUT/'sipp_audit/linkage_quality_diagnostics.csv',index=False)
    bounds=dict(lower=lower,upper=upper,width=upper-lower,observed_cohort=observed.sum().item(),missing_cohort=miss.sum().item(),
        weights='Fixed 2024 prior weights and prior age/sex standard; not follow-up or longitudinal weights',
        mathematical_definition='For a binary outcome assumed defined for every baseline-cohort record, assign each unavailable value in [0,1]; sum signed standardization coefficients',
        formula='L=sum_observed a_i*y_i + sum_missing min(a_i,0); U=sum_observed a_i*y_i + sum_missing max(a_i,0)',
        is_confidence_interval=False,identifies_current_resident_population=False,
        eligibility_caveat='Death, outmigration and current eligibility are not resolved by public nonmatches. These are hypothetical missing-outcome sensitivity bounds, not bounds for a verified live-resident target.',
        implication='Historical predictors or resampling of complete cases do not identify a full baseline-cohort contrast without further response/eligibility assumptions.')
    (OUT/'sipp_audit/missing_outcome_sensitivity_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n',encoding='utf8')
    print(json.dumps(bounds,indent=2))


if __name__=='__main__':main()

"""Post-score descriptive extension; recompute from frozen draws, never resample."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import norm,beta
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/acs_reporting183_2026-11-21';OUT=ROOT/'outputs/acs_hssc199_2026-11-22'
OUT.joinpath('analysis').mkdir(exist_ok=True)
assert (OUT/'revision_analysis_plan.json').exists()
def interval(k,n):return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.) if n else (None,None)
old=pd.read_csv(BASE/'analysis/final/reporting_results.csv');rows=[];paired=[]
for p in sorted((BASE/'draws/final').glob('*.npz')):
 with np.load(p) as z:
  point=z['point'][:,:,0];se=np.sqrt(z['variance'][:,:,0]);truth=float(z['truth'][0]);methods=z['methods'].astype(str).tolist();state=str(z['state']);m=int(z['m']);R=len(point)
 L=point-norm.ppf(.975)*se;U=point+norm.ppf(.975)*se
 for tau in [.01,.02,.05]:
  c=np.full(point.shape,3);c[L>tau]=0;c[U< -tau]=1;c[(L>=-tau)&(U<=tau)]=2;true=0 if truth>tau else 1 if truth< -tau else 2
  correct=c==true;wrong=(c!=3)&~correct
  for j,method in enumerate(methods):
   n=int(correct[:,j].sum());e=int(wrong[:,j].sum());un=int((c[:,j]==3).sum());o=old[(old.state==state)&(old.m==m)&(old.endpoint=='standardized')&(old.method==method)&np.isclose(old.tau,tau)].iloc[0]
   assert (n+e,e,un)==(o.claim_count,o.wrong_count,o.unresolved_count),(state,m,tau,method)
   lo,hi=interval(n,R)
   rows.append(dict(state=state,m=m,tau=tau,method=method,R=R,truth=truth,true_category=true,correct=n,wrong=e,unresolved=un,correct_yield=n/R,correct_yield_MC95_low=lo,correct_yield_MC95_high=hi,wrong_per_attempt=e/R,wrong_given_report=e/(n+e) if n+e else np.nan,correct_advantage=int(((c[:,j]==0)&correct[:,j]).sum()),correct_disadvantage=int(((c[:,j]==1)&correct[:,j]).sum()),correct_within=int(((c[:,j]==2)&correct[:,j]).sum())))
  j0=methods.index('SRS_HGB');j1=methods.index('Neyman_working_fine');d=correct[:,j1].astype(float)-correct[:,j0]
  paired.append(dict(state=state,m=m,tau=tau,increment=float(d.mean()),MC_variance_of_mean=float(d.var(ddof=1)/R),R=R))
data=pd.DataFrame(rows);data.to_csv(OUT/'analysis/correct_report_yield_all_cases.csv',index=False)
pair=pd.DataFrame(paired);pair.to_csv(OUT/'analysis/paired_correct_report_yield.csv',index=False)
summary=[]
for (m,tau),g in pair.groupby(['m','tau']):summary.append(dict(m=int(m),tau=float(tau),equal_state_increment_pp=100*g.increment.mean(),paired_MCSE_pp=100*np.sqrt(g.MC_variance_of_mean.sum())/len(g),states_positive=int((g.increment>0).sum()),states=len(g)))
pd.DataFrame(summary).to_csv(OUT/'analysis/paired_yield_summary.csv',index=False)
primary=data[(data.m==1000)&np.isclose(data.tau,.02)]
states=primary[primary.method=='Neyman_working_fine'].copy();states.to_csv(OUT/'analysis/primary_all_state_reporting.csv',index=False)
other=states[states.state!='sc'];srs=primary[primary.method=='SRS_HGB'];p=next(r for r in summary if r['m']==1000 and r['tau']==.02)
info={
 'classification':'Post-score descriptive extension using the already fixed v7 draws; no new independent evaluation',
 'all_old_report_counts_reproduced':True,'rows':len(data),'primary_comparison':p,
 'mature_correct_yield':float(states.correct.sum()/states.R.sum()),'SRS_correct_yield':float(srs.correct.sum()/srs.R.sum()),
 'mature_state_correct_yield_minmax':[float(states.correct_yield.min()),float(states.correct_yield.max())],
 'other_seven_states_correct_yield':float(other.correct.sum()/other.R.sum()),
 'SC':states[states.state=='sc'].to_dict('records')[0],
 'geographic_inference':'Eight purposive states, not a national probability sample. MCSE concerns the repeated sampling comparison only.',
 'API_external_table':'B06009 metadata available; numeric API returned non-JSON. No published-table numeric external validation is claimed.'
}
(OUT/'analysis/new_descriptive_results.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(info,ensure_ascii=False,indent=2))

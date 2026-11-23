"""Saved-summary extension audit; raw bootstrap reconstruction is a separate command."""
from pathlib import Path
from itertools import combinations,product
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;F=ROOT/'data/followup'
def compare(a,b):
    if a is None and isinstance(b,float) and np.isnan(b):return
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert np.isclose(a,b,rtol=2e-10,atol=2e-12,equal_nan=True),(a,b)
    else:assert a==b,(a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--check-case',type=Path);args=p.parse_args()
    if args.check_case:
        expected=F/'case_summaries'/args.check_case.name
        compare(json.loads(expected.read_text()),json.loads(args.check_case.read_text()));print('FRESH_RAW_CASE_MATCHES_REFERENCE',expected.name);return
    cases=sorted((F/'case_summaries').glob('*.json'));assert len(cases)==20
    rows=[];reports=[];precision=[]
    for path in cases:
        d=json.loads(path.read_text());assert d['reference_max_difference']<2e-12
        rows+=d['rows'];reports+=d['reports'];precision+=d['precision']
    for data,name,keys in [(rows,'interval_results.csv',['scope','m','method','interval','level']),
          (reports,'reporting_results.csv',['scope','m','method','interval','tau']),
          (precision,'bootstrap_precision.csv',['scope','m','method','interval','level','draw'])]:
        a=pd.DataFrame(data).sort_values(keys).reset_index(drop=True);b=pd.read_csv(F/name).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=2e-10,atol=2e-12)
    rr=pd.DataFrame(reports);assert (rr.correct+rr.wrong+rr.unresolved).eq(2000).all()
    m=pd.read_csv(F/'membership_results.csv');s=pd.read_csv(F/'membership_strong_sensitivity.csv');assert len(m)==40 and len(s)==30
    assert np.allclose(m.query("method=='complete_case'").estimate,m.query("method=='12_cell_calibration'").estimate)
    for values,name in [(m,'membership_summary.csv'),(s,'membership_strong_summary.csv')]:
        summary=values[values.clip_lower==.05].groupby('method').agg(estimate_pp=('estimate',lambda x:x.mean()*100),min_estimate_pp=('estimate',lambda x:x.min()*100),
            max_estimate_pp=('estimate',lambda x:x.max()*100),mean_absolute_error_pp=('absolute_error',lambda x:x.mean()*100))
        reference=pd.read_csv(F/name,index_col=0).sort_index()
        pd.testing.assert_frame_equal(summary.sort_index(),reference,check_exact=False,rtol=2e-10,atol=2e-12)
    bd=pd.read_csv(F/'missing_change_bounds.csv');assert np.all(np.diff(bd.lower)<=1e-12) and np.all(np.diff(bd.upper)>=-1e-12)
    x=np.array([0,0,0,1,1,2.]);n=len(x)
    for f in [0,.1,.7,1]:
        k=n//2;lam=np.sqrt(k*(1-f)/(n-k));dev=np.array([lam*(x[list(ix)].mean()-x.mean()) for ix in combinations(range(n),k)])
        rw=np.array([np.sqrt(1-f)*(x[list(ix)].mean()-x.mean()) for ix in product(range(n),repeat=n-1)])
        target=(1-f)*x.var(ddof=1)/n
        assert abs(dev.mean())<1e-14 and abs(dev.var()-target)<1e-14
        assert abs(rw.mean())<1e-14 and abs(rw.var()-target)<1e-14
    print('EXTENSION_SUMMARY_AUDIT_PASSED 20 cases; 120 intervals; 180 reports; 70 membership rows; conditional bootstrap moments')
    print('This saved-summary audit does not independently regenerate the full inner bootstrap or masked-outcome model fits.')
if __name__=='__main__':main()

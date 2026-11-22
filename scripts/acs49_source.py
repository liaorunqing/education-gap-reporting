"""Geography-versus-size controlled source sampling, prior to new target labels."""
from pathlib import Path
from datetime import datetime,timezone
import json,pickle,zipfile
import numpy as np,pandas as pd
from acs49_data import ROOT,OUT,RAW,STATES,FEATURES,path_for,sha

def source_pool():
    frames=[];rows=[]
    for state,fips in STATES.items():
        receipt=OUT/'receipts'/f'2017_{state}.json';assert receipt.exists(),state
        with zipfile.ZipFile(path_for(2017,state)) as z:
            name=next(n for n in z.namelist() if n.endswith('.csv'))
            with z.open(name) as stream:
                d=pd.read_csv(stream,usecols=['ST','PWGTP','SCHL','FSCHLP']+FEATURES)
        d['source_row']=np.arange(len(d),dtype=np.int32)
        d=d[d.AGEP.ge(25)&d.PWGTP.gt(0)].copy()
        assert len(d)==json.loads(receipt.read_text())['adult_covariates']
        assert d.ST.eq(fips).all() and d.FSCHLP.isin([0,1]).all() and d.SCHL.between(1,24).all()
        d['cell']=(np.where(d.AGEP<=39,0,np.where(d.AGEP<=59,1,2))*4+(d.SEX-1)*2+d.NATIVITY-1).astype(np.int16)
        d['group']=(d.ST*12+d.cell).astype(np.int16)
        d['y']=d.SCHL.ge(21).astype(np.int8)
        for c in ['ST','SEX','NATIVITY','RAC1P','CIT','FSCHLP','cell','y']:d[c]=d[c].astype(np.int16)
        d['AGEP']=d.AGEP.astype(np.int16);d['PWGTP']=d.PWGTP.astype(float)
        frames.append(d)
        rows.append({'state':state,'FIPS':fips,'adults':len(d),'allocated_count':int(d.FSCHLP.sum()),
            'allocated_weight_share':float(np.average(d.FSCHLP,weights=d.PWGTP)),
            'raw_source_sha256':sha(path_for(2017,state))})
    pool=pd.concat(frames,ignore_index=True);pd.DataFrame(rows).to_csv(OUT/'source2017_population_audit.csv',index=False)
    return pool

def allocation(groups,n):
    # Every nonempty source geographic-demographic stratum receives at least2.
    values,count=np.unique(groups,return_counts=True);base=np.minimum(count,2)
    assert base.sum()<=n<=count.sum()
    capacity=count-base;wanted=(n-base.sum())*capacity/capacity.sum()
    extra=np.minimum(np.floor(wanted).astype(int),capacity)
    left=n-int((base+extra).sum())
    order=np.argsort(-(wanted-extra),kind='stable')
    for k in order:
        if left==0:break
        if extra[k]<capacity[k]:extra[k]+=1;left-=1
    assert left==0 and (base+extra).sum()==n
    return values,count,base+extra

def sample(pool,n,seed):
    values,N,ng=allocation(pool.group.to_numpy(),n);rng=np.random.default_rng(seed)
    indices=[];prob=[];audit=[]
    group_array=pool.group.to_numpy()
    for group,Ns,ns in zip(values,N,ng):
        available=np.flatnonzero(group_array==group)
        selected=rng.choice(available,int(ns),replace=False)
        indices.extend(selected);prob.extend(np.repeat(ns/Ns,ns))
        audit.append({'group':int(group),'N':int(Ns),'n':int(ns),'source_selection_probability':float(ns/Ns)})
    d=pool.iloc[indices].copy();d['source_pi']=prob;d['fit_weight']=d.PWGTP/d.source_pi
    return d,audit

def main():
    assert (OUT/'source_training_plan.json').exists()
    dest=OUT/'source_training_frames.pkl';assert not dest.exists()
    pool=source_pool();assert pool.ST.nunique()==51
    source_sets={};audits={}
    narrow=pool[pool.ST.isin([10,44])].copy();assert len(narrow)==14123
    narrow['source_pi']=1.;narrow['fit_weight']=narrow.PWGTP
    source_sets['two_state']=narrow
    for arm,n,eligible in [('national_matched',14123,pool),('national_large',100000,pool),
        ('national_unallocated',100000,pool[pool.FSCHLP.eq(0)].copy())]:
        d,a=sample(eligible,n,739);source_sets[arm]=d;audits[arm]=a
    records=[]
    for arm,d in source_sets.items():
        assert not d.duplicated(['ST','source_row']).any()
        assert d.fit_weight.gt(0).all()
        records.append({'arm':arm,'training_adults':len(d),'states':int(d.ST.nunique()),'strata':int(d.group.nunique()),
            'allocated_record_fraction':float(d.FSCHLP.mean()),'weight_sum':float(d.fit_weight.sum()),
            'source_year':2017,'source_sampling_seed':739,
            'population_scope':'Unallocated-education subpopulation' if arm=='national_unallocated' else 'All released adult records'})
    dest.write_bytes(pickle.dumps(source_sets,protocol=5))
    pd.DataFrame(records).to_csv(OUT/'source_training_arm_audit.csv',index=False)
    (OUT/'source_sampling_audit.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'source_pool_adults':len(pool),'training_frames_sha256':sha(dest),'allocation':audits,
        'target2023_labels_read':False},indent=2),encoding='utf8')
    print('NATIONAL_SOURCE_TRAINING_FRAMES_FROZEN',len(pool),records,flush=True)
if __name__=='__main__':main()

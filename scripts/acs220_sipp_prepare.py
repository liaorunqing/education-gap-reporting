"""Read public longitudinal files under a frozen plan; write only aggregate audits publicly."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time,zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acs_research215_2026-11-22'
PRIVATE=OUT/'private_sipp'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def dump(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def nativity(d):
    out=np.full(len(d),-1,int)
    native=d.EBORNUS.eq(1)|d.ENATCIT.isin([4,5])
    foreign=d.EBORNUS.eq(2)&(d.ECITIZEN.eq(2)|d.ENATCIT.isin([1,2,3]))
    assert not (native&foreign).any(),'Contradictory birth/citizenship codes must be inspected.'
    out[native]=0;out[foreign]=1
    return out


def parse(year):
    PRIVATE.mkdir(parents=True,exist_ok=True)
    target=PRIVATE/f'december_{year}.csv.gz'
    if target.exists():return pd.read_csv(target,dtype={'SSUID':'string','SHHADID':'string'})
    plan=json.loads((OUT/'sipp_pilot_plan.json').read_text(encoding='utf8'))
    archive=globals().get('RAW_OVERRIDE',ROOT/'data/raw/sipp218_2026-11-22')/f'pu{year}_csv.zip'
    receipt=json.loads((OUT/f'sipp_acquisition_{year}.json').read_text())
    assert sha(archive)==receipt['sha256']
    dump(OUT/f'sipp_label_access_{year}.json',dict(utc=datetime.now(timezone.utc).isoformat(),year=year,
         plan_sha256=sha(OUT/'sipp_pilot_plan.json'),prior_q_frozen=(OUT/'sipp_prior_background_freeze.json').exists(),
         source_sha256=receipt['sha256'],status='Already downloaded official file; first outcome parsing starts here'))
    if year==2025:assert (OUT/'sipp_prior_background_freeze.json').exists()
    start=time.monotonic();pieces=[];total=0
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('.csv'))
        with z.open(name) as f:
            dtype={c:'string' if c in ['SSUID','SHHADID','TBORNPLACE'] else 'float64' for c in plan['columns']}
            for chunk in pd.read_csv(f,sep='|',usecols=plan['columns'],dtype=dtype,chunksize=40000):
                total+=len(chunk);pieces.append(chunk.loc[chunk.MONTHCODE.eq(12)].copy())
                print('SIPP_PARSE',year,total,'rows',round(time.monotonic()-start,1),'s',flush=True)
    df=pd.concat(pieces,ignore_index=True)
    assert not df.duplicated(['SPANEL','SSUID','PNUM']).any()
    assert df[['SPANEL','SSUID','PNUM','SHHADID']].notna().all().all()
    df.to_csv(target,index=False,compression='gzip')
    dump(OUT/f'sipp_parse_QA_{year}.json',dict(monthly_rows=total,December_records=len(df),unique_person_keys=True,
         source_bytes=archive.stat().st_size,private_derived_file_sha256=sha(target),seconds=round(time.monotonic()-start,2)))
    return df


def freeze_prior():
    d=parse(2024);d['nativity']=nativity(d)
    frame=d[d.SPANEL.isin([2022,2023,2024]) & d.TAGE_EHC.ge(25)&d.WPFINWGT.gt(0)&d.nativity.ge(0)&d.ESEX.isin([1,2])].copy()
    frame['ageband']=np.where(frame.TAGE_EHC<=39,0,np.where(frame.TAGE_EHC<=59,1,2))
    frame['cell']=frame.ageband*2+frame.ESEX.astype(int)-1
    frame['g']=2*frame.cell+frame.nativity
    W=np.bincount(frame.g,weights=frame.WPFINWGT,minlength=12).reshape(6,2)
    assert (W>0).all()
    q=.5*(W[:,0]/W[:,0].sum()+W[:,1]/W[:,1].sum())
    j=dict(prior_year=2024,reference_year=2023,panels=[2022,2023,2024],monthly_row=12,
        prior_adults=len(frame),prior_weight_sum=float(frame.WPFINWGT.sum()),q=q.tolist(),cell_weight_totals=W.tolist(),
        prior_frame_file_sha256=sha(PRIVATE/'december_2024.csv.gz'),plan_sha256=sha(OUT/'sipp_pilot_plan.json'),
        created_before_2025_outcome_parse=not (OUT/'sipp_label_access_2025.json').exists())
    assert j['created_before_2025_outcome_parse']
    dump(OUT/'sipp_prior_background_freeze.json',j)
    print('SIPP_PRIOR_FRAME_FROZEN',len(frame),flush=True)


def audit():
    prior=parse(2024);current=parse(2025)
    keys=['SPANEL','SSUID','PNUM']
    prior['nativity']=nativity(prior);current['nativity']=nativity(current)
    prior['cluster']=prior.SPANEL.astype(int).astype(str)+'|'+prior.SSUID+'|'+prior.SHHADID
    current['current_cluster']=current.SPANEL.astype(int).astype(str)+'|'+current.SSUID+'|'+current.SHHADID
    eligible=prior[prior.SPANEL.isin([2022,2023,2024])&prior.TAGE_EHC.ge(25)&prior.WPFINWGT.gt(0)&prior.nativity.ge(0)&prior.ESEX.isin([1,2])].copy()
    merged=eligible.merge(current,on=keys,how='left',suffixes=('_prior','_current'),indicator=True,validate='one_to_one')
    exists=merged['_merge'].eq('both')
    correct_wave=merged.SWAVE_current.eq(merged.SWAVE_prior+1)
    valid_prior=merged.EEDUC_prior.between(31,46)
    valid_current=merged.EEDUC_current.between(31,46)
    observed=exists&correct_wave&valid_current&valid_prior
    cluster=merged.groupby('cluster',sort=False).agg(n=('PNUM','size'))
    obs_n=merged.loc[observed].groupby('cluster').size().reindex(cluster.index,fill_value=0)
    complete=obs_n.eq(cluster.n)
    merged['observed_for_pilot']=observed
    merged['complete_cluster']=merged.cluster.isin(cluster.index[complete])
    merged.to_csv(PRIVATE/'linked_cohort_private.csv.gz',index=False,compression='gzip')
    m=merged.loc[observed].copy();m['prior_BA']=m.EEDUC_prior.ge(43).astype(int);m['current_BA']=m.EEDUC_current.ge(43).astype(int)
    trans=m.groupby(['prior_BA','current_BA','AEDUC_prior','AEDUC_current']).size().reset_index(name='persons')
    dest=OUT/'sipp_audit';dest.mkdir(exist_ok=True);trans.to_csv(dest/'education_transition_by_flags.csv',index=False)
    panel=[]
    for panel_year,g in merged.groupby('SPANEL'):
        ob=g.observed_for_pilot
        panel.append(dict(panel=int(panel_year),prior_adults=len(g),matched_persons=int(g['_merge'].eq('both').sum()),
                          valid_matched_outcomes=int(ob.sum()),prior_households=int(g.cluster.nunique()),
                          fully_observed_households=int(g.loc[g.complete_cluster,'cluster'].nunique())))
    pd.DataFrame(panel).to_csv(dest/'panel_retention.csv',index=False)
    group=[]
    for nat,g in merged.groupby('nativity_prior'):
        group.append(dict(nativity=int(nat),prior_adults=len(g),observed=int(g.observed_for_pilot.sum()),
            person_retention=float(g.observed_for_pilot.mean()),weighted_retention=float(np.average(g.observed_for_pilot,weights=g.WPFINWGT_prior)),
            prior_weight_sum=float(g.WPFINWGT_prior.sum())))
    pd.DataFrame(group).to_csv(dest/'group_retention.csv',index=False)
    new=current[current.SPANEL.isin([2022,2023,2024])].merge(prior[keys],on=keys,how='left',indicator=True,validate='one_to_one')
    same=m.groupby('cluster').current_cluster.nunique()
    education_rate=float((m.current_BA!=m.prior_BA).mean())
    both_reported=m.AEDUC_prior.eq(1)&m.AEDUC_current.eq(1)
    report=dict(prior_adults=len(merged),prior_clusters=len(cluster),matched_persons=int(exists.sum()),
        valid_matched_outcomes=int(observed.sum()),person_observed_fraction=float(observed.mean()),
        weighted_person_observed_fraction=float(np.average(observed,weights=merged.WPFINWGT_prior)),
        complete_clusters=int(complete.sum()),complete_cluster_fraction=float(complete.mean()),
        adults_in_complete_clusters=int(merged.complete_cluster.sum()),planned_rotation_panel2021_adults=int((prior.SPANEL.eq(2021)&prior.TAGE_EHC.ge(25)&prior.WPFINWGT.gt(0)).sum()),
        valid_prior_education_missing=int((~valid_prior).sum()),unknown_nativity_all_prior_december=int(prior.nativity.lt(0).sum()),
        new_person_identifiers_in_continuing_panels=int(new['_merge'].eq('left_only').sum()),
        prior_clusters_spanning_multiple_current_households=int(same.gt(1).sum()),
        common_person_birth_group_changed=int(m.nativity_prior.ne(m.nativity_current).sum()),
        common_person_sex_changed=int(m.ESEX_prior.ne(m.ESEX_current).sum()),
        age_change_outside_0to2=int((~(m.TAGE_EHC_current-m.TAGE_EHC_prior).between(0,2)).sum()),
        BA_status_change_fraction=education_rate,BA_gains=int(((m.prior_BA==0)&(m.current_BA==1)).sum()),
        BA_reversals=int(((m.prior_BA==1)&(m.current_BA==0)).sum()),
        current_education_as_reported_fraction=float(m.AEDUC_current.eq(1).mean()),both_education_as_reported_fraction=float(both_reported.mean()),
        both_reported_BA_change_fraction=float((m.loc[both_reported,'current_BA']!=m.loc[both_reported,'prior_BA']).mean()),
        prior_information_genuinely_observed_in_previous_wave=True,current_complete_roster_known_before_selection=False,
        all_prior_outcomes_observed=False,real_field_cost_validation=False,
        interpretation='Feasible historical-to-follow-up linkage among observed public panel records; incomplete prior cohort and retrospective subset selection remain unsolved',
        conditional_pilot_gate_passed=bool(complete.sum()>=1000))
    dump(dest/'feasibility_QA.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prior','audit']);a=p.parse_args()
    if a.mode=='prior':freeze_prior()
    else:audit()

"""Audited ACS household reporting study. Final education is opened only by score mode."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,pickle,re,urllib.request,zipfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from acs49_data import STATES as FIPS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acs_reporting183_2026-11-21'
RAW=ROOT/'data/raw/acs_reporting183_2026-11-21'
SOURCE=ROOT/'outputs/acs_operational49_2026-11-17'
FINAL='ct nh ia ks al sc az co'.split()
SEEN='ma pa il mn ga va ca wa'.split()
REGIONS=dict(zip(FINAL,['Northeast','Northeast','Midwest','Midwest','South','South','West','West']))
DEV=[(2022,s) for s in FINAL]+[(2023,s) for s in FINAL]+[(2024,s) for s in SEEN]
TARGET=[(2024,s) for s in FINAL]
R=2000
FEATURES=['n_adult','foreign_fraction','older_fraction','female_fraction','age25_39_fraction','mean_prediction','sd_prediction','abs_b_sum','b_square_sum','max_abs_b','weight_cv','mixed_nativity']

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def loadjson(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def log(event,**kw):
    with (OUT/'access_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=utc(),event=event,**kw),ensure_ascii=False)+'\n')
def zip_path(year,state):
    found=sorted((ROOT/'data/raw').rglob(f'acs_{year}_{state}.zip'))
    if found:return found[0]
    return RAW/f'acs_{year}_{state}.zip'

def init():
    assert not OUT.exists(),'Initialization never overwrites a prior protocol'
    OUT.mkdir(parents=True);RAW.mkdir(parents=True)
    inventory=[]
    for p in (ROOT/'data/raw').rglob('*.zip'):
        m=re.fullmatch(r'acs_(\d{4})_([a-z]{2})\.zip',p.name,re.I)
        if m:inventory.append(dict(year=int(m[1]),state=m[2].lower(),path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size))
    assert not [v for v in inventory if (v['year'],v['state']) in TARGET],'Prespecified confirmation data already downloaded'
    # Audit code and all old JSON records for named target-year accesses without opening label arrays.
    hits=[]
    for folder in ['scripts','outputs','data/raw']:
        for p in (ROOT/folder).rglob('*'):
            if OUT in p.parents or RAW in p.parents or p.suffix not in ['.py','.json','.jsonl']:continue
            if p.stat().st_size>12_000_000:continue
            text=p.read_text(encoding='utf-8',errors='replace')
            matches=re.findall(r'(?:acs_2024_|2024/1-Year/csv_p)(ct|nh|ia|ks|al|sc|az|co)(?:\.|["\s])',text,re.I)
            if matches:hits.append(dict(path=p.relative_to(ROOT).as_posix(),states=sorted(set(matches))))
    assert not hits, hits
    protected={}
    for folder in ['outputs/acs_hssc_review161_2026-11-19','outputs/acs_solution178_2026-11-20']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/'outputs').rglob('*manifest*.json'):
        if OUT not in p.parents:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [SOURCE/'models/national_large_739.pkl',SOURCE/'models/source_transformers.pkl',SOURCE/'historical_local_controls.json']:
        protected[p.relative_to(ROOT).as_posix()]=sha(p)
    dump('protected_before.json',protected)
    dump('prior_exposure_inventory.json',dict(utc=utc(),zip_inventory=inventory,named_target_access_hits=hits,scope='Local files/code/JSON access records; cannot establish absence of access outside workspace',target_pairs=TARGET))
    protocol=dict(utc=utc(),question='When can limited household samples support reliable reports of nativity educational-attainment gaps?',target_journal='HSSC; substantive education measurement application, no required ML win',
      development_pairs=DEV,confirmation_pairs=TARGET,confirmation_regions=REGIONS,minimum_confirmation_states=6,minimum_regions=3,
      population='Released positive-PWGTP adults AGE>=25 in HU serials only; GQ excluded and coverage disclosed; no recovered Census PSU claim',
      outcome='SCHL>=21; released label, not independently verified degree',
      standardization='6 age/SEX cells, age25-39/40-59/60+; q fixed from equal-nativity pooled2022development weighted composition',
      budgets=[300,1000,3000],primary_budget=1000,draws=R,seed=739,
      strata='any_foreign x any_age60plus x adult_count>1, eight background-only strata',
      estimand='foreign minus native standardized BA+ gap; fixed q and current background denominators; raw gap secondary',
      information='Entire released demographic/household background and weights assumed known; current unsampled education hidden to policies',
      point_predictor='Frozen2017 national_large_739 HGB and2017 full-source state-cell historical control',
      risk_learners={'RF':dict(n_estimators=200,min_samples_leaf=20,max_features=1.0,random_state=739),'HGB':dict(max_iter=200,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=739,loss='poisson')},
      risk_training='2022 complete households; five household-level folds for diagnostics;2023and seen2024 validation, no target2024Y',
      reporting=dict(primary_ci=.95,primary_tau=.02,sensitivity_tau=[.01,.05],categories=['advantage_above_tau','disadvantage_below_minus_tau','within_reporting_band','unresolved']),
      metrics=['MSE','bias','MAE','90/95coverage','width','claim_rate','wrong_claim_per_attempt','wrong_given_claim','statewise_and_equal_state_MCSE'],
      inference='Conditional repeated-sampling finite released frame; original ACS replicate-weight uncertainty separate',
      no_claims=['new algorithm from combining known methods','causal education mechanism','actual monetary savings','unseen calendar year/country/geography'],
      final_gate='All code/models/predictions/analysis and figure rules frozen and numerical checks passed before final label parse; single scored final evaluation',
      methods_details='Follow-up implementation contract must be frozen before final scoring; development changes logged',
      data_version='Current official standard1-year files, retrospective latest-vintage, not as-of historical releases')
    dump('protocol.json',protocol);log('initialized',labels_read=False);print('INITIALIZED',len(protected),flush=True)

def acquire_final():
    assert (OUT/'protocol.json').exists()
    for year,s in TARGET:
        p=RAW/f'acs_{year}_{s}.zip'
        if p.exists():continue
        url=f'https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{s}.zip'
        temp=p.with_suffix('.partial')
        req=urllib.request.Request(url,headers={'User-Agent':'Public education measurement research/1.0'})
        with urllib.request.urlopen(req,timeout=90) as res,temp.open('wb') as f:
            headers=dict(res.headers)
            while b:=res.read(2**20):f.write(b)
        with zipfile.ZipFile(temp) as z:
            assert z.testzip() is None
        temp.replace(p)
        dump(f'acquisition/{year}_{s}.json',dict(url=url,sha256=sha(p),bytes=p.stat().st_size,utc=utc(),headers=headers,labels_parsed=False))
        log('download',year=year,state=s,labels_read=False);print('ACQUIRED',s,p.stat().st_size,flush=True)

def read_background(year,s):
    dest=OUT/f'background/{year}_{s}.npz'
    if dest.exists():return
    p=zip_path(year,s);assert p.exists(),p
    with zipfile.ZipFile(p) as z:
        member=next(n for n in z.namelist() if n.endswith('.csv'))
        with z.open(member) as f:header=f.readline().decode('utf-8-sig').strip().split(',')
        st='STATE' if 'STATE' in header else 'ST'
        cols=[st,'SERIALNO','SPORDER','AGEP','SEX','NATIVITY','RAC1P','CIT','PWGTP','RELSHIPP']
        with z.open(member) as f:d=pd.read_csv(f,usecols=cols,dtype={'SERIALNO':str})
    d=d.rename(columns={st:'ST'})
    assert d.ST.eq(FIPS[s]).all()
    assert not d.duplicated(['SERIALNO','SPORDER']).any()
    hu=d.SERIALNO.str.match(r'^\d{4}HU\d+$')
    gq=d.SERIALNO.str.match(r'^\d{4}GQ\d+$')
    assert (hu|gq).all()
    audit=dict(year=year,state=s,rows=len(d),HU_rows=int(hu.sum()),GQ_rows=int(gq.sum()),adult_GQ=int((gq&d.AGEP.ge(25)).sum()),
      nonpositive_weight=int(d.PWGTP.le(0).sum()),background_missing=d.isna().sum().to_dict(),state_alias=st,
      file_sha256=sha(p),labels_read=False,utc=utc())
    # Retain all members in a separate roster to audit group completeness, then eligible adults.
    d=d[hu].copy();unit_sizes=d.groupby('SERIALNO').size();unique_units=d.SERIALNO.nunique()
    a=d[d.AGEP.ge(25)&d.PWGTP.gt(0)].copy()
    assert not a[cols[3:]].isna().any().any()
    assert a.SEX.isin([1,2]).all() and a.NATIVITY.isin([1,2]).all()
    units,h=np.unique(a.SERIALNO.to_numpy(str),return_inverse=True)
    a['cell']=(np.where(a.AGEP<=39,0,np.where(a.AGEP<=59,1,2))*2+(a.SEX-1)).astype(int)
    a['g']=(a.cell*2+a.NATIVITY-1).astype(int)
    a['group']=a.ST*12+a.g
    trans=pickle.loads((SOURCE/'models/source_transformers.pkl').read_bytes())
    models=pickle.loads((SOURCE/'models/national_large_739.pkl').read_bytes())
    with threadpool_limits(limits=1):f=models['HGB'].predict_proba(trans['transformer'].transform(a))[:,1]
    controls=loadjson(SOURCE/'historical_local_controls.json')['full_national_source']
    lookup={(r['state'],r['cell']):r['mean'] for r in controls}
    fc=np.array([lookup[FIPS[s],int(k)] for k in a.g])
    assert np.isfinite(f).all() and np.isfinite(fc).all()
    ids=(a.SERIALNO+'|'+a.SPORDER.astype(str)).to_numpy(str)
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,ids=ids,unit=units,h=h,w=a.PWGTP.to_numpy(float),g=a.g.to_numpy(int),age=a.AGEP.to_numpy(int),sex=a.SEX.to_numpy(int),nativity=a.NATIVITY.to_numpy(int),f=f,fc=fc,
      member_count=unit_sizes.loc[units].to_numpy(int))
    audit.update(eligible_adults=len(a),eligible_units=len(units),all_HU_units=unique_units,minimum_cell_n=int(a.groupby('g').size().min()),complete_HU_roster_loaded=True,background_sha256=sha(dest))
    dump(f'background/{year}_{s}_audit.json',audit);log('background',year=year,state=s,labels_read=False)
    print('BACKGROUND',year,s,len(a),len(units),flush=True)

def prepare(which):
    pairs=DEV if which=='dev' else TARGET
    for y,s in pairs:read_background(y,s)
    if which=='dev' and not (OUT/'standardization.json').exists():
        W=np.zeros(12)
        for s in FINAL:
            with np.load(OUT/f'background/2022_{s}.npz') as z:W+=np.bincount(z['g'],weights=z['w'],minlength=12)
        W=W.reshape(6,2);q=.5*(W[:,0]/W[:,0].sum()+W[:,1]/W[:,1].sum())
        dump('standardization.json',dict(q=q.tolist(),reference='Pooled8developmentstates2022HUadults; equal nativity distribution average',utc=utc()))

def read_labels(year,s,final=False):
    dest=OUT/f'labels/{year}_{s}.npz'
    if dest.exists():return
    if (year,s) in TARGET:
        assert final and (OUT/'final_freeze.json').exists()
        for rel,hv in loadjson(OUT/'final_freeze.json')['files'].items():assert sha(ROOT/rel)==hv,rel
    else:assert (year,s) in DEV
    with np.load(OUT/f'background/{year}_{s}.npz') as z:ids=z['ids']
    with zipfile.ZipFile(zip_path(year,s)) as z:
        n=next(n for n in z.namelist() if n.endswith('.csv'))
        cols=['SERIALNO','SPORDER','AGEP','PWGTP','SCHL','FSCHLP']+[f'PWGTP{i}' for i in range(1,81)]
        with z.open(n) as f:d=pd.read_csv(f,usecols=cols,dtype={'SERIALNO':str})
    a=d[d.SERIALNO.str.contains('HU')&d.AGEP.ge(25)&d.PWGTP.gt(0)].copy()
    check=(a.SERIALNO+'|'+a.SPORDER.astype(str)).to_numpy(str);assert np.array_equal(check,ids)
    assert a.SCHL.between(1,24).all() and a.FSCHLP.isin([0,1]).all()
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,ids=ids,y=a.SCHL.ge(21).to_numpy(float),flag=a.FSCHLP.to_numpy(int),repw=a[[f'PWGTP{i}' for i in range(1,81)]].to_numpy(float))
    log('education_labels',year=year,state=s,confirmation=final,freeze_sha256=sha(OUT/'final_freeze.json') if final else None)
    print('LABELS',year,s,'FINAL' if final else 'DEVELOPMENT',flush=True)

def verify_old():
    protected=loadjson(OUT/'protected_before.json');bad=[p for p,v in protected.items() if not (ROOT/p).exists() or sha(ROOT/p)!=v]
    assert not bad,bad;return len(protected)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','acquire','prepare_dev','prepare_final','labels_dev','verify']);a=p.parse_args()
    if a.mode=='init':init()
    elif a.mode=='acquire':acquire_final()
    elif a.mode.startswith('prepare'):prepare(a.mode.split('_')[1])
    elif a.mode=='labels_dev':
        for y,s in DEV:read_labels(y,s)
    else:print('PROTECTED_UNCHANGED',verify_old())

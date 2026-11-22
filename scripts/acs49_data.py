"""National-source expansion and new target acquisition with sealed education."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,re,shutil,time,urllib.request,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acs_operational49_2026-11-17'
RAW=ROOT/'data/raw/acs_operational49_2026-11-17'
STATES=dict(zip(
    'al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy'.split(),
    [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56]))
TARGETS=['ct','nh','ia','ks','al','sc','az','co']
REGIONS={'Northeast':['ct','nh'],'Midwest':['ia','ks'],'South':['al','sc'],'West':['az','co']}
FEATURES=['AGEP','SEX','NATIVITY','RAC1P','CIT']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    plan=OUT/'acquisition_plan.json'
    if plan.exists():return
    prior=[]
    for path in (ROOT/'data/raw').rglob('*.zip'):
        m=re.search(r'acs_(\d{4})_([a-z]{2})\.zip$',path.name,re.I)
        if m:prior.append({'year':int(m[1]),'state':m[2].lower(),'file':path.relative_to(ROOT).as_posix()})
    assert not [a for a in prior if a['year'] in [2022,2023] and a['state'] in TARGETS]
    (OUT/'prior_microdata_exposure_inventory.json').write_text(json.dumps(prior,indent=2),encoding='utf8')
    p={'utc':datetime.now(timezone.utc).isoformat(),'scope':'National2017source, new2022development and2023historical holdout state-years.',
        'source_states':list(STATES),'source_count':51,'target_states':TARGETS,'target_regions':REGIONS,
        'features':['ST']+FEATURES,'education':'SCHL>=21 amongAGEP>=25,PWGTP>0',
        'allocation_flag':'FSCHLP; no allocation status is treated as validation of degree truth.',
        'sealed':'Acquisition reads headers and covariates only in ALLyears.2023SCHL andFSCHLP values remain sealed until final freeze.',
        'source_access':'2017labels permitted in later source-sampling/training step.',
        'version':'Current public standard1-Year versions,not historical first-release vintages.',
        'source_training_arms':['DE_RI_all_14123','national_matched_14123','national_100000'],
        'training_control':'Fixed5learners as phase47; same features,source-year,hyperparameters; separate geography versus sample-size effects.',
        'imputation_sensitivity':'Released labels,unallocated-label subpopulation,latent-outcome worst-case bounds; no imputation flag treated as true label.',
        'holdout_state_years_previously_loaded':False,'pre_acquisition_inventory_sha256':sha(OUT/'prior_microdata_exposure_inventory.json')}
    plan.write_text(json.dumps(p,indent=2),encoding='utf8')
    # Protect the completed branch without modifying it.
    old=ROOT/'outputs/acs_forward47_2026-11-17'
    (OUT/'prior_branch_manifest_sha256.json').write_text(json.dumps({'file':str(old.relative_to(ROOT)/'artifact_manifest.json'),
        'sha256':sha(old/'artifact_manifest.json')},indent=2),encoding='utf8')

def path_for(year,state):
    if year==2017 and state in ['de','ri']:
        return ROOT/f'data/raw/acs_forward45_2026-11-17/acs_2017_{state}.zip'
    return RAW/f'acs_{year}_{state}.zip'

def acquire(year,state):
    path=path_for(year,state);url=f'https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{state}.zip'
    receipt={'utc':datetime.now(timezone.utc).isoformat(),'year':year,'state':state,'url':url,
        'source_reused':year==2017 and state in ['de','ri'],'SCHL_values_read':False,'FSCHLP_values_read':False}
    if not path.exists():
        partial=path.with_suffix('.download')
        if partial.exists():partial.unlink() # Exact temporary file in this new data directory only.
        request=urllib.request.Request(url,headers={'User-Agent':'AcademicResearch publicACS reproducibility'})
        try:
            with urllib.request.urlopen(request,timeout=90) as response,partial.open('wb') as dest:
                n=0
                while True:
                    chunk=response.read(2**20)
                    if not chunk:break
                    n+=len(chunk)
                    assert n<300_000_000,'Unexpected per-state file size'
                    dest.write(chunk)
                receipt['response_headers']={k:v for k,v in response.headers.items() if k.lower() in ['last-modified','etag','content-length','content-type']}
            with zipfile.ZipFile(partial) as z:assert z.testzip() is None
            partial.rename(path)
        except Exception:
            if partial.exists():partial.unlink()
            raise
    with zipfile.ZipFile(path) as z:
        members=[n for n in z.namelist() if n.lower().endswith('.csv')]
        assert len(members)==1
        with z.open(members[0]) as stream:header=stream.readline().decode('utf-8-sig').strip().split(',')
        statecol='STATE' if year>=2023 else 'ST'
        needed=['SERIALNO','SPORDER',statecol,'PWGTP','SCHL','FSCHLP']+FEATURES
        assert set(needed).issubset(header),(year,state,set(needed)-set(header))
        assert all(f'PWGTP{i}' in header for i in range(1,81))
        usecols=[statecol,'PWGTP']+FEATURES
        with z.open(members[0]) as stream:d=pd.read_csv(stream,usecols=usecols)
    assert d[statecol].eq(STATES[state]).all()
    d=d[d.AGEP.ge(25)&d.PWGTP.gt(0)].copy()
    assert d.SEX.isin([1,2]).all() and d.NATIVITY.isin([1,2]).all()
    d['age_band']=pd.cut(d.AGEP,[24,39,59,200],labels=['25-39','40-59','60+']).astype(str)
    support=d.groupby(['age_band','SEX','NATIVITY'],observed=True).agg(N=('PWGTP','size'),W=('PWGTP','sum')).reset_index()
    receipt.update(file=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path),
        header=header,adult_covariates=len(d),domain_count=len(support),minimum_domain_N=int(support.N.min()),
        dictionary_geography_column=statecol)
    (OUT/'receipts').mkdir(exist_ok=True)
    dest=OUT/'receipts'/f'{year}_{state}.json'
    if not dest.exists():dest.write_text(json.dumps(receipt,indent=2),encoding='utf8')
    support.assign(year=year,state=state).to_csv(OUT/'receipts'/f'{year}_{state}_support.csv',index=False)
    print('SEALED_COVARIATE_AUDIT',year,state,path.stat().st_size,len(d),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--years',nargs='+',type=int,default=[2017,2022,2023]);p.add_argument('--states',nargs='+')
    args=p.parse_args();initialize()
    assert set(args.years)<=set([2017,2022,2023])
    for year in args.years:
        states=args.states or (list(STATES) if year==2017 else TARGETS)
        assert set(states)<=set(STATES if year==2017 else TARGETS)
        for state in states:
            dest=OUT/'receipts'/f'{year}_{state}.json'
            if dest.exists():continue
            for attempt in range(3):
                try:acquire(year,state);break
                except Exception as e:
                    print('ACQUISITION_RETRY',year,state,attempt+1,repr(e),flush=True)
                    if attempt==2:raise
    receipts=[json.loads(p.read_text()) for p in sorted((OUT/'receipts').glob('*.json'))]
    (OUT/'acquisition_progress.json').write_text(json.dumps({'files_completed':len(receipts),
        'expected':67,'bytes':sum(r['bytes'] for r in receipts),'final_labels_read':False},indent=2),encoding='utf8')
if __name__=='__main__':main()

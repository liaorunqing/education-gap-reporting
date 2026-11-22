"""Raw-only rebuild of the used CT pipeline, separate from frozen experiments.

Input source ZIPs are checked against published checksums. Existing model or derived
array files are never copied into the reconstruction. Legacy implementation modules
are redirected to a new workspace; final-label guards remain active there.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,pickle,shutil,sys,time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/acs_research215_2026-11-22'
PUBLIC=ROOT/'outputs/acs_public210_2026-11-22/repository'
if not PUBLIC.exists():PUBLIC=ROOT  # Layout of the standalone public repository.
STATES='ct nh ia ks al sc az co'.split()


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def dump(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run(rawdir,workdir):
    import acs49_data as d49
    import acs49_source as source
    import acs49_train as train49
    import acs49_local_controls as controls
    import acs183_reporting_data as data
    import acs185_reporting_risk as risk
    import acs186_reporting_run as runner
    work=Path(workdir).resolve();assert not work.exists(),'Use a fresh reconstruction workspace.'
    plan_path=STUDY/'research_plan.json'
    if not plan_path.exists():plan_path=PUBLIC/'metadata/raw_reconstruction_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    work.mkdir(parents=True)
    src=work/'outputs/acs_operational49_2026-11-17';src.mkdir(parents=True)
    out=work/'outputs/acs_reporting183_2026-11-21';out.mkdir(parents=True)
    events=[];start=time.monotonic()
    def event(name,**kw):
        j=dict(stage=name,elapsed_seconds=round(time.monotonic()-start,2),**kw);events.append(j)
        dump(work/'reconstruction_events.json',events);print(json.dumps(j),flush=True)
    metadata={j['filename']:j for j in json.loads((PUBLIC/'metadata/census_sources.json').read_text())}
    pool_paths={}
    wanted=[(2017,s) for s in d49.STATES]+[(2022,s) for s in STATES]+[(2024,'ct')]
    for year,state in wanted:
        key=f'{year}_{state}';expected=metadata[key]['sha256']
        candidates=sorted(Path(rawdir).resolve().rglob(f'acs_{key}.zip'))
        if not candidates:raise FileNotFoundError('Official ZIP needed: '+metadata[key]['url'])
        path=candidates[0];assert sha(path)==expected,(key,'source vintage mismatch')
        pool_paths[year,state]=path
    dump(work/'input_provenance.json',[dict(year=y,state=s,url=metadata[f'{y}_{s}']['url'],sha256=sha(p),bytes=p.stat().st_size) for (y,s),p in pool_paths.items()])
    # Only aggregate count metadata are used to check original source-row definitions.
    audit_path=PUBLIC/'metadata/source2017_population_audit.csv'
    if not audit_path.exists():audit_path=ROOT/'outputs/acs_operational49_2026-11-17/source2017_population_audit.csv'
    popaudit=pd.read_csv(audit_path).set_index('state')
    for s in d49.STATES:
        dump(src/f'receipts/2017_{s}.json',dict(adult_covariates=int(popaudit.loc[s,'adults'])))
    source.OUT=src;source.path_for=lambda y,s:pool_paths[y,s]
    event('RAW_ZIPS_VERIFIED',files=len(pool_paths),total_compressed_bytes=sum(p.stat().st_size for p in pool_paths.values()))
    pool=source.source_pool();sample,_=source.sample(pool,100000,739)
    assert len(pool)==2254998 and sample.ST.nunique()==51
    control=controls.summarize(pool,pool.PWGTP)
    dump(src/'historical_local_controls.json',dict(full_national_source=control))
    source_fingerprint=hashlib.sha256(pd.util.hash_pandas_object(sample,index=True).values.tobytes()).hexdigest()
    event('SOURCE_SAMPLE_REBUILT',source_adults=len(pool),training_adults=len(sample),sample_fingerprint=source_fingerprint)
    del pool
    cats=['ST','SEX','NATIVITY','RAC1P','CIT']
    transform=ColumnTransformer([('age',StandardScaler(),['AGEP']),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)]).fit(sample)
    model,audit=train49.fit_models(sample,transform,None,seed=739,only=['HGB'])
    (src/'models').mkdir()
    (src/'models/source_transformers.pkl').write_bytes(pickle.dumps(dict(transformer=transform),protocol=5))
    (src/'models/national_large_739.pkl').write_bytes(pickle.dumps(model,protocol=5))
    dump(src/'source_HGB_fit_audit.json',audit)
    event('SOURCE_HGB_REFITTED',iterations=int(model['HGB'].n_iter_))
    del sample,model,transform
    data.ROOT=work;data.OUT=out;data.SOURCE=src;data.zip_path=lambda y,s:pool_paths[y,s]
    risk.OUT=out;runner.OUT=out
    for s in STATES:data.read_background(2022,s)
    W=np.zeros(12)
    for s in STATES:
        with np.load(out/f'background/2022_{s}.npz') as z:W+=np.bincount(z['g'],weights=z['w'],minlength=12)
    W=W.reshape(6,2);q=.5*(W[:,0]/W[:,0].sum()+W[:,1]/W[:,1].sum())
    dump(out/'standardization.json',dict(q=q.tolist(),reference='Raw-rebuilt pooled eight-state 2022 reference; equal nativity average'))
    for s in STATES:
        data.read_labels(2022,s)
        risk.background_units(2022,s);risk.score_arrays(2022,s)
    event('HISTORICAL_HOUSEHOLDS_REBUILT',states=8,q=q.tolist())
    X=[];y=[];g=[];T=[]
    for s in STATES:
        with np.load(out/f'units/2022_{s}.npz') as z:X.append(z['X']);g.append(z['stratum'])
        with np.load(out/f'scores/2022_{s}.npz') as z:y.append(z['T'][:,0]**2);T.append(z['T'][:,0])
    X=np.concatenate(X);y=np.concatenate(y);g=np.concatenate(g);T=np.concatenate(T)
    models,scale=risk.fit_models(X,y)
    prior_sd=np.array([T[g==k].std(ddof=1) if (g==k).sum()>1 else T.std(ddof=1) for k in range(8)])
    prior_risk=np.array([y[g==k].mean() if np.any(g==k) else y.mean() for k in range(8)])
    package=dict(models=models,scale=scale,prior_sd=prior_sd,prior_risk=prior_risk,features=data.FEATURES)
    (out/'risk_models.pkl').write_bytes(pickle.dumps(package))
    event('HOUSEHOLD_RISK_MODELS_REFITTED',households=len(y),RF_trees=len(models['RF'].estimators_),HGB_iterations=int(models['HGB'].n_iter_),
          scope='Final fitted risk models only; fivefold diagnostics do not feed this CT design and are not rerun')
    del X,y,g,T,models,package
    data.read_background(2024,'ct');risk.background_units(2024,'ct')
    risk.DEV=[];risk.TARGET=[(2024,'ct')];risk.predict_all()
    shutil.copy2(PUBLIC/'metadata/frozen/implementation_contract.json',out/'implementation_contract.json')
    runner.setup_designs(2024,'ct',1000)
    files={p.relative_to(work).as_posix():sha(p) for folder in [out/'background',out/'units',out/'risk_predictions',out/'designs',src/'models'] for p in folder.rglob('*') if p.is_file()}
    for p in [out/'standardization.json',out/'risk_models.pkl',out/'implementation_contract.json',src/'historical_local_controls.json']:
        files[p.relative_to(work).as_posix()]=sha(p)
    dump(out/'final_freeze.json',dict(status='Technical rebuild of already-scored CT; not a new confirmation freeze',files=files,plan_sha256=sha(plan_path)))
    event('REBUILT_PREDICTIONS_AND_DESIGN_FROZEN',target_education_opened_in_rebuild=False,files=len(files))
    data.read_labels(2024,'ct',final=True);risk.score_arrays(2024,'ct')
    runner.run_task((2024,'ct',1000,'raw_rebuild',2000))
    event('FULL_CT_DRAWS_REBUILT',repetitions=2000,methods=16)
    reference=PUBLIC/'data/saved_draws/2024_ct_1000.npz'
    if reference.exists():
        with np.load(reference,allow_pickle=False) as a,np.load(out/'draws/raw_rebuild/2024_ct_1000.npz',allow_pickle=False) as b:
            differences={}
            for key in ['point','variance','count','resources','GREG_cell_ranks','CPS_GREG_ranks']:
                differences[key]=float(np.max(np.abs(a[key]-b[key])))
                if key in ['point','variance']:assert np.allclose(a[key],b[key],rtol=2e-10,atol=2e-12),(key,differences[key])
                else:assert np.array_equal(a[key],b[key]),(key,differences[key])
        dump(work/'saved_reference_comparison.json',dict(differences=differences,tolerances=dict(rtol=2e-10,atol=2e-12),
               reference_used_only_after_refit_and_resampling=True))
        event('SAVED_REFERENCE_COMPARISON_PASSED',max_difference=max(differences.values()))
    dump(work/'rebuild_completed.json',dict(utc=datetime.now(timezone.utc).isoformat(),repetitions=2000,methods=16,
        source_HGB_retrained=True,household_risk_models_retrained=True,pretrained_or_derived_array_inputs=False,
        skipped_components=['Other unused source predictor families','Risk fivefold diagnostics','Other final states/budgets'],
        reconstruction_scope='Used source and design path for CT 1000-HU across all methods, not entire historical research programme',events=events))
    print('RAW_RECONSTRUCTION_COMPLETE',work,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',default=str(ROOT/'data/raw'));p.add_argument('--workdir',default=str(STUDY/'raw_CT_rebuild'));a=p.parse_args()
    run(a.raw_dir,a.workdir)

"""Prespecified development and once-only confirmation of household reporting."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,pickle,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from acs183_reporting_data import ROOT,OUT,FINAL,SEEN,TARGET,DEV,R,FIPS,loadjson,dump,log,sha,utc,read_labels,verify_old
from acs185_reporting_risk import score_arrays
from acs184_reporting_methods import integer_allocation,group_indices,strat_estimate,two_wave,cps_setup,cps_counts,cps_estimate,srs_greg,srs_greg_cells,cps_greg_cells,checks

METHODS=['SRS_HT','SRS_historical_cell','SRS_HGB','SRS_GREG','proportional_HGB','Neyman_HGB','two_wave_HGB','CPS_coarse','CPS_RF','CPS_HGB','CPS_robust_HGB','SRS_GREG_cells','Neyman_working_fine','CPS_working','CPS_working_GREG','CPS_RF_GREG']
ROBUST_RHOS=[0.,.25,.5,.75,1.]

def risk_bins(stratum,risk):
    # Background-only common bins; retain coarse cells while approximating continuous risk.
    g=np.zeros(len(risk),int);counter=0
    for k in range(8):
        ix=np.flatnonzero(stratum==k)
        if not len(ix):continue
        edges=np.unique(np.quantile(risk[ix],[.25,.5,.75]))
        parts=np.searchsorted(edges,risk[ix],side='right')
        for j in np.unique(parts):g[ix[parts==j]]=counter;counter+=1
    return g

def setup_designs(year,s,m):
    dest=OUT/f'designs/{year}_{s}_{m}.pkl'
    if dest.exists():
        existing=pickle.loads(dest.read_bytes())
        if 'working_groups' in existing:return
    with np.load(OUT/f'units/{year}_{s}.npz') as z:g=z['stratum'];H=len(g);working=z['working_risk']
    assert m<H
    with np.load(OUT/f'risk_predictions/{year}_{s}.npz') as z:risks=z['risk']
    models=pickle.loads((OUT/'risk_models.pkl').read_bytes());sd=models['prior_sd']
    groups=group_indices(g);N=np.array([len(x) for x in groups])
    proportional=integer_allocation(N,np.ones(len(N)),m)
    neyman=integer_allocation(N,sd[:len(N)],m)
    cps=[];cgroups=[];allbins=[]
    for j in range(3):
        bins=risk_bins(g,risks[:,j]);groups_j=group_indices(bins);N_j=np.array([len(ix) for ix in groups_j]);risk_j=np.array([risks[ix,j].mean() for ix in groups_j])
        cps.append(cps_setup(N_j,np.sqrt(risk_j),m));cgroups.append(groups_j);allbins.append(bins)
    bins=allbins[2];CN=np.array([len(ix) for ix in cgroups[2]]);B=np.array([risks[ix].mean(0) for ix in cgroups[2]])
    # Li et al.-style L2 misspecification penalty, fixed analyst-selected radius.
    # This is explicitly a binned fixed-size adaptation, not an exact rerun of original ASI.
    alternatives=[]
    for rho in ROBUST_RHOS:
        candidate=cps_setup(CN,np.sqrt(B[:,2]),m,rho)
        pi=candidate['pi'];radius=.2*np.sqrt(np.dot(CN,B[:,2]**2))
        objective=np.dot(CN,B[:,2]/pi)+radius*np.sqrt(np.dot(CN,1/pi**2))
        alternatives.append((objective,rho,candidate))
    best=min(alternatives,key=lambda a:a[0]);cps.append(best[2]);cgroups.append(cgroups[2])
    wb=risk_bins(g,working);wg=group_indices(wb);WN=np.array([len(ix) for ix in wg]);wr=np.array([working[ix].mean() for ix in wg])
    cps.append(cps_setup(WN,np.sqrt(wr),m));cgroups.append(wg)
    working_n=integer_allocation(WN,np.sqrt(wr),m)
    contract=dict(year=year,state=s,m=m,H=H,groups=groups,proportional=proportional,neyman=neyman,prior_sd=sd[:len(N)],
      bins=bins,cgroups=cgroups,cps=cps,working_groups=wg,working_n=working_n,robust_rho=best[1],robust_objectives=[dict(rho=rho,objective=float(obj)) for obj,rho,_ in alternatives],
      family='Exact prescribed marginals for bin-constant CPS; Deville approximate variance; no no-worse finite-sample guarantee',
      adaptation='8demographic strata × up to4own-model-risk bins;20%L2radius; one fixed predictor',
      labels_used=False)
    dest.parent.mkdir(exist_ok=True);dest.write_bytes(pickle.dumps(contract))
    dump(f'designs/{year}_{s}_{m}_audit.json',dict(H=H,m=m,strata=len(N),risk_bins=len(CN),robust_rho=best[1],
      max_marginal_error=max(x['marginal_error'] for x in cps),max_calibration_iterations=max(x['calibration_iterations'] for x in cps),
      labels_used=False,sha256=sha(dest)))

def prepare_designs(which):
    pairs=[(2023,s) for s in FINAL]+[(2024,s) for s in SEEN] if which=='dev' else TARGET
    for year,s in pairs:
        for m in [300,1000,3000]:setup_designs(year,s,m)
        print('DESIGNS',year,s,'READY',flush=True)

def run_task(task):
    year,s,m,phase,rr=task
    dest=OUT/f'draws/{phase}/{year}_{s}_{m}.npz'
    if dest.exists():
        with np.load(dest) as z:assert len(z['point'])==rr
        return
    start=time.monotonic();design=pickle.loads((OUT/f'designs/{year}_{s}_{m}.pkl').read_bytes())
    with np.load(OUT/f'units/{year}_{s}.npz') as z:F=z['F'];Fc=z['Fc'];Z=z['Z'];g=z['stratum'];adult=z['adult_counts'];foreign=z['X'][:,1]*adult
    with np.load(OUT/f'scores/{year}_{s}.npz') as z:Y=z['Y'];T=z['T'];Tc=z['Tc'];truth=z['truth']
    H=len(T);Fmean=F.mean(0);Fcmean=Fc.mean(0);Zmean=Z.mean(0);groups=design['groups'];point=np.full((rr,len(METHODS),2),np.nan);variance=point.copy()
    count=np.empty((rr,len(METHODS)),int);resources=np.empty((rr,len(METHODS),2));ranks=np.empty(rr,int);cps_ranks=np.empty((rr,2),int);unique_ok=True
    cps_draws=[]
    seed=739+100000*year+100*FIPS[s]+m
    for j,setup in enumerate(design['cps']):
        cps_draws.append(cps_counts(setup,rr,np.random.default_rng(seed+8000000+10000*j)))
    full_pi=[]
    for gg,setup in zip(design['cgroups'],design['cps']):
        ps=np.empty(H)
        for ix,prob in zip(gg,setup['pi']):ps[ix]=prob
        full_pi.append(ps)
    for r in range(rr):
        rng=np.random.default_rng(seed+100000000+r)
        # Reuse exactly the same SRS for all four estimator controls.
        si=rng.choice(H,m,replace=False)
        resources[r,:4]=[adult[si].sum(),foreign[si].sum()]
        for j,values,pred in [(0,Y,np.zeros(2)),(1,Tc,Fcmean),(2,T,Fmean)]:
            point[r,j]=pred+values[si].mean(0);variance[r,j]=(1-m/H)*values[si].var(0,ddof=1)/m;count[r,j]=m
        point[r,3],variance[r,3]=srs_greg(Y,F,si,H,Fmean);count[r,3]=m
        point[r,11],variance[r,11],ranks[r]=srs_greg_cells(Y,Z,si,H,Zmean);count[r,11]=m;resources[r,11]=resources[r,0]
        rng=np.random.default_rng(seed+200000000+r)
        wp,wv,wi=strat_estimate(design['working_groups'],design['working_n'],T,rng)
        point[r,12]=Fmean+wp;variance[r,12]=wv;count[r,12]=len(wi);resources[r,12]=[adult[wi].sum(),foreign[wi].sum()]
        for j,n in [(4,design['proportional']),(5,design['neyman'])]:
            rng=np.random.default_rng(seed+200000000+r)
            p,v,selected=strat_estimate(groups,n,T,rng);point[r,j]=Fmean+p;variance[r,j]=v;count[r,j]=len(selected)
            unique_ok &= len(np.unique(selected))==m
            resources[r,j]=[adult[selected].sum(),foreign[selected].sum()]
        rng=np.random.default_rng(seed+300000000+r)
        p,v,selected=two_wave(groups,m,T,design['prior_sd'],rng)
        point[r,6]=Fmean+p;variance[r,6]=v;count[r,6]=len(selected);unique_ok &= len(np.unique(selected))==m
        resources[r,6]=[adult[selected].sum(),foreign[selected].sum()]
        for j,(setup,counts) in enumerate(zip(design['cps'],cps_draws)):
            rng=np.random.default_rng(seed+400000000+r)
            p,v,selected=cps_estimate(design['cgroups'][j],counts[r],setup['pi'],T,rng)
            method_index=7+j if j<4 else 13
            point[r,method_index]=Fmean+p;variance[r,method_index]=v;count[r,method_index]=len(selected);unique_ok &=len(np.unique(selected))==m
            resources[r,method_index]=[adult[selected].sum(),foreign[selected].sum()]
            if j in [1,4]:
                entry=15 if j==1 else 14
                point[r,entry],variance[r,entry],rank=cps_greg_cells(Y,Z,selected,full_pi[j][selected],H,Zmean)
                cps_ranks[r,entry-14]=rank;count[r,entry]=m;resources[r,entry]=resources[r,method_index]
    assert np.isfinite(point).all() and np.isfinite(variance).all() and (variance>=0).all()
    assert (count==m).all() and unique_ok
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,point=point,variance=variance,truth=truth,methods=np.array(METHODS),count=count,resources=resources,GREG_cell_ranks=ranks,CPS_GREG_ranks=cps_ranks,year=year,state=s,m=m,seed=seed,
      freeze_sha256=np.array(sha(OUT/'final_freeze.json') if phase=='final' else sha(OUT/'implementation_contract.json')))
    dump(f'draws/{phase}/{year}_{s}_{m}_QA.json',dict(draws=rr,methods=len(METHODS),m=m,all_counts_exact=True,all_unique=True,all_finite=True,
      CPS_marginal_error=max(x['marginal_error'] for x in design['cps']),GREG_cell_rank_deficient_draws=int((ranks<14).sum()),seconds=time.monotonic()-start,sha256=sha(dest)))
    if phase=='dev16':
        with np.load(OUT/f'draws/dev/{year}_{s}_{m}.npz') as old:
            assert np.array_equal(point[:,:12],old['point']) and np.array_equal(variance[:,:12],old['variance']),'Original12method replay changed'
    print('RUN_COMPLETE',phase,year,s,m,rr,round(time.monotonic()-start,1),'s',flush=True)

def contract():
    assert not (OUT/'implementation_contract.json').exists()
    qa=checks();dump('numerical_checks.json',qa)
    dump('implementation_contract.json',dict(utc=utc(),methods=METHODS,primary_comparator='SRS_HGB; always also report SRS_GREG and Neyman_HGB',
      main_estimator='Fixed2017HGB generalized difference for all allocation policies; SRS_HT/historical_cell/GREG are separate strong estimator controls',
      main_budgets=[300,1000,3000],primary=1000,R=R,
      variance='Exact stratified finite-population for fixed predictor; conditional-remainder estimator/variance for adaptive two-wave; Deville approximate CPS; first-order GREG',
      two_wave='20% initial proportional sample, at least2per nonempty coarse stratum; variance shrinkage20prior df; bounded integer residual Neyman; at least2per nonempty remainder; count both waves',
      robust='Binned fixed-size adaptation of uniform/active path +20%L2uncertainty norm, rho grid0,.25,.5,.75,1; no target labels',
      CPS='Grouped independent binomial rejection conditional total=m; calibrate internal odds to prescribed marginal inclusion probabilities; max marginal error2e-10',
      sample_units='HU released unit; entire eligible household; m denotes HUs, not individual labels, contacts or currency',
      frozen_before_main_development=True,checks=qa))

def freeze():
    assert not (OUT/'final_freeze.json').exists(),'Final confirmation can only be frozen once'
    assert (OUT/'development_completed.json').exists()
    assert not any((OUT/f'labels/2024_{s}.npz').exists() for s in FINAL)
    assert len(FINAL)>=6 and len(set(REG for REG in ['NE','NE','MW','MW','S','S','W','W']))>=3
    files={}
    for directory in ['background','units','risk_predictions','designs']:
        for p in (OUT/directory).rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [OUT/'protocol.json',OUT/'implementation_contract.json',OUT/'standardization.json',OUT/'risk_models.pkl',OUT/'analysis_and_figure_contract.json',OUT/'development_selection.json',OUT/'roster_sequence_integrity.json',OUT/'risk_training_QA.json',OUT/'prior_exposure_inventory.json']+list((ROOT/'scripts').glob('acs18[3-9]_reporting*.py')):
        assert p.exists(),p;files[p.relative_to(ROOT).as_posix()]=sha(p)
    for s in FINAL:
        p=ROOT/f'data/raw/acs_reporting183_2026-11-21/acs_2024_{s}.zip';files[p.relative_to(ROOT).as_posix()]=sha(p)
    dump('final_freeze.json',dict(utc=utc(),files=files,final_pairs=TARGET,labels_parsed=False,final_once=True,old_files_unchanged=verify_old()))
    log('final_freeze',labels_read=False,sha256=sha(OUT/'final_freeze.json'));print('FINAL_FROZEN',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','designs_dev','designs_final','pilot','dev','freeze','final','replay']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.mode=='contract':contract()
    elif a.mode.startswith('designs'):prepare_designs(a.mode.split('_')[1])
    elif a.mode=='pilot':run_task((2023,'ct',1000,'pilot',32))
    elif a.mode=='freeze':freeze()
    elif a.mode in ['dev','final']:
        final=a.mode=='final';pairs=TARGET if final else [(2023,s) for s in FINAL]+[(2024,s) for s in SEEN]
        if final:
            assert (OUT/'final_freeze.json').exists() and not (OUT/'final_started.json').exists(),'Final evaluation already started'
            dump('final_started.json',dict(utc=utc(),freeze_sha256=sha(OUT/'final_freeze.json')))
            for y,s in pairs:read_labels(y,s,True);score_arrays(y,s)
        tasks=[(y,s,m,'final' if final else 'dev16',R) for y,s in pairs for m in [300,1000,3000]]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_task,tasks))
        dump('final_completed.json' if final else 'development_completed.json',dict(utc=utc(),state_years=len(pairs),tasks=len(tasks),draws_each=R,methods=len(METHODS),old_files_unchanged=verify_old()))
    else:
        target=OUT/'draws/final/2024_ct_1000.npz';assert target.exists()
        run_task((2024,'ct',1000,'replay',8))
        with np.load(target) as original,np.load(OUT/'draws/replay/2024_ct_1000.npz') as replay:
            dif=max(float(np.max(abs(original[k][:8]-replay[k]))) for k in ['point','variance','count','resources'])
            assert dif==0,dif
        dump('saved_draw_replay_QA.json',dict(first_draws_recomputed=8,all_methods=len(METHODS),max_point_variance_count_resource_difference=dif,
          no_new_label_parse=True,same_frozen_random_stream_prefix=True,sha256=sha(target)))

if __name__=='__main__':main()

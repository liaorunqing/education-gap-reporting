"""Background-only unit construction; historical risk learning; separate scoring arrays."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,pickle,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from acs183_reporting_data import ROOT,OUT,FINAL,SEEN,DEV,TARGET,FEATURES,loadjson,dump,log,sha,utc

def coefficients(bg):
    g,w,h=bg['g'],bg['w'],bg['h'];H=len(bg['unit']);q=np.array(loadjson(OUT/'standardization.json')['q'])
    W=np.bincount(g,weights=w,minlength=12);assert (W>0).all()
    sign=2*(g%2)-1
    a=q[g//2]*sign*w/W[g]
    Wn=W.reshape(6,2).sum(0);ar=sign*w/Wn[g%2]
    return H*np.column_stack([a,ar])

def background_units(year,s):
    dest=OUT/f'units/{year}_{s}.npz'
    if dest.exists():
        with np.load(dest) as old:
            if 'working_risk' in old.files:return
    with np.load(OUT/f'background/{year}_{s}.npz') as z:bg={k:z[k] for k in z.files}
    h=bg['h'];H=len(bg['unit']);b=coefficients(bg);f=bg['f'];fc=bg['fc'];n=np.bincount(h,minlength=H)
    def agg(v):return np.bincount(h,weights=v,minlength=H)
    F=np.column_stack([agg(b[:,j]*f) for j in range(2)])
    Fc=np.column_stack([agg(b[:,j]*fc) for j in range(2)])
    Z=np.column_stack([agg(b[:,0]*(bg['g']==k)) for k in range(12)]+[F[:,0],F[:,1]])
    working_risk=agg(b[:,0]**2*f*(1-f))
    foreign=agg(bg['nativity']==2);older=agg(bg['age']>=60);female=agg(bg['sex']==2)
    mu=agg(f)/n;sd=np.sqrt(np.maximum(agg(f*f)/n-mu*mu,0))
    maxb=np.zeros(H);np.maximum.at(maxb,h,abs(b[:,0]))
    wm=agg(bg['w'])/n;wcv=np.sqrt(np.maximum(agg(bg['w']**2)/n-wm*wm,0))/wm
    mixed=(foreign>0)&(foreign<n)
    X=np.column_stack([n,foreign/n,older/n,female/n,agg(bg['age']<40)/n,mu,sd,agg(abs(b[:,0])),agg(b[:,0]**2),maxb,wcv,mixed])
    stratum=((foreign>0)*4+(older>0)*2+(n>1)).astype(int)
    assert np.isfinite(X).all() and np.isfinite(F).all()
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,X=X,F=F,Fc=Fc,Z=Z,working_risk=working_risk,stratum=stratum,unit=bg['unit'],adult_counts=n,features=np.array(FEATURES))
    log('unit_background',year=year,state=s,labels_read=False,H=H)

def score_arrays(year,s):
    dest=OUT/f'scores/{year}_{s}.npz'
    description=OUT/f'scores/{year}_{s}_description.json'
    if dest.exists() and description.exists() and 'cell_outcome_means' in loadjson(description):return
    if (year,s) in TARGET:assert (OUT/'final_freeze.json').exists()
    with np.load(OUT/f'background/{year}_{s}.npz') as z:bg={k:z[k] for k in z.files}
    with np.load(OUT/f'labels/{year}_{s}.npz') as z:y=z['y'];flag=z['flag'];rw=z['repw'];assert np.array_equal(z['ids'],bg['ids'])
    with np.load(OUT/f'units/{year}_{s}.npz') as z:F=z['F'];Fc=z['Fc'];X=z['X'];stratum=z['stratum']
    b=coefficients(bg);h=bg['h'];H=len(F)
    Y=np.column_stack([np.bincount(h,weights=b[:,j]*y,minlength=H) for j in range(2)])
    truth=Y.mean(0);g=bg['g'];w=bg['w'];q=np.array(loadjson(OUT/'standardization.json')['q'])
    W=np.bincount(g,weights=w,minlength=12);YW=np.bincount(g,weights=w*y,minlength=12);mu=YW/W
    repW=np.array([rw[g==k].sum(0) for k in range(12)])
    repY=np.array([(rw[g==k]*y[g==k,None]).sum(0) for k in range(12)])
    repmu=repY/repW;repgap=(q[:,None]*(repmu[1::2]-repmu[::2])).sum(0)
    rW=repW.reshape(6,2,80).sum(0);rY=repY.reshape(6,2,80).sum(0)
    rawrep=rY[1]/rW[1]-rY[0]/rW[0]
    se=np.sqrt(.05*np.sum((np.column_stack([repgap,rawrep])-truth)**2,axis=0))
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,Y=Y,T=Y-F,Tc=Y-Fc,truth=truth)
    npair=W.reshape(6,2).sum(0);fw=npair[1]/npair.sum();ess=(w[g%2==1].sum()**2/(w[g%2==1]**2).sum())
    means=np.bincount(stratum,weights=Y[:,0]-F[:,0],minlength=8)/np.maximum(np.bincount(stratum,minlength=8),1)
    risk=(Y[:,0]-F[:,0])**2
    individual_diagonal=float(np.sum((b[:,0]*(y-bg['f']))**2))
    group_second=float(np.sum(risk))
    desc=dict(year=year,state=s,H=H,adults=len(y),foreign_adults=int((g%2==1).sum()),foreign_weight_share=float(fw),foreign_weight_ESS=float(ess),
      older_weight_share=float(w[bg['age']>=60].sum()/w.sum()),mixed_nativity_household_share=float(X[:,-1].mean()),
      standardized_gap=float(truth[0]),raw_gap=float(truth[1]),standardized_ACS_replicate_SE=float(se[0]),raw_ACS_replicate_SE=float(se[1]),
      cell_weight_totals=W.tolist(),cell_outcome_means=mu.tolist(),
      mean_HGB_prediction_bias=float((F.mean(0)-truth)[0]),mean_cell_prediction_bias=float((Fc.mean(0)-truth)[0]),
      allocation_flag_weight_share=float(np.sum(w*flag)/w.sum()),
      population_residual_variance=float((Y[:,0]-F[:,0]).var(ddof=1)),mean_residual_risk=float(risk.mean()),
      individual_diagonal_residual_sum=individual_diagonal,household_residual_squared_sum=group_second,
      within_household_cross_term_fraction=float((group_second-individual_diagonal)/group_second),
      true_gap_near_reporting_band=bool(abs(truth[0])<=.02),
      scope='Released HU finite-frame targets; replicate SE describes original survey with fixed reference q, not extra-label experiment SE')
    dump(f'scores/{year}_{s}_description.json',desc)
    log('score_array',year=year,state=s,confirmation=(year,s) in TARGET)

def fit_models(X,y):
    scale=max(float(np.mean(y)),1e-8);yn=y/scale
    models={
      'RF':RandomForestRegressor(n_estimators=200,min_samples_leaf=20,max_features=1.,n_jobs=1,random_state=739),
      'HGB':HistGradientBoostingRegressor(loss='poisson',max_iter=200,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=739)}
    with threadpool_limits(limits=1):
        for name,model in models.items():model.fit(X,yn)
    return models,scale

def fold_job(task):
    fold,train,test=task
    with np.load(OUT/'risk_training_arrays.npz') as z:X=z['X'];y=z['y'];g=z['stratum']
    models,scale=fit_models(X[train],y[train]);pred=[]
    pred.append(np.full(len(test),y[train].mean()))
    means=np.array([y[train][g[train]==k].mean() if np.any(g[train]==k) else y[train].mean() for k in range(8)])
    pred.append(means[g[test]])
    pred.extend([np.maximum(models[n].predict(X[test])*scale,1e-10) for n in ['RF','HGB']])
    np.savez_compressed(OUT/f'risk_folds/fold{fold}.npz',test=test,prediction=np.array(pred).T)
    print('RISK_FOLD',fold,'DONE',flush=True)
    return fold

def train():
    assert not (OUT/'risk_models.pkl').exists(),'Never refit silently'
    X=[];y=[];g=[];states=[];prior=[]
    for s in FINAL:
        with np.load(OUT/f'units/2022_{s}.npz') as z:X.append(z['X']);g.append(z['stratum']);states.extend([s]*len(z['X']))
        with np.load(OUT/f'scores/2022_{s}.npz') as z:y.append(z['T'][:,0]**2);prior.append(z['T'][:,0])
    X=np.concatenate(X);y=np.concatenate(y);g=np.concatenate(g);T=np.concatenate(prior)
    np.savez_compressed(OUT/'risk_training_arrays.npz',X=X,y=y,stratum=g,states=np.array(states))
    (OUT/'risk_folds').mkdir(exist_ok=True)
    tasks=[(j,a,b) for j,(a,b) in enumerate(KFold(5,shuffle=True,random_state=739).split(X))]
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(fold_job,tasks))
    prediction=np.empty((len(y),4))
    for j in range(5):
        with np.load(OUT/f'risk_folds/fold{j}.npz') as z:prediction[z['test']]=z['prediction']
    base=np.mean((y-prediction[:,0])**2)
    metrics=[]
    for j,name in enumerate(['constant','coarse_stratum','RF','HGB']):
        metrics.append(dict(model=name,MSE=float(np.mean((y-prediction[:,j])**2)),relative_MSE=float(np.mean((y-prediction[:,j])**2)/base),mean_prediction=float(prediction[:,j].mean()),mean_risk=float(y.mean())))
    models,scale=fit_models(X,y)
    prior_sd=np.array([T[g==k].std(ddof=1) if (g==k).sum()>1 else T.std(ddof=1) for k in range(8)])
    prior_risk=np.array([y[g==k].mean() if np.any(g==k) else y.mean() for k in range(8)])
    package=dict(models=models,scale=scale,prior_sd=prior_sd,prior_risk=prior_risk,features=FEATURES)
    (OUT/'risk_models.pkl').write_bytes(pickle.dumps(package))
    dump('risk_training_QA.json',dict(utc=utc(),households=len(y),folds=5,whole_unit_rows=True,models=metrics,model_sha256=sha(OUT/'risk_models.pkl'),source_year=2022,no_confirmation_labels=True))
    print('RISK_MODELS_TRAINED',len(y),metrics,flush=True)

def predict_all():
    package=pickle.loads((OUT/'risk_models.pkl').read_bytes())
    for year,s in DEV+TARGET:
        dest=OUT/f'risk_predictions/{year}_{s}.npz'
        if dest.exists():continue
        with np.load(OUT/f'units/{year}_{s}.npz') as z:X=z['X'];g=z['stratum']
        with threadpool_limits(limits=1):P=np.column_stack([package['prior_risk'][g]]+[np.maximum(package['models'][n].predict(X)*package['scale'],1e-10) for n in ['RF','HGB']])
        dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,risk=P,models=np.array(['coarse','RF','HGB']))
        log('risk_predictions',year=year,state=s,current_labels_used=False)
    print('ALL_RISK_PREDICTIONS_SAVED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['units','dev_scores','train','predict']);a=p.parse_args()
    if a.mode=='units':
        for y,s in DEV+TARGET:background_units(y,s)
    elif a.mode=='dev_scores':
        for y,s in DEV:score_arrays(y,s)
    elif a.mode=='train':train()
    else:predict_all()

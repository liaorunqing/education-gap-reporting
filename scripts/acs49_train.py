"""Frozen national-source fitting; preserve prior2-state models and experiments."""
from datetime import datetime,timezone
import json,pickle,time,warnings
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from acs49_data import ROOT,OUT,FEATURES,sha
from acs26_models import LogisticMixed

def fit_models(d,transformer,gam_transformer,seed=739,only=None):
    cats=['ST','SEX','NATIVITY','RAC1P','CIT'];y=d.y.to_numpy(float)
    w=d.fit_weight.to_numpy(float).copy();w/=w.mean()
    X=transformer.transform(d);models={};audits=[];start=time.monotonic()
    recipes={'logistic':LogisticRegression(C=1,max_iter=1000,random_state=seed),
        'RF':RandomForestClassifier(n_estimators=200,min_samples_leaf=20,max_features=1.,n_jobs=1,random_state=seed),
        'HGB':HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,
            l2_regularization=1,early_stopping=False,random_state=seed)}
    with threadpool_limits(limits=2):
      for name,model in recipes.items():
        if only and name not in only:continue
        with warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter('always',ConvergenceWarning)
            models[name]=model.fit(X,y,sample_weight=w)
        conv=[str(z.message) for z in wr if issubclass(z.category,ConvergenceWarning)]
        assert not conv,(name,conv)
        audits.append({'model':name,'fit_success':True,'seed':seed,
            'iterations':np.asarray(getattr(model,'n_iter_',[])).tolist(),'seconds':time.monotonic()-start})
      if only is None:
        gam=LogisticRegression(C=1,max_iter=2000,tol=1e-7)
        with warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter('always',ConvergenceWarning)
            gam.fit(gam_transformer.transform(d),y,sample_weight=w)
        conv=[str(z.message) for z in wr if issubclass(z.category,ConvergenceWarning)]
        assert not conv and gam.n_iter_[0]<2000,conv
        models['GAM']=gam;audits.append({'model':'GAM','fit_success':True,'seed':seed,'iterations':gam.n_iter_.tolist()})
        mixed=LogisticMixed().fit(X,y,w,d.group.astype(str).to_numpy())
        models['GLMM']=mixed;audits.append({'model':'GLMM',**mixed.audit,'seed':seed})
    return models,audits

def main():
    path=OUT/'source_training_frames.pkl';assert path.exists()
    assert not (OUT/'national_training_completed.json').exists()
    frames=pickle.loads(path.read_bytes());cats=['ST','SEX','NATIVITY','RAC1P','CIT']
    reference=frames['national_large']
    transform=ColumnTransformer([('age',StandardScaler(),['AGEP']),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)]).fit(reference)
    gt=ColumnTransformer([('age',SplineTransformer(n_knots=5,degree=3,include_bias=False),['AGEP']),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)]).fit(reference)
    modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
    transformer_path=modeldir/'source_transformers.pkl'
    if not transformer_path.exists():transformer_path.write_bytes(pickle.dumps({'transformer':transform,'gam_transformer':gt},protocol=5))
    freeze={'utc':datetime.now(timezone.utc).isoformat(),'source_frames_sha256':sha(path),
        'training_plan_sha256':sha(OUT/'source_training_plan.json'),'code_sha256':sha(ROOT/'scripts/acs49_train.py'),
        'working_GLMM_code_sha256':sha(ROOT/'scripts/acs26_models.py'),'transformer_sha256':sha(transformer_path),
        'target_education_or_flags_read':False}
    fp=OUT/'source_fit_freeze.json'
    if fp.exists():assert json.loads(fp.read_text())['code_sha256']==freeze['code_sha256']
    else:fp.write_text(json.dumps(freeze,indent=2),encoding='utf8')
    audit=[]
    for arm,d in frames.items():
        dest=modeldir/f'{arm}_739.pkl';ap=modeldir/f'{arm}_739_audit.json'
        if dest.exists():
            assert ap.exists();audit+=json.loads(ap.read_text());continue
        models,records=fit_models(d,transform,gt)
        dest.write_bytes(pickle.dumps(models,protocol=5))
        controls={}
        for cell,v in d.groupby('cell'):
            controls[str(cell)]={'N':len(v),'W':float(v.fit_weight.sum()),'mean':float(np.average(v.y,weights=v.fit_weight))}
        (modeldir/f'{arm}_controls.json').write_text(json.dumps({'global_mean':float(np.average(d.y,weights=d.fit_weight)),
            'cells':controls,'source_year':2017,'training_arm':arm},indent=2),encoding='utf8')
        for r in records:r.update(arm=arm,model_file_sha256=sha(dest),source_adults=len(d),source_states=int(d.ST.nunique()))
        ap.write_text(json.dumps(records,indent=2),encoding='utf8');audit+=records
        print('SOURCE_ARM_FITTED',arm,len(d),d.ST.nunique(),flush=True)
    pd.DataFrame(audit).to_csv(OUT/'source_model_fit_audit.csv',index=False)
    (OUT/'national_training_completed.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'arms':4,'primary_model_fits':len(audit),'all_primary_models_fitted':len(audit)==20,
        'target2023_education_and_flags_read':False,'secondary_seed_fits':'Pending as predeclared; do not treat primary as source-sampling uncertainty assessment.',
        'model_hashes':{p.name:sha(p) for p in modeldir.glob('*.pkl')}},indent=2),encoding='utf8')
if __name__=='__main__':main()

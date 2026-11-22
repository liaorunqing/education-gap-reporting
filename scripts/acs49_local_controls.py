"""Strong local historical controls enabled by national source geography."""
import json,pickle
import numpy as np,pandas as pd
from acs49_data import OUT
from acs49_source import source_pool

def summarize(d,weights):
    t=d.copy();t['_W']=weights;t['_WY']=weights*t.y
    g=t.groupby(['ST','cell']).agg(N=('y','size'),W=('_W','sum'),WY=('_WY','sum'))
    g['mean']=g.WY/g.W
    return [{'state':int(st),'cell':int(c),'N':int(r.N),'W':float(r.W),'mean':float(r['mean'])}
        for (st,c),r in g.iterrows()]
def main():
    path=OUT/'historical_local_controls.json';assert not path.exists()
    frames=pickle.loads((OUT/'source_training_frames.pkl').read_bytes());pool=source_pool()
    out={'full_national_source':summarize(pool,pool.PWGTP),
        'by_training_arm':{arm:summarize(d,d.fit_weight) for arm,d in frames.items()},
        'scope':'Frozen2017weighted local state-age-SEX-nativity rates; not new calibration.',
        'fallback':'two_state has no local labels for8new targetstates; explicitly use its pooledcell control only, markfallback.',
        'volume_note':'Full-national-source control usesall2254998sourceadults; same-training-arm localcontrol matches availabletraininglabels.'}
    path.write_text(json.dumps(out,indent=2),encoding='utf8')
    print('LOCAL_HISTORICAL_BASELINES_FROZEN',len(out['full_national_source']),flush=True)
if __name__=='__main__':main()

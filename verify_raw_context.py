"""Reconstruct descriptive targets directly from a public Census ZIP; no ML refitting."""
from pathlib import Path
import argparse, hashlib, json, urllib.request, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def run(state, filename=None):
    sources = json.loads((ROOT/'metadata/census_sources.json').read_text(encoding='utf8'))
    meta = next(m for m in sources if m['filename'] == '2024_'+state)
    p = Path(filename).resolve() if filename else ROOT/'raw_downloads'/('acs_2024_'+state+'.zip')
    if not p.exists():
        if filename:
            raise FileNotFoundError(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(meta['url']) as src, p.open('wb') as dst:
            import shutil
            shutil.copyfileobj(src, dst)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == meta['sha256'], 'Source vintage differs; do not silently substitute a new file.'
    with zipfile.ZipFile(p) as z:
        member = next(n for n in z.namelist() if n.endswith('.csv'))
        with z.open(member) as f:
            df = pd.read_csv(f, usecols=['SERIALNO','SPORDER','AGEP','SEX','NATIVITY','PWGTP','SCHL'],
                             dtype={'SERIALNO':str})
    assert not df.duplicated(['SERIALNO','SPORDER']).any()
    a = df[df.SERIALNO.str.match(r'^\d{4}HU\d+$') & df.AGEP.ge(25) & df.PWGTP.gt(0)].copy()
    assert a[['AGEP','SEX','NATIVITY','PWGTP','SCHL']].notna().all().all()
    assert a.SCHL.between(1,24).all() and a.NATIVITY.isin([1,2]).all() and a.SEX.isin([1,2]).all()
    age = np.where(a.AGEP < 40, 0, np.where(a.AGEP < 60, 1, 2))
    g = 2*(2*age + a.SEX.to_numpy(int)-1) + a.NATIVITY.to_numpy(int)-1
    weight = a.PWGTP.to_numpy(float)
    w = np.bincount(g, weights=weight, minlength=12).reshape(6,2)
    y = np.bincount(g, weights=weight*a.SCHL.ge(21).to_numpy(float), minlength=12).reshape(6,2)
    assert (w > 0).all(), 'Empty cells remain a failure, not zero outcomes.'
    q = np.array(json.loads((ROOT/'metadata/frozen/standardization.json').read_text(encoding='utf8'))['q'])
    rates = y/w
    standardized = float(q @ (rates[:,1]-rates[:,0]))
    raw = float(y[:,1].sum()/w[:,1].sum() - y[:,0].sum()/w[:,0].sum())
    reference = pd.read_csv(ROOT/'data/tables/population_context.csv').set_index('state').loc[state]
    assert len(a) == reference.adults and a.SERIALNO.nunique() == reference.H
    assert np.isclose(standardized, reference.standardized_gap, rtol=0, atol=1e-12)
    assert np.isclose(raw, reference.raw_gap, rtol=0, atol=1e-12)
    result = dict(state=state, households=int(a.SERIALNO.nunique()), adults=len(a),
                  standardized_gap=standardized, raw_gap=raw, source_sha256=h,
                  scope='Raw public-source reconstruction of fixed targets; not a new holdout, field validation or replicate-SE reproduction')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--state', choices='ct nh ia ks al sc az co'.split(), default='ct')
    p.add_argument('--file', help='Optional existing public ACS 2024 state ZIP; otherwise download from Census')
    a = p.parse_args()
    run(a.state, a.file)

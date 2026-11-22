"""Fetch checksum-pinned official files; never save session headers."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,urllib.request

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/acs_public210_2026-11-22/repository'
if not PUBLIC.exists():PUBLIC=ROOT


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def one(meta,folder):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    p=folder/meta['local_name'];part=p.with_suffix('.partial')
    if p.exists():
        assert sha(p)==meta['sha256'],str(p)+' has a different vintage'
        return p
    total=meta.get('bytes')
    if total is None:
        with urllib.request.urlopen(urllib.request.Request(meta['url'],method='HEAD'),timeout=60) as r:total=int(r.headers['Content-Length'])
    size=part.stat().st_size if part.exists() else 0
    assert size<=total
    while size<total:
        end=min(size+4*2**20,total)-1
        for attempt in range(3):
            try:
                req=urllib.request.Request(meta['url'],headers={'User-Agent':'EducationGapReproducibility/1.1','Range':f'bytes={size}-{end}'})
                with urllib.request.urlopen(req,timeout=60) as r:
                    if r.status==200 and size==0:
                        block=r.read();assert len(block)==total
                    else:
                        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {size}-')
                        block=r.read();assert len(block)==end-size+1
                with part.open('ab') as f:f.write(block)
                size+=len(block);break
            except Exception:
                if attempt==2:raise
    assert sha(part)==meta['sha256'],'Official file vintage/hash mismatch: '+meta['url']
    part.replace(p);print('SOURCE_VERIFIED',p.name,total,flush=True)
    return p


def required_acs():
    sources={m['filename']:m for m in json.loads((PUBLIC/'metadata/census_sources.json').read_text(encoding='utf8'))}
    from acs49_data import STATES
    keys=[f'2017_{s}' for s in STATES]+[f'2022_{s}' for s in 'ct nh ia ks al sc az co'.split()]+['2024_ct']
    return [dict(sources[k],local_name='acs_'+k+'.zip') for k in keys]


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    items=required_acs()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:list(ex.map(lambda x:one(x,a.raw_dir),items))
    print('ALL_REQUIRED_ACS_ZIPS_VERIFIED',len(items))


if __name__=='__main__':main()

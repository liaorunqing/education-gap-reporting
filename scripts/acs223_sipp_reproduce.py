"""Raw-file entry point for the separately scoped SIPP feasibility pilot."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
from acs222_fetch_sources import one,sha

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/acs_public210_2026-11-22/repository'
if not PUBLIC.exists():PUBLIC=ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True);p.add_argument('--workdir',required=True);p.add_argument('--fetch',action='store_true');a=p.parse_args()
    work=Path(a.workdir).resolve();assert not work.exists();work.mkdir(parents=True)
    metadata=PUBLIC/'metadata/sipp'
    shutil.copy2(metadata/'sipp_pilot_plan.json',work/'sipp_pilot_plan.json')
    raw=Path(a.raw_dir).resolve();raw.mkdir(parents=True,exist_ok=True)
    for y in [2024,2025]:
        receipt=json.loads((metadata/f'sipp_acquisition_{y}.json').read_text(encoding='utf8'))
        file=raw/f'pu{y}_csv.zip'
        if a.fetch:one(dict(receipt,local_name=file.name),raw)
        assert file.is_file() and sha(file)==receipt['sha256'],file
        shutil.copy2(metadata/f'sipp_acquisition_{y}.json',work/f'sipp_acquisition_{y}.json')
    import acs220_sipp_prepare as prepare
    import acs221_sipp_pilot as pilot
    prepare.OUT=work;prepare.PRIVATE=work/'private_sipp'
    prepare.RAW_OVERRIDE=raw
    prepare.freeze_prior();prepare.audit()
    pilot.OUT=work;pilot.PRIVATE=work/'private_sipp';pilot.main()
    expected=PUBLIC/'data/sipp_pilot'
    for name in ['point_interval_results.csv','reporting_results.csv','paired_comparisons.csv']:
        x=pd.read_csv(work/'sipp_pilot'/name);y=pd.read_csv(expected/name)
        pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=2e-10,atol=2e-12)
        for c in y.select_dtypes(include='integer').columns:assert np.array_equal(x[c],y[c]),(name,c)
    print('RAW_SIPP_COHORT_AUDIT_AND_PILOT_REPRODUCED',work)


if __name__=='__main__':main()

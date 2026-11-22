"""Reproduce published summaries from saved simulation draws, without person records."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf8'))
    for name, expected in manifest.items():
        p = (ROOT/name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file(), name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
    print('RELEASE_HASHES_VERIFIED', len(manifest))


def numerical_checks():
    sys.path.insert(0, str(ROOT/'scripts'))
    from acs184_reporting_methods import checks
    print('EXACT_FINITE_POPULATION_CHECKS', json.dumps(checks(), sort_keys=True))
    expected = pd.read_csv(ROOT/'data/tables/point_interval_results.csv')
    checked = 0
    for p in sorted((ROOT/'data/saved_draws').glob('*.npz')):
        with np.load(p, allow_pickle=False) as z:
            for ei, endpoint in enumerate(['standardized','raw']):
                for j, method in enumerate(z['methods'].tolist()):
                    e = z['point'][:,j,ei] - z['truth'][ei]
                    v = z['variance'][:,j,ei]
                    row = expected[(expected.state == str(z['state'])) & (expected.m == int(z['m']))
                                   & (expected.endpoint == endpoint) & (expected.method == method)].iloc[0]
                    values = dict(MSE=np.mean(e**2), bias=e.mean(), MAE=np.abs(e).mean(),
                                  empirical_SD=e.std(ddof=1), coverage95=(np.abs(e)<=norm.ppf(.975)*np.sqrt(v)).mean(),
                                  coverage90=(np.abs(e)<=norm.ppf(.95)*np.sqrt(v)).mean(),
                                  width95=2*norm.ppf(.975)*np.sqrt(v).mean(),
                                  width90=2*norm.ppf(.95)*np.sqrt(v).mean(), mean_estimated_variance=v.mean())
                    for k, value in values.items():
                        assert np.isclose(value, row[k], rtol=2e-12, atol=2e-14), (p.name, method, endpoint, k)
                    checked += 1
    assert checked == 768
    print('SAVED_DRAW_POINT_INTERVAL_ROWS_VERIFIED', checked)


def tables(workdir):
    dest = Path(workdir).resolve()
    if dest.exists():
        raise SystemExit('Choose a NEW workdir to preserve existing results.')
    dest.mkdir(parents=True)
    base = dest/'outputs/acs_reporting183_2026-11-21'
    out = dest/'outputs/acs_hssc199_2026-11-22'
    (base/'analysis/final').mkdir(parents=True)
    (base/'draws/final').mkdir(parents=True)
    (dest/'scripts').mkdir()
    out.mkdir(parents=True)
    shutil.copy2(ROOT/'data/tables/reporting_results.csv', base/'analysis/final/reporting_results.csv')
    for p in (ROOT/'data/saved_draws').glob('*.npz'):
        shutil.copy2(p, base/'draws/final'/p.name)
    shutil.copy2(ROOT/'scripts/acs201_hssc_analysis.py', dest/'scripts/acs201_hssc_analysis.py')
    shutil.copy2(ROOT/'metadata/v8_postscore_plan.json', out/'revision_analysis_plan.json')
    subprocess.run([sys.executable, str(dest/'scripts/acs201_hssc_analysis.py')], cwd=dest, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    expected = json.loads((ROOT/'metadata/expected_v8_tables.json').read_text(encoding='utf8'))
    byte_matches = 0
    for name, h in expected.items():
        generated = out/'analysis'/name
        byte_matches += hashlib.sha256(generated.read_bytes()).hexdigest() == h
        actual = pd.read_csv(generated)
        reference = pd.read_csv(ROOT/'data/tables'/name)
        # CSV line endings and final floating-point digits can vary by platform.
        # Counts/categories remain exact; floating quantities use a roundoff tolerance.
        pd.testing.assert_frame_equal(actual, reference, check_exact=False, rtol=2e-12, atol=2e-14)
        for col in reference.select_dtypes(include='integer').columns:
            assert np.array_equal(actual[col], reference[col]), (name, col)
    print('POSTSCORE_TABLES_NUMERICALLY_VERIFIED', len(expected), 'BYTE_IDENTICAL', byte_matches)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    p.add_argument('--checks', action='store_true')
    p.add_argument('--tables', action='store_true')
    p.add_argument('--workdir', default='reproduced_tables')
    a = p.parse_args()
    if a.verify:
        verify()
    if a.checks:
        numerical_checks()
    if a.tables:
        print('OUTPUT', tables(a.workdir))


if __name__ == '__main__':
    main()

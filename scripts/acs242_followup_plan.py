"""Freeze a bounded extension on accessed data; not a new holdout."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,argparse
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--pilot-dir',type=Path,default=R/'outputs/acs_research215_2026-11-22');parser.add_argument('--baseline-dir',type=Path,default=R/'outputs/acs_boundaries235_2026-11-22');parser.add_argument('--out-dir',type=Path,default=R/'outputs/acs_followup242_2026-11-22');args=parser.parse_args()
O=args.out_dir.resolve();O.mkdir(exist_ok=True)
p=O/'analysis_plan.json';assert not p.exists()
paths=list(args.baseline_dir.glob('*.csv'))+list((args.pilot_dir/'sipp_pilot').glob('*'))
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Extension on accessed SIPP; analysis plan, not preregistration or external confirmation',
 phase1=dict(scopes=['all_released','reported_only'],budgets=[300,1000],methods=['SRS_HT','SRS_prior_cells','SRS_prior_BA','proportional_prior_BA','working_Neyman_prior_BA'],R=2000,B=999,precision_B=3999,precision_draws='first 100 per case, not outcome-selected',levels=[.90,.95],report_resolutions=[.01,.02,.05],
 bootstrap=['Preston one-stage rescaled SRSWOR subset; floor(n/2) within sampled stratum; lambda=sqrt[k(1-f)/(n-k)]','Rao-Wu n-1 with-replacement variance-matching sensitivity'],
 intervals=['Preston_basic','RW_basic','empirical_Bernstein_Serfling'],quantiles='numpy linear',
 bounds='Prior-only binary household bounds from 235; theorem4.3 delta=alpha/(10*strata); kappa=7/3+3/sqrt(2); union over signs/strata; census contribution zero',
 limits='Bootstrap is approximate, not distribution-free. Household clusters are second-phase experiment units, not recovered original Census PSUs. Full-model refitting not needed because all predictors/allocations are frozen.'),
 phase2=dict(target='Known prior BA gap of ALL baseline adults, scorer reference; not missing later education',mask='Actual subsequent complete-household membership',
 X='Only baseline age, sex, nativity, citizenship, household size, mean age, shares female/foreign/older and log prior weight. NO prior education/allocation flag or follow-up education/weights.',
 methods=['complete_case','12_cell_calibration','IPW_logit','AIPW_logit_HGB'],crossfit='3 folds by entire historical household; seeds739..743; fitting variation, not independent populations',
 response='Household logistic response fit, standard scaled features L2 C=1, max_iter1000',outcome='HGB fixed 200 iterations, lr .05, 7 leaves, min_leaf20, l2=1, early_stopping=False, seed739',
 propensity_clip=[.05,.95],clip_sensitivity=[.02,.98],metrics='Known target error, weight diagnostics, fold variability; NO confidence interval or missing-at-random validation claim',
 gate='Run after phase1. No masked label enters a fitted predictor. Masked prior-outcome target differs from actual future-outcome task.'),
 interpretation='Mature controls may solve the problem; do not select by one favorable metric; no new ML claim, no cost or causal claim',
 preserved={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths if x.is_file()})
p.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8');print('FOLLOWUP_PLAN_FROZEN',hashlib.sha256(p.read_bytes()).hexdigest())

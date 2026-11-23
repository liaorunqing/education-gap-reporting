from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];O=R/'outputs/acs_followup242_2026-11-22';p=O/'strong_membership_sensitivity_plan.json';assert not p.exists()
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Added after inspecting initial membership results; not a prespecified confirmatory comparison',
    reason='Check whether unnormalized model response weights account for weak IPW/AIPW before attributing failure to all mature correction methods',
    methods=['IPW_cell_normalized','AIPW_cell_calibrated','OR_HGB'],
    calibration='Within each of12 age/sex/nativity cells normalize sum w*R/p to known baseline W_g; no extra fitting or outcome-derived margins',
    models_and_splits='Exactly reuse first-stage model settings, seed739..743 and3 household folds; no retuning; reproduce all40 prior estimates',
    limits='Known-prior-outcome mask only. Cannot identify missing future outcomes, validate MAR or claim new method',
    original_results_sha256=hashlib.sha256((O/'membership_results.csv').read_bytes()).hexdigest())
p.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8');print('STRONG_CONTROL_SENSITIVITY_FIXED')

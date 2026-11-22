# Numerical portability of raw refitting

The frozen study ran on Windows. In the original environment, the raw-only CT reconstruction refits the used source/risk models and reproduces all 2,000 draws/16 methods exactly; a separate public-code work directory also passed. No old model or derived observation array is used as an input.

An additional Linux raw-refit audit did **not** satisfy the same strict pathwise criterion. Restoring thread environment initialization before importing NumPy did not remove the difference. The source ZIPs, selected 2017 training sample fingerprint and standard-population vector matched. Twelve methods, including direct/historical controls, HGB-assisted SRS and the principal mature working Neyman comparator, agreed within approximately 4e-15 per point. Differences were concentrated in CPS_RF, CPS_HGB, CPS_robust_HGB and CPS_RF_GREG. Maximum point differences in these variants were approximately 0.121, 6.92e-8, 0.147 and 0.115, respectively; some selected adult counts differed. All household budgets remained exactly 1,000.

This isolates the sensitivity to learned-risk CPS paths but does not establish a unique mechanism. Tiny changes in learned risks, bin membership or calibration can change a discrete sampling path; further stage-level diagnosis would be needed before assigning causation. No model hyperparameter, source result, tolerance or favorable-method inclusion was changed to make this audit pass. We do not claim cross-platform refit parity for all methods or infer that the four aggregate method rankings remain identical.

The default manual raw workflow uses the reference Windows platform. Ubuntu remains selectable and the strict failure history is retained. This platform choice is explicit, not a reclassification of the failed Linux criterion. The saved-output summary replay and the separate SIPP raw workflow passed Linux checks. Software-environment fidelity, fixed-model replay and refit portability are distinct claims.

Diagnostic run: https://github.com/liaorunqing/education-gap-reporting/actions/runs/44992364172

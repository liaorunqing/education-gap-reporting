# Release history

## 1.0.1

Separate archive byte integrity from numerical reproduction across operating systems. All frozen data are unchanged. Recomputed integer counts/categories must match exactly; floating CSV values use rtol=2e-12, atol=2e-14. The original Windows reconstruction remains byte-identical for all four CSVs.

## 1.0.0

Initial public release. Local checks passed. The Linux workflow passed release hashes, exact finite-population checks and all 768 point/interval rows, but failed an overly platform-specific CSV-byte comparison. Version 1.0.1 corrects that check; it does not change estimates, variance formulas, methods or source outputs.

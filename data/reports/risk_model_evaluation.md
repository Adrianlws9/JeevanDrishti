# JEEVAN-DRISHTI Phase 5: Outbreak Risk Classification Evaluation Report

- **Generated At**: `2026-10-31T14:40:26`
- **Model Version**: `phase5-v1.0.0`
- **Source Dataset**: `data/final_training/risk_classification_dataset.csv` (`SHA-256: 9fd2af84d3275525a7c0821b1f2b715bf8a584badd8486345e4916e4626114da`)
- **Selected Primary Model (by Validation PR-AUC)**: `HistGradientBoosting_Classifier`
- **Production Readiness Verdict**: `PILOT_DECISION_SUPPORT_ONLY (NOT_PRODUCTION_AUTONOMOUS)`

## 1. Methodology, Leakage Prevention & Label Defensibility

1. **Defensible Empirical Labels Only**: No arbitrary case cutoffs were invented. The primary target (`target_severe_outbreak_p75`) labels whether an outbreak's case count exceeds the **Training-split (`2023-W01..2024-W33`) 75th percentile (`p75`) for that specific disease**, restricted to diseases with `>= 5` training observations (`481` valid rows across 12 supported diseases; `30` rows across 16 rare `<5`-train-obs diseases are explicitly excluded as `NaN` / `UNSUPPORTED`). Three secondary defensible targets (`target_fatal_outbreak`, `target_above_median_outbreak_p50`, and `target_next_4w_recurrence`) are also benchmarked.
2. **Training-Only Preprocessing**: `preprocessing_pipeline.pkl` (`ColumnTransformer` with `SimpleImputer(strategy='most_frequent') + OneHotEncoder(min_frequency=5)` for 6 categorical features and `SimpleImputer(strategy='median', add_indicator=True) + StandardScaler()` for 69 stationary numeric features) is fitted **strictly on `split == 'train'` (`period_idx <= 84`)**.
3. **Prevention of Temporal Covariate Shift**: Monotonic time counters (`report_year`, `calendar_year`, `period_idx`, and expanding cumulative panel sums starting at 0 in `2023-W01`) and ultra-sparse columns (`>70%` missing in train, such as single-station Pune RH and 52-week exact calendar lags) are excluded so models generalize across `2023 -> 2024 -> 2025` using stationary seasonal, climate, and prior-burden rate features.
4. **Untouched Held-Out Test Split (`2025-W01..2025-W52`)**: All hyperparameter choices, F1 decision thresholds, and Platt sigmoid probability calibrators were fitted strictly using `train` (`2023-W01..2024-W33`) and `validation` (`2024-W34..2024-W52`).

## 2. Primary Target Performance: `target_severe_outbreak_p75` (Upper-Quartile Disease Severity)

| Model | Split | N (Pos) | Pos Rate | Thr | Acc | Prec | Recall | F1 | F1@0.5 | PR-AUC | ROC-AUC | Brier (Cal) | ECE (Cal) | Confusion (TN, FP, FN, TP) | Δ PR-AUC vs Base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Logistic_Regression_Baseline` | `train` | 340 (82) | 24.1% | 0.212 | 0.4176 | 0.2929 | 1.0000 | **0.4530** | 0.6009 | **0.5971** | 0.8308 | 0.1655 | 0.0753 | (60, 198, 0, 82) | +0.0000 |
| `Logistic_Regression_Baseline` | `validation` | 76 (15) | 19.7% | 0.212 | 0.4079 | 0.2500 | 1.0000 | **0.4000** | 0.3077 | **0.2717** | 0.6153 | 0.1553 | 0.0217 | (16, 45, 0, 15) | +0.0000 |
| `Logistic_Regression_Baseline` | `test` | 65 (9) | 13.9% | 0.212 | 0.2615 | 0.1455 | 0.8889 | **0.2500** | 0.3000 | **0.2166** | 0.6290 | 0.1255 | 0.0925 | (9, 47, 1, 8) | +0.0000 |
| `Random_Forest_Classifier` | `train` | 340 (82) | 24.1% | 0.353 | 0.4941 | 0.3228 | 1.0000 | **0.4881** | 0.7362 | **0.7966** | 0.9298 | 0.1602 | 0.0737 | (86, 172, 0, 82) | +0.1995 |
| `Random_Forest_Classifier` | `validation` | 76 (15) | 19.7% | 0.353 | 0.3816 | 0.2241 | 0.8667 | **0.3562** | 0.0870 | **0.2359** | 0.5519 | 0.1588 | 0.0264 | (16, 45, 2, 13) | -0.0358 |
| `Random_Forest_Classifier` | `test` | 65 (9) | 13.9% | 0.353 | 0.2308 | 0.1404 | 0.8889 | **0.2424** | 0.1667 | **0.1680** | 0.4841 | 0.1321 | 0.1072 | (7, 49, 1, 8) | -0.0486 |
| `XGBoost_Classifier` | `train` | 340 (82) | 24.1% | 0.344 | 0.4353 | 0.2993 | 1.0000 | **0.4607** | 0.6738 | **0.7557** | 0.9059 | 0.1744 | 0.0480 | (66, 192, 0, 82) | +0.1586 |
| `XGBoost_Classifier` | `validation` | 76 (15) | 19.7% | 0.344 | 0.3684 | 0.2295 | 0.9333 | **0.3684** | 0.1379 | **0.2388** | 0.5530 | 0.1586 | 0.0226 | (14, 47, 1, 14) | -0.0329 |
| `XGBoost_Classifier` | `test` | 65 (9) | 13.9% | 0.344 | 0.2000 | 0.1228 | 0.7778 | **0.2121** | 0.2286 | **0.2094** | 0.5397 | 0.1266 | 0.0867 | (6, 50, 2, 7) | -0.0072 |
| `HistGradientBoosting_Classifier` | `train` | 340 (82) | 24.1% | 0.265 | 0.6029 | 0.3779 | 1.0000 | **0.5485** | 0.8696 | **0.9612** | 0.9878 | 0.1649 | 0.0850 | (123, 135, 0, 82) | +0.3641 |
| `HistGradientBoosting_Classifier` | `validation` | 76 (15) | 19.7% | 0.265 | 0.5395 | 0.2727 | 0.8000 | **0.4068** | 0.1667 | **0.3017** | 0.6087 | 0.1570 | 0.0206 | (29, 32, 3, 12) | +0.0300 |
| `HistGradientBoosting_Classifier` | `test` | 65 (9) | 13.9% | 0.265 | 0.3538 | 0.1333 | 0.6667 | **0.2222** | 0.2667 | **0.1765** | 0.5119 | 0.1269 | 0.0864 | (17, 39, 3, 6) | -0.0401 |
| `SoftVoting_Ensemble_Calibrated` | `train` | 340 (82) | 24.1% | 0.380 | 0.6912 | 0.4385 | 1.0000 | **0.6097** | 0.7935 | **0.8887** | 0.9654 | 0.1547 | 0.1253 | (153, 105, 0, 82) | +0.2916 |
| `SoftVoting_Ensemble_Calibrated` | `validation` | 76 (15) | 19.7% | 0.380 | 0.5921 | 0.2778 | 0.6667 | **0.3922** | 0.1600 | **0.2778** | 0.6306 | 0.1558 | 0.0561 | (35, 26, 5, 10) | +0.0061 |
| `SoftVoting_Ensemble_Calibrated` | `test` | 65 (9) | 13.9% | 0.380 | 0.4154 | 0.1282 | 0.5556 | **0.2083** | 0.2581 | **0.2152** | 0.5575 | 0.1275 | 0.0925 | (22, 34, 4, 5) | -0.0014 |

## 3. Secondary Target Benchmarks

### 3.1 `target_fatal_outbreak` (Observed Mortality `reported_deaths >= 1`, 511 Valid Rows)

| Model | Split | N (Pos) | Pos Rate | Thr | Acc | Prec | Recall | F1 | F1@0.5 | PR-AUC | ROC-AUC | Brier (Cal) | ECE (Cal) | Confusion (TN, FP, FN, TP) | Δ PR-AUC vs Base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Logistic_Regression_Baseline` | `train` | 356 (114) | 32.0% | 0.638 | 0.8511 | 0.8211 | 0.6842 | **0.7464** | 0.7490 | **0.8457** | 0.9087 | 0.1249 | 0.1172 | (225, 17, 36, 78) | +0.0000 |
| `Logistic_Regression_Baseline` | `validation` | 77 (19) | 24.7% | 0.638 | 0.7532 | 0.5000 | 0.6316 | **0.5581** | 0.4643 | **0.5943** | 0.7196 | 0.1804 | 0.1579 | (46, 12, 7, 12) | +0.0000 |
| `Logistic_Regression_Baseline` | `test` | 78 (13) | 16.7% | 0.638 | 0.7564 | 0.2857 | 0.3077 | **0.2963** | 0.3226 | **0.3028** | 0.6083 | 0.1738 | 0.1234 | (55, 10, 9, 4) | +0.0000 |
| `Random_Forest_Classifier` | `train` | 356 (114) | 32.0% | 0.637 | 0.8399 | 0.9385 | 0.5351 | **0.6816** | 0.7913 | **0.8587** | 0.9241 | 0.1143 | 0.0932 | (238, 4, 53, 61) | +0.0130 |
| `Random_Forest_Classifier` | `validation` | 77 (19) | 24.7% | 0.637 | 0.8312 | 0.8000 | 0.4211 | **0.5517** | 0.4211 | **0.5799** | 0.6887 | 0.1767 | 0.1523 | (56, 2, 11, 8) | -0.0144 |
| `Random_Forest_Classifier` | `test` | 78 (13) | 16.7% | 0.637 | 0.7821 | 0.0000 | 0.0000 | **0.0000** | 0.3333 | **0.2800** | 0.5811 | 0.1508 | 0.1068 | (61, 4, 13, 0) | -0.0228 |
| `XGBoost_Classifier` | `train` | 356 (114) | 32.0% | 0.728 | 0.8202 | 0.9310 | 0.4737 | **0.6279** | 0.7680 | **0.8457** | 0.9147 | 0.1189 | 0.0761 | (238, 4, 60, 54) | +0.0000 |
| `XGBoost_Classifier` | `validation` | 77 (19) | 24.7% | 0.728 | 0.8312 | 0.7500 | 0.4737 | **0.5806** | 0.4746 | **0.6449** | 0.7613 | 0.1518 | 0.1110 | (55, 3, 10, 9) | +0.0506 |
| `XGBoost_Classifier` | `test` | 78 (13) | 16.7% | 0.728 | 0.8077 | 0.3750 | 0.2308 | **0.2857** | 0.3889 | **0.2870** | 0.6000 | 0.1511 | 0.0794 | (60, 5, 10, 3) | -0.0158 |
| `HistGradientBoosting_Classifier` | `train` | 356 (114) | 32.0% | 0.767 | 0.8792 | 0.9494 | 0.6579 | **0.7772** | 0.8780 | **0.9330** | 0.9718 | 0.1023 | 0.1642 | (238, 4, 39, 75) | +0.0873 |
| `HistGradientBoosting_Classifier` | `validation` | 77 (19) | 24.7% | 0.767 | 0.7792 | 0.5625 | 0.4737 | **0.5143** | 0.4681 | **0.6223** | 0.7223 | 0.1612 | 0.0947 | (51, 7, 10, 9) | +0.0280 |
| `HistGradientBoosting_Classifier` | `test` | 78 (13) | 16.7% | 0.767 | 0.8205 | 0.4545 | 0.3846 | **0.4167** | 0.3429 | **0.2793** | 0.6213 | 0.1532 | 0.0829 | (59, 6, 8, 5) | -0.0235 |
| `SoftVoting_Ensemble_Calibrated` | `train` | 356 (114) | 32.0% | 0.722 | 0.8539 | 0.9306 | 0.5877 | **0.7204** | 0.8319 | **0.8999** | 0.9518 | 0.0968 | 0.1042 | (237, 5, 47, 67) | +0.0542 |
| `SoftVoting_Ensemble_Calibrated` | `validation` | 77 (19) | 24.7% | 0.722 | 0.8312 | 0.7143 | 0.5263 | **0.6061** | 0.4800 | **0.6422** | 0.7323 | 0.1651 | 0.1083 | (54, 4, 9, 10) | +0.0479 |
| `SoftVoting_Ensemble_Calibrated` | `test` | 78 (13) | 16.7% | 0.722 | 0.8205 | 0.4444 | 0.3077 | **0.3636** | 0.3125 | **0.2975** | 0.6059 | 0.1591 | 0.1005 | (60, 5, 9, 4) | -0.0053 |

### 3.2 `target_above_median_outbreak_p50` (Above-Median Disease Severity, 481 Valid Rows)

| Model | Split | N (Pos) | Pos Rate | Thr | Acc | Prec | Recall | F1 | F1@0.5 | PR-AUC | ROC-AUC | Brier (Cal) | ECE (Cal) | Confusion (TN, FP, FN, TP) | Δ PR-AUC vs Base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Logistic_Regression_Baseline` | `train` | 340 (163) | 47.9% | 0.213 | 0.5382 | 0.5094 | 0.9939 | **0.6736** | 0.7041 | **0.7609** | 0.7885 | 0.2312 | 0.0532 | (21, 156, 1, 162) | +0.0000 |
| `Logistic_Regression_Baseline` | `validation` | 76 (28) | 36.8% | 0.213 | 0.4605 | 0.4058 | 1.0000 | **0.5773** | 0.3704 | **0.3835** | 0.5409 | 0.2362 | 0.0794 | (7, 41, 0, 28) | +0.0000 |
| `Logistic_Regression_Baseline` | `test` | 65 (30) | 46.2% | 0.213 | 0.5385 | 0.5000 | 1.0000 | **0.6667** | 0.6377 | **0.6141** | 0.6486 | 0.2355 | 0.0374 | (5, 30, 0, 30) | +0.0000 |
| `Random_Forest_Classifier` | `train` | 340 (163) | 47.9% | 0.414 | 0.6529 | 0.5801 | 1.0000 | **0.7342** | 0.7862 | **0.9135** | 0.9173 | 0.2443 | 0.0271 | (59, 118, 0, 163) | +0.1526 |
| `Random_Forest_Classifier` | `validation` | 76 (28) | 36.8% | 0.414 | 0.3684 | 0.3649 | 0.9643 | **0.5294** | 0.1364 | **0.3440** | 0.4308 | 0.2398 | 0.0818 | (1, 47, 1, 27) | -0.0395 |
| `Random_Forest_Classifier` | `test` | 65 (30) | 46.2% | 0.414 | 0.4615 | 0.4615 | 1.0000 | **0.6316** | 0.5574 | **0.4690** | 0.5495 | 0.2484 | 0.0078 | (0, 35, 0, 30) | -0.1451 |
| `XGBoost_Classifier` | `train` | 340 (163) | 47.9% | 0.391 | 0.6147 | 0.5548 | 0.9939 | **0.7121** | 0.7446 | **0.8127** | 0.8356 | 0.2289 | 0.0897 | (47, 130, 1, 162) | +0.0518 |
| `XGBoost_Classifier` | `validation` | 76 (28) | 36.8% | 0.391 | 0.3947 | 0.3784 | 1.0000 | **0.5490** | 0.3871 | **0.3804** | 0.4821 | 0.2414 | 0.0832 | (2, 46, 0, 28) | -0.0031 |
| `XGBoost_Classifier` | `test` | 65 (30) | 46.2% | 0.391 | 0.4923 | 0.4762 | 1.0000 | **0.6452** | 0.4615 | **0.4731** | 0.5181 | 0.2491 | 0.0043 | (2, 33, 0, 30) | -0.1410 |
| `HistGradientBoosting_Classifier` | `train` | 340 (163) | 47.9% | 0.321 | 0.7118 | 0.6245 | 1.0000 | **0.7689** | 0.8982 | **0.9709** | 0.9714 | 0.2182 | 0.1321 | (79, 98, 0, 163) | +0.2100 |
| `HistGradientBoosting_Classifier` | `validation` | 76 (28) | 36.8% | 0.321 | 0.4342 | 0.3846 | 0.8929 | **0.5376** | 0.4262 | **0.3968** | 0.5260 | 0.2395 | 0.0858 | (8, 40, 3, 25) | +0.0133 |
| `HistGradientBoosting_Classifier` | `test` | 65 (30) | 46.2% | 0.321 | 0.4923 | 0.4737 | 0.9000 | **0.6207** | 0.5672 | **0.6076** | 0.6305 | 0.2443 | 0.0134 | (5, 30, 3, 27) | -0.0065 |
| `SoftVoting_Ensemble_Calibrated` | `train` | 340 (163) | 47.9% | 0.364 | 0.6794 | 0.5993 | 1.0000 | **0.7494** | 0.8606 | **0.9423** | 0.9438 | 0.2145 | 0.1301 | (68, 109, 0, 163) | +0.1814 |
| `SoftVoting_Ensemble_Calibrated` | `validation` | 76 (28) | 36.8% | 0.364 | 0.4474 | 0.3939 | 0.9286 | **0.5532** | 0.4000 | **0.3968** | 0.5394 | 0.2389 | 0.0789 | (8, 40, 2, 26) | +0.0133 |
| `SoftVoting_Ensemble_Calibrated` | `test` | 65 (30) | 46.2% | 0.364 | 0.5385 | 0.5000 | 0.9667 | **0.6591** | 0.6269 | **0.5754** | 0.6381 | 0.2406 | 0.0247 | (6, 29, 1, 29) | -0.0387 |

### 3.3 `target_next_4w_recurrence` (4-Week District-Disease Recurrence, 476 Valid Rows)

| Model | Split | N (Pos) | Pos Rate | Thr | Acc | Prec | Recall | F1 | F1@0.5 | PR-AUC | ROC-AUC | Brier (Cal) | ECE (Cal) | Confusion (TN, FP, FN, TP) | Δ PR-AUC vs Base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Logistic_Regression_Baseline` | `train` | 332 (79) | 23.8% | 0.240 | 0.5030 | 0.3193 | 0.9620 | **0.4795** | 0.6400 | **0.7034** | 0.8693 | 0.1676 | 0.0444 | (91, 162, 3, 76) | +0.0000 |
| `Logistic_Regression_Baseline` | `validation` | 67 (9) | 13.4% | 0.240 | 0.4478 | 0.1667 | 0.7778 | **0.2745** | 0.1739 | **0.1673** | 0.5536 | 0.1270 | 0.0956 | (23, 35, 2, 7) | +0.0000 |
| `Logistic_Regression_Baseline` | `test` | 77 (8) | 10.4% | 0.240 | 0.3506 | 0.0800 | 0.5000 | **0.1379** | 0.1481 | **0.1152** | 0.3659 | 0.1110 | 0.1441 | (23, 46, 4, 4) | +0.0000 |
| `Random_Forest_Classifier` | `train` | 332 (79) | 23.8% | 0.442 | 0.7952 | 0.5426 | 0.8861 | **0.6731** | 0.6927 | **0.8139** | 0.9119 | 0.1475 | 0.1158 | (194, 59, 9, 70) | +0.1105 |
| `Random_Forest_Classifier` | `validation` | 67 (9) | 13.4% | 0.442 | 0.5373 | 0.2105 | 0.8889 | **0.3404** | 0.2105 | **0.1706** | 0.5881 | 0.1297 | 0.1229 | (28, 30, 1, 8) | +0.0033 |
| `Random_Forest_Classifier` | `test` | 77 (8) | 10.4% | 0.442 | 0.6494 | 0.0476 | 0.1250 | **0.0690** | 0.0833 | **0.0985** | 0.3659 | 0.1129 | 0.1326 | (49, 20, 7, 1) | -0.0167 |
| `XGBoost_Classifier` | `train` | 332 (79) | 23.8% | 0.448 | 0.7831 | 0.5259 | 0.8987 | **0.6636** | 0.6875 | **0.7594** | 0.8951 | 0.1479 | 0.0833 | (189, 64, 8, 71) | +0.0560 |
| `XGBoost_Classifier` | `validation` | 67 (9) | 13.4% | 0.448 | 0.6119 | 0.2258 | 0.7778 | **0.3500** | 0.1875 | **0.1765** | 0.6073 | 0.1307 | 0.1209 | (34, 24, 2, 7) | +0.0092 |
| `XGBoost_Classifier` | `test` | 77 (8) | 10.4% | 0.448 | 0.6623 | 0.0909 | 0.2500 | **0.1333** | 0.1667 | **0.1261** | 0.3877 | 0.1132 | 0.1328 | (49, 20, 6, 2) | +0.0109 |
| `HistGradientBoosting_Classifier` | `train` | 332 (79) | 23.8% | 0.522 | 0.9367 | 0.8295 | 0.9241 | **0.8743** | 0.8690 | **0.9626** | 0.9862 | 0.1345 | 0.1273 | (238, 15, 6, 73) | +0.2592 |
| `HistGradientBoosting_Classifier` | `validation` | 67 (9) | 13.4% | 0.522 | 0.7313 | 0.2632 | 0.5556 | **0.3571** | 0.3871 | **0.2223** | 0.6820 | 0.1255 | 0.1123 | (44, 14, 4, 5) | +0.0550 |
| `HistGradientBoosting_Classifier` | `test` | 77 (8) | 10.4% | 0.522 | 0.8312 | 0.2222 | 0.2500 | **0.2353** | 0.2105 | **0.1523** | 0.4946 | 0.1078 | 0.1170 | (62, 7, 6, 2) | +0.0371 |
| `SoftVoting_Ensemble_Calibrated` | `train` | 332 (79) | 23.8% | 0.434 | 0.8343 | 0.5984 | 0.9241 | **0.7264** | 0.8114 | **0.8981** | 0.9549 | 0.1367 | 0.1269 | (204, 49, 6, 73) | +0.1947 |
| `SoftVoting_Ensemble_Calibrated` | `validation` | 67 (9) | 13.4% | 0.434 | 0.6269 | 0.2333 | 0.7778 | **0.3590** | 0.2308 | **0.2051** | 0.6475 | 0.1267 | 0.1088 | (35, 23, 2, 7) | +0.0378 |
| `SoftVoting_Ensemble_Calibrated` | `test` | 77 (8) | 10.4% | 0.434 | 0.7013 | 0.1053 | 0.2500 | **0.1481** | 0.2000 | **0.1422** | 0.3714 | 0.1109 | 0.1304 | (52, 17, 6, 2) | +0.0270 |

## 4. Subgroup Evaluation by Disease (`target_severe_outbreak_p75`)

| Disease | Total Valid | Train | Val | Test | Positives (%) | Status | Val+Test Acc | Val+Test Prec | Val+Test Rec | Val+Test F1 | Val+Test PR-AUC | Val+Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dengue | 151 | 134 | 10 | 7 | 33 (21.9%) | `EVALUATED_VAL_AND_TEST` | 0.5294 | 0.2222 | 0.6667 | 0.3333 | 0.2214 | 0.5000 |
| Acute Diarrhoeal Disease | 110 | 67 | 23 | 20 | 23 (20.9%) | `EVALUATED_VAL_AND_TEST` | 0.6047 | 0.2381 | 0.8333 | 0.3704 | 0.4528 | 0.7252 |
| Food Poisoning | 50 | 23 | 10 | 17 | 14 (28.0%) | `EVALUATED_VAL_AND_TEST` | 0.4074 | 0.2778 | 0.6250 | 0.3846 | 0.2678 | 0.3882 |
| Malaria | 39 | 29 | 4 | 6 | 9 (23.1%) | `EVALUATED_VAL_AND_TEST` | 0.5000 | 0.2000 | 0.5000 | 0.2857 | 0.3333 | 0.6250 |
| Chikungunya | 34 | 20 | 9 | 5 | 7 (20.6%) | `EVALUATED_VAL_AND_TEST` | 0.2857 | 0.1667 | 1.0000 | 0.2857 | 0.1736 | 0.4167 |
| Hepatitis A | 24 | 9 | 10 | 5 | 4 (16.7%) | `EVALUATED_VAL_AND_TEST` | 0.2667 | 0.1538 | 1.0000 | 0.2667 | 0.3611 | 0.6923 |
| Cholera | 18 | 13 | 4 | 1 | 3 (16.7%) | `EVALUATED_VAL_AND_TEST` | 0.4000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| Zika Virus | 17 | 15 | 2 | 0 | 4 (23.5%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.4118 | 0.2857 | 1.0000 | 0.4444 | N/A | N/A |
| Fever (unspecified) | 14 | 11 | 0 | 3 | 3 (21.4%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.5000 | 0.3000 | 1.0000 | 0.4615 | N/A | N/A |
| Leptospirosis | 9 | 5 | 3 | 1 | 2 (22.2%) | `EVALUATED_VAL_AND_TEST` | 0.2500 | 0.2500 | 1.0000 | 0.4000 | 0.5000 | 0.6667 |
| Scrub Typhus | 9 | 8 | 1 | 0 | 2 (22.2%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.5556 | 0.3333 | 1.0000 | 0.5000 | N/A | N/A |
| Dengue + Chikungunya | 6 | 6 | 0 | 0 | 2 (33.3%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.6667 | 0.5000 | 1.0000 | 0.6667 | N/A | N/A |

## 5. Subgroup Evaluation by District (Top Districts by Sample Size, `target_severe_outbreak_p75`)

| District | Total Valid | Train | Val | Test | Positives (%) | Status | Val+Test Acc | Val+Test F1 | Val+Test PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Kolhapur | 39 | 27 | 5 | 7 | 13 (33.3%) | `EVALUATED_VAL_AND_TEST` | 0.2500 | 0.3077 | 0.5750 |
| Thane | 31 | 18 | 8 | 5 | 2 (6.5%) | `EVALUATED_VAL_AND_TEST` | 0.5385 | 0.0000 | 0.0909 |
| Pune | 30 | 23 | 6 | 1 | 8 (26.7%) | `EVALUATED_VAL_AND_TEST` | 0.5714 | 0.4000 | 0.3333 |
| Gadchiroli | 27 | 18 | 3 | 6 | 7 (25.9%) | `EVALUATED_VAL_AND_TEST` | 0.6667 | 0.0000 | 0.1429 |
| Amravati | 24 | 15 | 4 | 5 | 6 (25.0%) | `EVALUATED_VAL_AND_TEST` | 0.4444 | 0.2857 | 0.2500 |
| Dharashiv | 24 | 19 | 2 | 3 | 5 (20.8%) | `EVALUATED_VAL_AND_TEST` | 0.8000 | 0.6667 | 0.5000 |
| Raigad | 24 | 16 | 3 | 5 | 7 (29.2%) | `EVALUATED_VAL_AND_TEST` | 0.5000 | 0.6000 | 0.7470 |
| Nanded | 22 | 13 | 4 | 5 | 3 (13.6%) | `EVALUATED_VAL_AND_TEST` | 0.5556 | 0.0000 | N/A |
| Satara | 19 | 18 | 0 | 1 | 3 (15.8%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.5789 | 0.4286 | N/A |
| Nagpur | 16 | 12 | 1 | 3 | 5 (31.2%) | `EVALUATED_VAL_AND_TEST` | 0.7500 | 0.8000 | 1.0000 |
| Yavatmal | 16 | 15 | 0 | 1 | 2 (12.5%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.4375 | 0.3077 | N/A |
| Sangli | 14 | 7 | 3 | 4 | 2 (14.3%) | `EVALUATED_VAL_AND_TEST` | 0.1429 | 0.2500 | 0.1667 |
| Ahmednagar | 13 | 10 | 3 | 0 | 6 (46.2%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.6154 | 0.7059 | N/A |
| Akola | 13 | 13 | 0 | 0 | 6 (46.2%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.5385 | 0.6667 | N/A |
| Chhatrapati Sambhajinagar | 13 | 9 | 1 | 3 | 2 (15.4%) | `EVALUATED_VAL_AND_TEST` | 0.5000 | 0.6667 | 0.4167 |
| Palghar | 13 | 7 | 4 | 2 | 2 (15.4%) | `EVALUATED_VAL_AND_TEST` | 0.5000 | 0.4000 | 0.5000 |
| Washim | 13 | 10 | 2 | 1 | 3 (23.1%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.5385 | 0.5000 | N/A |
| Bhandara | 12 | 5 | 3 | 4 | 1 (8.3%) | `EVALUATED_VAL_AND_TEST` | 0.4286 | 0.0000 | N/A |
| Chandrapur | 12 | 11 | 1 | 0 | 2 (16.7%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.6667 | 0.5000 | N/A |
| Parbhani | 12 | 9 | 3 | 0 | 2 (16.7%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.9167 | 0.8000 | N/A |
| Beed | 11 | 9 | 2 | 0 | 0 (0.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.8182 | 0.0000 | N/A |
| Nashik | 10 | 6 | 4 | 0 | 1 (10.0%) | `EVALUATED_VAL_ONLY` | 0.7500 | 0.6667 | 1.0000 |
| Wardha | 9 | 5 | 3 | 1 | 2 (22.2%) | `EVALUATED_VAL_AND_TEST` | 0.0000 | 0.0000 | N/A |
| Jalgaon | 8 | 5 | 0 | 3 | 4 (50.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.5000 | 0.6667 | N/A |
| Latur | 8 | 6 | 2 | 0 | 1 (12.5%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.8750 | 0.6667 | N/A |
| Buldhana | 7 | 3 | 2 | 2 | 3 (42.9%) | `EVALUATED_VAL_AND_TEST` | 0.0000 | 0.0000 | 0.2500 |
| Sindhudurg | 7 | 5 | 2 | 0 | 1 (14.3%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.2857 | 0.2857 | N/A |
| Dhule | 6 | 5 | 1 | 0 | 3 (50.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.6667 | 0.7500 | N/A |
| Jalna | 6 | 6 | 0 | 0 | 0 (0.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.8333 | 0.0000 | N/A |
| Nandurbar | 6 | 3 | 2 | 1 | 3 (50.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.6667 | 0.7500 | N/A |
| Gondia | 5 | 3 | 1 | 1 | 1 (20.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.2000 | 0.3333 | N/A |
| Hingoli | 4 | 3 | 1 | 0 | 0 (0.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.7500 | 0.0000 | N/A |
| Solapur | 4 | 4 | 0 | 0 | 0 (0.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.2500 | 0.0000 | N/A |
| Mumbai Suburban | 3 | 2 | 0 | 1 | 0 (0.0%) | `INSUFFICIENT_HOLDOUT_SAMPLE (<4 val+test rows)` | 0.3333 | 0.0000 | N/A |

## 6. Top 20 Contributing Features (Global Normalized Importance)

| Rank | Transformed Feature Name | Normalized Importance |
|---|---|---|
| 1 | `numerical__district_lat` | 0.021272 |
| 2 | `numerical__tmin_c_mean_lag_8w` | 0.020923 |
| 3 | `numerical__rain_mm_yoy_abs_diff_lag_1w` | 0.018715 |
| 4 | `numerical__rain_mm_deviation_expanding_month_abs_lag_1w` | 0.018579 |
| 5 | `numerical__tmin_c_mean_lag_1w` | 0.018117 |
| 6 | `numerical__seasonal_state_disease_season_mean_prev` | 0.017653 |
| 7 | `numerical__temp_diurnal_range_lag_1w` | 0.017628 |
| 8 | `numerical__rain_mm_expanding_month_baseline_prev` | 0.016954 |
| 9 | `numerical__is_marathwada` | 0.016827 |
| 10 | `numerical__tavg_c_mean_lag_1w` | 0.016705 |
| 11 | `numerical__temp_diurnal_range_lag_4w` | 0.016524 |
| 12 | `numerical__tavg_c_roll_mean_prev_2w` | 0.016352 |
| 13 | `numerical__rain_mm_sum_lag_8w` | 0.015853 |
| 14 | `numerical__prev_outbreak_roll_mean_2` | 0.015738 |
| 15 | `numerical__tavg_c_roll_mean_prev_8w` | 0.015337 |
| 16 | `numerical__district_disease_prior_deaths_cum` | 0.015300 |
| 17 | `numerical__tavg_c_mean_lag_8w` | 0.015172 |
| 18 | `numerical__temp_diurnal_range_lag_2w` | 0.015042 |
| 19 | `numerical__tmax_c_mean_lag_2w` | 0.014367 |
| 20 | `categorical__administrative_division_Marathwada` | 0.014332 |

## 7. Unsupported Diseases & Districts Inventory

- **Unsupported Core Target Disease (0 records in 2023–2025 IDSP)**: `Heat-related illness`.
- **Unsupported Rare Diseases for `target_severe_outbreak_p75` (`< 5` training observations; 16 diseases, 30 total rows)**: `Acute Encephalitis Syndrome`, `Acute Hepatitis (unspecified)`, `Chickenpox`, `Dengue + Malaria`, `Dysentery`, `Fever of Unknown Cause`, `Fever with Rash`, `Hepatitis A + E`, `Japanese Encephalitis`, `Jaundice`, `Measles`, `Measles + Rubella`, `Mumps`, `Pertussis`, `Rabies (human)`, `Typhoid`. (Note: these 16 diseases ARE supported for `target_fatal_outbreak` where mortality is directly observed.)
- **Unsupported Districts (0 outbreak records in 2023–2025 IDSP)**: `Mumbai City`, `Ratnagiri`.
- **Partially Supported District (No climate coordinates in lookup; disease-history & temporal features only)**: `Mumbai Suburban` (3 rows).

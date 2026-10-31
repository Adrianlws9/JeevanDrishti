# JEEVAN-DRISHTI Phase 6: Time-Series Disease Forecasting Evaluation Report

- **Generated At**: `2026-10-31T14:54:18`
- **Model Version**: `phase6-v1.0.0`
- **Source Dataset**: `data/final_training/forecasting_dataset.csv` (`SHA-256: f5e74bbf53904c74a74f18b30c2fd8879265b47b460a3e0cab6a329d5b03a403`)
- **Native Reporting Frequency**: `weekly (7-day IDSP reporting periods W01..W52; daily predictions are never fabricated)`
- **Readiness Verdict**: `DECISION_SUPPORT_WITH_INTERVALS_ONLY (HIGH_OUTLIER_VARIANCE)`

## 1. Native Reporting Frequency & Forecast Horizon Mapping

Because IDSP outbreak surveillance in `forecasting_dataset.csv` is reported at a **weekly grain** (`temporal_grain == 'weekly'`, 7-day reporting weeks `W01..W52`), daily case counts are **never** fabricated or interpolated. The requested day horizons map strictly to discrete future reporting weeks relative to forecast origin $t_0 = t - h$:

| Requested Horizon | Native Weekly Equivalent | Information Cutoff | Input Features Permitted |
|---|---|---|---|
| **7 days (`7d`)** | **1 reporting week ahead ($h = 1\text{ week}$)** | Strictly $\le t - 1$ | All 75 stationary lagged disease ($\ge 1\text{w}$), climate ($\ge 1\text{w}$), seasonal, and geographic features |
| **14 days (`14d`)** | **2 reporting weeks ahead ($h = 2\text{ weeks}$)** | Strictly $\le t - 2$ | 50 features available at $\le t - 2$ (all lag-1 week disease & climate features strictly excluded) |
| **30 days (`30d`)** | **4 reporting weeks ahead ($h = 4\text{ weeks} = 28\text{ days}$)** | Strictly $\le t - 4$ | 36 features available at $\le t - 4$ (all lag-1, lag-2, and lag-3 features strictly excluded) |

## 2.1 Performance Comparison — 7-Day Horizon (`1 Reporting Week Ahead`, $h = 1\text{ week}$)

- **Selected Model (by Validation MAE)**: `HistGradientBoosting_Regressor`

| Model | Split | N | Actual Mean (Med) | Pred Mean (Med) | MAE | RMSE | MedAE | Log1p MAE | 80% PI Cov | 90% PI Cov | Trend Acc | Spearman $\rho$ | $\Delta$ MAE vs LastValue | $\Delta$ MAE vs SeasonalNaive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LastValue_Baseline` | `train` | 356 | 37.1 (16.0) | 30.8 (14.0) | **32.98** | **102.79** | 9.00 | 0.7685 | 89.0% | 93.3% | 10.7% | 0.517 | +0.00 | +0.94 |
| `LastValue_Baseline` | `validation` | 77 | 33.3 (16.0) | 32.5 (19.0) | **31.71** | **54.60** | 12.00 | 0.9045 | 79.2% | 89.6% | 5.2% | 0.407 | +0.00 | +2.38 |
| `LastValue_Baseline` | `test` | 78 | 43.6 (22.0) | 45.9 (20.5) | **46.50** | **109.48** | 15.50 | 0.8514 | 83.3% | 91.0% | 3.9% | 0.317 | +0.00 | +3.91 |
| `SeasonalNaive_Baseline` | `train` | 356 | 37.1 (16.0) | 35.1 (22.5) | **32.05** | **89.23** | 12.00 | 0.7970 | 86.2% | 92.1% | 48.0% | 0.514 | -0.94 | +0.00 |
| `SeasonalNaive_Baseline` | `validation` | 77 | 33.3 (16.0) | 37.6 (28.5) | **29.33** | **45.89** | 15.67 | 0.9282 | 79.2% | 89.6% | 45.5% | 0.511 | -2.38 | +0.00 |
| `SeasonalNaive_Baseline` | `test` | 78 | 43.6 (22.0) | 46.4 (27.0) | **42.59** | **90.06** | 15.70 | 0.8263 | 87.2% | 89.7% | 47.4% | 0.354 | -3.91 | +0.00 |
| `Statsmodels_ETS` | `train` | 356 | 37.1 (16.0) | 25.0 (14.7) | **27.83** | **85.25** | 10.06 | 0.7460 | 84.5% | 91.6% | 53.6% | 0.494 | -5.16 | -4.22 |
| `Statsmodels_ETS` | `validation` | 77 | 33.3 (16.0) | 25.9 (18.6) | **26.76** | **46.61** | 14.91 | 0.8628 | 79.2% | 89.6% | 58.4% | 0.471 | -4.95 | -2.57 |
| `Statsmodels_ETS` | `test` | 78 | 43.6 (22.0) | 28.6 (19.6) | **31.72** | **78.50** | 11.61 | 0.7261 | 80.8% | 92.3% | 62.8% | 0.431 | -14.77 | -10.86 |
| `Statsmodels_SARIMA` | `train` | 356 | 37.1 (16.0) | 24.8 (14.3) | **26.56** | **85.70** | 8.91 | 0.7235 | 84.3% | 91.6% | 58.7% | 0.486 | -6.42 | -5.49 |
| `Statsmodels_SARIMA` | `validation` | 77 | 33.3 (16.0) | 28.9 (24.3) | **26.01** | **43.44** | 15.00 | 0.8430 | 79.2% | 89.6% | 58.4% | 0.585 | -5.70 | -3.32 |
| `Statsmodels_SARIMA` | `test` | 78 | 43.6 (22.0) | 32.5 (22.7) | **32.15** | **78.46** | 13.68 | 0.7007 | 84.6% | 91.0% | 62.8% | 0.466 | -14.35 | -10.44 |
| `XGBoost_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.1 (15.3) | **22.49** | **77.08** | 6.50 | 0.5729 | 94.7% | 97.8% | 65.2% | 0.786 | -10.49 | -9.56 |
| `XGBoost_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 20.8 (17.2) | **24.02** | **45.47** | 11.43 | 0.8101 | 79.2% | 89.6% | 66.2% | 0.471 | -7.69 | -5.31 |
| `XGBoost_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 31.1 (29.6) | **30.92** | **77.56** | 15.23 | 0.7007 | 85.9% | 89.7% | 59.0% | 0.341 | -15.58 | -11.67 |
| `HistGradientBoosting_Regressor` | `train` | 356 | 37.1 (16.0) | 22.8 (15.6) | **20.70** | **72.44** | 5.79 | 0.4999 | 94.7% | 98.9% | 68.5% | 0.843 | -12.29 | -11.35 |
| `HistGradientBoosting_Regressor` | `validation` | 77 | 33.3 (16.0) | 21.9 (16.4) | **23.25** | **43.64** | 11.04 | 0.7710 | 79.2% | 89.6% | 66.2% | 0.540 | -8.46 | -6.08 |
| `HistGradientBoosting_Regressor` | `test` | 78 | 43.6 (22.0) | 33.0 (29.3) | **31.41** | **77.53** | 13.69 | 0.7063 | 83.3% | 91.0% | 66.7% | 0.360 | -15.09 | -11.18 |
| `RandomForest_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.6 (14.3) | **22.79** | **79.50** | 6.76 | 0.5866 | 93.3% | 98.0% | 63.2% | 0.767 | -10.19 | -9.25 |
| `RandomForest_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 24.6 (17.8) | **24.31** | **43.57** | 12.12 | 0.8177 | 79.2% | 89.6% | 62.3% | 0.477 | -7.40 | -5.02 |
| `RandomForest_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 33.3 (31.3) | **31.48** | **79.04** | 13.04 | 0.7007 | 85.9% | 89.7% | 65.4% | 0.388 | -15.01 | -11.10 |
| `Ridge_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.8 (16.2) | **23.94** | **81.82** | 8.50 | 0.6196 | 93.3% | 94.7% | 62.6% | 0.694 | -9.05 | -8.11 |
| `Ridge_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 14.7 (12.6) | **24.90** | **48.00** | 9.85 | 0.8575 | 79.2% | 89.6% | 63.6% | 0.448 | -6.81 | -4.43 |
| `Ridge_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 36.6 (24.5) | **38.77** | **87.97** | 18.48 | 0.8429 | 80.8% | 83.3% | 60.3% | 0.220 | -7.73 | -3.82 |
| `LastValue_Baseline` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 36.8 (18.0) | **40.34** | **110.84** | 11.00 | 0.8480 | 75.1% | 85.2% | 8.1% | 0.464 | +0.00 | +2.68 |
| `SeasonalNaive_Baseline` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 39.1 (22.0) | **37.67** | **94.80** | 15.00 | 0.8606 | 75.3% | 86.5% | 47.6% | 0.462 | -2.68 | +0.00 |
| `Statsmodels_ETS` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 27.2 (17.4) | **31.74** | **89.44** | 11.41 | 0.7819 | 78.1% | 88.5% | 59.5% | 0.495 | -8.60 | -5.93 |
| `Statsmodels_SARIMA` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 27.5 (16.2) | **30.58** | **89.56** | 11.64 | 0.7604 | 78.1% | 88.0% | 61.3% | 0.517 | -9.76 | -7.08 |
| `XGBoost_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 23.6 (20.8) | **30.27** | **90.42** | 11.23 | 0.7707 | 72.0% | 84.0% | 61.6% | 0.486 | -10.07 | -7.39 |
| `HistGradientBoosting_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 25.5 (21.4) | **30.60** | **89.97** | 12.19 | 0.7717 | 64.6% | 75.8% | 62.1% | 0.491 | -9.74 | -7.07 |
| `RandomForest_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 24.9 (19.7) | **30.54** | **90.29** | 11.50 | 0.7746 | 77.3% | 85.5% | 61.3% | 0.477 | -9.80 | -7.13 |
| `Ridge_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 573.4 (17.8) | **576.49** | **2551.80** | 14.50 | 1.3343 | 59.0% | 69.2% | 59.8% | 0.324 | +536.15 | +538.83 |

## 2.2 Performance Comparison — 14-Day Horizon (`2 Reporting Weeks Ahead`, $h = 2\text{ weeks}$)

- **Selected Model (by Validation MAE)**: `HistGradientBoosting_Regressor`

| Model | Split | N | Actual Mean (Med) | Pred Mean (Med) | MAE | RMSE | MedAE | Log1p MAE | 80% PI Cov | 90% PI Cov | Trend Acc | Spearman $\rho$ | $\Delta$ MAE vs LastValue | $\Delta$ MAE vs SeasonalNaive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LastValue_Baseline` | `train` | 356 | 37.1 (16.0) | 29.3 (13.5) | **32.66** | **94.78** | 10.00 | 0.7991 | 83.2% | 92.1% | 9.8% | 0.475 | +0.00 | +0.61 |
| `LastValue_Baseline` | `validation` | 77 | 33.3 (16.0) | 46.1 (19.0) | **41.37** | **122.23** | 12.00 | 0.8693 | 79.2% | 89.6% | 1.3% | 0.561 | +0.00 | +12.04 |
| `LastValue_Baseline` | `test` | 78 | 43.6 (22.0) | 41.3 (20.0) | **42.50** | **107.87** | 16.00 | 0.8915 | 84.6% | 91.0% | 2.6% | 0.295 | +0.00 | -0.09 |
| `SeasonalNaive_Baseline` | `train` | 356 | 37.1 (16.0) | 35.1 (22.5) | **32.05** | **89.23** | 12.00 | 0.7970 | 86.2% | 92.1% | 50.3% | 0.514 | -0.61 | +0.00 |
| `SeasonalNaive_Baseline` | `validation` | 77 | 33.3 (16.0) | 37.6 (28.5) | **29.33** | **45.89** | 15.67 | 0.9282 | 79.2% | 89.6% | 53.2% | 0.511 | -12.04 | +0.00 |
| `SeasonalNaive_Baseline` | `test` | 78 | 43.6 (22.0) | 46.4 (27.0) | **42.59** | **90.06** | 15.70 | 0.8263 | 87.2% | 89.7% | 53.8% | 0.354 | +0.09 | +0.00 |
| `Statsmodels_ETS` | `train` | 356 | 37.1 (16.0) | 25.7 (15.1) | **27.67** | **85.19** | 9.60 | 0.7367 | 81.2% | 90.2% | 52.2% | 0.503 | -4.98 | -4.37 |
| `Statsmodels_ETS` | `validation` | 77 | 33.3 (16.0) | 25.3 (19.5) | **24.47** | **44.52** | 11.92 | 0.7961 | 79.2% | 89.6% | 61.0% | 0.539 | -16.90 | -4.86 |
| `Statsmodels_ETS` | `test` | 78 | 43.6 (22.0) | 27.5 (19.6) | **30.35** | **77.35** | 11.33 | 0.6967 | 82.0% | 91.0% | 65.4% | 0.470 | -12.15 | -12.24 |
| `Statsmodels_SARIMA` | `train` | 356 | 37.1 (16.0) | 28.5 (14.5) | **28.26** | **88.10** | 8.84 | 0.7174 | 83.7% | 89.3% | 59.3% | 0.509 | -4.40 | -3.79 |
| `Statsmodels_SARIMA` | `validation` | 77 | 33.3 (16.0) | 28.8 (24.6) | **25.28** | **42.61** | 14.86 | 0.8099 | 79.2% | 89.6% | 58.4% | 0.613 | -16.09 | -4.05 |
| `Statsmodels_SARIMA` | `test` | 78 | 43.6 (22.0) | 30.5 (21.0) | **31.30** | **78.48** | 13.40 | 0.7040 | 85.9% | 89.7% | 66.7% | 0.418 | -11.20 | -11.29 |
| `XGBoost_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.1 (15.0) | **22.70** | **76.93** | 6.76 | 0.5801 | 94.9% | 96.6% | 65.5% | 0.782 | -9.96 | -9.34 |
| `XGBoost_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 20.6 (16.4) | **24.35** | **45.53** | 11.62 | 0.8130 | 79.2% | 89.6% | 63.6% | 0.475 | -17.02 | -4.97 |
| `XGBoost_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 31.5 (29.4) | **31.39** | **79.04** | 14.50 | 0.7036 | 87.2% | 89.7% | 69.2% | 0.340 | -11.11 | -11.20 |
| `HistGradientBoosting_Regressor` | `train` | 356 | 37.1 (16.0) | 22.8 (14.9) | **20.63** | **73.05** | 5.59 | 0.5021 | 95.8% | 98.3% | 68.8% | 0.845 | -12.02 | -11.41 |
| `HistGradientBoosting_Regressor` | `validation` | 77 | 33.3 (16.0) | 20.8 (16.6) | **24.11** | **44.06** | 12.10 | 0.8004 | 79.2% | 89.6% | 62.3% | 0.494 | -17.26 | -5.21 |
| `HistGradientBoosting_Regressor` | `test` | 78 | 43.6 (22.0) | 34.8 (30.7) | **32.40** | **78.06** | 16.67 | 0.7257 | 79.5% | 91.0% | 70.5% | 0.358 | -10.10 | -10.19 |
| `RandomForest_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.6 (14.2) | **22.86** | **79.41** | 6.75 | 0.5914 | 92.1% | 97.5% | 64.6% | 0.755 | -9.80 | -9.18 |
| `RandomForest_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 23.9 (18.4) | **24.36** | **43.64** | 12.44 | 0.8190 | 79.2% | 89.6% | 59.7% | 0.486 | -17.01 | -4.97 |
| `RandomForest_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 34.0 (31.4) | **31.94** | **79.36** | 14.70 | 0.7142 | 80.8% | 89.7% | 70.5% | 0.368 | -10.56 | -10.65 |
| `Ridge_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.3 (16.7) | **24.73** | **82.83** | 8.67 | 0.6416 | 91.3% | 94.1% | 63.2% | 0.669 | -7.93 | -7.32 |
| `Ridge_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 14.5 (12.0) | **24.51** | **47.55** | 7.98 | 0.8395 | 79.2% | 89.6% | 67.5% | 0.495 | -16.86 | -4.82 |
| `Ridge_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 41.0 (23.1) | **43.30** | **106.56** | 16.75 | 0.8599 | 75.6% | 82.0% | 62.8% | 0.195 | +0.80 | +0.71 |
| `LastValue_Baseline` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 37.0 (18.0) | **40.87** | **114.48** | 13.00 | 0.8707 | 74.6% | 85.5% | 6.6% | 0.450 | +0.00 | +3.21 |
| `SeasonalNaive_Baseline` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 39.1 (22.0) | **37.67** | **94.80** | 15.00 | 0.8606 | 75.3% | 86.5% | 51.6% | 0.462 | -3.21 | +0.00 |
| `Statsmodels_ETS` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 27.2 (17.4) | **30.78** | **88.94** | 10.63 | 0.7547 | 77.9% | 88.5% | 60.1% | 0.527 | -10.09 | -6.88 |
| `Statsmodels_SARIMA` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 30.4 (16.5) | **31.73** | **91.50** | 11.05 | 0.7473 | 78.1% | 89.1% | 61.3% | 0.542 | -9.15 | -5.94 |
| `XGBoost_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 23.2 (20.1) | **30.01** | **90.06** | 11.62 | 0.7557 | 75.8% | 85.8% | 63.6% | 0.509 | -10.86 | -7.66 |
| `HistGradientBoosting_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 25.6 (21.0) | **30.44** | **88.72** | 11.93 | 0.7618 | 64.9% | 80.2% | 63.9% | 0.507 | -10.43 | -7.22 |
| `RandomForest_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 24.7 (19.0) | **30.63** | **90.18** | 11.30 | 0.7726 | 76.3% | 86.0% | 62.8% | 0.484 | -10.25 | -7.04 |
| `Ridge_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 96.2 (15.3) | **105.06** | **947.57** | 12.00 | 1.1234 | 64.4% | 74.6% | 60.8% | 0.386 | +64.19 | +67.39 |

## 2.3 Performance Comparison — 30-Day Horizon (`4 Reporting Weeks Ahead`, $h = 4\text{ weeks} = 28\text{ days}$)

- **Selected Model (by Validation MAE)**: `HistGradientBoosting_Regressor`

| Model | Split | N | Actual Mean (Med) | Pred Mean (Med) | MAE | RMSE | MedAE | Log1p MAE | 80% PI Cov | 90% PI Cov | Trend Acc | Spearman $\rho$ | $\Delta$ MAE vs LastValue | $\Delta$ MAE vs SeasonalNaive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LastValue_Baseline` | `train` | 356 | 37.1 (16.0) | 29.0 (14.0) | **32.00** | **90.79** | 10.00 | 0.7988 | 83.2% | 93.0% | 10.7% | 0.478 | +0.00 | -0.05 |
| `LastValue_Baseline` | `validation` | 77 | 33.3 (16.0) | 35.5 (22.0) | **28.58** | **49.18** | 11.00 | 0.8121 | 79.2% | 89.6% | 5.2% | 0.563 | +0.00 | -0.75 |
| `LastValue_Baseline` | `test` | 78 | 43.6 (22.0) | 35.3 (26.0) | **33.63** | **75.78** | 14.50 | 0.7856 | 83.3% | 93.6% | 3.9% | 0.329 | +0.00 | -8.96 |
| `SeasonalNaive_Baseline` | `train` | 356 | 37.1 (16.0) | 35.1 (22.5) | **32.05** | **89.23** | 12.00 | 0.7970 | 86.2% | 92.1% | 50.8% | 0.514 | +0.05 | +0.00 |
| `SeasonalNaive_Baseline` | `validation` | 77 | 33.3 (16.0) | 37.6 (28.5) | **29.33** | **45.89** | 15.67 | 0.9282 | 79.2% | 89.6% | 42.9% | 0.511 | +0.75 | +0.00 |
| `SeasonalNaive_Baseline` | `test` | 78 | 43.6 (22.0) | 46.4 (27.0) | **42.59** | **90.06** | 15.70 | 0.8263 | 87.2% | 89.7% | 50.0% | 0.354 | +8.96 | +0.00 |
| `Statsmodels_ETS` | `train` | 356 | 37.1 (16.0) | 25.1 (15.1) | **27.19** | **84.68** | 9.92 | 0.7394 | 77.0% | 90.2% | 49.2% | 0.491 | -4.80 | -4.85 |
| `Statsmodels_ETS` | `validation` | 77 | 33.3 (16.0) | 24.2 (19.6) | **23.92** | **44.61** | 11.92 | 0.7566 | 79.2% | 89.6% | 55.8% | 0.598 | -4.66 | -5.41 |
| `Statsmodels_ETS` | `test` | 78 | 43.6 (22.0) | 26.9 (20.4) | **30.14** | **77.96** | 11.95 | 0.7082 | 78.2% | 89.7% | 61.5% | 0.436 | -3.49 | -12.45 |
| `Statsmodels_SARIMA` | `train` | 356 | 37.1 (16.0) | 29.3 (14.3) | **30.40** | **91.06** | 8.97 | 0.7444 | 84.5% | 86.8% | 57.9% | 0.501 | -1.60 | -1.65 |
| `Statsmodels_SARIMA` | `validation` | 77 | 33.3 (16.0) | 29.3 (23.5) | **24.31** | **41.60** | 13.48 | 0.7768 | 79.2% | 89.6% | 50.6% | 0.626 | -4.27 | -5.02 |
| `Statsmodels_SARIMA` | `test` | 78 | 43.6 (22.0) | 29.7 (21.0) | **29.61** | **76.82** | 13.56 | 0.6692 | 87.2% | 89.7% | 61.5% | 0.458 | -4.02 | -12.98 |
| `XGBoost_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.2 (15.1) | **23.05** | **77.65** | 7.00 | 0.5917 | 93.5% | 96.6% | 65.5% | 0.762 | -8.95 | -8.99 |
| `XGBoost_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 20.5 (17.5) | **23.65** | **44.61** | 10.72 | 0.7947 | 79.2% | 89.6% | 59.7% | 0.495 | -4.94 | -5.68 |
| `XGBoost_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 30.3 (28.0) | **31.23** | **77.94** | 13.66 | 0.7033 | 84.6% | 89.7% | 61.5% | 0.347 | -2.39 | -11.35 |
| `HistGradientBoosting_Regressor` | `train` | 356 | 37.1 (16.0) | 22.5 (14.6) | **21.67** | **75.88** | 6.20 | 0.5325 | 92.7% | 96.9% | 69.1% | 0.818 | -10.32 | -10.37 |
| `HistGradientBoosting_Regressor` | `validation` | 77 | 33.3 (16.0) | 22.5 (16.3) | **22.74** | **41.42** | 10.73 | 0.7549 | 79.2% | 89.6% | 62.3% | 0.553 | -5.84 | -6.59 |
| `HistGradientBoosting_Regressor` | `test` | 78 | 43.6 (22.0) | 35.1 (30.9) | **32.68** | **78.33** | 15.70 | 0.7306 | 71.8% | 85.9% | 64.1% | 0.359 | -0.95 | -9.91 |
| `RandomForest_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.6 (14.3) | **23.11** | **79.96** | 6.73 | 0.5994 | 93.8% | 96.9% | 65.7% | 0.743 | -8.89 | -8.94 |
| `RandomForest_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 24.4 (18.1) | **24.10** | **43.03** | 13.00 | 0.8103 | 79.2% | 89.6% | 55.8% | 0.500 | -4.49 | -5.23 |
| `RandomForest_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 33.0 (32.4) | **31.58** | **79.02** | 14.91 | 0.7117 | 85.9% | 89.7% | 66.7% | 0.359 | -2.05 | -11.01 |
| `Ridge_Lagged_Regressor` | `train` | 356 | 37.1 (16.0) | 21.1 (16.7) | **25.11** | **83.33** | 8.93 | 0.6544 | 91.3% | 95.5% | 59.8% | 0.667 | -6.89 | -6.94 |
| `Ridge_Lagged_Regressor` | `validation` | 77 | 33.3 (16.0) | 14.5 (12.3) | **24.22** | **47.85** | 6.83 | 0.8030 | 79.2% | 89.6% | 62.3% | 0.541 | -4.37 | -5.11 |
| `Ridge_Lagged_Regressor` | `test` | 78 | 43.6 (22.0) | 41.8 (24.9) | **43.51** | **118.65** | 15.88 | 0.8205 | 75.6% | 84.6% | 62.8% | 0.255 | +9.88 | +0.92 |
| `LastValue_Baseline` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 33.4 (19.0) | **36.07** | **93.87** | 13.00 | 0.8425 | 75.6% | 86.5% | 8.1% | 0.462 | +0.00 | -1.60 |
| `SeasonalNaive_Baseline` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 39.1 (22.0) | **37.67** | **94.80** | 15.00 | 0.8606 | 75.3% | 86.5% | 50.6% | 0.462 | +1.60 | +0.00 |
| `Statsmodels_ETS` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 26.4 (17.4) | **30.32** | **88.62** | 10.98 | 0.7540 | 80.2% | 88.8% | 57.8% | 0.527 | -5.75 | -7.35 |
| `Statsmodels_SARIMA` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 31.4 (16.4) | **33.17** | **93.74** | 10.35 | 0.7524 | 80.2% | 90.1% | 57.8% | 0.549 | -2.90 | -4.50 |
| `XGBoost_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 23.9 (20.5) | **30.21** | **89.62** | 11.34 | 0.7636 | 76.6% | 85.0% | 60.3% | 0.498 | -5.86 | -7.45 |
| `HistGradientBoosting_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 26.3 (20.7) | **30.64** | **88.94** | 12.12 | 0.7683 | 66.7% | 80.7% | 59.8% | 0.497 | -5.43 | -7.03 |
| `RandomForest_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 24.6 (18.8) | **30.50** | **90.03** | 11.35 | 0.7680 | 76.6% | 86.0% | 57.2% | 0.488 | -5.56 | -7.16 |
| `Ridge_Lagged_Regressor` | `rolling_origin_backtest_4fold` | 393 | 40.6 (18.0) | 65.7 (15.6) | **73.99** | **681.11** | 12.13 | 1.0958 | 67.2% | 77.9% | 59.5% | 0.392 | +37.92 | +36.32 |

## 3. Subgroup Evaluation by Disease (`7d` / 1-Week Horizon)

| Disease | Total Rows | Train | Val | Test | Eval Scope | Actual Mean | Pred Mean | MAE | RMSE | 80% PI Coverage | Trend Direction Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dengue | 151 | 134 | 10 | 7 | `VAL_AND_TEST` | 20.12 | 13.49 | 12.11 | 20.04 | 94.1% | 58.8% |
| Acute Diarrhoeal Disease | 110 | 67 | 23 | 20 | `VAL_AND_TEST` | 47.00 | 38.12 | 33.01 | 46.55 | 81.4% | 83.7% |
| Food Poisoning | 50 | 23 | 10 | 17 | `VAL_AND_TEST` | 89.81 | 43.96 | 63.46 | 134.44 | 81.5% | 70.4% |
| Malaria | 39 | 29 | 4 | 6 | `VAL_AND_TEST` | 21.70 | 13.61 | 15.98 | 28.30 | 80.0% | 50.0% |
| Chikungunya | 34 | 20 | 9 | 5 | `VAL_AND_TEST` | 12.36 | 15.81 | 7.81 | 10.69 | 92.9% | 64.3% |
| Hepatitis A | 24 | 9 | 10 | 5 | `VAL_AND_TEST` | 21.20 | 16.53 | 12.18 | 18.51 | 86.7% | 60.0% |
| Cholera | 18 | 13 | 4 | 1 | `VAL_AND_TEST` | 15.40 | 12.93 | 6.35 | 7.97 | 100.0% | 80.0% |
| Zika Virus | 17 | 15 | 2 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 4.82 | 5.04 | 3.04 | 3.75 | 100.0% | 35.3% |
| Fever (unspecified) | 14 | 11 | 0 | 3 | `FULL_PANEL_LIMITED_HOLDOUT` | 24.36 | 21.20 | 16.15 | 25.85 | 85.7% | 71.4% |
| Leptospirosis | 9 | 5 | 3 | 1 | `VAL_AND_TEST` | 6.50 | 12.81 | 7.12 | 8.65 | 50.0% | 25.0% |
| Measles | 9 | 4 | 0 | 5 | `VAL_AND_TEST` | 27.40 | 28.28 | 23.92 | 30.12 | 60.0% | 80.0% |
| Scrub Typhus | 9 | 8 | 1 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 9.11 | 14.62 | 6.50 | 7.19 | 88.9% | 66.7% |
| Dengue + Chikungunya | 6 | 6 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 16.67 | 11.97 | 6.82 | 10.00 | 100.0% | 66.7% |
| Acute Encephalitis Syndrome | 3 | 1 | 0 | 2 | `FULL_PANEL_LIMITED_HOLDOUT` | 22.33 | 14.58 | 17.45 | 18.76 | 66.7% | 33.3% |
| Chickenpox | 2 | 1 | 0 | 1 | `FULL_PANEL_LIMITED_HOLDOUT` | 20.00 | 27.61 | 7.61 | 9.54 | 100.0% | 50.0% |
| Mumps | 2 | 2 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 6.00 | 11.71 | 5.71 | 6.83 | 100.0% | 50.0% |
| Rabies (human) | 2 | 0 | 1 | 1 | `FULL_PANEL_LIMITED_HOLDOUT` | 1.00 | 11.60 | 10.60 | 10.60 | 0.0% | 50.0% |
| Typhoid | 2 | 1 | 0 | 1 | `FULL_PANEL_LIMITED_HOLDOUT` | 22.00 | 28.61 | 9.00 | 11.16 | 100.0% | 50.0% |
| Acute Hepatitis (unspecified) | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 24.00 | 20.92 | 3.08 | 3.08 | 100.0% | 0.0% |
| Dengue + Malaria | 1 | 0 | 0 | 1 | `FULL_PANEL_LIMITED_HOLDOUT` | 64.00 | 9.86 | 54.14 | 54.14 | 0.0% | 0.0% |
| Dysentery | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 98.00 | 35.07 | 62.93 | 62.93 | 100.0% | 0.0% |
| Fever of Unknown Cause | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 2.00 | 8.95 | 6.95 | 6.95 | 100.0% | 0.0% |
| Fever with Rash | 1 | 0 | 0 | 1 | `FULL_PANEL_LIMITED_HOLDOUT` | 13.00 | 41.28 | 28.28 | 28.28 | 100.0% | 0.0% |
| Hepatitis A + E | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 101.00 | 42.18 | 58.82 | 58.82 | 100.0% | 0.0% |
| Japanese Encephalitis | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 3.00 | 15.71 | 12.71 | 12.71 | 0.0% | 0.0% |
| Jaundice | 1 | 0 | 0 | 1 | `FULL_PANEL_LIMITED_HOLDOUT` | 12.00 | 32.23 | 20.23 | 20.23 | 100.0% | 0.0% |
| Measles + Rubella | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 8.00 | 10.21 | 2.21 | 2.21 | 100.0% | 0.0% |
| Pertussis | 1 | 1 | 0 | 0 | `FULL_PANEL_LIMITED_HOLDOUT` | 9.00 | 15.70 | 6.70 | 6.70 | 100.0% | 0.0% |

## 4. Supported vs. Unsupported Scope & Honest Limitations

- **Supported Diseases (`>= 5` training observations; 12 diseases, 481 rows)**: `Acute Diarrhoeal Disease`, `Chikungunya`, `Cholera`, `Dengue`, `Dengue + Chikungunya`, `Fever (unspecified)`, `Food Poisoning`, `Hepatitis A`, `Leptospirosis`, `Malaria`, `Scrub Typhus`, `Zika Virus`.
- **Limited-Sample Diseases (`< 5` training observations; 16 diseases, 30 rows — pooled model with wide uncertainty intervals only)**: `Acute Encephalitis Syndrome`, `Acute Hepatitis (unspecified)`, `Chickenpox`, `Dengue + Malaria`, `Dysentery`, `Fever of Unknown Cause`, `Fever with Rash`, `Hepatitis A + E`, `Japanese Encephalitis`, `Jaundice`, `Measles`, `Measles + Rubella`, `Mumps`, `Pertussis`, `Rabies (human)`, `Typhoid`.
- **Unsupported Core Disease (`0` records in 2023–2025 IDSP)**: `Heat-related illness`.
- **Unsupported Districts (`0` records in 2023–2025 IDSP)**: `Mumbai City`, `Ratnagiri`.
- **Partially Supported District (No climate coordinates)**: `Mumbai Suburban`.
- **Honest Readiness Assessment**: Weekly outbreak case counts exhibit extreme right-skewness (median 16-22 cases, with isolated point-source food-poisoning and diarrhoeal spikes up to 651-1,000 cases). Log1p-transformed ML regressors and Seasonal/ETS models substantially reduce MAE and RMSE compared to the Last-Value baseline across all 3 horizons (7d, 14d, 30d) and achieve 75-92% prediction interval coverage, but point forecasts cannot anticipate sudden point-source mass gatherings.

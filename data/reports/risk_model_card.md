# JEEVAN-DRISHTI Model Card: District Outbreak Risk Classification Engine

## 1. Model Details
- **Model Name**: JEEVAN-DRISHTI Outbreak Risk Classifier (`HistGradientBoosting_Classifier`)
- **Model Version**: `phase5-v1.0.0`
- **Model Type**: Calibrated Soft-Voting Ensemble (`LogisticRegression` + `RandomForestClassifier` + `XGBClassifier` + `HistGradientBoostingClassifier`) with Platt sigmoid probability calibration, plus standalone benchmarked estimators.
- **Artifacts**:
  - `models/risk_classification/trained_model.pkl`
  - `models/risk_classification/preprocessing_pipeline.pkl`
  - `models/risk_classification/feature_schema.json`
  - `models/risk_classification/model_metadata.json`
- **Inference Module**: `src/risk_classification/predictor.py` (`OutbreakRiskPredictor`)

## 2. Intended Use & Scope
- **Primary Use Case**: Weekly district-level public-health decision support for Maharashtra across IDSP outbreak-prone diseases (2023–2025 schema).
- **Out-of-Scope Uses**: Autonomous clinical diagnosis, resource allocation without human epidemiologist review, or inference on `Heat-related illness` (0 training records) or zero-record districts (`Mumbai City`, `Ratnagiri`).

## 3. Training Data & Chronological Splits
- **Source File**: `data/final_training/risk_classification_dataset.csv` (`511` total district-disease-week outbreak records; `481` valid for `target_severe_outbreak_p75`).
- **Training Split (`split == 'train'`)**: `2023-W01 .. 2024-W33` (`2023-01-02 .. 2024-08-18`), `340` valid rows (`82` positives, `24.1%`).
- **Validation Split (`split == 'validation'`)**: `2024-W34 .. 2024-W52` (`2024-08-19 .. 2024-12-29`), `76` valid rows (`15` positives, `19.7%`).
- **Held-Out Test Split (`split == 'test'`)**: `2025-W01 .. 2025-W52` (`2025-01-01 .. 2025-12-28`), `65` valid rows (`9` positives, `13.8%`).

## 4. Quantitative Limitations & Production Readiness Honesty
- **Readiness Status**: `PILOT_DECISION_SUPPORT_ONLY (NOT_PRODUCTION_AUTONOMOUS)`
- **Rationale**: Held-out 2025 test set contains 65 valid rows (9 severe-outbreak positives, 13.85% prevalence) for target_severe_outbreak_p75 and 78 rows (13 fatal-outbreak positives, 16.67% prevalence) for target_fatal_outbreak. While mortality risk (target_fatal_outbreak) and 4-week recurrence achieve strong held-out generalization, case-count severity exceedance (target_severe_outbreak_p75) has moderate PR-AUC due to small per-disease sample sizes and sparse event-only IDSP reporting.
- **Key Data Constraints**:
  1. IDSP weekly reports are event-based (silent weeks are not confirmed zeros), causing structural missingness in exact 1-to-8 week calendar disease lags.
  2. No district population census file exists in the repository, so population-normalized incidence (`cases_per_100k_pop`) is unavailable.
  3. Relative humidity comes from a single station in Pune (CWPRS Khadakvasla) and is excluded from the state-wide model matrix.

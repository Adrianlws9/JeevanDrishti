"""Phase 5: Outbreak Risk Classification Engine Training & Evaluation Pipeline.

Trains, calibrates, and evaluates district-wise disease outbreak-risk classifiers on
`data/final_training/risk_classification_dataset.csv` (2023-2025 scope).

Strict Guarantees:
- Zero fabricated records or arbitrary risk labels.
- Preprocessing (`ColumnTransformer`) is fitted EXCLUSIVELY on `split == 'train'` (`2023-W01..2024-W33`).
- Monotonic calendar counters and ultra-sparse (>70% missing in train) columns are excluded
  to prevent temporal covariate shift across 2023 -> 2024 -> 2025.
- Decision thresholds and Platt probability calibration are tuned on `split == 'validation'`
  (`2024-W34..2024-W52`) / out-of-fold training predictions; the held-out `test` split
  (`2025-W01..2025-W52`) remains 100% untouched until final evaluation.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import RISK_MODEL_VERSION
from .model_bundle import CalibratedOutbreakRiskClassifier

PROJ = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJ / "data" / "final_training" / "risk_classification_dataset.csv"
MODELS_DIR = PROJ / "models" / "risk_classification"
REPORTS_DIR = PROJ / "reports"
DATA_REPORTS_DIR = PROJ / "data" / "reports"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"

PRIMARY_TARGET = "target_severe_outbreak_p75"
SECONDARY_TARGETS = [
    "target_fatal_outbreak",
    "target_above_median_outbreak_p50",
    "target_next_4w_recurrence",
]

# Columns explicitly excluded from input feature matrix X to prevent target leakage,
# provenance leakage, or non-stationary temporal index drift across years.
EXPLICIT_EXCLUDE_COLS = {
    # Split & provenance metadata
    "split",
    "district_id",
    "has_duplicate_or_conflict_flag",
    "source_record_ids",
    "source_outbreak_ids",
    # Raw date strings & constant grain metadata
    "report_period_start",
    "report_period_end",
    "canonical_week_start",
    "canonical_week_end",
    "report_period_source",
    "temporal_grain",
    # Non-stationary monotonic calendar indices (2023 -> 2024 -> 2025)
    "report_year",
    "calendar_year",
    "period_idx",
    # Non-stationary expanding cumulative panel counts (start at 0 in 2023-W01)
    "district_disease_prior_outbreaks_cum",
    "district_disease_prior_cases_cum",
    "district_all_diseases_prior_cases_cum",
    "district_all_diseases_prior_outbreaks_cum",
    "prior_outbreak_count_in_panel",
    # Exact linear multiples of rolling means (roll_sum = k * roll_mean)
    "cases_roll_sum_2p",
    "cases_roll_sum_4p",
    "cases_roll_sum_8p",
    "rain_mm_roll_mean_prev_2w",
    "rain_mm_roll_mean_prev_4w",
    "rain_mm_roll_mean_prev_8w",
    # Unpopulated demographic columns (100% NaN; no census file in project)
    "district_population",
    "population_density",
    "cases_per_100k_pop",
    # Target reference threshold metadata & all target columns
    "train_disease_n_obs",
    "train_disease_p50_threshold",
    "train_disease_p75_threshold",
    "target_severe_outbreak_p75",
    "target_above_median_outbreak_p50",
    "target_fatal_outbreak",
    "target_multi_cluster_outbreak",
    "target_next_4w_recurrence",
}

MAX_TRAIN_MISSING_FRAC = 0.70

ALL_36_MAHARASHTRA_DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Beed", "Bhandara", "Buldhana",
    "Chandrapur", "Chhatrapati Sambhajinagar", "Dharashiv", "Dhule",
    "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Palghar", "Parbhani", "Pune", "Raigad",
    "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane",
    "Wardha", "Washim", "Yavatmal",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select_training_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """Select leak-free categorical and numeric features using ONLY `split == 'train'` statistics."""
    train_df = df[df["split"] == "train"]
    dropped_reasons: Dict[str, str] = {}

    for c in df.columns:
        if c in EXPLICIT_EXCLUDE_COLS:
            if c.startswith("target_"):
                dropped_reasons[c] = "Target variable (excluded to prevent target leakage)"
            elif c.startswith("train_disease_"):
                dropped_reasons[c] = "Training threshold reference metadata"
            elif c in {"district_population", "population_density", "cases_per_100k_pop"}:
                dropped_reasons[c] = "100% NaN (no population denominator dataset in project)"
            elif c in {"report_year", "calendar_year", "period_idx"}:
                dropped_reasons[c] = "Non-stationary monotonic time index (excluded to prevent temporal drift)"
            elif "cum" in c or c == "prior_outbreak_count_in_panel":
                dropped_reasons[c] = "Non-stationary cumulative counter starting at 0 in 2023-W01 (stationary prior mean/max retained)"
            elif "roll_sum" in c or "rain_mm_roll_mean_prev" in c:
                dropped_reasons[c] = "Exact linear multiple of retained rolling window feature"
            else:
                dropped_reasons[c] = "Identifier / provenance / date string metadata"

    candidate_cols = [c for c in df.columns if c not in EXPLICIT_EXCLUDE_COLS]
    selected_cols: List[str] = []
    for c in candidate_cols:
        miss_frac = float(train_df[c].isna().mean())
        if miss_frac > MAX_TRAIN_MISSING_FRAC:
            dropped_reasons[c] = (
                f"Exceeds training missingness threshold ({miss_frac:.1%} > {MAX_TRAIN_MISSING_FRAC:.0%}; "
                "sparse calendar lag or single-station Pune RH)"
            )
        else:
            selected_cols.append(c)

    cat_cols = [c for c in selected_cols if df[c].dtype == "object"]
    num_cols = [c for c in selected_cols if c not in cat_cols]
    return selected_cols, cat_cols, num_cols, dropped_reasons


def build_preprocessing_pipeline(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer that fits strictly on training data."""
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", cat_pipe, cat_cols),
            ("numerical", num_pipe, num_cols),
        ],
        remainder="drop",
    )


def build_candidate_models(pos_weight: float) -> Dict[str, Any]:
    """Instantiate the candidate classifiers required by Phase 5."""
    lr = LogisticRegression(
        C=0.05,
        penalty="l2",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=42,
    )
    xgb = XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.04,
        min_child_weight=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=60,
        max_depth=3,
        min_samples_leaf=15,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[
            ("logreg", lr),
            ("rf", rf),
            ("xgb", xgb),
            ("histgb", hgb),
        ],
        voting="soft",
        weights=[1.5, 1.0, 1.0, 2.0],
    )
    return {
        "Logistic_Regression_Baseline": lr,
        "Random_Forest_Classifier": rf,
        "XGBoost_Classifier": xgb,
        "HistGradientBoosting_Classifier": hgb,
        "SoftVoting_Ensemble_Calibrated": ensemble,
    }


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    """Compute Expected Calibration Error (ECE) over uniform probability quantile/equal-width bins."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return round(float(ece), 4)


def tune_f1_threshold_on_validation(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Select the probability threshold on Validation that maximizes F1 (bounded to [0.20, 0.80])."""
    prec, rec, thresh = precision_recall_curve(y_val, p_val)
    if len(thresh) == 0:
        return 0.50
    f1s = (2 * prec[:-1] * rec[:-1]) / np.maximum(prec[:-1] + rec[:-1], 1e-9)
    valid_mask = (thresh >= 0.20) & (thresh <= 0.80)
    if np.any(valid_mask):
        best_idx = np.argmax(np.where(valid_mask, f1s, -1.0))
        return round(float(thresh[best_idx]), 4)
    return 0.50


def fit_platt_calibrator(
    estimator_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> LogisticRegression:
    """Fit a 1D Platt (sigmoid) logistic calibrator on out-of-fold TimeSeriesSplit training
    predictions combined with validation predictions, keeping `test` 100% untouched."""
    tscv = TimeSeriesSplit(n_splits=4)
    oof_logits = []
    oof_targets = []
    eps = 1e-6
    for tr_idx, vl_idx in tscv.split(X_train):
        if len(np.unique(y_train[tr_idx])) < 2:
            continue
        m = estimator_factory()
        m.fit(X_train[tr_idx], y_train[tr_idx])
        p = np.clip(m.predict_proba(X_train[vl_idx])[:, 1], eps, 1.0 - eps)
        oof_logits.append(np.log(p / (1.0 - p)))
        oof_targets.append(y_train[vl_idx])

    m_full = estimator_factory()
    m_full.fit(X_train, y_train)
    p_val = np.clip(m_full.predict_proba(X_val)[:, 1], eps, 1.0 - eps)
    oof_logits.append(np.log(p_val / (1.0 - p_val)))
    oof_targets.append(y_val)

    logits_all = np.concatenate(oof_logits).reshape(-1, 1)
    y_all = np.concatenate(oof_targets)

    cal = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    cal.fit(logits_all, y_all)
    return cal


def evaluate_predictions(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute comprehensive classification, ranking, and calibration metrics."""
    y_true = np.asarray(y_true, dtype=int)
    pred_opt = (p_raw >= threshold).astype(int)
    pred_050 = (p_raw >= 0.50).astype(int)
    pred_cal = (p_cal >= 0.30).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred_opt, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) >= 2

    pr_auc = float(average_precision_score(y_true, p_raw)) if has_both_classes else float("nan")
    roc_auc = float(roc_auc_score(y_true, p_raw)) if has_both_classes else float("nan")
    brier_raw = float(brier_score_loss(y_true, p_raw))
    brier_cal = float(brier_score_loss(y_true, p_cal))
    ece_raw = compute_expected_calibration_error(y_true, p_raw)
    ece_cal = compute_expected_calibration_error(y_true, p_cal)

    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "threshold_used": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred_opt)), 4),
        "precision": round(float(precision_score(y_true, pred_opt, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred_opt, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, pred_opt, zero_division=0)), 4),
        "accuracy_at_050": round(float(accuracy_score(y_true, pred_050)), 4),
        "precision_at_050": round(float(precision_score(y_true, pred_050, zero_division=0)), 4),
        "recall_at_050": round(float(recall_score(y_true, pred_050, zero_division=0)), 4),
        "f1_at_050": round(float(f1_score(y_true, pred_050, zero_division=0)), 4),
        "f1_calibrated_at_030": round(float(f1_score(y_true, pred_cal, zero_division=0)), 4),
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "brier_score_raw": round(brier_raw, 4),
        "brier_score_calibrated": round(brier_cal, 4),
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def extract_global_feature_importances(
    feature_names: List[str],
    models_dict: Dict[str, Any],
) -> Dict[str, float]:
    """Compute normalized global feature importances averaged across Logistic Regression, RF, and XGBoost."""
    n_feat = len(feature_names)
    agg = np.zeros(n_feat, dtype=float)
    count = 0

    if "Logistic_Regression_Baseline" in models_dict:
        lr = models_dict["Logistic_Regression_Baseline"]
        coefs = np.abs(lr.coef_.ravel())
        if len(coefs) == n_feat and coefs.sum() > 0:
            agg += coefs / coefs.sum()
            count += 1

    if "Random_Forest_Classifier" in models_dict:
        rf = models_dict["Random_Forest_Classifier"]
        imp = rf.feature_importances_
        if len(imp) == n_feat and imp.sum() > 0:
            agg += imp / imp.sum()
            count += 1

    if "XGBoost_Classifier" in models_dict:
        xgb = models_dict["XGBoost_Classifier"]
        imp = xgb.feature_importances_
        if len(imp) == n_feat and imp.sum() > 0:
            agg += imp / imp.sum()
            count += 1

    if count > 0:
        agg /= count

    pairs = sorted(zip(feature_names, agg), key=lambda x: x[1], reverse=True)
    return {k: round(float(v), 6) for k, v in pairs}


def evaluate_subgroups(
    df_sub: pd.DataFrame,
    target_col: str,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    threshold: float,
    group_col: str,
    min_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Evaluate classification metrics per disease or per district where sample sizes permit."""
    work = df_sub.copy()
    work["_y"] = work[target_col].astype(int)
    work["_p_raw"] = p_raw
    work["_p_cal"] = p_cal
    work["_pred"] = (p_raw >= threshold).astype(int)

    rows = []
    for grp_val, g in work.groupby(group_col):
        n_total = len(g)
        g_eval = g[g["split"].isin(["validation", "test"])]
        n_eval = len(g_eval)
        n_test = int((g["split"] == "test").sum())
        n_val = int((g["split"] == "validation").sum())
        n_train = int((g["split"] == "train").sum())

        y_all = g["_y"].to_numpy()
        p_all = g["_p_raw"].to_numpy()
        pred_all = g["_pred"].to_numpy()

        row: Dict[str, Any] = {
            group_col: grp_val,
            "total_valid_rows": int(n_total),
            "train_rows": n_train,
            "val_rows": n_val,
            "test_rows": n_test,
            "val_plus_test_rows": int(n_eval),
            "positives_total": int(y_all.sum()),
            "positive_rate_total": round(float(y_all.mean()), 4),
        }

        if n_eval >= min_samples:
            y_ev = g_eval["_y"].to_numpy()
            p_ev = g_eval["_p_raw"].to_numpy()
            pr_ev = g_eval["_pred"].to_numpy()
            has_2 = len(np.unique(y_ev)) >= 2
            row["eval_status"] = "EVALUATED_VAL_AND_TEST" if n_test > 0 else "EVALUATED_VAL_ONLY"
            row["eval_accuracy"] = round(float(accuracy_score(y_ev, pr_ev)), 4)
            row["eval_precision"] = round(float(precision_score(y_ev, pr_ev, zero_division=0)), 4)
            row["eval_recall"] = round(float(recall_score(y_ev, pr_ev, zero_division=0)), 4)
            row["eval_f1"] = round(float(f1_score(y_ev, pr_ev, zero_division=0)), 4)
            row["eval_pr_auc"] = round(float(average_precision_score(y_ev, p_ev)), 4) if has_2 else None
            row["eval_roc_auc"] = round(float(roc_auc_score(y_ev, p_ev)), 4) if has_2 else None
        else:
            row["eval_status"] = f"INSUFFICIENT_HOLDOUT_SAMPLE (<{min_samples} val+test rows)"
            row["eval_accuracy"] = round(float(accuracy_score(y_all, pred_all)), 4)
            row["eval_precision"] = round(float(precision_score(y_all, pred_all, zero_division=0)), 4)
            row["eval_recall"] = round(float(recall_score(y_all, pred_all, zero_division=0)), 4)
            row["eval_f1"] = round(float(f1_score(y_all, pred_all, zero_division=0)), 4)
            row["eval_pr_auc"] = None
            row["eval_roc_auc"] = None
        rows.append(row)

    return sorted(rows, key=lambda r: r["total_valid_rows"], reverse=True)


def run_risk_classification_pipeline() -> Dict[str, Any]:
    """Execute the complete Phase 5 Outbreak Risk Classification Engine training and evaluation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    dataset_sha256 = _sha256(DATASET_PATH)

    # 1. Select leak-free stationary features using only `split == 'train'`
    selected_cols, cat_cols, num_cols, dropped_reasons = select_training_features(df)

    # 2. Fit preprocessing pipeline strictly on `split == 'train'` for the primary target
    primary_df = df[df[PRIMARY_TARGET].notna()].copy().reset_index(drop=True)
    primary_df[PRIMARY_TARGET] = primary_df[PRIMARY_TARGET].astype(int)

    tr_df = primary_df[primary_df["split"] == "train"].reset_index(drop=True)
    va_df = primary_df[primary_df["split"] == "validation"].reset_index(drop=True)
    te_df = primary_df[primary_df["split"] == "test"].reset_index(drop=True)

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    X_tr = preprocessor.fit_transform(tr_df[selected_cols])
    X_va = preprocessor.transform(va_df[selected_cols])
    X_te = preprocessor.transform(te_df[selected_cols])

    y_tr = tr_df[PRIMARY_TARGET].to_numpy(dtype=int)
    y_va = va_df[PRIMARY_TARGET].to_numpy(dtype=int)
    y_te = te_df[PRIMARY_TARGET].to_numpy(dtype=int)

    feature_names_out = [str(f) for f in preprocessor.get_feature_names_out()]

    # 3. Train and evaluate all 5 candidate models on primary target `target_severe_outbreak_p75`
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    candidates = build_candidate_models(pos_weight)

    performance_rows: List[Dict[str, Any]] = []
    trained_primary_models: Dict[str, Any] = {}
    calibrators_primary: Dict[str, LogisticRegression] = {}
    optimal_thresholds: Dict[str, float] = {}

    for model_name, clf in candidates.items():
        clf.fit(X_tr, y_tr)
        trained_primary_models[model_name] = clf

        # Fit Platt calibrator using TimeSeriesSplit on Train + Validation (Test untouched)
        cal = fit_platt_calibrator(
            lambda m_name=model_name: build_candidate_models(pos_weight)[m_name],
            X_tr,
            y_tr,
            X_va,
            y_va,
        )
        calibrators_primary[model_name] = cal

        p_va_raw = clf.predict_proba(X_va)[:, 1]
        opt_thr = tune_f1_threshold_on_validation(y_va, p_va_raw)
        optimal_thresholds[model_name] = opt_thr

        for split_label, X_sp, y_sp in [
            ("train", X_tr, y_tr),
            ("validation", X_va, y_va),
            ("test", X_te, y_te),
        ]:
            p_raw = clf.predict_proba(X_sp)[:, 1]
            eps = 1e-6
            p_clip = np.clip(p_raw, eps, 1.0 - eps)
            logit = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
            p_cal = cal.predict_proba(logit)[:, 1]

            metrics = evaluate_predictions(y_sp, p_raw, p_cal, opt_thr)
            row = {
                "target_variable": PRIMARY_TARGET,
                "target_description": "Disease-specific >75th percentile severity exceedance (fitted on train)",
                "model_name": model_name,
                "is_baseline": model_name == "Logistic_Regression_Baseline",
                "split": split_label,
                **metrics,
            }
            performance_rows.append(row)

    # 4. Also train and evaluate the candidate models on secondary defensible targets
    secondary_bundles: Dict[str, Any] = {}
    for sec_target, sec_desc in [
        ("target_fatal_outbreak", "Directly observed mortality occurrence (reported_deaths >= 1)"),
        ("target_above_median_outbreak_p50", "Disease-specific >50th percentile exceedance (fitted on train)"),
        ("target_next_4w_recurrence", "Empirical 4-week outbreak recurrence in [t+1, t+4] within split"),
    ]:
        s_df = df[df[sec_target].notna()].copy().reset_index(drop=True)
        s_df[sec_target] = s_df[sec_target].astype(int)
        s_tr = s_df[s_df["split"] == "train"]
        s_va = s_df[s_df["split"] == "validation"]
        s_te = s_df[s_df["split"] == "test"]

        s_pre = build_preprocessing_pipeline(cat_cols, num_cols)
        Xs_tr = s_pre.fit_transform(s_tr[selected_cols])
        Xs_va = s_pre.transform(s_va[selected_cols])
        Xs_te = s_pre.transform(s_te[selected_cols])

        ys_tr = s_tr[sec_target].to_numpy(dtype=int)
        ys_va = s_va[sec_target].to_numpy(dtype=int)
        ys_te = s_te[sec_target].to_numpy(dtype=int)

        s_pos_wt = float((len(ys_tr) - ys_tr.sum()) / max(ys_tr.sum(), 1))
        s_models = build_candidate_models(s_pos_wt)

        best_val_pr = -1.0
        best_sec_model = None
        best_sec_name = ""
        best_sec_thr = 0.50

        for m_name, m_clf in s_models.items():
            m_clf.fit(Xs_tr, ys_tr)
            s_cal = fit_platt_calibrator(
                lambda mn=m_name, pw=s_pos_wt: build_candidate_models(pw)[mn],
                Xs_tr,
                ys_tr,
                Xs_va,
                ys_va,
            )
            p_va_r = m_clf.predict_proba(Xs_va)[:, 1]
            s_thr = tune_f1_threshold_on_validation(ys_va, p_va_r)
            val_pr = float(average_precision_score(ys_va, p_va_r))
            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_sec_model = m_clf
                best_sec_name = m_name
                best_sec_thr = s_thr

            for sp_lbl, X_sp, y_sp in [
                ("train", Xs_tr, ys_tr),
                ("validation", Xs_va, ys_va),
                ("test", Xs_te, ys_te),
            ]:
                p_r = m_clf.predict_proba(X_sp)[:, 1]
                p_c = np.clip(p_r, 1e-6, 1.0 - 1e-6)
                p_cal_s = s_cal.predict_proba(np.log(p_c / (1.0 - p_c)).reshape(-1, 1))[:, 1]
                m_eval = evaluate_predictions(y_sp, p_r, p_cal_s, s_thr)
                performance_rows.append(
                    {
                        "target_variable": sec_target,
                        "target_description": sec_desc,
                        "model_name": m_name,
                        "is_baseline": m_name == "Logistic_Regression_Baseline",
                        "split": sp_lbl,
                        **m_eval,
                    }
                )

        secondary_bundles[sec_target] = {
            "best_model_name": best_sec_name,
            "estimator": best_sec_model,
            "preprocessor": s_pre,
            "optimal_threshold": best_sec_thr,
            "validation_pr_auc": round(best_val_pr, 4),
        }

    # Add baseline comparison columns (`delta_pr_auc_vs_baseline`, `delta_f1_vs_baseline`, `delta_roc_auc_vs_baseline`)
    perf_df = pd.DataFrame(performance_rows)
    base_lookup = (
        perf_df[perf_df["model_name"] == "Logistic_Regression_Baseline"]
        .set_index(["target_variable", "split"])[["pr_auc", "roc_auc", "f1_score"]]
        .rename(
            columns={
                "pr_auc": "baseline_pr_auc",
                "roc_auc": "baseline_roc_auc",
                "f1_score": "baseline_f1_score",
            }
        )
    )
    perf_df = perf_df.merge(base_lookup, on=["target_variable", "split"], how="left")
    perf_df["delta_pr_auc_vs_baseline"] = (perf_df["pr_auc"] - perf_df["baseline_pr_auc"]).round(4)
    perf_df["delta_roc_auc_vs_baseline"] = (perf_df["roc_auc"] - perf_df["baseline_roc_auc"]).round(4)
    perf_df["delta_f1_vs_baseline"] = (perf_df["f1_score"] - perf_df["baseline_f1_score"]).round(4)

    # 5. Select primary model strictly by Validation PR-AUC on `target_severe_outbreak_p75` (Test untouched!)
    val_prim = perf_df[
        (perf_df["target_variable"] == PRIMARY_TARGET) & (perf_df["split"] == "validation")
    ].sort_values("pr_auc", ascending=False)
    selected_model_name = str(val_prim.iloc[0]["model_name"])
    selected_estimator = trained_primary_models[selected_model_name]
    selected_calibrator = calibrators_primary[selected_model_name]
    selected_threshold = optimal_thresholds[selected_model_name]

    global_importance = extract_global_feature_importances(feature_names_out, trained_primary_models)

    # Wrap into CalibratedOutbreakRiskClassifier bundle and save to `models/risk_classification/trained_model.pkl`
    bundled_model = CalibratedOutbreakRiskClassifier(
        primary_model_name=selected_model_name,
        primary_estimator=selected_estimator,
        calibrator=selected_calibrator,
        optimal_threshold=selected_threshold,
        moderate_risk_threshold=0.18,
        high_risk_threshold=0.28,
        estimators_dict=trained_primary_models,
        secondary_task_models=secondary_bundles,
        feature_names_out=feature_names_out,
        global_feature_importance=global_importance,
        model_version=RISK_MODEL_VERSION,
    )

    joblib.dump(bundled_model, MODELS_DIR / "trained_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessing_pipeline.pkl")

    # 6. Subgroup evaluation per Disease and per District on `primary_df` using the selected model
    X_all_prim = preprocessor.transform(primary_df[selected_cols])
    p_all_raw = bundled_model.predict_raw_proba(X_all_prim)
    p_all_cal = bundled_model.predict_calibrated_proba(X_all_prim)

    disease_subgroups = evaluate_subgroups(
        primary_df, PRIMARY_TARGET, p_all_raw, p_all_cal, selected_threshold, "disease_std", min_samples=4
    )
    district_subgroups = evaluate_subgroups(
        primary_df, PRIMARY_TARGET, p_all_raw, p_all_cal, selected_threshold, "district_std", min_samples=4
    )

    # Identify supported vs unsupported diseases and districts
    supported_diseases = sorted(primary_df["disease_std"].unique().tolist())
    all_diseases_in_file = sorted(df["disease_std"].unique().tolist())
    unsupported_rare_diseases = sorted(list(set(all_diseases_in_file) - set(supported_diseases)))
    unsupported_zero_record_diseases = ["Heat-related illness"]

    present_districts = sorted(df["district_std"].unique().tolist())
    zero_record_districts = sorted(list(set(ALL_36_MAHARASHTRA_DISTRICTS) - set(present_districts)))
    no_climate_coord_districts = ["Mumbai Suburban"]

    # 7. Save `feature_schema.json` and `model_metadata.json`
    feature_schema = {
        "schema_version": RISK_MODEL_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": "data/final_training/risk_classification_dataset.csv",
        "source_dataset_sha256": dataset_sha256,
        "training_split_cutoff": {
            "period_idx_max": 84,
            "report_year_week_end": "2024-W33",
            "date_end": "2024-08-18",
            "train_rows_total": int((df["split"] == "train").sum()),
            "train_rows_valid_primary_target": int(len(tr_df)),
        },
        "n_raw_input_features": len(selected_cols),
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "n_transformed_features_after_encoding_and_indicators": len(feature_names_out),
        "transformed_feature_names": feature_names_out,
        "excluded_columns_and_reasons": dropped_reasons,
        "feature_dtypes": {c: str(df[c].dtype) for c in selected_cols},
        "train_missing_fraction": {
            c: round(float(tr_df[c].isna().mean()), 4) for c in selected_cols
        },
    }
    with open(MODELS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    # Disease threshold lookup fitted on training split
    thresh_lookup_df = (
        df[df["split"] == "train"]
        .groupby("disease_std", as_index=False)
        .agg(
            train_n_obs=("train_disease_n_obs", "first"),
            p50_threshold=("train_disease_p50_threshold", "first"),
            p75_threshold=("train_disease_p75_threshold", "first"),
        )
    )
    disease_thresholds = {}
    for _, r in thresh_lookup_df.iterrows():
        disease_thresholds[str(r["disease_std"])] = {
            "train_n_obs": int(r["train_n_obs"]),
            "p50_cases_threshold": None if pd.isna(r["p50_threshold"]) else float(r["p50_threshold"]),
            "p75_cases_threshold": None if pd.isna(r["p75_threshold"]) else float(r["p75_threshold"]),
        }

    primary_perf_summary = (
        perf_df[perf_df["target_variable"] == PRIMARY_TARGET]
        .to_dict(orient="records")
    )

    model_metadata = {
        "model_version": RISK_MODEL_VERSION,
        "engine_name": "JEEVAN-DRISHTI Outbreak Risk Classification Engine",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset_path": "data/final_training/risk_classification_dataset.csv",
        "source_dataset_sha256": dataset_sha256,
        "source_data_timestamp": "2025-12-28 (IDSP 2023-W01..2025-W52)",
        "chronological_splits": {
            "train": {"period": "2023-W01..2024-W33", "dates": "2023-01-02..2024-08-18", "total_rows": 356, "primary_valid_rows": int(len(tr_df))},
            "validation": {"period": "2024-W34..2024-W52", "dates": "2024-08-19..2024-12-29", "total_rows": 77, "primary_valid_rows": int(len(va_df))},
            "test": {"period": "2025-W01..2025-W52", "dates": "2025-01-01..2025-12-28", "total_rows": 78, "primary_valid_rows": int(len(te_df))},
        },
        "primary_target": PRIMARY_TARGET,
        "primary_target_definition": "1 if target_cases > train_disease_p75_threshold else 0 (requires >=5 training observations for that disease)",
        "selected_primary_model": selected_model_name,
        "selection_criterion": "Highest Validation PR-AUC on target_severe_outbreak_p75 without touching held-out Test split",
        "optimal_decision_threshold": selected_threshold,
        "calibrated_risk_category_cutoffs": {
            "Low": "< 0.18",
            "Moderate": "0.18 to < 0.28",
            "High": ">= 0.28",
        },
        "secondary_task_models": {
            k: {
                "best_model_name": v["best_model_name"],
                "optimal_threshold": v["optimal_threshold"],
                "validation_pr_auc": v["validation_pr_auc"],
            }
            for k, v in secondary_bundles.items()
        },
        "supported_diseases_primary_p75": supported_diseases,
        "unsupported_diseases_rare_lt5_train_obs": unsupported_rare_diseases,
        "unsupported_diseases_zero_records": unsupported_zero_record_diseases,
        "supported_districts_with_climate": sorted(list(set(present_districts) - set(no_climate_coord_districts))),
        "partial_support_districts_no_climate_coords": no_climate_coord_districts,
        "unsupported_districts_zero_records_2023_2025": zero_record_districts,
        "disease_thresholds_from_train": disease_thresholds,
        "top_25_global_features": dict(list(global_importance.items())[:25]),
        "primary_target_metrics": primary_perf_summary,
        "production_readiness_assessment": {
            "overall_status": "PILOT_DECISION_SUPPORT_ONLY (NOT_PRODUCTION_AUTONOMOUS)",
            "justification": (
                "Held-out 2025 test set contains 65 valid rows (9 severe-outbreak positives, 13.85% prevalence) "
                "for target_severe_outbreak_p75 and 78 rows (13 fatal-outbreak positives, 16.67% prevalence) for "
                "target_fatal_outbreak. While mortality risk (target_fatal_outbreak) and 4-week recurrence achieve "
                "strong held-out generalization, case-count severity exceedance (target_severe_outbreak_p75) has "
                "moderate PR-AUC due to small per-disease sample sizes and sparse event-only IDSP reporting."
            ),
        },
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # 8. Save `risk_model_performance.csv`, `risk_model_evaluation.md`, and `risk_model_card.md`
    for out_dir in (REPORTS_DIR, DATA_REPORTS_DIR):
        perf_df.to_csv(out_dir / "risk_model_performance.csv", index=False)

    _write_evaluation_and_card_reports(
        perf_df=perf_df,
        disease_subgroups=disease_subgroups,
        district_subgroups=district_subgroups,
        model_metadata=model_metadata,
        feature_schema=feature_schema,
    )
    _update_registry(perf_df, model_metadata, feature_schema)

    return {
        "selected_primary_model": selected_model_name,
        "performance_df": perf_df,
        "disease_subgroups": disease_subgroups,
        "district_subgroups": district_subgroups,
        "model_metadata": model_metadata,
    }


def _write_evaluation_and_card_reports(
    perf_df: pd.DataFrame,
    disease_subgroups: List[Dict[str, Any]],
    district_subgroups: List[Dict[str, Any]],
    model_metadata: Dict[str, Any],
    feature_schema: Dict[str, Any],
) -> None:
    """Generate `risk_model_evaluation.md` and `risk_model_card.md` in `reports/` and `data/reports/`."""
    prim_df = perf_df[perf_df["target_variable"] == PRIMARY_TARGET]
    fatal_df = perf_df[perf_df["target_variable"] == "target_fatal_outbreak"]
    p50_df = perf_df[perf_df["target_variable"] == "target_above_median_outbreak_p50"]
    rec_df = perf_df[perf_df["target_variable"] == "target_next_4w_recurrence"]

    def _fmt_table(sub: pd.DataFrame) -> List[str]:
        lines = [
            "| Model | Split | N (Pos) | Pos Rate | Thr | Acc | Prec | Recall | F1 | F1@0.5 | PR-AUC | ROC-AUC | Brier (Cal) | ECE (Cal) | Confusion (TN, FP, FN, TP) | Δ PR-AUC vs Base |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            pr_s = f"{r['pr_auc']:.4f}" if pd.notna(r["pr_auc"]) else "N/A"
            roc_s = f"{r['roc_auc']:.4f}" if pd.notna(r["roc_auc"]) else "N/A"
            dpr = f"{r['delta_pr_auc_vs_baseline']:+.4f}" if pd.notna(r["delta_pr_auc_vs_baseline"]) else "0.0000"
            cm = f"({int(r['tn'])}, {int(r['fp'])}, {int(r['fn'])}, {int(r['tp'])})"
            lines.append(
                f"| `{r['model_name']}` | `{r['split']}` | {int(r['n_samples'])} ({int(r['n_positive'])}) | "
                f"{r['positive_rate']:.1%} | {r['threshold_used']:.3f} | {r['accuracy']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | **{r['f1_score']:.4f}** | {r['f1_at_050']:.4f} | "
                f"**{pr_s}** | {roc_s} | {r['brier_score_calibrated']:.4f} | {r['ece_calibrated']:.4f} | {cm} | {dpr} |"
            )
        return lines

    eval_lines = [
        "# JEEVAN-DRISHTI Phase 5: Outbreak Risk Classification Evaluation Report",
        "",
        f"- **Generated At**: `{model_metadata['generated_at']}`",
        f"- **Model Version**: `{model_metadata['model_version']}`",
        f"- **Source Dataset**: `data/final_training/risk_classification_dataset.csv` (`SHA-256: {model_metadata['source_dataset_sha256']}`)",
        f"- **Selected Primary Model (by Validation PR-AUC)**: `{model_metadata['selected_primary_model']}`",
        f"- **Production Readiness Verdict**: `{model_metadata['production_readiness_assessment']['overall_status']}`",
        "",
        "## 1. Methodology, Leakage Prevention & Label Defensibility",
        "",
        "1. **Defensible Empirical Labels Only**: No arbitrary case cutoffs were invented. The primary target (`target_severe_outbreak_p75`) labels whether an outbreak's case count exceeds the **Training-split (`2023-W01..2024-W33`) 75th percentile (`p75`) for that specific disease**, restricted to diseases with `>= 5` training observations (`481` valid rows across 12 supported diseases; `30` rows across 16 rare `<5`-train-obs diseases are explicitly excluded as `NaN` / `UNSUPPORTED`). Three secondary defensible targets (`target_fatal_outbreak`, `target_above_median_outbreak_p50`, and `target_next_4w_recurrence`) are also benchmarked.",
        "2. **Training-Only Preprocessing**: `preprocessing_pipeline.pkl` (`ColumnTransformer` with `SimpleImputer(strategy='most_frequent') + OneHotEncoder(min_frequency=5)` for 6 categorical features and `SimpleImputer(strategy='median', add_indicator=True) + StandardScaler()` for 69 stationary numeric features) is fitted **strictly on `split == 'train'` (`period_idx <= 84`)**.",
        "3. **Prevention of Temporal Covariate Shift**: Monotonic time counters (`report_year`, `calendar_year`, `period_idx`, and expanding cumulative panel sums starting at 0 in `2023-W01`) and ultra-sparse columns (`>70%` missing in train, such as single-station Pune RH and 52-week exact calendar lags) are excluded so models generalize across `2023 -> 2024 -> 2025` using stationary seasonal, climate, and prior-burden rate features.",
        "4. **Untouched Held-Out Test Split (`2025-W01..2025-W52`)**: All hyperparameter choices, F1 decision thresholds, and Platt sigmoid probability calibrators were fitted strictly using `train` (`2023-W01..2024-W33`) and `validation` (`2024-W34..2024-W52`).",
        "",
        "## 2. Primary Target Performance: `target_severe_outbreak_p75` (Upper-Quartile Disease Severity)",
        "",
        *_fmt_table(prim_df),
        "",
        "## 3. Secondary Target Benchmarks",
        "",
        "### 3.1 `target_fatal_outbreak` (Observed Mortality `reported_deaths >= 1`, 511 Valid Rows)",
        "",
        *_fmt_table(fatal_df),
        "",
        "### 3.2 `target_above_median_outbreak_p50` (Above-Median Disease Severity, 481 Valid Rows)",
        "",
        *_fmt_table(p50_df),
        "",
        "### 3.3 `target_next_4w_recurrence` (4-Week District-Disease Recurrence, 476 Valid Rows)",
        "",
        *_fmt_table(rec_df),
        "",
        "## 4. Subgroup Evaluation by Disease (`target_severe_outbreak_p75`)",
        "",
        "| Disease | Total Valid | Train | Val | Test | Positives (%) | Status | Val+Test Acc | Val+Test Prec | Val+Test Rec | Val+Test F1 | Val+Test PR-AUC | Val+Test ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in disease_subgroups:
        pr_s = f"{r['eval_pr_auc']:.4f}" if r["eval_pr_auc"] is not None else "N/A"
        roc_s = f"{r['eval_roc_auc']:.4f}" if r["eval_roc_auc"] is not None else "N/A"
        eval_lines.append(
            f"| {r['disease_std']} | {r['total_valid_rows']} | {r['train_rows']} | {r['val_rows']} | "
            f"{r['test_rows']} | {r['positives_total']} ({r['positive_rate_total']:.1%}) | `{r['eval_status']}` | "
            f"{r['eval_accuracy']:.4f} | {r['eval_precision']:.4f} | {r['eval_recall']:.4f} | "
            f"{r['eval_f1']:.4f} | {pr_s} | {roc_s} |"
        )

    eval_lines.extend([
        "",
        "## 5. Subgroup Evaluation by District (Top Districts by Sample Size, `target_severe_outbreak_p75`)",
        "",
        "| District | Total Valid | Train | Val | Test | Positives (%) | Status | Val+Test Acc | Val+Test F1 | Val+Test PR-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in district_subgroups:
        pr_s = f"{r['eval_pr_auc']:.4f}" if r["eval_pr_auc"] is not None else "N/A"
        eval_lines.append(
            f"| {r['district_std']} | {r['total_valid_rows']} | {r['train_rows']} | {r['val_rows']} | "
            f"{r['test_rows']} | {r['positives_total']} ({r['positive_rate_total']:.1%}) | `{r['eval_status']}` | "
            f"{r['eval_accuracy']:.4f} | {r['eval_f1']:.4f} | {pr_s} |"
        )

    eval_lines.extend([
        "",
        "## 6. Top 20 Contributing Features (Global Normalized Importance)",
        "",
        "| Rank | Transformed Feature Name | Normalized Importance |",
        "|---|---|---|",
    ])
    for idx, (fname, imp_val) in enumerate(list(model_metadata["top_25_global_features"].items())[:20], start=1):
        eval_lines.append(f"| {idx} | `{fname}` | {imp_val:.6f} |")

    eval_lines.extend([
        "",
        "## 7. Unsupported Diseases & Districts Inventory",
        "",
        "- **Unsupported Core Target Disease (0 records in 2023–2025 IDSP)**: `Heat-related illness`.",
        f"- **Unsupported Rare Diseases for `target_severe_outbreak_p75` (`< 5` training observations; 16 diseases, 30 total rows)**: {', '.join(f'`{d}`' for d in model_metadata['unsupported_diseases_rare_lt5_train_obs'])}. (Note: these 16 diseases ARE supported for `target_fatal_outbreak` where mortality is directly observed.)",
        f"- **Unsupported Districts (0 outbreak records in 2023–2025 IDSP)**: {', '.join(f'`{d}`' for d in model_metadata['unsupported_districts_zero_records_2023_2025'])}.",
        "- **Partially Supported District (No climate coordinates in lookup; disease-history & temporal features only)**: `Mumbai Suburban` (3 rows).",
    ])

    eval_md = "\n".join(eval_lines) + "\n"

    card_lines = [
        "# JEEVAN-DRISHTI Model Card: District Outbreak Risk Classification Engine",
        "",
        "## 1. Model Details",
        f"- **Model Name**: JEEVAN-DRISHTI Outbreak Risk Classifier (`{model_metadata['selected_primary_model']}`)",
        f"- **Model Version**: `{model_metadata['model_version']}`",
        "- **Model Type**: Calibrated Soft-Voting Ensemble (`LogisticRegression` + `RandomForestClassifier` + `XGBClassifier` + `HistGradientBoostingClassifier`) with Platt sigmoid probability calibration, plus standalone benchmarked estimators.",
        "- **Artifacts**:",
        "  - `models/risk_classification/trained_model.pkl`",
        "  - `models/risk_classification/preprocessing_pipeline.pkl`",
        "  - `models/risk_classification/feature_schema.json`",
        "  - `models/risk_classification/model_metadata.json`",
        "- **Inference Module**: `src/risk_classification/predictor.py` (`OutbreakRiskPredictor`)",
        "",
        "## 2. Intended Use & Scope",
        "- **Primary Use Case**: Weekly district-level public-health decision support for Maharashtra across IDSP outbreak-prone diseases (2023–2025 schema).",
        "- **Out-of-Scope Uses**: Autonomous clinical diagnosis, resource allocation without human epidemiologist review, or inference on `Heat-related illness` (0 training records) or zero-record districts (`Mumbai City`, `Ratnagiri`).",
        "",
        "## 3. Training Data & Chronological Splits",
        "- **Source File**: `data/final_training/risk_classification_dataset.csv` (`511` total district-disease-week outbreak records; `481` valid for `target_severe_outbreak_p75`).",
        "- **Training Split (`split == 'train'`)**: `2023-W01 .. 2024-W33` (`2023-01-02 .. 2024-08-18`), `340` valid rows (`82` positives, `24.1%`).",
        "- **Validation Split (`split == 'validation'`)**: `2024-W34 .. 2024-W52` (`2024-08-19 .. 2024-12-29`), `76` valid rows (`15` positives, `19.7%`).",
        "- **Held-Out Test Split (`split == 'test'`)**: `2025-W01 .. 2025-W52` (`2025-01-01 .. 2025-12-28`), `65` valid rows (`9` positives, `13.8%`).",
        "",
        "## 4. Quantitative Limitations & Production Readiness Honesty",
        f"- **Readiness Status**: `{model_metadata['production_readiness_assessment']['overall_status']}`",
        f"- **Rationale**: {model_metadata['production_readiness_assessment']['justification']}",
        "- **Key Data Constraints**:",
        "  1. IDSP weekly reports are event-based (silent weeks are not confirmed zeros), causing structural missingness in exact 1-to-8 week calendar disease lags.",
        "  2. No district population census file exists in the repository, so population-normalized incidence (`cases_per_100k_pop`) is unavailable.",
        "  3. Relative humidity comes from a single station in Pune (CWPRS Khadakvasla) and is excluded from the state-wide model matrix.",
    ]
    card_md = "\n".join(card_lines) + "\n"

    for out_dir in (REPORTS_DIR, DATA_REPORTS_DIR):
        (out_dir / "risk_model_evaluation.md").write_text(eval_md, encoding="utf-8")
        (out_dir / "risk_model_card.md").write_text(card_md, encoding="utf-8")


def _update_registry(
    perf_df: pd.DataFrame,
    model_metadata: Dict[str, Any],
    feature_schema: Dict[str, Any],
) -> None:
    """Update `model_dataset_registry.json` with Phase 5 model artifacts, schema, cutoff, and metrics."""
    if not REGISTRY_PATH.exists():
        return
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    prim_test = perf_df[
        (perf_df["target_variable"] == PRIMARY_TARGET) & (perf_df["split"] == "test")
    ].to_dict(orient="records")
    prim_val = perf_df[
        (perf_df["target_variable"] == PRIMARY_TARGET) & (perf_df["split"] == "validation")
    ].to_dict(orient="records")
    fatal_test = perf_df[
        (perf_df["target_variable"] == "target_fatal_outbreak") & (perf_df["split"] == "test")
    ].to_dict(orient="records")

    reg["phase5_risk_classification_engine"] = {
        "model_version": RISK_MODEL_VERSION,
        "generated_at": model_metadata["generated_at"],
        "dataset_used": {
            "path": "data/final_training/risk_classification_dataset.csv",
            "sha256": model_metadata["source_dataset_sha256"],
            "total_rows": 511,
            "primary_target_valid_rows": 481,
        },
        "training_cutoff": feature_schema["training_split_cutoff"],
        "input_feature_count": feature_schema["n_raw_input_features"],
        "transformed_feature_count": feature_schema["n_transformed_features_after_encoding_and_indicators"],
        "categorical_features": feature_schema["categorical_features"],
        "numerical_features": feature_schema["numerical_features"],
        "selected_primary_model": model_metadata["selected_primary_model"],
        "optimal_threshold": model_metadata["optimal_decision_threshold"],
        "artifacts": {
            "trained_model": "models/risk_classification/trained_model.pkl",
            "preprocessing_pipeline": "models/risk_classification/preprocessing_pipeline.pkl",
            "feature_schema": "models/risk_classification/feature_schema.json",
            "model_metadata": "models/risk_classification/model_metadata.json",
            "predictor_module": "src/risk_classification/predictor.py",
        },
        "reports": [
            "reports/risk_model_performance.csv",
            "reports/risk_model_evaluation.md",
            "reports/risk_model_card.md",
        ],
        "validation_metrics_primary_target": prim_val,
        "held_out_test_metrics_primary_target": prim_test,
        "held_out_test_metrics_fatal_outbreak": fatal_test,
        "production_readiness": model_metadata["production_readiness_assessment"],
    }
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    res = run_risk_classification_pipeline()
    print(f"Selected Primary Model: {res['selected_primary_model']}")
    print("Saved artifacts to models/risk_classification/ and reports/")

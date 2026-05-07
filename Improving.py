#!/usr/bin/env python3
"""
Credit score classification — production-style training pipeline.

Improves on typical baselines (e.g. single models without proper imbalance handling)
by: customer-level aggregation, SMOTENC + class weights, gradient boosting (XGBoost,
LightGBM) and Random Forest, and Optuna-tuned hyperparameters with stratified CV.

Run: python Improving.py [--fast] [--row-level]

macOS: XGBoost/LightGBM need OpenMP (`brew install libomp`). If those imports fail, RF-only still runs.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import xgboost as xgb
except Exception:  # ImportError or XGBoostError (e.g. missing libomp on macOS)
    xgb = None  # type: ignore

try:
    import lightgbm as lgb
except Exception:
    lgb = None  # type: ignore


# -----------------------------------------------------------------------------
# Data loading & cleaning
# -----------------------------------------------------------------------------


def _strip_numeric_noise(val: Any) -> Any:
    """Remove common OCR/noise patterns from numeric-like strings."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return val
    s = str(val).strip()
    s = re.sub(r"[^\d.\-+eE]", "", s.replace("_", ""))
    if s in {"", ".", "-", "+"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_credit_history_age(text: Any) -> float:
    """Convert '22 Years and 1 Months' into total months (float)."""
    if pd.isna(text) or text == "" or str(text).upper() == "NA":
        return np.nan
    s = str(text)
    y = re.search(r"(\d+)\s*Years?", s, re.I)
    m = re.search(r"(\d+)\s*Months?", s, re.I)
    years = float(y.group(1)) if y else 0.0
    months = float(m.group(1)) if m else 0.0
    return years * 12.0 + months


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce noisy Kaggle-style fields; keep target if present."""
    out = df.copy()
    if "Credit_History_Age" in out.columns:
        out["Credit_History_Age"] = out["Credit_History_Age"].map(parse_credit_history_age)

    numeric_like = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    for c in numeric_like:
        if c in out.columns:
            out[c] = out[c].map(_strip_numeric_noise)
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def aggregate_by_customer(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    One row per Customer_ID to reduce label noise and temporal leakage.

    The raw file repeats customers monthly; Credit_Score sometimes changes across
    months (data quality). After ordering by calendar month, we take the **last**
    month's label and categorical features (panel endpoint) and **median** numerics
    over the window — fast and stable; modal label is similar in practice.
    """
    if "Customer_ID" not in df.columns:
        raise ValueError("Expected Customer_ID for aggregation.")

    target_col = "Credit_Score"
    work = df.copy()
    if "Month" in work.columns:
        month_order = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12,
        }
        work["_mord"] = work["Month"].map(month_order).fillna(0)
        work = work.sort_values(["Customer_ID", "_mord"])

    drop_cols = {"ID", "Customer_ID", "Name", "SSN", target_col}
    if "_mord" in work.columns:
        drop_cols.add("_mord")
    feature_cols = [c for c in work.columns if c not in drop_cols]

    num_cols = work[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    g = work.groupby("Customer_ID", sort=False)
    num_agg = g[num_cols].median(numeric_only=True) if num_cols else pd.DataFrame()
    cat_agg = g[cat_cols].last() if cat_cols else pd.DataFrame()
    y_series = g[target_col].last()

    X = pd.concat([num_agg, cat_agg], axis=1)
    y = y_series.astype(str).values
    return X, y


def row_level_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Kaggle-style: each month is one sample (faster baseline, more leakage risk)."""
    target_col = "Credit_Score"
    drop_cols = [c for c in ("ID", "Customer_ID", "Name", "SSN") if c in df.columns]
    X = df.drop(columns=drop_cols + [target_col], errors="ignore")
    y = df[target_col].values
    return X, y


# -----------------------------------------------------------------------------
# Preprocessing & imbalance (SMOTENC)
# -----------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Numeric: median impute + scale (helps k-NN in SMOTENC). Categorical: ordinal."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric_cols, categorical_cols


def smotenc_categorical_indices(n_num: int, n_cat: int) -> List[int]:
    """In ColumnTransformer output, categoricals follow numerics."""
    return list(range(n_num, n_num + n_cat))


def class_sample_weights(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Balanced sample weights for tree trainers that support sample_weight."""
    from sklearn.utils.class_weight import compute_sample_weight

    return compute_sample_weight(class_weight="balanced", y=y)


# -----------------------------------------------------------------------------
# Models & Optuna objectives
# -----------------------------------------------------------------------------


def make_plain_pipeline(preprocessor: ColumnTransformer, clf: Any) -> Pipeline:
    """Preprocess + classifier only — used for Optuna (SMOTENC each trial is prohibitively slow)."""
    return Pipeline(steps=[("pre", clone(preprocessor)), ("clf", clf)])


def make_resampled_pipeline(
    preprocessor: ColumnTransformer,
    n_num: int,
    n_cat: int,
    clf: Any,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Production path: preprocessing, SMOTENC on mixed numeric/ordinal features, then classifier.
    SMOTENC is applied in final CV/test evaluation, not inside every Optuna trial.
    """
    cat_idx = smotenc_categorical_indices(n_num, n_cat)
    return ImbPipeline(
        steps=[
            ("pre", clone(preprocessor)),
            (
                "smote",
                SMOTENC(
                    categorical_features=cat_idx,
                    random_state=random_state,
                    k_neighbors=5,
                ),
            ),
            ("clf", clf),
        ]
    )


def _fold_metric(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="macro"))


def cv_model_score(
    pipeline_template: Pipeline,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    le: LabelEncoder,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weight: bool = True,
    metric: str = "f1_macro",
) -> float:
    """
    Stratified K-fold mean score on a sklearn Pipeline (pre + clf).
    metric: 'f1_macro' (default, good for imbalance), 'accuracy', or 'balanced_accuracy'.
    If use_sample_weight, pass balanced weights to the classifier (e.g. XGBoost).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    for train_idx, val_idx in cv.split(X, y_enc):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
        model = clone(pipeline_template)
        if use_sample_weight:
            sw = class_sample_weights(y_tr, le.classes_)
            model.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        fold_scores.append(_fold_metric(y_val, pred, metric))

    return float(np.mean(fold_scores))


def cv_macro_f1(
    pipeline_template: Pipeline,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    le: LabelEncoder,
    n_splits: int = 5,
    random_state: int = 42,
    use_sample_weight: bool = True,
) -> float:
    """Backward-compatible wrapper: macro-F1."""
    return cv_model_score(
        pipeline_template,
        X,
        y_enc,
        le,
        n_splits=n_splits,
        random_state=random_state,
        use_sample_weight=use_sample_weight,
        metric="f1_macro",
    )


def optuna_tune_xgb(
    X_train: pd.DataFrame,
    y_enc: np.ndarray,
    le: LabelEncoder,
    preprocessor: ColumnTransformer,
    n_num: int,
    n_cat: int,
    n_classes: int,
    n_trials: int,
    cv_splits: int,
    random_state: int,
    fast: bool = False,
    optimize_metric: str = "f1_macro",
) -> Dict[str, Any]:
    if xgb is None:
        raise RuntimeError("xgboost is not installed.")

    def objective(trial: optuna.Trial) -> float:
        ne_hi = 200 if fast else 600
        ne_lo = 80 if fast else 200
        params = {
            "n_estimators": trial.suggest_int("n_estimators", ne_lo, ne_hi),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
            "objective": "multi:softprob",
            "num_class": n_classes,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": 1,
            "eval_metric": "mlogloss",
        }
        clf = xgb.XGBClassifier(**params)
        pipe = make_plain_pipeline(preprocessor, clf)
        return cv_model_score(
            pipe,
            X_train,
            y_enc,
            le,
            n_splits=cv_splits,
            random_state=random_state,
            use_sample_weight=True,
            metric=optimize_metric,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_tune_rf(
    X_train: pd.DataFrame,
    y_enc: np.ndarray,
    le: LabelEncoder,
    preprocessor: ColumnTransformer,
    n_num: int,
    n_cat: int,
    n_trials: int,
    cv_splits: int,
    random_state: int,
    fast: bool = False,
) -> Dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        max_feat = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.4, 0.7])
        ne_hi = 200 if fast else 600
        ne_lo = 80 if fast else 200
        md_hi = 25 if fast else 40
        params = {
            "n_estimators": trial.suggest_int("n_estimators", ne_lo, ne_hi),
            "max_depth": trial.suggest_int("max_depth", 8, md_hi),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": max_feat,
            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": 1,
        }
        clf = RandomForestClassifier(**params)
        pipe = make_plain_pipeline(preprocessor, clf)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores: List[float] = []
        for train_idx, val_idx in cv.split(X_train, y_enc):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
            model = clone(pipe)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(f1_score(y_val, pred, average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_tune_lgbm(
    X_train: pd.DataFrame,
    y_enc: np.ndarray,
    le: LabelEncoder,
    preprocessor: ColumnTransformer,
    n_num: int,
    n_cat: int,
    n_classes: int,
    n_trials: int,
    cv_splits: int,
    random_state: int,
    fast: bool = False,
) -> Dict[str, Any]:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed.")

    def objective(trial: optuna.Trial) -> float:
        ne_hi = 250 if fast else 800
        ne_lo = 80 if fast else 200
        nl_hi = 127 if fast else 256
        params = {
            "n_estimators": trial.suggest_int("n_estimators", ne_lo, ne_hi),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, nl_hi),
            "max_depth": trial.suggest_int("max_depth", 4, 10 if fast else 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        pipe = make_plain_pipeline(preprocessor, clf)
        return cv_macro_f1(
            pipe,
            X_train,
            y_enc,
            le,
            n_splits=cv_splits,
            random_state=random_state,
            use_sample_weight=False,
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# -----------------------------------------------------------------------------
# Evaluation & feature importance
# -----------------------------------------------------------------------------


def evaluate_multiclass(
    name: str,
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    le: LabelEncoder,
    use_sample_weight: bool,
    cv_splits: int = 5,
) -> None:
    """Accuracy, macro-F1, macro-OVR ROC-AUC; CV + hold-out test."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_pred = np.empty_like(y_train_enc)

    for train_idx, val_idx in cv.split(X_train, y_train_enc):
        model = clone(pipeline)
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train_enc[train_idx]
        if use_sample_weight:
            sw = class_sample_weights(y_tr, le.classes_)
            model.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        oof_pred[val_idx] = model.predict(X_val)

    cv_pred = oof_pred

    print(f"\n===== {name} — {cv_splits}-fold CV (OOF) =====")
    print(classification_report(y_train_enc, cv_pred, target_names=le.classes_))
    cv_acc = accuracy_score(y_train_enc, cv_pred)
    cv_f1 = f1_score(y_train_enc, cv_pred, average="macro")
    cv_bacc = balanced_accuracy_score(y_train_enc, cv_pred)
    print(f"CV Accuracy: {cv_acc:.4f} | CV Macro-F1: {cv_f1:.4f} | CV Balanced acc: {cv_bacc:.4f}")

    final_model = clone(pipeline)
    if use_sample_weight:
        sw_full = class_sample_weights(y_train_enc, le.classes_)
        final_model.fit(X_train, y_train_enc, clf__sample_weight=sw_full)
    else:
        final_model.fit(X_train, y_train_enc)

    y_test_pred = final_model.predict(X_test)
    print(f"\n----- {name} — hold-out test -----")
    print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_))
    acc = accuracy_score(y_test_enc, y_test_pred)
    f1 = f1_score(y_test_enc, y_test_pred, average="macro")
    b_acc = balanced_accuracy_score(y_test_enc, y_test_pred)
    print(f"Test Accuracy: {acc:.4f} | Test Macro-F1: {f1:.4f}")
    print(
        f"Test Balanced accuracy: {b_acc:.4f}  "
        "(mesatarja e recall për klasë — më e qëndrueshme kur klasat janë të pabalancuara)"
    )

    if hasattr(final_model.named_steps["clf"], "predict_proba"):
        proba = final_model.predict_proba(X_test)
        try:
            auc = roc_auc_score(
                y_test_enc,
                proba,
                multi_class="ovr",
                average="macro",
            )
            print(f"Test Macro ROC-AUC (OVR): {auc:.4f}")
        except ValueError as e:
            print(f"ROC-AUC skipped: {e}")


def print_xgb_feature_importance(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train_enc: np.ndarray,
    le: LabelEncoder,
    top_k: int = 15,
) -> None:
    """Fit XGBoost pipeline and print gain-based importances with feature names."""
    model = clone(pipeline)
    sw = class_sample_weights(y_train_enc, le.classes_)
    model.fit(X_train, y_train_enc, clf__sample_weight=sw)
    pre = model.named_steps["pre"]
    names = pre.get_feature_names_out()
    clf = model.named_steps["clf"]
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]
    print(f"\n===== Top {top_k} XGBoost feature importances (gain-based) =====")
    for i in order:
        print(f"  {names[i]}: {imp[i]:.5f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Credit score ML pipeline (Improving.py)")
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "train.csv",
        help="Path to train.csv",
    )
    p.add_argument(
        "--row-level",
        action="store_true",
        help="Use each month as one row (Kaggle-style) instead of customer aggregation",
    )
    p.add_argument("--fast", action="store_true", help="Fewer Optuna trials for quick runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--xgb-only",
        action="store_true",
        help="Train/tune vetëm XGBoost (më i shpejtë kur duhet vetëm ai).",
    )
    p.add_argument(
        "--optimize-metric",
        choices=("f1_macro", "accuracy", "balanced_accuracy"),
        default="f1_macro",
        help=(
            "Metrika që Optuna maksimizon për XGBoost. "
            "'accuracy' = përqindja e saktë të parashikimeve; "
            "'balanced_accuracy' më e barabartë për çdo klasë kur data është e pabalancuar."
        ),
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    csv_path = args.data
    if not csv_path.is_file():
        logging.error("Data file not found: %s", csv_path)
        return 1

    # Trial counts: full study vs. quick smoke run (narrower spaces + fewer trees in --fast)
    if args.fast:
        n_xgb, n_rf, n_lgb, cv_splits = 4, 4, 4, 2
        eval_cv = 3
    else:
        n_xgb, n_rf, n_lgb, cv_splits = 25, 20, 20, 5
        eval_cv = 5

    df = clean_dataframe(load_raw_data(csv_path))
    if args.row_level:
        X, y = row_level_xy(df)
        logging.info("Mode: row-level samples (n=%d)", len(X))
    else:
        X, y = aggregate_by_customer(df)
        logging.info("Mode: one row per Customer_ID (n=%d)", len(X))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    counts = pd.Series(y).value_counts(normalize=True)
    logging.info("Class distribution:\n%s", counts.to_string())
    logging.info(
        "Imbalance: max/min class share = %.2f (SMOTENC + balanced weights mitigate minority loss)",
        counts.max() / counts.min(),
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    n_num, n_cat = len(num_cols), len(cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=args.seed,
    )

    logging.info("Optuna trials: XGB=%d, RF=%d, LGBM=%d | inner CV folds=%d", n_xgb, n_rf, n_lgb, cv_splits)
    logging.info(
        "XGBoost Optuna optimizon: %s (për accuracy 'të saktë' sipas një metrike, përdor --optimize-metric)",
        args.optimize_metric,
    )

    # --- XGBoost ---
    if xgb is not None:
        logging.info("Tuning XGBoost...")
        best_xgb = optuna_tune_xgb(
            X_train,
            y_train,
            le,
            preprocessor,
            n_num,
            n_cat,
            n_classes,
            n_trials=n_xgb,
            cv_splits=cv_splits,
            random_state=args.seed,
            fast=args.fast,
            optimize_metric=args.optimize_metric,
        )
        xgb_clf = xgb.XGBClassifier(
            **best_xgb,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb_pipe = make_resampled_pipeline(
            preprocessor, n_num, n_cat, xgb_clf, random_state=args.seed
        )
        logging.info("Best XGBoost params: %s", best_xgb)
        evaluate_multiclass(
            "XGBoost + SMOTENC",
            xgb_pipe,
            X_train,
            X_test,
            y_train,
            y_test,
            le,
            use_sample_weight=True,
            cv_splits=eval_cv,
        )
        print_xgb_feature_importance(xgb_pipe, X_train, y_train, le, top_k=15)
    else:
        logging.warning("Skipping XGBoost (import/load failed; install OpenMP on macOS: brew install libomp).")

    if args.xgb_only:
        if xgb is None:
            logging.error(
                "--xgb-only kërkon XGBoost të ngarkuar; në macOS: brew install libomp, pastaj ri-instaloni xgboost."
            )
            return 1
        logging.info("Mbaroi (--xgb-only): RF dhe LightGBM janë anashkaluar.")
        return 0

    # --- Random Forest ---
    logging.info("Tuning Random Forest...")
    best_rf = optuna_tune_rf(
        X_train,
        y_train,
        le,
        preprocessor,
        n_num,
        n_cat,
        n_trials=n_rf,
        cv_splits=cv_splits,
        random_state=args.seed,
        fast=args.fast,
    )
    rf_clf = RandomForestClassifier(
        **best_rf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf_pipe = make_resampled_pipeline(preprocessor, n_num, n_cat, rf_clf, random_state=args.seed)
    logging.info("Best RandomForest params: %s", best_rf)
    evaluate_multiclass(
        "Random Forest + SMOTENC",
        rf_pipe,
        X_train,
        X_test,
        y_train,
        y_test,
        le,
        use_sample_weight=False,
        cv_splits=eval_cv,
    )

    # --- LightGBM ---
    if lgb is not None:
        logging.info("Tuning LightGBM...")
        best_lgb = optuna_tune_lgbm(
            X_train,
            y_train,
            le,
            preprocessor,
            n_num,
            n_cat,
            n_classes,
            n_trials=n_lgb,
            cv_splits=cv_splits,
            random_state=args.seed,
            fast=args.fast,
        )
        lgb_clf = lgb.LGBMClassifier(
            **best_lgb,
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_pipe = make_resampled_pipeline(
            preprocessor, n_num, n_cat, lgb_clf, random_state=args.seed
        )
        logging.info("Best LightGBM params: %s", best_lgb)
        evaluate_multiclass(
            "LightGBM + SMOTENC",
            lgb_pipe,
            X_train,
            X_test,
            y_train,
            y_test,
            le,
            use_sample_weight=False,
            cv_splits=eval_cv,
        )
    else:
        logging.warning("Skipping LightGBM (import/load failed; install OpenMP on macOS: brew install libomp).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

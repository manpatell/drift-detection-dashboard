"""
ML model training and evaluation utilities for CreditGuard ML Monitoring Platform.

Trains a Gradient Boosted Classifier on reference data and evaluates it
on production data to surface performance degradation alongside data drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


FEATURE_COLS: List[str] = [
    "annual_income", "credit_score", "loan_amount", "age",
    "employment_years", "debt_to_income", "num_credit_lines",
    "monthly_payment", "payment_history", "loan_purpose",
    "home_ownership", "loan_grade",
]

CAT_COLS: List[str] = ["payment_history", "loan_purpose", "home_ownership", "loan_grade"]


def _encode(df: pd.DataFrame, le_map: dict, fit: bool = False) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_map[col] = le
        else:
            le = le_map[col]
            df[col] = df[col].apply(lambda x: x if x in set(le.classes_) else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    return df, le_map


def _make_target(df: pd.DataFrame, rng: np.random.Generator, noise_std: float = 0.3) -> np.ndarray:
    """Derive a synthetic loan-default label from risk factors."""
    risk = (
        -0.30 * (df["credit_score"] - 700) / 60
        - 0.20 * np.log(df["annual_income"].clip(1) / 60_000 + 1)
        + 0.40 * df["debt_to_income"]
        + 0.10 * (df["loan_amount"] / df["annual_income"].clip(1))
        - 0.15 * df["employment_years"] / 10
        + rng.normal(0, noise_std, len(df))
    )
    return (1 / (1 + np.exp(-risk)) > 0.35).astype(int)


@st.cache_resource
def train_model(ref_df: pd.DataFrame, seed: int = 42):
    """
    Train a GBM model on reference data.

    Returns: (model, le_map, baseline_metrics, feature_cols)
    """
    rng  = np.random.default_rng(seed)
    df, le_map = _encode(ref_df.copy(), {}, fit=True)
    df["target"] = _make_target(df, rng, noise_std=0.3)

    X = df[FEATURE_COLS]
    y = df["target"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.85, random_state=seed,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    fpr, tpr, roc_thresh = roc_curve(y_te, y_prob)
    prec, rec, pr_thresh  = precision_recall_curve(y_te, y_prob)

    baseline: Dict = {
        "accuracy":         round(accuracy_score(y_te, y_pred),  4),
        "f1_score":         round(f1_score(y_te, y_pred),        4),
        "auc_roc":          round(roc_auc_score(y_te, y_prob),   4),
        "precision":        round(precision_score(y_te, y_pred), 4),
        "recall":           round(recall_score(y_te, y_pred),    4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "roc_curve":        {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve":         {"precision": prec.tolist(), "recall": rec.tolist()},
        "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
        "n_train": len(X_tr),
        "n_test":  len(X_te),
        "default_rate": round(float(y.mean()), 4),
    }

    return model, le_map, baseline, FEATURE_COLS


def evaluate_on_production(
    model,
    prod_df: pd.DataFrame,
    le_map: dict,
    feature_cols: List[str],
    seed: int = 99,
) -> Dict:
    """Evaluate model on production data with slightly higher noise (concept drift)."""
    rng  = np.random.default_rng(seed)
    df, _ = _encode(prod_df.copy(), le_map, fit=False)
    y_true = _make_target(df, rng, noise_std=0.5)  # more noise → worse perf

    X_prod = df[feature_cols]
    y_pred = model.predict(X_prod)
    y_prob = model.predict_proba(X_prod)[:, 1]

    fpr, tpr, _  = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    return {
        "accuracy":         round(accuracy_score(y_true, y_pred),  4),
        "f1_score":         round(f1_score(y_true, y_pred),        4),
        "auc_roc":          round(roc_auc_score(y_true, y_prob),   4),
        "precision":        round(precision_score(y_true, y_pred), 4),
        "recall":           round(recall_score(y_true, y_pred),    4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve":        {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve":         {"precision": prec.tolist(), "recall": rec.tolist()},
        "y_prob":           y_prob,
        "y_pred":           y_pred,
        "y_true":           y_true,
        "default_rate":     round(float(y_true.mean()), 4),
    }

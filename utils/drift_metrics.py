"""
Statistical drift detection metrics for CreditGuard ML Monitoring Platform.

Implements:
  - Population Stability Index (PSI)
  - Kolmogorov-Smirnov Test (KS)
  - Jensen-Shannon Divergence (JS)
  - Wasserstein Distance (Earth Mover's Distance)
  - Chi-Square Test (categorical features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple

from config import THRESHOLDS


# ─── Numeric Metrics ──────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index.
    PSI < 0.10 → stable | 0.10–0.20 → moderate | > 0.20 → significant drift
    """
    eps = 1e-6
    mn  = min(expected.min(), actual.min())
    mx  = max(expected.max(), actual.max())
    if mn == mx:
        return 0.0
    bins    = np.linspace(mn, mx, buckets + 1)
    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected) + eps
    act_pct = np.histogram(actual,   bins=bins)[0] / len(actual)   + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_js_divergence(expected: np.ndarray, actual: np.ndarray, buckets: int = 50) -> float:
    """Jensen-Shannon Divergence (symmetric, bounded [0, 1])."""
    mn   = min(expected.min(), actual.min())
    mx   = max(expected.max(), actual.max())
    if mn == mx:
        return 0.0
    bins = np.linspace(mn, mx, buckets + 1)
    p    = np.histogram(expected, bins=bins)[0].astype(float) + 1e-10
    q    = np.histogram(actual,   bins=bins)[0].astype(float) + 1e-10
    p   /= p.sum()
    q   /= q.sum()
    return float(jensenshannon(p, q))


def compute_ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    stat, p = stats.ks_2samp(expected, actual)
    return float(stat), float(p)


def compute_wasserstein(expected: np.ndarray, actual: np.ndarray) -> float:
    """Wasserstein-1 distance normalised by reference std."""
    raw = wasserstein_distance(expected, actual)
    std = expected.std()
    return float(raw / std) if std > 0 else float(raw)


def _drift_status(psi_val: float, ks_p: float) -> str:
    if psi_val >= THRESHOLDS["psi"]["warning"] or ks_p < THRESHOLDS["ks_p_value"]:
        return "critical"
    if psi_val >= THRESHOLDS["psi"]["stable"]:
        return "warning"
    return "stable"


def compute_numeric_drift(ref: np.ndarray, prod: np.ndarray) -> Dict:
    """Full drift report for a single numeric feature."""
    ks_stat, ks_p  = compute_ks_test(ref, prod)
    psi_val        = compute_psi(ref, prod)
    js_val         = compute_js_divergence(ref, prod)
    wdist          = compute_wasserstein(ref, prod)
    mean_shift_pct = (prod.mean() - ref.mean()) / (abs(ref.mean()) + 1e-9) * 100

    return {
        "ks_statistic":   round(ks_stat, 4),
        "ks_p_value":     round(ks_p,    6),
        "psi":            round(psi_val, 4),
        "js_divergence":  round(js_val,  4),
        "wasserstein":    round(wdist,   4),
        "status":         _drift_status(psi_val, ks_p),
        "ref_mean":       round(float(ref.mean()),  2),
        "prod_mean":      round(float(prod.mean()), 2),
        "ref_std":        round(float(ref.std()),   2),
        "prod_std":       round(float(prod.std()),  2),
        "ref_median":     round(float(np.median(ref)),  2),
        "prod_median":    round(float(np.median(prod)), 2),
        "mean_shift_pct": round(float(mean_shift_pct), 1),
    }


# ─── Categorical Metrics ──────────────────────────────────────────────────────

def compute_categorical_drift(ref: pd.Series, prod: pd.Series) -> Dict:
    """Drift report for a categorical feature (Chi-Square + PSI)."""
    all_cats   = sorted(set(ref.unique()) | set(prod.unique()))
    ref_counts = ref.value_counts()
    prod_counts = prod.value_counts()

    ref_freq  = np.array([ref_counts.get(c, 0)  for c in all_cats], dtype=float)
    prod_freq = np.array([prod_counts.get(c, 0) for c in all_cats], dtype=float)

    ref_pct  = ref_freq  / ref_freq.sum()
    prod_pct = prod_freq / prod_freq.sum()

    expected_freq = ref_pct * prod_freq.sum()
    try:
        chi2, p_val = stats.chisquare(prod_freq + 1e-9, f_exp=expected_freq + 1e-9)
    except Exception:
        chi2, p_val = 0.0, 1.0

    eps     = 1e-6
    psi_val = float(np.sum((prod_pct + eps - ref_pct - eps) * np.log((prod_pct + eps) / (ref_pct + eps))))

    if psi_val >= THRESHOLDS["psi"]["warning"] or p_val < 0.05:
        status = "critical"
    elif psi_val >= THRESHOLDS["psi"]["stable"]:
        status = "warning"
    else:
        status = "stable"

    distribution = {
        "reference":  {c: round(ref_counts.get(c, 0)  / len(ref)  * 100, 1) for c in all_cats},
        "production": {c: round(prod_counts.get(c, 0) / len(prod) * 100, 1) for c in all_cats},
    }

    return {
        "chi2_statistic": round(float(chi2),  4),
        "p_value":        round(float(p_val), 6),
        "psi":            round(psi_val, 4),
        "status":         status,
        "distribution":   distribution,
    }


# ─── Full Report ─────────────────────────────────────────────────────────────

def compute_full_drift_report(
    ref_df:  pd.DataFrame,
    prod_df: pd.DataFrame,
    numeric_features:     list,
    categorical_features: list,
) -> Dict:
    """Compute drift for every monitored feature and return a structured report."""
    report: Dict = {"numeric": {}, "categorical": {}, "summary": {}}

    for feat in numeric_features:
        if feat in ref_df.columns and feat in prod_df.columns:
            report["numeric"][feat] = compute_numeric_drift(
                ref_df[feat].dropna().values,
                prod_df[feat].dropna().values,
            )

    for feat in categorical_features:
        if feat in ref_df.columns and feat in prod_df.columns:
            report["categorical"][feat] = compute_categorical_drift(
                ref_df[feat].dropna(),
                prod_df[feat].dropna(),
            )

    all_statuses = (
        [v["status"] for v in report["numeric"].values()] +
        [v["status"] for v in report["categorical"].values()]
    )
    all_psi = (
        [v["psi"] for v in report["numeric"].values()] +
        [v["psi"] for v in report["categorical"].values()]
    )

    report["summary"] = {
        "total_features": len(all_statuses),
        "critical":       all_statuses.count("critical"),
        "warning":        all_statuses.count("warning"),
        "stable":         all_statuses.count("stable"),
        "avg_psi":        round(float(np.mean(all_psi)), 4) if all_psi else 0.0,
        "max_psi":        round(float(np.max(all_psi)),  4) if all_psi else 0.0,
    }

    return report

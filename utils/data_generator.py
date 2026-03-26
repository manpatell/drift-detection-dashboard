"""
Synthetic data generator for CreditGuard ML Monitoring Platform.

Simulates a realistic credit-risk lending dataset with configurable
distributional drift to mimic real-world model degradation scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

from config import REFERENCE_DAYS, PRODUCTION_DAYS, TIMESERIES_DAYS


# ─── Reference (Training) Data ────────────────────────────────────────────────

def generate_reference_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a stable reference dataset representing the training-time
    population of loan applicants.
    """
    rng = np.random.default_rng(seed)

    age            = rng.normal(42, 12, n).clip(18, 75).astype(int)
    annual_income  = rng.lognormal(11.0, 0.5, n).clip(20_000, 500_000)
    credit_score   = rng.normal(700, 60, n).clip(300, 850).astype(int)
    loan_amount    = rng.lognormal(10.5, 0.6, n).clip(1_000, 100_000)
    emp_years      = rng.exponential(7, n).clip(0, 40).astype(int)
    dti            = rng.beta(2, 5, n) * 0.8
    num_lines      = rng.poisson(5, n).clip(0, 20)
    monthly_pmt    = (loan_amount * 0.020 + rng.normal(0, 50, n)).clip(50, 5_000)

    payment_hist   = rng.choice(
        ["Excellent", "Good", "Fair", "Poor"], n, p=[0.30, 0.40, 0.20, 0.10]
    )
    loan_purpose   = rng.choice(
        ["home", "auto", "education", "personal", "medical"], n,
        p=[0.20, 0.25, 0.15, 0.30, 0.10]
    )
    home_ownership = rng.choice(
        ["Own", "Mortgage", "Rent"], n, p=[0.25, 0.40, 0.35]
    )
    loan_grade     = rng.choice(
        ["A", "B", "C", "D", "E"], n, p=[0.20, 0.30, 0.25, 0.15, 0.10]
    )

    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(days=int(rng.integers(0, REFERENCE_DAYS))) for _ in range(n)]

    return pd.DataFrame({
        "age":              age,
        "annual_income":    annual_income,
        "credit_score":     credit_score,
        "loan_amount":      loan_amount,
        "employment_years": emp_years,
        "debt_to_income":   dti,
        "num_credit_lines": num_lines,
        "monthly_payment":  monthly_pmt,
        "payment_history":  payment_hist,
        "loan_purpose":     loan_purpose,
        "home_ownership":   home_ownership,
        "loan_grade":       loan_grade,
        "timestamp":        timestamps,
    })


# ─── Production Data (with drift) ────────────────────────────────────────────

def generate_production_data(
    n: int = 2000,
    drift_magnitude: float = 1.5,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Generate production data with configurable distributional drift.

    drift_magnitude=0 → identical to reference distribution
    drift_magnitude=3 → severe drift (economic downturn scenario)
    """
    rng = np.random.default_rng(seed)
    dm  = drift_magnitude  # shorthand

    # Income drops, credit quality degrades, riskier applicant pool
    annual_income = (
        rng.lognormal(10.90, 0.60, n) - dm * 12_000
    ).clip(15_000, 500_000)

    credit_score  = (
        rng.normal(700 - dm * 35, 70, n)
    ).clip(300, 850).astype(int)

    dti           = (
        rng.beta(3, 5, n) * 0.9 + dm * 0.07
    ).clip(0, 1.0)

    age            = rng.normal(38, 14, n).clip(18, 75).astype(int)
    loan_amount    = rng.lognormal(10.7, 0.7, n).clip(1_000, 100_000)
    emp_years      = rng.exponential(5, n).clip(0, 40).astype(int)
    num_lines      = rng.poisson(4, n).clip(0, 20)
    monthly_pmt    = (loan_amount * 0.022 + rng.normal(0, 60, n)).clip(50, 5_000)

    # Categorical distributions shift toward riskier categories
    poor_p   = min(0.10 + dm * 0.06, 0.35)
    fair_p   = min(0.20 + dm * 0.04, 0.40)
    good_p   = max(0.40 - dm * 0.05, 0.15)
    excel_p  = max(0.30 - dm * 0.05, 0.10)
    total    = poor_p + fair_p + good_p + excel_p
    payment_hist = rng.choice(
        ["Excellent", "Good", "Fair", "Poor"], n,
        p=[excel_p / total, good_p / total, fair_p / total, poor_p / total],
    )

    loan_purpose   = rng.choice(
        ["home", "auto", "education", "personal", "medical"], n,
        p=[0.15, 0.20, 0.10, 0.40, 0.15],
    )
    home_ownership = rng.choice(
        ["Own", "Mortgage", "Rent"], n, p=[0.20, 0.35, 0.45]
    )

    d_p  = min(0.10 + dm * 0.05, 0.30)
    e_p  = min(0.10 + dm * 0.04, 0.25)
    c_p  = 0.30
    b_p  = max(0.30 - dm * 0.04, 0.10)
    a_p  = max(0.20 - dm * 0.05, 0.05)
    gsum = a_p + b_p + c_p + d_p + e_p
    loan_grade = rng.choice(
        ["A", "B", "C", "D", "E"], n,
        p=[a_p / gsum, b_p / gsum, c_p / gsum, d_p / gsum, e_p / gsum],
    )

    start = datetime(2025, 3, 1)
    timestamps = [start + timedelta(days=int(rng.integers(0, PRODUCTION_DAYS))) for _ in range(n)]

    return pd.DataFrame({
        "age":              age,
        "annual_income":    annual_income,
        "credit_score":     credit_score,
        "loan_amount":      loan_amount,
        "employment_years": emp_years,
        "debt_to_income":   dti,
        "num_credit_lines": num_lines,
        "monthly_payment":  monthly_pmt,
        "payment_history":  payment_hist,
        "loan_purpose":     loan_purpose,
        "home_ownership":   home_ownership,
        "loan_grade":       loan_grade,
        "timestamp":        timestamps,
    })


# ─── Time-Series Drift Simulation ─────────────────────────────────────────────

def generate_time_series_drift(days: int = 60, seed: int = 123) -> pd.DataFrame:
    """
    Simulate daily drift and model performance metrics over time,
    showing a realistic gradual degradation pattern.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1, days)

    start = datetime(2025, 2, 1)
    dates = [start + timedelta(days=i) for i in range(days)]

    avg_psi             = (0.04 + 0.28 * t ** 1.8 + rng.normal(0, 0.008, days)).clip(0)
    ks_income           = (0.03 + 0.32 * t ** 1.6 + rng.normal(0, 0.009, days)).clip(0)
    ks_credit           = (0.02 + 0.22 * t ** 1.3 + rng.normal(0, 0.008, days)).clip(0)
    accuracy            = (0.88 - 0.09 * t ** 1.5 + rng.normal(0, 0.004, days)).clip(0.5, 1.0)
    f1_score            = (0.86 - 0.11 * t ** 1.5 + rng.normal(0, 0.004, days)).clip(0.5, 1.0)
    auc_roc             = (0.91 - 0.07 * t ** 1.5 + rng.normal(0, 0.003, days)).clip(0.5, 1.0)
    precision           = (0.87 - 0.08 * t ** 1.4 + rng.normal(0, 0.004, days)).clip(0.5, 1.0)
    recall              = (0.85 - 0.10 * t ** 1.5 + rng.normal(0, 0.005, days)).clip(0.5, 1.0)
    prediction_volume   = (800 + 400 * t + rng.normal(0, 30, days)).clip(100).astype(int)
    alert_count         = (rng.poisson(3 * t ** 2 + 0.2, days)).astype(int)

    return pd.DataFrame({
        "date":               dates,
        "avg_psi":            avg_psi,
        "ks_annual_income":   ks_income,
        "ks_credit_score":    ks_credit,
        "accuracy":           accuracy,
        "f1_score":           f1_score,
        "auc_roc":            auc_roc,
        "precision":          precision,
        "recall":             recall,
        "prediction_volume":  prediction_volume,
        "alert_count":        alert_count,
    })

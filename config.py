"""
Central configuration for CreditGuard ML Monitoring Platform.
"""

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "reference":  "#3B82F6",
    "production": "#EF4444",
    "stable":     "#10B981",
    "warning":    "#F59E0B",
    "critical":   "#EF4444",
    "neutral":    "#6B7280",
    "primary":    "#6366F1",
    "accent":     "#8B5CF6",
    "surface":    "#1E293B",
    "bg":         "#0F172A",
}

# ─── Drift Thresholds ─────────────────────────────────────────────────────────
THRESHOLDS = {
    "psi":          {"stable": 0.1, "warning": 0.2},
    "ks_p_value":   0.05,
    "js_divergence":{"stable": 0.05, "warning": 0.1},
    "wasserstein":  {"stable": 0.1,  "warning": 0.3},
    "accuracy_drop": 0.03,
    "f1_drop":       0.05,
}

# ─── Feature Definitions ──────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "annual_income",
    "credit_score",
    "loan_amount",
    "age",
    "employment_years",
    "debt_to_income",
    "num_credit_lines",
    "monthly_payment",
]

CATEGORICAL_FEATURES = [
    "payment_history",
    "loan_purpose",
    "home_ownership",
    "loan_grade",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "annual_income":    "Applicant's gross annual income (USD)",
    "credit_score":     "FICO credit score (300–850)",
    "loan_amount":      "Requested loan principal (USD)",
    "age":              "Applicant age in years",
    "employment_years": "Years at current employer",
    "debt_to_income":   "Total monthly debt / monthly income",
    "num_credit_lines": "Number of open credit lines",
    "monthly_payment":  "Estimated monthly loan payment (USD)",
    "payment_history":  "Historical payment reliability rating",
    "loan_purpose":     "Stated purpose for the loan",
    "home_ownership":   "Applicant's housing situation",
    "loan_grade":       "Internal risk grade assigned at origination",
}

# ─── Use Case Metadata ────────────────────────────────────────────────────────
USE_CASE = {
    "name":              "CreditGuard ML Platform",
    "description":       "Real-time loan default risk scoring",
    "model_type":        "Gradient Boosted Classifier",
    "model_version":     "v2.4.1",
    "trained_on":        "2025-01-15",
    "monitoring_since":  "2025-02-01",
    "owner":             "ML Platform Team",
    "sla_accuracy":      0.85,
    "sla_latency_ms":    120,
}

# ─── Time Windows ─────────────────────────────────────────────────────────────
REFERENCE_DAYS  = 90
PRODUCTION_DAYS = 30
TIMESERIES_DAYS = 60

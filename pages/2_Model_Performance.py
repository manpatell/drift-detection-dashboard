"""
Model Performance Monitoring — CreditGuard ML Monitoring Platform

Tracks accuracy, F1, AUC-ROC, precision, and recall over time.
Compares baseline (reference) vs. production performance, and surfaces
prediction score distribution shifts and confusion matrix changes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Model Performance · CreditGuard",
    page_icon="🎯", layout="wide",
)

from utils.data_generator import (
    generate_reference_data, generate_production_data, generate_time_series_drift,
)
from utils.model_utils import train_model, evaluate_on_production
from utils.visualization import (
    gauge_chart, confusion_matrix_plot, roc_curve_plot,
    feature_importance_chart, time_series_chart,
    multi_metric_time_series, prediction_score_distribution,
)
from config import USE_CASE

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0F172A; color: #E2E8F0; }
  [data-testid="stSidebar"] { background-color: #1E293B; }
  [data-testid="metric-container"] {
    background:#1E293B; border:1px solid #334155; border-radius:12px; padding:16px;
  }
  .section-title {
    font-size:18px; font-weight:700; color:#C7D2FE;
    border-left:4px solid #6366F1; padding-left:10px; margin:24px 0 12px 0;
  }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Model Performance")
    st.divider()
    n_ref          = st.slider("Reference samples",  1_000, 10_000, 5_000, 500)
    n_prod         = st.slider("Production samples",   500,  5_000, 2_000, 250)
    drift_magnitude= st.slider("Drift magnitude",      0.0,    3.0,   1.5, 0.1)
    seed           = st.number_input("Seed", 0, 999, 42)

    st.divider()
    st.markdown("**Model Info**")
    st.markdown(f"Version `{USE_CASE['model_version']}`")
    st.markdown(f"Type: {USE_CASE['model_type']}")
    st.markdown(f"SLA Accuracy: ≥ {USE_CASE['sla_accuracy']:.0%}")


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating data…")
def get_data(n_ref, n_prod, drift_mag, seed):
    ref_df  = generate_reference_data(n=n_ref,  seed=seed)
    prod_df = generate_production_data(n=n_prod, drift_magnitude=drift_mag, seed=seed + 57)
    ts_df   = generate_time_series_drift(days=60, seed=seed + 99)
    return ref_df, prod_df, ts_df


ref_df, prod_df, ts_df          = get_data(n_ref, n_prod, drift_magnitude, int(seed))
model, le_map, baseline, fcols  = train_model(ref_df, seed=int(seed))
prod_metrics                    = evaluate_on_production(model, prod_df, le_map, fcols, seed=int(seed) + 57)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 🎯 Model Performance Monitoring")
st.markdown(
    "Track and compare **baseline** vs **production** model performance metrics. "
    "Identify accuracy degradation, threshold violations, and distributional shifts."
)
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Performance KPIs — Baseline vs Production</p>',
            unsafe_allow_html=True)

metrics_table = {
    "Metric":     ["Accuracy", "F1 Score", "AUC-ROC", "Precision", "Recall", "Default Rate"],
    "Baseline":   [
        f"{baseline['accuracy']:.4f}",
        f"{baseline['f1_score']:.4f}",
        f"{baseline['auc_roc']:.4f}",
        f"{baseline['precision']:.4f}",
        f"{baseline['recall']:.4f}",
        f"{baseline['default_rate']:.4f}",
    ],
    "Production": [
        f"{prod_metrics['accuracy']:.4f}",
        f"{prod_metrics['f1_score']:.4f}",
        f"{prod_metrics['auc_roc']:.4f}",
        f"{prod_metrics['precision']:.4f}",
        f"{prod_metrics['recall']:.4f}",
        f"{prod_metrics['default_rate']:.4f}",
    ],
    "Delta": [
        f"{prod_metrics['accuracy']  - baseline['accuracy']:+.4f}",
        f"{prod_metrics['f1_score']  - baseline['f1_score']:+.4f}",
        f"{prod_metrics['auc_roc']   - baseline['auc_roc']:+.4f}",
        f"{prod_metrics['precision'] - baseline['precision']:+.4f}",
        f"{prod_metrics['recall']    - baseline['recall']:+.4f}",
        f"{prod_metrics['default_rate'] - baseline['default_rate']:+.4f}",
    ],
}

def _style_delta(val):
    try:
        v = float(val)
        if v < -0.02:
            return "color:#fca5a5"
        if v < 0:
            return "color:#fcd34d"
        return "color:#86efac"
    except Exception:
        return ""

st.dataframe(
    pd.DataFrame(metrics_table).style.applymap(_style_delta, subset=["Delta"]),
    use_container_width=True, hide_index=True,
)

st.divider()

# ── Gauge Charts ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Production Performance Gauges</p>', unsafe_allow_html=True)

g1, g2, g3, g4, g5 = st.columns(5)
with g1:
    st.plotly_chart(gauge_chart(prod_metrics["accuracy"],  0.5, 1.0, "Accuracy",
                                warn_threshold=0.85, crit_threshold=0.75),
                    use_container_width=True)
with g2:
    st.plotly_chart(gauge_chart(prod_metrics["auc_roc"],   0.5, 1.0, "AUC-ROC",
                                warn_threshold=0.85, crit_threshold=0.75),
                    use_container_width=True)
with g3:
    st.plotly_chart(gauge_chart(prod_metrics["f1_score"],  0.5, 1.0, "F1 Score",
                                warn_threshold=0.82, crit_threshold=0.72),
                    use_container_width=True)
with g4:
    st.plotly_chart(gauge_chart(prod_metrics["precision"], 0.5, 1.0, "Precision",
                                warn_threshold=0.82, crit_threshold=0.72),
                    use_container_width=True)
with g5:
    st.plotly_chart(gauge_chart(prod_metrics["recall"],    0.5, 1.0, "Recall",
                                warn_threshold=0.80, crit_threshold=0.70),
                    use_container_width=True)

st.divider()

# ── Confusion Matrices ────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Confusion Matrix Comparison</p>', unsafe_allow_html=True)

cm1, cm2 = st.columns(2)
with cm1:
    st.plotly_chart(
        confusion_matrix_plot(baseline["confusion_matrix"], "Baseline (Reference)"),
        use_container_width=True,
    )
with cm2:
    st.plotly_chart(
        confusion_matrix_plot(prod_metrics["confusion_matrix"], "Production"),
        use_container_width=True,
    )

st.divider()

# ── ROC Curve Comparison ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">ROC Curve — Baseline vs Production</p>', unsafe_allow_html=True)

rc1, rc2 = st.columns([2, 1])
with rc1:
    st.plotly_chart(
        roc_curve_plot(
            baseline["roc_curve"], prod_metrics["roc_curve"],
            baseline["auc_roc"],   prod_metrics["auc_roc"],
        ),
        use_container_width=True,
    )
with rc2:
    st.markdown("**ROC Curve Summary**")
    auc_delta = prod_metrics["auc_roc"] - baseline["auc_roc"]
    st.metric("Baseline AUC",   f"{baseline['auc_roc']:.4f}")
    st.metric("Production AUC", f"{prod_metrics['auc_roc']:.4f}",
              delta=f"{auc_delta:+.4f}",
              delta_color="normal" if auc_delta >= 0 else "inverse")
    st.markdown("---")
    if auc_delta < -0.05:
        st.error("⚠️ Significant AUC degradation — retrain recommended.")
    elif auc_delta < -0.02:
        st.warning("⚡ Moderate AUC drop — monitor closely.")
    else:
        st.success("✅ AUC within acceptable range.")

st.divider()

# ── Prediction Score Distribution ─────────────────────────────────────────────
st.markdown('<p class="section-title">Prediction Score Distribution Shift</p>', unsafe_allow_html=True)

# Reconstruct reference probabilities using baseline model on ref data
from utils.model_utils import FEATURE_COLS, _encode
ref_enc, _ = _encode(ref_df.copy(), le_map, fit=False)
ref_probs  = model.predict_proba(ref_enc[FEATURE_COLS])[:, 1]
prod_probs = prod_metrics["y_prob"]

st.plotly_chart(
    prediction_score_distribution(ref_probs, prod_probs),
    use_container_width=True,
)

pd1, pd2, pd3 = st.columns(3)
pd1.metric("Ref Avg P(Default)",  f"{ref_probs.mean():.4f}")
pd2.metric("Prod Avg P(Default)", f"{prod_probs.mean():.4f}")
pd3.metric("Score Distribution Δ",
           f"{prod_probs.mean() - ref_probs.mean():+.4f}",
           delta_color="inverse" if prod_probs.mean() > ref_probs.mean() else "normal")

st.divider()

# ── Time-Series Performance ────────────────────────────────────────────────────
st.markdown('<p class="section-title">Performance Trend — Last 60 Days</p>', unsafe_allow_html=True)

ts1, ts2 = st.columns(2)
with ts1:
    st.plotly_chart(
        multi_metric_time_series(
            ts_df, ["accuracy", "f1_score", "auc_roc"],
            "Core Metrics Over Time",
        ),
        use_container_width=True,
    )
with ts2:
    st.plotly_chart(
        multi_metric_time_series(
            ts_df, ["precision", "recall"],
            "Precision & Recall Over Time",
        ),
        use_container_width=True,
    )

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)

st.plotly_chart(
    feature_importance_chart(
        baseline["feature_importance"],
        "Feature Importance — Gradient Boosted Classifier",
    ),
    use_container_width=True,
)

st.caption(
    "Feature importance is computed as mean impurity decrease (MDI) "
    "across all 150 gradient boosted trees."
)

st.divider()
st.caption("Model Performance Monitoring · CreditGuard ML Monitoring Platform")

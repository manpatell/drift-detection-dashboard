"""
CreditGuard ML Monitoring Platform
====================================
Production-grade MLOps dashboard for real-time model drift detection,
data quality monitoring, and model performance tracking.

Stack : Streamlit · Plotly · scikit-learn · SciPy · pandas · NumPy
Domain: Credit Risk — Loan Default Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CreditGuard ML Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "CreditGuard ML Monitoring Platform — Production MLOps Dashboard"},
)

from utils.data_generator import (
    generate_reference_data,
    generate_production_data,
    generate_time_series_drift,
)
from utils.drift_metrics import compute_full_drift_report
from utils.model_utils import train_model, evaluate_on_production
from utils.visualization import (
    drift_status_heatmap,
    psi_bar_chart,
    time_series_chart,
    multi_metric_time_series,
    gauge_chart,
)
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, USE_CASE, COLORS

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0F172A; color: #E2E8F0; }
  [data-testid="stSidebar"] { background-color: #1E293B; }
  [data-testid="metric-container"] {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 12px; padding: 16px;
  }
  .section-title {
    font-size: 18px; font-weight: 700; color: #C7D2FE;
    border-left: 4px solid #6366F1; padding-left: 10px;
    margin: 24px 0 12px 0;
  }
  .hero-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F1E35 100%);
    border: 1px solid #334155; border-radius: 16px; padding: 24px;
  }
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CreditGuard")
    st.markdown("**ML Monitoring Platform**")
    st.divider()

    st.markdown("#### Simulation Controls")
    n_ref     = st.slider("Reference samples",  1_000, 10_000, 5_000, 500)
    n_prod    = st.slider("Production samples",   500,  5_000, 2_000, 250)
    drift_mag = st.slider("Drift magnitude",      0.0,    3.0,   1.5, 0.1,
                          help="0 = no drift · 3 = severe drift")
    seed      = st.number_input("Random seed", 0, 999, 42)

    st.divider()
    st.markdown("#### Model Info")
    st.markdown(f"**Version** `{USE_CASE['model_version']}`")
    st.markdown(f"**Type** {USE_CASE['model_type']}")
    st.markdown(f"**Trained** {USE_CASE['trained_on']}")
    st.markdown(f"**Monitor since** {USE_CASE['monitoring_since']}")

    st.divider()
    st.markdown("#### Alert Thresholds")
    st.markdown("🔴 **PSI > 0.20** — High drift")
    st.markdown("🟡 **PSI > 0.10** — Moderate drift")
    st.markdown("🟢 **PSI < 0.10** — Stable")

    st.divider()
    st.caption("Navigate using the **Pages** menu above ↑")


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating datasets…")
def load_datasets(n_ref, n_prod, drift_mag, seed):
    ref_df  = generate_reference_data(n=n_ref,  seed=seed)
    prod_df = generate_production_data(n=n_prod, drift_magnitude=drift_mag, seed=seed + 57)
    ts_df   = generate_time_series_drift(days=60, seed=seed + 99)
    return ref_df, prod_df, ts_df


@st.cache_data(show_spinner="Computing drift metrics…")
def cached_drift(_ref_df, _prod_df):
    return compute_full_drift_report(
        _ref_df, _prod_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )


# ── Load everything ───────────────────────────────────────────────────────────
ref_df, prod_df, ts_df          = load_datasets(n_ref, n_prod, drift_mag, int(seed))
model, le_map, baseline, fcols  = train_model(ref_df, seed=int(seed))
prod_metrics                    = evaluate_on_production(model, prod_df, le_map, fcols, seed=int(seed) + 57)
drift_report                    = cached_drift(ref_df, prod_df)
summary                         = drift_report["summary"]

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🛡️ CreditGuard ML Monitoring Platform")
    st.markdown(
        f"**{USE_CASE['description']}** &nbsp;|&nbsp; "
        f"Model `{USE_CASE['model_version']}` &nbsp;|&nbsp; "
        f"Owner: {USE_CASE['owner']}"
    )
with col_h2:
    st.markdown(
        f"<br>**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        unsafe_allow_html=True,
    )
    if summary["critical"] > 0:
        st.error(f"⚠️ {summary['critical']} Critical Alert(s) Active")
    elif summary["warning"] > 0:
        st.warning(f"⚡ {summary['warning']} Warning(s) Active")
    else:
        st.success("✅ All Systems Nominal")

st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">System Health KPIs</p>', unsafe_allow_html=True)

acc_d  = round(prod_metrics["accuracy"]  - baseline["accuracy"],  4)
f1_d   = round(prod_metrics["f1_score"]  - baseline["f1_score"],  4)
auc_d  = round(prod_metrics["auc_roc"]   - baseline["auc_roc"],   4)
prec_d = round(prod_metrics["precision"] - baseline["precision"], 4)
rec_d  = round(prod_metrics["recall"]    - baseline["recall"],    4)

k = st.columns(8)
k[0].metric("Features",       summary["total_features"])
k[1].metric("🔴 Critical",    summary["critical"],
            delta=f"{summary['critical']} alerts" if summary["critical"] else "Clear",
            delta_color="inverse")
k[2].metric("🟡 Warnings",    summary["warning"],
            delta=f"{summary['warning']}" if summary["warning"] else "Clear",
            delta_color="inverse")
k[3].metric("Accuracy",  f"{prod_metrics['accuracy']:.2%}",
            delta=f"{acc_d:+.2%}",  delta_color="normal" if acc_d  >= 0 else "inverse")
k[4].metric("F1 Score",  f"{prod_metrics['f1_score']:.4f}",
            delta=f"{f1_d:+.4f}",   delta_color="normal" if f1_d   >= 0 else "inverse")
k[5].metric("AUC-ROC",  f"{prod_metrics['auc_roc']:.4f}",
            delta=f"{auc_d:+.4f}",  delta_color="normal" if auc_d  >= 0 else "inverse")
k[6].metric("Precision", f"{prod_metrics['precision']:.4f}",
            delta=f"{prec_d:+.4f}", delta_color="normal" if prec_d >= 0 else "inverse")
k[7].metric("Recall",    f"{prod_metrics['recall']:.4f}",
            delta=f"{rec_d:+.4f}",  delta_color="normal" if rec_d  >= 0 else "inverse")

st.divider()

# ── Gauge Charts ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Model Performance Gauges</p>', unsafe_allow_html=True)

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.plotly_chart(
        gauge_chart(prod_metrics["accuracy"], 0.5, 1.0, "Accuracy",
                    warn_threshold=0.85, crit_threshold=0.75),
        use_container_width=True,
    )
with g2:
    st.plotly_chart(
        gauge_chart(prod_metrics["auc_roc"], 0.5, 1.0, "AUC-ROC",
                    warn_threshold=0.85, crit_threshold=0.75),
        use_container_width=True,
    )
with g3:
    st.plotly_chart(
        gauge_chart(prod_metrics["f1_score"], 0.5, 1.0, "F1 Score",
                    warn_threshold=0.82, crit_threshold=0.72),
        use_container_width=True,
    )
with g4:
    drift_health = max(0.0, 1.0 - summary["avg_psi"] / 0.30)
    st.plotly_chart(
        gauge_chart(drift_health, 0.0, 1.0, "Data Health Index",
                    warn_threshold=0.70, crit_threshold=0.40),
        use_container_width=True,
    )

st.divider()

# ── Drift Status + PSI ────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Feature Drift Overview</p>', unsafe_allow_html=True)

st.plotly_chart(drift_status_heatmap(drift_report), use_container_width=True)

drift_rows = []
for feat, m in {**drift_report["numeric"], **drift_report["categorical"]}.items():
    drift_rows.append({"Feature": feat, "PSI": m["psi"]})
df_psi = pd.DataFrame(drift_rows)
st.plotly_chart(psi_bar_chart(df_psi), use_container_width=True)

st.divider()

# ── Time-Series Row ───────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Trend Analysis — Last 60 Days</p>', unsafe_allow_html=True)

tc1, tc2 = st.columns(2)
with tc1:
    st.plotly_chart(
        time_series_chart(ts_df, "avg_psi", "Average PSI Over Time"),
        use_container_width=True,
    )
with tc2:
    st.plotly_chart(
        multi_metric_time_series(
            ts_df, ["accuracy", "f1_score", "auc_roc"],
            "Model Performance Metrics Over Time",
        ),
        use_container_width=True,
    )

st.divider()

# ── Active Alerts Table ───────────────────────────────────────────────────────
st.markdown('<p class="section-title">Active Drift Alerts</p>', unsafe_allow_html=True)

alerts = []
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
for feat, m in drift_report["numeric"].items():
    if m["status"] != "stable":
        alerts.append({
            "Timestamp":    now_str,
            "Feature":      feat,
            "Type":         "Numeric",
            "Status":       m["status"].upper(),
            "PSI":          m["psi"],
            "KS Statistic": m["ks_statistic"],
            "KS p-value":   m["ks_p_value"],
            "Mean Shift":   f"{m['mean_shift_pct']:+.1f}%",
        })
for feat, m in drift_report["categorical"].items():
    if m["status"] != "stable":
        alerts.append({
            "Timestamp":    now_str,
            "Feature":      feat,
            "Type":         "Categorical",
            "Status":       m["status"].upper(),
            "PSI":          m["psi"],
            "KS Statistic": m["chi2_statistic"],
            "KS p-value":   m["p_value"],
            "Mean Shift":   "N/A",
        })

if alerts:
    def _style_status(val):
        if val == "CRITICAL":
            return "background-color:#450a0a; color:#fca5a5"
        if val == "WARNING":
            return "background-color:#451a03; color:#fcd34d"
        return ""

    alerts_df = pd.DataFrame(alerts)
    st.dataframe(
        alerts_df.style.applymap(_style_status, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )
    csv = alerts_df.to_csv(index=False)
    st.download_button("⬇️ Export Alerts CSV", csv, "active_alerts.csv", "text/csv")
else:
    st.success("✅ No active drift alerts — all features are within acceptable thresholds.")

st.divider()

# ── Data Snapshot ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Data Snapshot</p>', unsafe_allow_html=True)

tab_ref, tab_prod = st.tabs(["📦 Reference Data", "🚀 Production Data"])
with tab_ref:
    st.caption(f"{len(ref_df):,} rows · {ref_df.shape[1]} columns")
    st.dataframe(ref_df.head(200), use_container_width=True, hide_index=True)
with tab_prod:
    st.caption(f"{len(prod_df):,} rows · {prod_df.shape[1]} columns")
    st.dataframe(prod_df.head(200), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    f"CreditGuard ML Monitoring Platform · "
    f"Model {USE_CASE['model_version']} · "
    "Built with Streamlit + Plotly + scikit-learn"
)

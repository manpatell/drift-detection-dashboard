"""
Alerts & Configuration — CreditGuard ML Monitoring Platform

Unified alert management interface: active alerts, historical alert timeline,
configurable thresholds, and simulated notification channels.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Alerts & Config · CreditGuard",
    page_icon="🚨", layout="wide",
)

from utils.data_generator import (
    generate_reference_data, generate_production_data, generate_time_series_drift,
)
from utils.drift_metrics import compute_full_drift_report
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, THRESHOLDS, COLORS, USE_CASE

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
  .alert-critical {
    background:#450a0a; border-left:4px solid #EF4444;
    padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:8px;
  }
  .alert-warning {
    background:#451a03; border-left:4px solid #F59E0B;
    padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:8px;
  }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 Alerts & Config")
    st.divider()
    n_ref           = st.slider("Reference samples",  1_000, 10_000, 5_000, 500)
    n_prod          = st.slider("Production samples",   500,  5_000, 2_000, 250)
    drift_magnitude = st.slider("Drift magnitude",      0.0,    3.0,   1.5, 0.1)
    seed            = st.number_input("Seed", 0, 999, 42)

    st.divider()
    st.markdown("#### Threshold Configuration")
    psi_warn  = st.number_input("PSI Warning",  0.01, 0.50, 0.10, 0.01, format="%.2f")
    psi_crit  = st.number_input("PSI Critical", 0.05, 1.00, 0.20, 0.01, format="%.2f")
    ks_alpha  = st.number_input("KS α (p-value)", 0.001, 0.10, 0.05, 0.005, format="%.3f")

    st.divider()
    st.markdown("#### Notification Channels")
    notify_email  = st.toggle("Email Alerts",   value=True)
    notify_slack  = st.toggle("Slack Alerts",   value=True)
    notify_pager  = st.toggle("PagerDuty",      value=False)
    notify_webhook= st.toggle("Custom Webhook", value=False)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing alerts…")
def get_data_and_drift(n_ref, n_prod, drift_mag, seed):
    ref_df  = generate_reference_data(n=n_ref,  seed=seed)
    prod_df = generate_production_data(n=n_prod, drift_magnitude=drift_mag, seed=seed + 57)
    ts_df   = generate_time_series_drift(days=60, seed=seed + 99)
    report  = compute_full_drift_report(ref_df, prod_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    return ref_df, prod_df, ts_df, report


ref_df, prod_df, ts_df, drift_report = get_data_and_drift(
    n_ref, n_prod, drift_magnitude, int(seed)
)
summary = drift_report["summary"]

# ── Dynamic thresholds from sidebar ───────────────────────────────────────────
def _status(psi: float, p_val: float) -> str:
    if psi >= psi_crit or p_val < ks_alpha:
        return "CRITICAL"
    if psi >= psi_warn:
        return "WARNING"
    return "STABLE"


# ── Build alerts list ─────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
alerts  = []

for feat, m in drift_report["numeric"].items():
    status = _status(m["psi"], m["ks_p_value"])
    if status != "STABLE":
        alerts.append({
            "ID":          f"ALT-{abs(hash(feat)) % 100000:05d}",
            "Timestamp":   now_str,
            "Feature":     feat,
            "Type":        "Numeric",
            "Status":      status,
            "PSI":         m["psi"],
            "Test Stat":   m["ks_statistic"],
            "p-value":     m["ks_p_value"],
            "Mean Shift":  f"{m['mean_shift_pct']:+.1f}%",
            "Severity":    1 if status == "CRITICAL" else 2,
        })

for feat, m in drift_report["categorical"].items():
    status = _status(m["psi"], m["p_value"])
    if status != "STABLE":
        alerts.append({
            "ID":          f"ALT-{abs(hash(feat)) % 100000:05d}",
            "Timestamp":   now_str,
            "Feature":     feat,
            "Type":        "Categorical",
            "Status":      status,
            "PSI":         m["psi"],
            "Test Stat":   m["chi2_statistic"],
            "p-value":     m["p_value"],
            "Mean Shift":  "N/A",
            "Severity":    1 if status == "CRITICAL" else 2,
        })

alerts.sort(key=lambda x: x["Severity"])
n_critical = sum(1 for a in alerts if a["Status"] == "CRITICAL")
n_warning  = sum(1 for a in alerts if a["Status"] == "WARNING")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 🚨 Alerts & Configuration")
st.markdown(
    "Centralized alert management for drift events, threshold violations, "
    "and model performance degradation signals."
)
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k = st.columns(5)
k[0].metric("Total Active Alerts", len(alerts))
k[1].metric("🔴 Critical",         n_critical)
k[2].metric("🟡 Warning",          n_warning)
k[3].metric("Features Monitored",  summary["total_features"])
k[4].metric("Max PSI",             f"{summary['max_psi']:.4f}")

st.divider()

# ── Active Alerts ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Active Alerts</p>', unsafe_allow_html=True)

if alerts:
    # Critical first
    for a in alerts:
        css_class = "alert-critical" if a["Status"] == "CRITICAL" else "alert-warning"
        icon      = "🔴" if a["Status"] == "CRITICAL" else "🟡"
        st.markdown(
            f'<div class="{css_class}">'
            f'<strong>{icon} [{a["ID"]}] {a["Feature"]}</strong> — {a["Status"]} '
            f'| PSI: {a["PSI"]:.4f} | Mean Shift: {a["Mean Shift"]} '
            f'| {a["Type"]} | {a["Timestamp"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    alerts_df = pd.DataFrame(
        [{k: v for k, v in a.items() if k != "Severity"} for a in alerts]
    )

    def _s(v):
        if v == "CRITICAL": return "background-color:#450a0a; color:#fca5a5"
        if v == "WARNING":  return "background-color:#451a03; color:#fcd34d"
        return ""

    st.markdown("")
    st.dataframe(
        alerts_df.style.applymap(_s, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Export Alerts CSV",
        alerts_df.to_csv(index=False),
        "alerts_export.csv", "text/csv",
    )
else:
    st.success(
        "✅ No active alerts — all features are within configured thresholds. "
        f"(PSI warning: {psi_warn:.2f}, critical: {psi_crit:.2f})"
    )

st.divider()

# ── Historical Alert Timeline ─────────────────────────────────────────────────
st.markdown('<p class="section-title">Historical Alert Timeline — Last 60 Days</p>',
            unsafe_allow_html=True)

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Bar(
    x=ts_df["date"], y=ts_df["alert_count"],
    marker_color=[
        COLORS["critical"] if v >= 4
        else COLORS["warning"] if v >= 2
        else COLORS["stable"]
        for v in ts_df["alert_count"]
    ],
    name="Daily Alerts",
    hovertemplate="%{x|%b %d}<br>Alerts: %{y}<extra></extra>",
))
fig_timeline.add_trace(go.Scatter(
    x=ts_df["date"],
    y=ts_df["alert_count"].rolling(7, min_periods=1).mean(),
    mode="lines", name="7-day MA",
    line=dict(color=COLORS["accent"], width=2, dash="dot"),
))
fig_timeline.update_layout(
    title="Daily Alert Count",
    xaxis_title="Date", yaxis_title="Alert Count",
    template="plotly_dark",
    height=320, margin=dict(l=50, r=20, t=60, b=50),
)
st.plotly_chart(fig_timeline, use_container_width=True)

# Alert stats
st.columns(4)[0].metric("Total Alerts (60d)", int(ts_df["alert_count"].sum()))
st.columns(4)[1].metric("Avg Alerts/Day",    f"{ts_df['alert_count'].mean():.1f}")
st.columns(4)[2].metric("Peak Alerts/Day",   int(ts_df["alert_count"].max()))
st.columns(4)[3].metric("Days with Alerts",  int((ts_df["alert_count"] > 0).sum()))

st.divider()

# ── Threshold Configuration Panel ─────────────────────────────────────────────
st.markdown('<p class="section-title">Alert Threshold Configuration</p>', unsafe_allow_html=True)

th_col1, th_col2 = st.columns(2)

with th_col1:
    st.markdown("**Drift Detection Thresholds**")
    th_df = pd.DataFrame({
        "Metric":          ["PSI — Warning", "PSI — Critical", "KS p-value",
                            "JS Divergence — Warning", "JS Divergence — Critical",
                            "Wasserstein — Warning",   "Wasserstein — Critical"],
        "Current Value":   [psi_warn, psi_crit, ks_alpha,
                            THRESHOLDS["js_divergence"]["stable"],
                            THRESHOLDS["js_divergence"]["warning"],
                            THRESHOLDS["wasserstein"]["stable"],
                            THRESHOLDS["wasserstein"]["warning"]],
        "Default":         [0.10, 0.20, 0.05, 0.05, 0.10, 0.10, 0.30],
    })
    st.dataframe(th_df, use_container_width=True, hide_index=True)

with th_col2:
    st.markdown("**Model Performance Thresholds**")
    perf_df = pd.DataFrame({
        "Metric":        ["Accuracy SLA", "F1 Score floor", "AUC-ROC floor",
                          "Accuracy drop alert", "F1 drop alert"],
        "Threshold":     [USE_CASE["sla_accuracy"], 0.82, 0.82, 0.03, 0.05],
        "Action":        ["Page on-call", "Page on-call", "Slack warning",
                          "Email team", "Email team"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.divider()

# ── Notification Channel Status ───────────────────────────────────────────────
st.markdown('<p class="section-title">Notification Channel Status</p>', unsafe_allow_html=True)

channels = [
    ("📧 Email",        notify_email,   "ml-alerts@company.com"),
    ("💬 Slack",        notify_slack,   "#ml-monitoring-alerts"),
    ("📟 PagerDuty",    notify_pager,   "ML Platform On-Call"),
    ("🌐 Webhook",      notify_webhook, "https://hooks.example.com/ml-alerts"),
]

ch_cols = st.columns(4)
for (name, enabled, target), col in zip(channels, ch_cols):
    with col:
        if enabled:
            st.success(f"**{name}**\n\n✅ Active\n\n`{target}`")
        else:
            st.info(f"**{name}**\n\n⭕ Disabled\n\n`{target}`")

st.divider()

# ── Alert Policy Summary ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">Alert Policy</p>', unsafe_allow_html=True)

policy_df = pd.DataFrame({
    "Condition":          [
        "PSI > {:.2f} on any feature".format(psi_crit),
        "PSI > {:.2f} on any feature".format(psi_warn),
        "KS p-value < {:.3f}".format(ks_alpha),
        "Accuracy drops > 3%",
        "AUC-ROC drops > 2%",
        "F1 Score drops > 5%",
    ],
    "Severity":           ["Critical", "Warning", "Warning", "Critical", "Warning", "Critical"],
    "Notification":       ["All Channels", "Email + Slack", "Email", "All Channels",
                           "Email + Slack", "All Channels"],
    "Auto-Retrain":       ["Recommended", "No", "No", "Trigger", "No", "Trigger"],
    "SLA Response (min)": [15, 60, 60, 15, 60, 15],
})
st.dataframe(policy_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Alerts & Configuration · CreditGuard ML Monitoring Platform")

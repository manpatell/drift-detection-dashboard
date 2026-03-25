"""
Day 05 — Model Drift Detection Dashboard
Theme: MLOps
Tags: mlops, monitoring, drift-detection, streamlit

Demonstrates: Detecting data drift and model performance degradation over
time using statistical tests (KS test, PSI) with live alerting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Drift Detection Dashboard", page_icon="📡", layout="wide")
st.title("📡 Model Drift Detection Dashboard")
st.markdown("Monitor your model for **data drift** and **performance degradation** over time.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Config")
n_ref      = st.sidebar.slider("Reference samples", 500, 5000, 1000, 500)
n_prod     = st.sidebar.slider("Production samples", 500, 5000, 1000, 500)
drift_mag  = st.sidebar.slider("Drift magnitude", 0.0, 3.0, 1.0, 0.1)
n_features = st.sidebar.slider("Features to monitor", 2, 10, 5)
seed       = st.sidebar.number_input("Random seed", 0, 999, 42)
np.random.seed(int(seed))

# ── Generate data ──────────────────────────────────────────────────────────────
feature_names = [f"feature_{i+1}" for i in range(n_features)]
ref_data  = pd.DataFrame(np.random.randn(n_ref,  n_features), columns=feature_names)
prod_data = pd.DataFrame(np.random.randn(n_prod, n_features) + drift_mag * np.random.choice([-1,1], n_features),
                          columns=feature_names)

# ── KS test + PSI ─────────────────────────────────────────────────────────────
def psi(expected, actual, buckets=10):
    eps = 1e-6
    mn, mx = min(expected.min(), actual.min()), max(expected.max(), actual.max())
    bins = np.linspace(mn, mx, buckets + 1)
    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected) + eps
    act_pct = np.histogram(actual,   bins=bins)[0] / len(actual)   + eps
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

drift_results = []
for feat in feature_names:
    ks_stat, ks_p = stats.ks_2samp(ref_data[feat], prod_data[feat])
    psi_val = psi(ref_data[feat].values, prod_data[feat].values)
    drift_detected = ks_p < 0.05 or psi_val > 0.2
    drift_results.append({
        "Feature": feat, "KS Statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 4), "PSI": round(psi_val, 4),
        "Drift": "🔴 Yes" if drift_detected else "🟢 No"
    })

df_drift = pd.DataFrame(drift_results)
n_drifted = (df_drift["Drift"] == "🔴 Yes").sum()

# ── KPIs ───────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Features monitored", n_features)
col2.metric("Drifted features", n_drifted, delta=f"{n_drifted} alerts" if n_drifted else "All clear")
col3.metric("Avg PSI", round(df_drift["PSI"].mean(), 4))
col4.metric("Avg KS stat", round(df_drift["KS Statistic"].mean(), 4))

if n_drifted > 0:
    st.error(f"⚠️ **Drift detected** in {n_drifted}/{n_features} features. Consider retraining your model.")
else:
    st.success("✅ No significant drift detected. Model inputs are stable.")

st.divider()

# ── Distribution plots ─────────────────────────────────────────────────────────
st.subheader("📊 Feature Distribution Comparison")
cols_per_row = 3
feat_chunks = [feature_names[i:i+cols_per_row] for i in range(0, len(feature_names), cols_per_row)]

for chunk in feat_chunks:
    cols = st.columns(len(chunk))
    for col, feat in zip(cols, chunk):
        with col:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.hist(ref_data[feat],  bins=30, alpha=0.6, color="#378ADD", label="Reference", density=True)
            ax.hist(prod_data[feat], bins=30, alpha=0.6, color="#D85A30", label="Production", density=True)
            psi_val = df_drift.loc[df_drift["Feature"]==feat, "PSI"].values[0]
            ax.set_title(f"{feat}\nPSI={psi_val}", fontsize=9)
            ax.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)

# ── Drift summary table ────────────────────────────────────────────────────────
st.subheader("📋 Drift Summary Table")
st.dataframe(df_drift.style.apply(
    lambda x: ["background-color: #fde8e8" if v == "🔴 Yes" else "" for v in x], subset=["Drift"]
), use_container_width=True)

st.subheader("📈 PSI Score by Feature")
fig3, ax3 = plt.subplots(figsize=(8, 3))
colors = ["#D85A30" if p > 0.2 else ("#F0C419" if p > 0.1 else "#1D9E75") for p in df_drift["PSI"]]
ax3.bar(df_drift["Feature"], df_drift["PSI"], color=colors)
ax3.axhline(0.2, color="#D85A30", linestyle="--", label="High drift (PSI>0.2)")
ax3.axhline(0.1, color="#F0C419", linestyle="--", label="Moderate drift (PSI>0.1)")
ax3.set_ylabel("PSI Score")
ax3.legend(fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig3)

st.caption("PSI > 0.2 = significant drift | 0.1–0.2 = moderate | < 0.1 = stable")

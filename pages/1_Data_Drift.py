"""
Data Drift Analysis — CreditGuard ML Monitoring Platform

Deep statistical comparison of reference vs. production feature distributions
using KS test, PSI, Jensen-Shannon Divergence, and Wasserstein Distance.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Data Drift Analysis · CreditGuard",
    page_icon="📊", layout="wide",
)

from utils.data_generator import generate_reference_data, generate_production_data
from utils.drift_metrics import compute_full_drift_report
from utils.visualization import (
    distribution_plot, box_plot_comparison,
    categorical_bar_comparison, psi_bar_chart,
)
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, COLORS

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
    st.markdown("## 📊 Data Drift")
    st.divider()
    n_ref          = st.slider("Reference samples",  1_000, 10_000, 5_000, 500)
    n_prod         = st.slider("Production samples",   500,  5_000, 2_000, 250)
    drift_magnitude= st.slider("Drift magnitude",      0.0,    3.0,   1.5, 0.1)
    seed           = st.number_input("Seed", 0, 999, 42)

    if st.button("🔄 Regenerate Data", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    selected_feature = st.selectbox(
        "Feature deep-dive:",
        NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    )
    st.divider()
    st.markdown("**Drift Tests Used**")
    st.markdown("- KS Test (numeric)")
    st.markdown("- Chi-Square (categorical)")
    st.markdown("- Population Stability Index")
    st.markdown("- Jensen-Shannon Divergence")
    st.markdown("- Wasserstein Distance")


# ── Data & metrics ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating data…")
def get_data(n_ref, n_prod, drift_magnitude, seed):
    ref_df  = generate_reference_data(n=n_ref,  seed=seed)
    prod_df = generate_production_data(n=n_prod, drift_magnitude=drift_magnitude, seed=seed + 57)
    return ref_df, prod_df

@st.cache_data(show_spinner="Computing drift…")
def get_drift(_ref_df, _prod_df):
    return compute_full_drift_report(_ref_df, _prod_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)


ref_df, prod_df = get_data(n_ref, n_prod, drift_magnitude, int(seed))
drift_report    = get_drift(ref_df, prod_df)
summary         = drift_report["summary"]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 📊 Data Drift Analysis")
st.markdown(
    "Statistical comparison between the **reference** (training) population "
    "and the current **production** distribution."
)
st.divider()

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k = st.columns(6)
k[0].metric("Total Features",  summary["total_features"])
k[1].metric("🔴 Critical",     summary["critical"])
k[2].metric("🟡 Warning",      summary["warning"])
k[3].metric("🟢 Stable",       summary["stable"])
k[4].metric("Avg PSI",         f"{summary['avg_psi']:.4f}")
k[5].metric("Max PSI",         f"{summary['max_psi']:.4f}")

st.divider()

# ── Full Drift Table ──────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Drift Summary Table</p>', unsafe_allow_html=True)

rows = []
for feat, m in drift_report["numeric"].items():
    rows.append({
        "Feature":       feat,
        "Type":          "Numeric",
        "KS Stat":       m["ks_statistic"],
        "KS p-value":    m["ks_p_value"],
        "PSI":           m["psi"],
        "JS Divergence": m["js_divergence"],
        "Wasserstein":   m["wasserstein"],
        "Ref Mean":      m["ref_mean"],
        "Prod Mean":     m["prod_mean"],
        "Mean Shift %":  f"{m['mean_shift_pct']:+.1f}%",
        "Status":        m["status"].upper(),
    })
for feat, m in drift_report["categorical"].items():
    rows.append({
        "Feature":       feat,
        "Type":          "Categorical",
        "KS Stat":       m["chi2_statistic"],
        "KS p-value":    m["p_value"],
        "PSI":           m["psi"],
        "JS Divergence": "—",
        "Wasserstein":   "—",
        "Ref Mean":      "—",
        "Prod Mean":     "—",
        "Mean Shift %":  "—",
        "Status":        m["status"].upper(),
    })

drift_df = pd.DataFrame(rows)

def _style_row(val):
    if val == "CRITICAL":
        return "background-color:#450a0a; color:#fca5a5"
    if val == "WARNING":
        return "background-color:#451a03; color:#fcd34d"
    if val == "STABLE":
        return "background-color:#052e16; color:#86efac"
    return ""

st.dataframe(
    drift_df.style.applymap(_style_row, subset=["Status"]),
    use_container_width=True, hide_index=True, height=420,
)

col_dl1, col_dl2 = st.columns([1, 5])
with col_dl1:
    st.download_button(
        "⬇️ Export CSV", drift_df.to_csv(index=False),
        "drift_report.csv", "text/csv",
    )

st.divider()

# ── PSI Bar Chart ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">PSI by Feature</p>', unsafe_allow_html=True)
psi_rows = [{"Feature": r["Feature"], "PSI": r["PSI"]}
            for r in rows if isinstance(r["PSI"], float)]
st.plotly_chart(psi_bar_chart(pd.DataFrame(psi_rows)), use_container_width=True)

st.divider()

# ── Feature Deep Dive ─────────────────────────────────────────────────────────
st.markdown(
    f'<p class="section-title">Feature Deep Dive — <code>{selected_feature}</code></p>',
    unsafe_allow_html=True,
)

if selected_feature in NUMERIC_FEATURES:
    m = drift_report["numeric"][selected_feature]

    mx = st.columns(5)
    mx[0].metric("KS Statistic",   m["ks_statistic"])
    mx[1].metric("KS p-value",     m["ks_p_value"])
    mx[2].metric("PSI",            m["psi"])
    mx[3].metric("JS Divergence",  m["js_divergence"])
    mx[4].metric("Wasserstein",    m["wasserstein"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            distribution_plot(
                ref_df[selected_feature].dropna().values,
                prod_df[selected_feature].dropna().values,
                selected_feature,
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            box_plot_comparison(
                ref_df[selected_feature].dropna().values,
                prod_df[selected_feature].dropna().values,
                selected_feature,
            ),
            use_container_width=True,
        )

    stats_df = pd.DataFrame({
        "Metric":     ["Mean", "Median", "Std Dev", "Min", "Max", "Skewness", "Kurtosis"],
        "Reference":  [
            f"{ref_df[selected_feature].mean():.3f}",
            f"{ref_df[selected_feature].median():.3f}",
            f"{ref_df[selected_feature].std():.3f}",
            f"{ref_df[selected_feature].min():.3f}",
            f"{ref_df[selected_feature].max():.3f}",
            f"{ref_df[selected_feature].skew():.4f}",
            f"{ref_df[selected_feature].kurtosis():.4f}",
        ],
        "Production": [
            f"{prod_df[selected_feature].mean():.3f}",
            f"{prod_df[selected_feature].median():.3f}",
            f"{prod_df[selected_feature].std():.3f}",
            f"{prod_df[selected_feature].min():.3f}",
            f"{prod_df[selected_feature].max():.3f}",
            f"{prod_df[selected_feature].skew():.4f}",
            f"{prod_df[selected_feature].kurtosis():.4f}",
        ],
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

else:
    m = drift_report["categorical"][selected_feature]
    mx = st.columns(3)
    mx[0].metric("Chi² Statistic", m["chi2_statistic"])
    mx[1].metric("p-value",        m["p_value"])
    mx[2].metric("PSI",            m["psi"])
    st.plotly_chart(
        categorical_bar_comparison(m["distribution"], selected_feature),
        use_container_width=True,
    )

st.divider()

# ── Grid of all distributions ─────────────────────────────────────────────────
st.markdown('<p class="section-title">All Numeric Distributions</p>', unsafe_allow_html=True)

for i in range(0, len(NUMERIC_FEATURES), 2):
    chunk = NUMERIC_FEATURES[i:i+2]
    cols  = st.columns(2)
    for col, feat in zip(cols, chunk):
        with col:
            st.plotly_chart(
                distribution_plot(
                    ref_df[feat].dropna().values,
                    prod_df[feat].dropna().values,
                    feat, bins=40,
                ),
                use_container_width=True,
            )

st.markdown('<p class="section-title">All Categorical Distributions</p>', unsafe_allow_html=True)

for i in range(0, len(CATEGORICAL_FEATURES), 2):
    chunk = CATEGORICAL_FEATURES[i:i+2]
    cols  = st.columns(2)
    for col, feat in zip(cols, chunk):
        with col:
            st.plotly_chart(
                categorical_bar_comparison(
                    drift_report["categorical"][feat]["distribution"], feat
                ),
                use_container_width=True,
            )

st.divider()
st.caption("Data Drift Analysis · CreditGuard ML Monitoring Platform")

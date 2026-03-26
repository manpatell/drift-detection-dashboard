"""
Feature Analysis — CreditGuard ML Monitoring Platform

In-depth exploration of feature relationships, correlation drift,
outlier patterns, and multivariate interactions between the reference
and production populations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Feature Analysis · CreditGuard",
    page_icon="🔍", layout="wide",
)

from utils.data_generator import generate_reference_data, generate_production_data
from utils.visualization import correlation_heatmap, distribution_plot
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
    st.markdown("## 🔍 Feature Analysis")
    st.divider()
    n_ref           = st.slider("Reference samples",  1_000, 10_000, 5_000, 500)
    n_prod          = st.slider("Production samples",   500,  5_000, 2_000, 250)
    drift_magnitude = st.slider("Drift magnitude",      0.0,    3.0,   1.5, 0.1)
    seed            = st.number_input("Seed", 0, 999, 42)

    st.divider()
    st.markdown("**Scatter Plot Config**")
    x_feat = st.selectbox("X axis", NUMERIC_FEATURES, index=0)
    y_feat = st.selectbox("Y axis", NUMERIC_FEATURES, index=1)

    st.divider()
    st.markdown("**Outlier Config**")
    zscore_thresh = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.5)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def get_data(n_ref, n_prod, drift_mag, seed):
    ref_df  = generate_reference_data(n=n_ref,  seed=seed)
    prod_df = generate_production_data(n=n_prod, drift_magnitude=drift_mag, seed=seed + 57)
    return ref_df, prod_df


ref_df, prod_df = get_data(n_ref, n_prod, drift_magnitude, int(seed))

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 🔍 Feature Analysis")
st.markdown(
    "Explore **correlation structures**, **feature interactions**, "
    "and **outlier patterns** across the reference and production populations."
)
st.divider()

# ── Descriptive Stats ─────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)

tab_ref, tab_prod = st.tabs(["📦 Reference", "🚀 Production"])
with tab_ref:
    st.dataframe(
        ref_df[NUMERIC_FEATURES].describe().T.round(3),
        use_container_width=True,
    )
with tab_prod:
    st.dataframe(
        prod_df[NUMERIC_FEATURES].describe().T.round(3),
        use_container_width=True,
    )

st.divider()

# ── Correlation Heatmaps ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">Correlation Matrix — Reference vs Production</p>',
            unsafe_allow_html=True)

ch1, ch2 = st.columns(2)
with ch1:
    st.plotly_chart(
        correlation_heatmap(ref_df[NUMERIC_FEATURES], "Reference Correlation Matrix"),
        use_container_width=True,
    )
with ch2:
    st.plotly_chart(
        correlation_heatmap(prod_df[NUMERIC_FEATURES], "Production Correlation Matrix"),
        use_container_width=True,
    )

# Correlation delta
ref_corr  = ref_df[NUMERIC_FEATURES].corr()
prod_corr = prod_df[NUMERIC_FEATURES].corr()
delta_corr = (prod_corr - ref_corr).abs()

st.markdown("**Correlation Change (|Δ|) — largest shifts indicate structural drift**")
fig_delta = go.Figure(go.Heatmap(
    z=delta_corr.values,
    x=list(delta_corr.columns),
    y=list(delta_corr.index),
    colorscale="YlOrRd", zmin=0, zmax=0.5,
    text=delta_corr.values.round(3),
    texttemplate="%{text}", textfont_size=9,
    colorbar=dict(title="|Δr|"),
))
fig_delta.update_layout(
    title="|Δ| Correlation Drift Heatmap",
    template="plotly_dark", height=460,
    margin=dict(l=120, r=20, t=60, b=80),
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_delta, use_container_width=True)

st.divider()

# ── Scatter Plot ──────────────────────────────────────────────────────────────
st.markdown(
    f'<p class="section-title">Scatter Plot — {x_feat} vs {y_feat}</p>',
    unsafe_allow_html=True,
)

scatter_sample = 1_000
ref_s  = ref_df.sample(min(scatter_sample,  len(ref_df)),  random_state=42)
prod_s = prod_df.sample(min(scatter_sample, len(prod_df)), random_state=42)

ref_s["Dataset"]  = "Reference"
prod_s["Dataset"] = "Production"
combined = pd.concat([ref_s, prod_s], ignore_index=True)

fig_scatter = px.scatter(
    combined, x=x_feat, y=y_feat, color="Dataset",
    color_discrete_map={"Reference": COLORS["reference"], "Production": COLORS["production"]},
    opacity=0.5, template="plotly_dark",
    title=f"Scatter: {x_feat} vs {y_feat}",
    marginal_x="histogram", marginal_y="histogram",
)
fig_scatter.update_layout(height=520, margin=dict(l=60, r=20, t=60, b=60))
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Outlier Analysis ──────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Outlier Analysis (Z-Score Method)</p>',
            unsafe_allow_html=True)


def flag_outliers(df: pd.DataFrame, features: list, thresh: float) -> pd.DataFrame:
    z = df[features].apply(lambda col: (col - col.mean()) / (col.std() + 1e-9))
    return (z.abs() > thresh).sum().reset_index(columns=["Feature", "Outlier Count"])


ref_out_counts  = {
    f: int(((ref_df[f]  - ref_df[f].mean())  / (ref_df[f].std()  + 1e-9)).abs().gt(zscore_thresh).sum())
    for f in NUMERIC_FEATURES
}
prod_out_counts = {
    f: int(((prod_df[f] - prod_df[f].mean()) / (prod_df[f].std() + 1e-9)).abs().gt(zscore_thresh).sum())
    for f in NUMERIC_FEATURES
}

outlier_df = pd.DataFrame({
    "Feature":         NUMERIC_FEATURES,
    "Ref Outliers":    [ref_out_counts[f]  for f in NUMERIC_FEATURES],
    "Prod Outliers":   [prod_out_counts[f] for f in NUMERIC_FEATURES],
    "Ref Outlier %":   [round(ref_out_counts[f]  / len(ref_df)  * 100, 2) for f in NUMERIC_FEATURES],
    "Prod Outlier %":  [round(prod_out_counts[f] / len(prod_df) * 100, 2) for f in NUMERIC_FEATURES],
})
outlier_df["Δ Outlier %"] = (outlier_df["Prod Outlier %"] - outlier_df["Ref Outlier %"]).round(2)

def _style_delta(val):
    if val > 2:
        return "color:#fca5a5"
    if val > 0.5:
        return "color:#fcd34d"
    return "color:#86efac"

st.dataframe(
    outlier_df.style.applymap(_style_delta, subset=["Δ Outlier %"]),
    use_container_width=True, hide_index=True,
)

fig_out = go.Figure()
fig_out.add_trace(go.Bar(
    name="Reference", x=outlier_df["Feature"], y=outlier_df["Ref Outlier %"],
    marker_color=COLORS["reference"], opacity=0.8,
))
fig_out.add_trace(go.Bar(
    name="Production", x=outlier_df["Feature"], y=outlier_df["Prod Outlier %"],
    marker_color=COLORS["production"], opacity=0.8,
))
fig_out.update_layout(
    barmode="group", title=f"Outlier Rates by Feature (|z| > {zscore_thresh})",
    yaxis_title="Outlier %", template="plotly_dark",
    height=360, margin=dict(l=50, r=20, t=60, b=60),
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_out, use_container_width=True)

st.divider()

# ── Missing Values ────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Missing Value Analysis</p>', unsafe_allow_html=True)

mv_ref  = ref_df.isnull().sum()
mv_prod = prod_df.isnull().sum()
mv_df   = pd.DataFrame({
    "Feature":           ref_df.columns,
    "Ref Missing":       mv_ref.values,
    "Prod Missing":      mv_prod.values,
    "Ref Missing %":     (mv_ref  / len(ref_df)  * 100).round(3).values,
    "Prod Missing %":    (mv_prod / len(prod_df) * 100).round(3).values,
})
st.dataframe(mv_df, use_container_width=True, hide_index=True)

total_mv = mv_ref.sum() + mv_prod.sum()
if total_mv == 0:
    st.success("✅ No missing values detected in either dataset.")
else:
    st.warning(f"⚡ {total_mv} missing values detected across both datasets.")

st.divider()

# ── Pairwise Feature Importance by Category ───────────────────────────────────
st.markdown('<p class="section-title">Value Distribution by Categorical Group</p>',
            unsafe_allow_html=True)

cat_col = st.selectbox("Group by:", CATEGORICAL_FEATURES, key="cat_group")
num_col = st.selectbox("Numeric feature:", NUMERIC_FEATURES, key="num_grp")

ref_df["Dataset"]  = "Reference"
prod_df["Dataset"] = "Production"
viz_df = pd.concat([
    ref_df[[cat_col, num_col, "Dataset"]],
    prod_df[[cat_col, num_col, "Dataset"]],
], ignore_index=True)

fig_box_cat = px.box(
    viz_df, x=cat_col, y=num_col, color="Dataset",
    color_discrete_map={"Reference": COLORS["reference"], "Production": COLORS["production"]},
    template="plotly_dark",
    title=f"{num_col} Distribution by {cat_col}",
)
fig_box_cat.update_layout(height=420, margin=dict(l=60, r=20, t=60, b=60))
st.plotly_chart(fig_box_cat, use_container_width=True)

st.divider()
st.caption("Feature Analysis · CreditGuard ML Monitoring Platform")

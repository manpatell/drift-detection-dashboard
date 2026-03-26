"""
Plotly visualisation helpers for CreditGuard ML Monitoring Platform.
All charts use plotly_dark theme for a consistent professional look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

from config import COLORS, THRESHOLDS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _psi_color(val: float) -> str:
    if val >= THRESHOLDS["psi"]["warning"]:
        return COLORS["critical"]
    if val >= THRESHOLDS["psi"]["stable"]:
        return COLORS["warning"]
    return COLORS["stable"]


# ─── Distribution Charts ──────────────────────────────────────────────────────

def distribution_plot(
    ref_data:  np.ndarray,
    prod_data: np.ndarray,
    feature_name: str,
    bins: int = 50,
) -> go.Figure:
    """Overlapping density histogram — Reference vs Production."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ref_data,  name="Reference", nbinsx=bins,
        marker_color=COLORS["reference"], opacity=0.65, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=prod_data, name="Production", nbinsx=bins,
        marker_color=COLORS["production"], opacity=0.65, histnorm="probability density",
    ))
    fig.update_layout(
        barmode="overlay", title=f"{feature_name}",
        xaxis_title=feature_name, yaxis_title="Density",
        template="plotly_dark", legend=dict(x=0.72, y=0.95),
        height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def box_plot_comparison(
    ref_data: np.ndarray,
    prod_data: np.ndarray,
    feature_name: str,
) -> go.Figure:
    """Side-by-side box plots."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=ref_data,  name="Reference",
        marker_color=COLORS["reference"], boxmean="sd", line_width=1.5,
    ))
    fig.add_trace(go.Box(
        y=prod_data, name="Production",
        marker_color=COLORS["production"], boxmean="sd", line_width=1.5,
    ))
    fig.update_layout(
        title=f"Box Plot — {feature_name}", template="plotly_dark",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def categorical_bar_comparison(distribution: Dict, feature_name: str) -> go.Figure:
    """Grouped bar chart for categorical distribution comparison."""
    cats     = list(distribution["reference"].keys())
    ref_vals = [distribution["reference"].get(c,  0) for c in cats]
    prd_vals = [distribution["production"].get(c, 0) for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Reference", x=cats, y=ref_vals,
        marker_color=COLORS["reference"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Production", x=cats, y=prd_vals,
        marker_color=COLORS["production"], opacity=0.85,
    ))
    fig.update_layout(
        barmode="group", title=feature_name,
        yaxis_title="Share (%)", template="plotly_dark",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─── Drift Summary Charts ─────────────────────────────────────────────────────

def psi_bar_chart(df_psi: pd.DataFrame) -> go.Figure:
    """Bar chart of PSI scores with reference threshold lines."""
    colors = [_psi_color(v) for v in df_psi["PSI"]]

    fig = go.Figure(go.Bar(
        x=df_psi["Feature"], y=df_psi["PSI"],
        marker_color=colors,
        text=df_psi["PSI"].round(3), textposition="outside",
        hovertemplate="<b>%{x}</b><br>PSI: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(
        y=THRESHOLDS["psi"]["warning"], line_dash="dash",
        line_color=COLORS["critical"], line_width=1.5,
        annotation_text="High drift threshold (0.20)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=THRESHOLDS["psi"]["stable"], line_dash="dot",
        line_color=COLORS["warning"], line_width=1.5,
        annotation_text="Moderate drift threshold (0.10)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Population Stability Index (PSI) by Feature",
        xaxis_title="Feature", yaxis_title="PSI Score",
        template="plotly_dark", showlegend=False,
        height=420, margin=dict(l=40, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )
    return fig


def drift_status_heatmap(drift_report: Dict) -> go.Figure:
    """Horizontal status bar — coloured by drift level."""
    status_map = {"stable": 0, "warning": 1, "critical": 2}
    colour_map = {0: COLORS["stable"], 1: COLORS["warning"], 2: COLORS["critical"]}
    label_map  = {0: "Stable", 1: "Warning", 2: "Critical"}

    features, statuses = [], []
    for feat, m in {**drift_report["numeric"], **drift_report["categorical"]}.items():
        features.append(feat)
        statuses.append(status_map[m["status"]])

    fig = go.Figure(go.Bar(
        x=features, y=[1] * len(features),
        marker_color=[colour_map[s] for s in statuses],
        text=[label_map[s] for s in statuses],
        textposition="inside", textfont_size=11,
        hovertemplate="<b>%{x}</b><br>Status: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Drift Status at a Glance",
        template="plotly_dark", showlegend=False,
        yaxis=dict(showticklabels=False, showgrid=False),
        height=160, margin=dict(l=20, r=20, t=50, b=60),
        xaxis_tickangle=-30,
    )
    return fig


# ─── Time Series ──────────────────────────────────────────────────────────────

def time_series_chart(
    ts_df:  pd.DataFrame,
    metric: str,
    title:  str,
    show_thresholds: bool = True,
) -> go.Figure:
    """Line chart with optional threshold reference lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["date"], y=ts_df[metric],
        mode="lines+markers", name=metric,
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=4),
        hovertemplate="%{x|%b %d}<br>" + metric + ": %{y:.4f}<extra></extra>",
    ))

    if show_thresholds:
        if "psi" in metric or "ks_" in metric:
            fig.add_hline(y=0.20, line_dash="dash", line_color=COLORS["critical"],
                          annotation_text="Critical (0.20)")
            fig.add_hline(y=0.10, line_dash="dot",  line_color=COLORS["warning"],
                          annotation_text="Warning (0.10)")
        elif metric in ("accuracy", "f1_score", "auc_roc", "precision", "recall"):
            fig.add_hline(y=0.85, line_dash="dash", line_color=COLORS["warning"],
                          annotation_text="SLA floor (0.85)")

    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Date", yaxis_title=metric.replace("_", " ").title(),
        height=340, margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def multi_metric_time_series(ts_df: pd.DataFrame, metrics: List[str], title: str) -> go.Figure:
    """Overlay multiple metrics on a single time-series chart."""
    palette = [COLORS["primary"], COLORS["stable"], COLORS["warning"],
               COLORS["accent"], COLORS["reference"]]
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=ts_df["date"], y=ts_df[m], mode="lines",
            name=m.replace("_", " ").title(),
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Date", height=360,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ─── Model Performance ────────────────────────────────────────────────────────

def gauge_chart(
    value: float,
    min_val: float,
    max_val: float,
    title: str,
    warn_threshold: float = 0.85,
    crit_threshold: float = 0.75,
    fmt: str = ".2%",
) -> go.Figure:
    """Gauge chart for a single KPI value."""
    if value >= warn_threshold:
        bar_color = COLORS["stable"]
    elif value >= crit_threshold:
        bar_color = COLORS["warning"]
    else:
        bar_color = COLORS["critical"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": fmt, "font": {"size": 26}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickformat": fmt},
            "bar":  {"color": bar_color, "thickness": 0.25},
            "steps": [
                {"range": [min_val,        crit_threshold], "color": "#2D1B1B"},
                {"range": [crit_threshold, warn_threshold], "color": "#2D2510"},
                {"range": [warn_threshold, max_val],        "color": "#0F2D1B"},
            ],
            "threshold": {
                "line":  {"color": COLORS["warning"], "width": 3},
                "value": warn_threshold,
                "thickness": 0.75,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        template="plotly_dark", height=240,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


def confusion_matrix_plot(cm: list, title: str = "Confusion Matrix") -> go.Figure:
    """Annotated heatmap for a 2×2 confusion matrix."""
    z     = np.array(cm)
    total = z.sum()
    pct   = z / total * 100
    text  = [[f"{z[r][c]}<br>({pct[r][c]:.1f}%)" for c in range(2)] for r in range(2)]

    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted: No Default", "Predicted: Default"],
        y=["Actual: No Default",   "Actual: Default"],
        colorscale="RdYlGn", reversescale=True,
        text=text, texttemplate="%{text}", textfont={"size": 14},
        showscale=False,
        hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        height=320, margin=dict(l=100, r=20, t=60, b=70),
    )
    return fig


def roc_curve_plot(
    ref_roc:  Dict,
    prod_roc: Dict,
    ref_auc:  float,
    prod_auc: float,
) -> go.Figure:
    """ROC curve comparison — reference vs production."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ref_roc["fpr"],  y=ref_roc["tpr"],
        mode="lines", name=f"Reference (AUC={ref_auc:.4f})",
        line=dict(color=COLORS["reference"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=prod_roc["fpr"], y=prod_roc["tpr"],
        mode="lines", name=f"Production (AUC={prod_auc:.4f})",
        line=dict(color=COLORS["production"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color=COLORS["neutral"], width=1, dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve — Reference vs Production",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template="plotly_dark", height=400,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def feature_importance_chart(importance: Dict, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for feature importances."""
    sorted_items = sorted(importance.items(), key=lambda x: x[1])
    features = [i[0] for i in sorted_items]
    values   = [i[1] for i in sorted_items]

    norm_values = np.array(values) / (max(values) + 1e-9)
    colors = [
        COLORS["critical"] if v > 0.7 else
        COLORS["warning"]  if v > 0.4 else
        COLORS["primary"]
        for v in norm_values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Importance Score",
        height=420, margin=dict(l=160, r=60, t=60, b=40),
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    """Correlation matrix heatmap."""
    corr  = df.corr(numeric_only=True)
    feats = list(corr.columns)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=feats, y=feats,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}", textfont_size=9,
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        height=500, margin=dict(l=120, r=20, t=60, b=80),
        xaxis_tickangle=-30,
    )
    return fig


def prediction_score_distribution(
    ref_probs:  np.ndarray,
    prod_probs: np.ndarray,
    bins: int = 40,
) -> go.Figure:
    """Predicted probability distribution — reference vs production."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ref_probs,  name="Reference", nbinsx=bins,
        marker_color=COLORS["reference"], opacity=0.65, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=prod_probs, name="Production", nbinsx=bins,
        marker_color=COLORS["production"], opacity=0.65, histnorm="probability density",
    ))
    fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Decision threshold (0.35)")
    fig.update_layout(
        barmode="overlay",
        title="Predicted Default Probability Distribution",
        xaxis_title="P(Default)", yaxis_title="Density",
        template="plotly_dark",
        height=360, margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig

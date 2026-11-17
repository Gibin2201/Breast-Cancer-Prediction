"""
Visualization Module for Breast Cancer Prediction System.
Generates responsive, interactive Plotly visualizations for EDA, model benchmarking,
confusion matrices, ROC curves, and radar charts matching the Healthcare SaaS palette:
- Primary: Deep Navy (#142D4E)
- Secondary: Medical Blue (#2878C7)
- Accent: Teal (#16A6A1)
- Benign: Clinical Green (#10B981)
- Malignant: Warning Red (#E63946)
- Background: #FFFFFF / #F5F7FA
- Text: #253247
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import (
    THEME_COLORS,
    FEATURE_DISPLAY_NAMES,
    TARGET_COLUMN,
)


def _apply_theme(fig: go.Figure, title: str = "", height: int = 460) -> go.Figure:
    """Apply consistent healthcare SaaS typography and theme styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 17, "family": "Inter, system-ui, sans-serif", "color": "#142D4E"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin={"l": 55, "r": 35, "t": 65, "b": 55},
        font={"family": "Inter, system-ui, sans-serif", "color": "#253247", "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": "#253247"},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(size=11, color="#253247"),
        title_font=dict(size=12, color="#142D4E"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(size=11, color="#253247"),
        title_font=dict(size=12, color="#142D4E"),
    )
    return fig


def plot_diagnosis_distribution(df: pd.DataFrame) -> go.Figure:
    """Interactive Donut chart for Benign vs Malignant cases with center KPI annotation."""
    target_series = df[TARGET_COLUMN]

    if target_series.dtype == object:
        counts = target_series.replace({"B": "Benign", "M": "Malignant"}).value_counts()
    else:
        counts = target_series.replace({0: "Benign", 1: "Malignant"}).value_counts()

    labels = counts.index.tolist()
    values = counts.values.tolist()

    colors = [
        THEME_COLORS["benign"] if l == "Benign" else THEME_COLORS["malignant"]
        for l in labels
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
                textinfo="label+percent+value",
                textfont=dict(size=13, color="#FFFFFF", family="Inter"),
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text=f"<span style='font-size:12px;color:#5A6E85;'>TOTAL COHORT</span><br><b style='font-size:24px;color:#142D4E;'>{len(df)}</b>",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        ]
    )
    return _apply_theme(fig, "Tumor Biopsy Diagnosis Distribution", height=380)


def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None) -> go.Figure:
    """Generate interactive correlation heatmap using the clinical divergent palette."""
    if features is None:
        features = [col for col in df.columns if col.startswith("mean ")]

    corr = df[features].corr()
    clean_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale=[
                [0.0, "#142D4E"],   # Deep Navy
                [0.5, "#FFFFFF"],   # White neutral
                [1.0, "#16A6A1"],   # Teal
            ],
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10, "color": "#253247", "family": "Inter"},
            colorbar=dict(title="Pearson r", thickness=14, tickfont=dict(size=11, color="#253247")),
        )
    )
    return _apply_theme(fig, "Feature Pearson Correlation Matrix", height=520)


def plot_feature_distribution(df: pd.DataFrame, feature_col: str) -> go.Figure:
    """KDE / Histogram comparison between Benign and Malignant for a given feature."""
    df_copy = df.copy()
    if df_copy[TARGET_COLUMN].dtype == object:
        df_copy["Diagnosis"] = df_copy[TARGET_COLUMN].replace({"B": "Benign", "M": "Malignant"})
    else:
        df_copy["Diagnosis"] = df_copy[TARGET_COLUMN].replace({0: "Benign", 1: "Malignant"})

    display_name = FEATURE_DISPLAY_NAMES.get(feature_col, feature_col)

    fig = px.histogram(
        df_copy,
        x=feature_col,
        color="Diagnosis",
        marginal="box",
        barmode="overlay",
        opacity=0.68,
        color_discrete_map={
            "Benign": THEME_COLORS["benign"],
            "Malignant": THEME_COLORS["malignant"],
        },
        labels={feature_col: display_name},
    )
    return _apply_theme(fig, f"Morphological Distribution: {display_name}", height=420)


def plot_box_comparison(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Multi-feature box plot comparisons grouped by tumor diagnosis."""
    df_copy = df.copy()
    if df_copy[TARGET_COLUMN].dtype == object:
        df_copy["Diagnosis"] = df_copy[TARGET_COLUMN].replace({"B": "Benign", "M": "Malignant"})
    else:
        df_copy["Diagnosis"] = df_copy[TARGET_COLUMN].replace({0: "Benign", 1: "Malignant"})

    fig = go.Figure()
    for feat in features:
        disp_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        for diag, color in [("Benign", THEME_COLORS["benign"]), ("Malignant", THEME_COLORS["malignant"])]:
            subset = df_copy[df_copy["Diagnosis"] == diag][feat]
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=f"{disp_name} ({diag[:3]})",
                    marker_color=color,
                    boxmean=True,
                )
            )

    return _apply_theme(fig, "Cell Nucleus Morphological Variance (Box Plots)", height=450)


def plot_scatter_features(
    df: pd.DataFrame, feature_x: str, feature_y: str, feature_size: Optional[str] = None
) -> go.Figure:
    """2D Scatter plot showing class separation across key features."""
    df_copy = df.copy()
    if df_copy[TARGET_COLUMN].dtype == object:
        df_copy["Diagnosis"] = df_copy[TARGET_COLUMN].replace({"B": "Benign", "M": "Malignant"})
    else:
        df_copy["Diagnosis"] = df_copy[TARGET_COLUMN].replace({0: "Benign", 1: "Malignant"})

    disp_x = FEATURE_DISPLAY_NAMES.get(feature_x, feature_x)
    disp_y = FEATURE_DISPLAY_NAMES.get(feature_y, feature_y)

    fig = px.scatter(
        df_copy,
        x=feature_x,
        y=feature_y,
        color="Diagnosis",
        size=feature_size,
        color_discrete_map={
            "Benign": THEME_COLORS["benign"],
            "Malignant": THEME_COLORS["malignant"],
        },
        labels={feature_x: disp_x, feature_y: disp_y},
        hover_data=[c for c in df_copy.columns[:4]],
    )
    fig.update_traces(marker=dict(line=dict(width=0.8, color="#FFFFFF"), opacity=0.85))
    return _apply_theme(fig, f"Bi-Variate Boundary Separation: {disp_x} vs. {disp_y}", height=450)


def plot_model_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    """Multi-metric grouped bar chart comparing all 5 algorithms side by side."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    df_melt = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    palette = [
        "#142D4E",  # Deep Navy
        "#2878C7",  # Medical Blue
        "#16A6A1",  # Teal Accent
        "#8B5CF6",  # Purple
        "#10B981",  # Emerald Green
    ]

    fig = px.bar(
        df_melt,
        x="Algorithm",
        y="Score",
        color="Metric",
        barmode="group",
        text=df_melt["Score"].apply(lambda v: f"{v:.1%}"),
        color_discrete_sequence=palette,
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10, family="Inter"))
    fig.update_yaxes(range=[0.75, 1.05])
    return _apply_theme(fig, "Algorithm Benchmark Comparison (Unseen Test Evaluation)", height=450)


def plot_model_radar_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing models across 5 evaluation dimensions."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    fig = go.Figure()

    colors = ["#2878C7", "#16A6A1", "#142D4E", "#8B5CF6", "#F59E0B"]

    for idx, row in comparison_df.iterrows():
        values = [row[m] for m in available_metrics]
        # Close the loop
        values.append(values[0])
        radar_metrics = available_metrics + [available_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_metrics,
                fill="toself",
                name=row["Algorithm"],
                line=dict(color=colors[idx % len(colors)], width=2),
                opacity=0.55,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.85, 1.0], tickfont=dict(size=10, color="#253247")),
            angularaxis=dict(tickfont=dict(size=12, color="#142D4E", family="Inter")),
        ),
        showlegend=True,
        paper_bgcolor="#FFFFFF",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        title=dict(
            text="<b>Multi-Metric Radar Comparison</b>",
            font=dict(size=17, family="Inter", color="#142D4E"),
            x=0.02,
        ),
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm_data: Dict[str, Any], model_name: str = "Support Vector Machine"
) -> go.Figure:
    """Confusion matrix heatmap with detailed clinical diagnostic annotations."""
    matrix = cm_data["matrix"]
    z = matrix
    x = ["Predicted Benign", "Predicted Malignant"]
    y = ["Actual Benign", "Actual Malignant"]

    annotations = [
        [
            f"<b>TN = {matrix[0][0]}</b><br><span style='font-size:11px;color:#5A6E85;'>Correct Benign</span>",
            f"<b>FP = {matrix[0][1]}</b><br><span style='font-size:11px;color:#E63946;'>Type I Error</span>",
        ],
        [
            f"<b>FN = {matrix[1][0]}</b><br><span style='font-size:11px;color:#E63946;'>Critical Miss (Type II)</span>",
            f"<b>TP = {matrix[1][1]}</b><br><span style='font-size:11px;color:#0E7055;'>Correct Malignant</span>",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0.0, "#F5F7FA"], [0.5, "#BCE0FD"], [1.0, "#142D4E"]],
            showscale=False,
        )
    )

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=x[j],
                y=y[i],
                text=annotations[i][j],
                showarrow=False,
                font=dict(
                    size=15,
                    color="#142D4E" if z[i][j] < (np.max(z) * 0.5) else "#FFFFFF",
                    family="Inter",
                ),
            )

    return _apply_theme(fig, f"Clinical Confusion Matrix: {model_name}", height=390)


def plot_roc_curves(roc_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot multi-model ROC curves with baseline random chance line."""
    fig = go.Figure()

    # Diagonal chance baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#CBD5E1", dash="dash", width=1.5),
            name="Chance Baseline (AUC = 0.50)",
        )
    )

    colors = [
        "#142D4E",  # Deep Navy
        "#2878C7",  # Medical Blue
        "#16A6A1",  # Teal Accent
        "#8B5CF6",  # Purple
        "#E63946",  # Warning Red
    ]

    for idx, (algo_name, curve_info) in enumerate(roc_dict.items()):
        fpr = curve_info["fpr"]
        tpr = curve_info["tpr"]
        auc_val = curve_info.get("auc", 0.0)
        color = colors[idx % len(colors)]
        width = 3.0 if "Support Vector" in algo_name else 1.8

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color=color, width=width),
                name=f"{algo_name} (AUC = {auc_val:.4f})",
            )
        )

    fig.update_xaxes(title="False Positive Rate (1 - Specificity)", range=[-0.01, 1.01])
    fig.update_yaxes(title="True Positive Rate (Sensitivity / Recall)", range=[-0.01, 1.05])

    return _apply_theme(fig, "Receiver Operating Characteristic (ROC) Benchmark", height=450)


def plot_prediction_gauge(benign_prob: float, malignant_prob: float) -> go.Figure:
    """Indicator Gauge chart visualizing model classification probability."""
    bar_color = "#E63946" if malignant_prob >= 0.5 else "#10B981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=malignant_prob * 100,
            title={"text": "<b>Malignancy Risk Probability</b>", "font": {"size": 17, "color": "#142D4E", "family": "Inter"}},
            number={"suffix": "%", "font": {"size": 38, "color": "#142D4E", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1.5, "tickcolor": "#5A6E85", "tickfont": {"size": 11, "color": "#5A6E85"}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 35], "color": "#E8F8F5"},
                    {"range": [35, 65], "color": "#FEF9C3"},
                    {"range": [65, 100], "color": "#FEE8E8"},
                ],
                "threshold": {
                    "line": {"color": "#921E28", "width": 4},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=270, margin={"l": 25, "r": 25, "t": 45, "b": 25}, paper_bgcolor="#FFFFFF")
    return fig

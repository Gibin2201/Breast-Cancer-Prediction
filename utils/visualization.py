"""
Visualization Module for Breast Cancer Prediction.
Generates responsive, interactive Plotly visualizations for EDA, model comparison,
confusion matrices, and ROC curves using the healthcare-inspired color palette.
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


# Healthcare High-Contrast Theme Plotly Template
def _apply_theme(fig: go.Figure, title: str = "", height: int = 460) -> go.Figure:
    """Apply consistent high-contrast typography and theme styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 18, "family": "Inter, system-ui, sans-serif", "color": "#0f172a"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=height,
        margin={"l": 55, "r": 35, "t": 65, "b": 55},
        font={"family": "Inter, system-ui, sans-serif", "color": "#0f172a", "size": 13},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 13, "color": "#0f172a"},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1.5,
        gridcolor="#e2e8f0",
        zeroline=False,
        tickfont=dict(size=12, color="#0f172a"),
        title_font=dict(size=13, color="#0f172a"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1.5,
        gridcolor="#e2e8f0",
        zeroline=False,
        tickfont=dict(size=12, color="#0f172a"),
        title_font=dict(size=13, color="#0f172a"),
    )
    return fig


def plot_diagnosis_distribution(df: pd.DataFrame) -> go.Figure:
    """Interactive Donut and Bar chart for Benign vs Malignant cases."""
    target_series = df[TARGET_COLUMN]
    
    # Standardize labels
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
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent+value",
                textfont=dict(size=14, color="#ffffff"),
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text=f"Total<br><b>{len(df)}</b>",
                x=0.5,
                y=0.5,
                font_size=18,
                showarrow=False,
                font_color=THEME_COLORS["navy"],
            )
        ]
    )
    return _apply_theme(fig, "Tumor Diagnosis Distribution (WDBC Cohort)", height=380)


def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None) -> go.Figure:
    """Generate interactive correlation heatmap for selected feature subset."""
    if features is None:
        # Default to 10 mean features for readability
        features = [col for col in df.columns if col.startswith("mean ")]

    corr = df[features].corr()
    clean_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale=[
                [0.0, "#1e3a8a"],
                [0.5, "#ffffff"],
                [1.0, "#0f766e"],
            ],
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#0f172a"},
            colorbar=dict(title="Correlation", thickness=16, tickfont=dict(size=12, color="#0f172a")),
        )
    )
    return _apply_theme(fig, "Feature Pearson Correlation Heatmap", height=540)


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
        opacity=0.65,
        color_discrete_map={
            "Benign": THEME_COLORS["benign"],
            "Malignant": THEME_COLORS["malignant"],
        },
        labels={feature_col: display_name},
    )
    return _apply_theme(fig, f"Distribution: {display_name} by Diagnosis", height=420)


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
                    name=f"{disp_name} ({diag})",
                    marker_color=color,
                    boxmean=True,
                )
            )

    return _apply_theme(fig, "Tumor Morphological Variations (Box Plots)", height=450)


def plot_scatter_features(
    df: pd.DataFrame, feature_x: str, feature_y: str, feature_size: Optional[str] = None
) -> go.Figure:
    """2D / 3D Scatter plot showing class separation across key features."""
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
        hover_data=df_copy.columns[:5],
    )
    return _apply_theme(fig, f"Bi-variate Separation: {disp_x} vs. {disp_y}", height=450)


def plot_model_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    """Multi-metric bar chart comparing all 5 algorithms side by side."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    df_melt = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    palette = [
        THEME_COLORS["primary"],
        THEME_COLORS["secondary"],
        "#8b5cf6",  # Purple
        "#f59e0b",  # Amber
        THEME_COLORS["benign"],
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
    fig.update_yaxes(range=[0.75, 1.02])
    return _apply_theme(fig, "Algorithm Performance Comparison (Test Evaluation)", height=450)


def plot_confusion_matrix_heatmap(
    cm_data: Dict[str, Any], model_name: str = "Best Model"
) -> go.Figure:
    """Confusion matrix heatmap with True Negative, False Positive, FN, TP."""
    matrix = cm_data["matrix"]
    z = matrix
    x = ["Predicted Benign", "Predicted Malignant"]
    y = ["Actual Benign", "Actual Malignant"]

    annotations = [
        [
            f"<b>TN</b>: {matrix[0][0]}<br>(Correct Benign)",
            f"<b>FP</b>: {matrix[0][1]}<br>(Type I Error)",
        ],
        [
            f"<b>FN</b>: {matrix[1][0]}<br>(Critical Miss)",
            f"<b>TP</b>: {matrix[1][1]}<br>(Correct Malignant)",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0.0, "#f8fafc"], [0.5, "#93c5fd"], [1.0, "#1e3a8a"]],
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
                font=dict(size=16, color="#0f172a" if z[i][j] < (np.max(z) * 0.5) else "#ffffff"),
            )

    return _apply_theme(fig, f"Confusion Matrix: {model_name}", height=400)


def plot_roc_curves(roc_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot multi-model ROC curves with baseline random chance line."""
    fig = go.Figure()

    # Diagonal baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash"),
            name="Chance Baseline (AUC = 0.50)",
        )
    )

    colors = [
        THEME_COLORS["primary"],
        THEME_COLORS["secondary"],
        "#e11d48",  # Rose
        "#8b5cf6",  # Violet
        "#10b981",  # Emerald
    ]

    for idx, (algo_name, curve_info) in enumerate(roc_dict.items()):
        fpr = curve_info["fpr"]
        tpr = curve_info["tpr"]
        auc_val = curve_info.get("auc", 0.0)
        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{algo_name} (AUC = {auc_val:.3f})",
            )
        )

    fig.update_xaxes(title="False Positive Rate (1 - Specificity)", range=[-0.01, 1.01])
    fig.update_yaxes(title="True Positive Rate (Sensitivity / Recall)", range=[-0.01, 1.05])

    return _apply_theme(fig, "Receiver Operating Characteristic (ROC) Curves", height=450)


def plot_prediction_gauge(benign_prob: float, malignant_prob: float) -> go.Figure:
    """Indicator Gauge chart visualizing model classification probability with high contrast."""
    bar_color = "#dc2626" if malignant_prob >= 0.5 else "#16a34a"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=malignant_prob * 100,
            title={"text": "<b>Malignancy Probability (%)</b>", "font": {"size": 18, "color": "#0f172a"}},
            number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#0f172a", "tickfont": {"size": 13, "color": "#0f172a"}},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#ffffff",
                "borderwidth": 2,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#b91c1c", "width": 5},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=280, margin={"l": 25, "r": 25, "t": 45, "b": 25}, paper_bgcolor="#ffffff")
    return fig

"""
Page 3: Exploratory Data Analysis (EDA) & Morphological Visualization.
Provides interactive Plotly visualizations: Pearson correlation matrix,
morphological kernel density comparisons, violin/box plots, and bi-variate scatter separation.
"""

import streamlit as st
import pandas as pd
from typing import List

from config import (
    TARGET_COLUMN,
    FEATURE_NAMES,
    MEAN_FEATURES,
    FEATURE_DISPLAY_NAMES,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    render_website_footer,
)
from utils.data_loader import load_dataset
from utils.visualization import (
    plot_diagnosis_distribution,
    plot_correlation_heatmap,
    plot_feature_distribution,
    plot_box_comparison,
    plot_scatter_features,
)


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Dataset Analysis (EDA)",
        category="Exploratory Analysis",
        active_badge="Plotly Interactive Analytics",
    )

    render_header(
        title="Exploratory Data Analysis (EDA)",
        subtitle="Interactive Correlation Heatmaps, Morphological Distributions, Box Variations, and Bi-Variate Separation",
        badge="Exploratory Data Science",
    )

    # Load dataset
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Error loading clinical dataset: {e}")
        return

    # Tabs for EDA Sections
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "🔥 Pearson Correlation Matrix",
        "📊 Morphological Distributions",
        "📦 Box & Variance Comparisons",
        "📈 Bi-Variate Scatter Separation",
    ])

    # TAB 1: Correlation Matrix
    with eda_tab1:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>🔥</span> Pearson Correlation Matrix Heatmap
                </div>
                <p style="font-size: 0.90rem; color: #5A6E85; margin-bottom: 8px;">
                    Features with high mutual correlation (> 0.90) indicate geometric collinearity (e.g., radius, perimeter, and area).
                    Select which feature vector group to inspect:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        corr_group = st.radio(
            "Select Feature Subset:",
            options=["10 Mean Nuclear Features", "10 Standard Error Features", "10 Worst Nuclear Features", "All 30 Features"],
            horizontal=True,
        )

        if "mean" in corr_group.lower():
            selected_features = [c for c in df.columns if c.startswith("mean ")]
        elif "error" in corr_group.lower():
            selected_features = [c for c in df.columns if "error" in c]
        elif "worst" in corr_group.lower():
            selected_features = [c for c in df.columns if c.startswith("worst ")]
        else:
            selected_features = FEATURE_NAMES

        fig_corr = plot_correlation_heatmap(df, selected_features)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.info(
            "💡 **Key Collinearity Observation:** Mean Radius, Mean Perimeter, and Mean Area have near-perfect correlation (r = 0.98 - 0.99). "
            "Algorithms like SVM and Logistic Regression benefit significantly from **StandardScaler** to normalize these high-variance dimensions."
        )

    # TAB 2: Morphological Distributions
    with eda_tab2:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>📊</span> Benign vs Malignant Feature Kernel Density & Histogram
                </div>
                <p style="font-size: 0.90rem; color: #5A6E85; margin-bottom: 8px;">
                    Inspect the distribution overlap between Benign (Green) and Malignant (Red) tumors across any of the 30 features:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_feat = st.selectbox(
            "Select Feature to Visualize Distribution:",
            options=FEATURE_NAMES,
            format_func=lambda f: f"{FEATURE_DISPLAY_NAMES.get(f, f)} ({f})",
            index=0,
        )

        fig_dist = plot_feature_distribution(df, selected_feat)
        st.plotly_chart(fig_dist, use_container_width=True)

    # TAB 3: Box Comparisons
    with eda_tab3:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>📦</span> Multi-Feature Morphological Variance (Box Plots)
                </div>
                <p style="font-size: 0.90rem; color: #5A6E85; margin-bottom: 8px;">
                    Compare interquartile ranges, medians, and pathological outliers across major morphological categories:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        box_preset = st.selectbox(
            "Select Comparison Category:",
            options=[
                "Nuclear Core Size (Radius, Texture, Perimeter)",
                "Boundary Complexity (Smoothness, Compactness, Concavity)",
                "Pathological Extremes (Worst Radius, Worst Texture, Worst Perimeter)",
            ],
        )

        if "Core Size" in box_preset:
            feats = ["mean radius", "mean texture", "mean perimeter"]
        elif "Boundary" in box_preset:
            feats = ["mean smoothness", "mean compactness", "mean concavity"]
        else:
            feats = ["worst radius", "worst texture", "worst perimeter"]

        fig_box = plot_box_comparison(df, feats)
        st.plotly_chart(fig_box, use_container_width=True)

    # TAB 4: Bi-Variate Scatter Separation
    with eda_tab4:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>📈</span> Bi-Variate Diagnostic Boundary Separation
                </div>
                <p style="font-size: 0.90rem; color: #5A6E85; margin-bottom: 8px;">
                    Select any two features to project tumor biopsies into a 2D clinical coordinate plane and visualize decision boundaries:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            x_feat = st.selectbox("X-Axis Dimension:", options=FEATURE_NAMES, index=0)
        with sc_col2:
            y_feat = st.selectbox("Y-Axis Dimension:", options=FEATURE_NAMES, index=3)

        fig_scatter = plot_scatter_features(df, x_feat, y_feat)
        st.plotly_chart(fig_scatter, use_container_width=True)

    render_website_footer()


if __name__ == "__main__":
    main()

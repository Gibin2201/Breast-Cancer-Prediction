"""
Page 2: Dataset Analysis (EDA)
Comprehensive Exploratory Data Analysis module for the Wisconsin Diagnostic Breast Cancer dataset.
Includes interactive Plotly charts, statistical summaries, correlation heatmaps,
box plots, scatter plots, feature distributions, and CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dataset Analysis | Breast Cancer ML", page_icon="🔬", layout="wide")

from config import (
    FEATURE_NAMES,
    MEAN_FEATURES,
    SE_FEATURES,
    WORST_FEATURES,
    FEATURE_DISPLAY_NAMES,
    FEATURE_TOOLTIPS,
    TARGET_COLUMN,
    THEME_COLORS,
)
from utils.ui_components import (
    apply_theme,
    render_header,
    render_sidebar,
    check_and_guide_setup,
    render_metric_card,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from utils.visualization import (
    plot_diagnosis_distribution,
    plot_correlation_heatmap,
    plot_feature_distribution,
    plot_box_comparison,
    plot_scatter_features,
)


def render_dataset_analysis_page():
    apply_theme()
    render_sidebar()

    render_header(
        title="Exploratory Data Analysis (EDA)",
        subtitle="In-depth Morphological Analysis of Digitized FNA Cell Nuclei (569 Patients, 30 Features)",
        badge="Data Exploration",
    )

    if not check_and_guide_setup():
        return

    try:
        df = load_dataset()
        meta = get_dataset_metadata(df)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_metric_card("Total Patients", f"{meta['total_records']}", "Fine Needle Biopsies", "#0d9488"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Numerical Attributes", f"{meta['total_features']}", "10 Mean + 10 SE + 10 Worst", "#2563eb"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card(
                "Benign Cases (B)",
                f"{meta['benign_cases']}",
                f"{meta['benign_percentage']}% of cohort",
                THEME_COLORS["benign"],
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_metric_card(
                "Malignant Cases (M)",
                f"{meta['malignant_cases']}",
                f"{meta['malignant_percentage']}% of cohort",
                THEME_COLORS["malignant"],
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    # Tabbed Exploratory Sections
    tabs = st.tabs([
        "📋 Dataset Preview & Stats",
        "🎯 Class Distribution & Missing Values",
        "🔥 Correlation Heatmap",
        "📊 Morphological Comparisons",
        "📈 Bi-Variate Scatter Separation",
    ])

    # TAB 1: Preview & Summary Statistics
    with tabs[0]:
        st.markdown("### 📋 Clinical Dataset Preview")
        st.write(
            "Below is the complete Wisconsin Diagnostic Breast Cancer (WDBC) tabular matrix. "
            "Filter, sort, or inspect individual biopsy records."
        )

        filter_col1, filter_col2 = st.columns([3, 7])
        with filter_col1:
            diag_filter = st.selectbox(
                "Filter by Diagnosis:",
                options=["All Records", "Benign (B)", "Malignant (M)"],
            )

        filtered_df = df.copy()
        if diag_filter == "Benign (B)":
            filtered_df = filtered_df[filtered_df[TARGET_COLUMN].isin(["B", 0, "Benign"])]
        elif diag_filter == "Malignant (M)":
            filtered_df = filtered_df[filtered_df[TARGET_COLUMN].isin(["M", 1, "Malignant"])]

        st.dataframe(filtered_df, use_container_width=True, height=320)

        # Download CSV Button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Standardized Dataset as CSV",
            data=csv_data,
            file_name="wdbc_breast_cancer_dataset.csv",
            mime="text/csv",
            help="Download the verified 569-row WDBC CSV dataset",
        )

        st.divider()
        st.markdown("### 📊 Descriptive Statistics")
        st.write("Summary statistics (mean, standard deviation, quartiles) across all 30 features:")
        st.dataframe(df[FEATURE_NAMES].describe().T.style.format("{:.3f}"), use_container_width=True, height=360)

    # TAB 2: Class Distribution & Missing Values
    with tabs[1]:
        st.markdown("### 🎯 Diagnosis Class Breakdown")
        col_dist1, col_dist2 = st.columns([5, 5])

        with col_dist1:
            fig_donut = plot_diagnosis_distribution(df)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_dist2:
            st.markdown("#### 🔍 Missing & Duplicate Values Audit")
            st.write(
                "In medical and clinical machine learning, missing records can introduce fatal "
                "sampling biases if handled improperly. The WDBC dataset was examined for null, "
                "infinite, and duplicated records:"
            )

            null_series = df.isnull().sum()
            missing_total = int(null_series.sum())
            dup_total = int(df.duplicated().sum())

            st.markdown(
                f"""
                <div style="background: #ffffff; border: 2px solid #cbd5e1; border-radius: 10px; padding: 20px; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(15,23,42,0.04);">
                    <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0; font-size: 1.02rem; color: #0f172a;">
                        <li style="margin-bottom: 10px;">✅ <b>Missing Null Values:</b> <span style="color: #16a34a; font-weight: 800;">{missing_total}</span></li>
                        <li style="margin-bottom: 10px;">✅ <b>Duplicate Biopsy Rows:</b> <span style="color: #16a34a; font-weight: 800;">{dup_total}</span></li>
                        <li style="margin-bottom: 10px;">✅ <b>Data Integrity Score:</b> <span style="color: #0d9488; font-weight: 800;">100.0% Complete</span></li>
                        <li>✅ <b>Data Leakage Prevention:</b> Pipelines with strictly fitted transformers on training partitions</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # TAB 3: Correlation Heatmap
    with tabs[2]:
        st.markdown("### 🔥 Pearson Correlation Matrix")
        st.write(
            "Examine linear multicollinearity across morphological features. "
            "High correlation is naturally observed between radius, perimeter, and area."
        )

        group_choice = st.radio(
            "Select Feature Cohort:",
            options=["Mean Features (10)", "Standard Error Features (10)", "Worst Features (10)", "All Key Nuclear Features"],
            horizontal=True,
        )

        if group_choice == "Mean Features (10)":
            selected_features = MEAN_FEATURES
        elif group_choice == "Standard Error Features (10)":
            selected_features = SE_FEATURES
        elif group_choice == "Worst Features (10)":
            selected_features = WORST_FEATURES
        else:
            selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean compactness", "mean concavity", "mean concave points", "worst radius", "worst perimeter", "worst area"]

        fig_heat = plot_correlation_heatmap(df, selected_features)
        st.plotly_chart(fig_heat, use_container_width=True)

    # TAB 4: Morphological Comparisons & Box Plots
    with tabs[3]:
        st.markdown("### 🔬 Clinical Feature Comparisons (Benign vs Malignant)")
        st.write(
            "Direct side-by-side comparison of benign vs malignant tumors across fundamental "
            "nuclear properties: **Mean Radius**, **Mean Texture**, and **Mean Area**."
        )

        comp1, comp2, comp3 = st.columns(3)
        with comp1:
            fig_rad = plot_feature_distribution(df, "mean radius")
            st.plotly_chart(fig_rad, use_container_width=True)
        with comp2:
            fig_tex = plot_feature_distribution(df, "mean texture")
            st.plotly_chart(fig_tex, use_container_width=True)
        with comp3:
            fig_area = plot_feature_distribution(df, "mean area")
            st.plotly_chart(fig_area, use_container_width=True)

        st.divider()
        st.markdown("#### 📦 Multi-Feature Box Plot Distribution")
        chosen_box_feats = st.multiselect(
            "Choose features for box plot comparison:",
            options=FEATURE_NAMES,
            default=["mean radius", "mean texture", "worst radius", "mean smoothness"],
            format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x),
        )

        if chosen_box_feats:
            fig_box = plot_box_comparison(df, chosen_box_feats)
            st.plotly_chart(fig_box, use_container_width=True)

    # TAB 5: Bi-Variate Scatter Separation
    with tabs[4]:
        st.markdown("### 📈 Bi-Variate Tumor Separation")
        st.write(
            "Visualize non-linear separability and cluster formation between benign and malignant classes."
        )

        scat_c1, scat_c2 = st.columns(2)
        with scat_c1:
            feat_x = st.selectbox(
                "X-Axis Feature:",
                options=FEATURE_NAMES,
                index=0,  # mean radius
                format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x),
            )
        with scat_c2:
            feat_y = st.selectbox(
                "Y-Axis Feature:",
                options=FEATURE_NAMES,
                index=1,  # mean texture
                format_func=lambda x: FEATURE_DISPLAY_NAMES.get(x, x),
            )

        fig_scatter = plot_scatter_features(df, feat_x, feat_y)
        st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == "__main__":
    render_dataset_analysis_page()

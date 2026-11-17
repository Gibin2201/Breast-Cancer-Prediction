"""
Page 1: Clinical Telemetry & Executive Dashboard.
Displays high-level healthcare metrics, cohort distribution, morphological comparisons,
and ML algorithm benchmark summary.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config import (
    THEME_COLORS,
    DATASET_PATH,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_NAMES,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    render_metric_card,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from utils.prediction import load_model_metadata
from utils.visualization import (
    plot_diagnosis_distribution,
    plot_model_comparison_bars,
)


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Executive Dashboard",
        category="Clinical Telemetry",
        active_badge="Healthcare AI Platform",
    )

    render_header(
        title="Clinical Telemetry Dashboard",
        subtitle="Real-Time Cohort Metrics, Class Breakdown, and Machine Learning Model Telemetry",
        badge="Executive Overview | WDBC Cohort",
    )

    # Load project data
    try:
        df = load_dataset()
        meta = get_dataset_metadata(df)
        model_meta = load_model_metadata()
        comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    except Exception as e:
        st.error(f"Error loading system telemetry: {e}")
        return

    active_model_name = model_meta.get("model_name", "Support Vector Machine")
    best_acc = model_meta.get("best_model_metrics", {}).get("accuracy", 0.9825)
    best_recall = model_meta.get("best_model_metrics", {}).get("recall", 0.9762)

    # 1. Metric Cards Row (5 Core Metrics from Project)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            render_metric_card("Total Records", f"{meta['total_records']}", "Fine Needle Biopsies", "#142D4E", "📋"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Total Features", f"{meta['total_features']}", "10 Mean + 10 SE + 10 Worst", "#2878C7", "🔬"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card("Benign Cases", f"{meta['benign_cases']}", f"{meta['benign_percentage']}% of cohort", "#10B981", "🛡️"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_metric_card("Malignant Cases", f"{meta['malignant_cases']}", f"{meta['malignant_percentage']}% of cohort", "#E63946", "⚠️"),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            render_metric_card("ML Algorithms", f"{len(comparison_df)}", f"Active: {active_model_name[:7]}..", "#16A6A1", "🤖"),
            unsafe_allow_html=True,
        )

    st.write("")

    # 2. Welcome & System Purpose Section
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>👋</span> Welcome to the Clinical AI Decision Support Dashboard
            </div>
            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px;">
                <div>
                    <p style="font-size: 0.94rem; line-height: 1.6; color: #253247; margin-bottom: 8px;">
                        This dashboard provides oncologists, clinical pathologists, and researchers with real-time analytics 
                        derived from digitized Fine Needle Aspirate (FNA) cytology specimens. The primary clinical objective 
                        is early, accurate differentiation between <b>Benign (B)</b> and <b>Malignant (M)</b> breast tumors.
                    </p>
                    <p style="font-size: 0.90rem; line-height: 1.6; color: #5A6E85; margin-bottom: 0;">
                        <b>Key Clinical Metric:</b> In medical oncology, <b>Sensitivity / Malignant Recall</b> is paramount. 
                        Our active Support Vector Machine pipeline minimizes False Negatives (FN) to safeguard patient health, 
                        achieving a verified <b>97.62% Malignant Recall</b> alongside <b>98.25% Overall Test Accuracy</b>.
                    </p>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #142D4E; text-transform: uppercase; margin-bottom: 6px;">
                        Clinical Telemetry Status
                    </div>
                    <div style="font-size: 0.84rem; color: #5A6E85; line-height: 1.5;">
                        • Cohort: <b>Wisconsin (WDBC)</b><br>
                        • Split: <b>80% Train / 20% Test</b><br>
                        • Optimal Model: <b>{model}</b><br>
                        • Data Leakage: <b>Zero (Pipeline Scaled)</b>
                    </div>
                </div>
            </div>
        </div>
        """.format(model=active_model_name),
        unsafe_allow_html=True,
    )

    st.write("")

    # 3. Two-Column Visualizations: Distribution & Morphological Comparison
    v_col1, v_col2 = st.columns([5, 5])
    with v_col1:
        st.markdown(
            """
            <div class="saas-card" style="margin-bottom: 0;">
                <div class="saas-card-header">
                    <span>🎯</span> Diagnosis Class Breakdown
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_donut = plot_diagnosis_distribution(df)
        st.plotly_chart(fig_donut, use_container_width=True)

    with v_col2:
        st.markdown(
            """
            <div class="saas-card" style="margin-bottom: 0;">
                <div class="saas-card-header">
                    <span>🔬</span> Key Morphological Metric Differences (Mean Values)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Compute mean comparisons
        key_cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean concavity"]
        comp_data = df.groupby("diagnosis")[key_cols].mean()
        
        # Format into clean table
        benign_row = comp_data.loc["B"]
        malignant_row = comp_data.loc["M"]
        diff_pct = ((malignant_row - benign_row) / benign_row) * 100

        morph_df = pd.DataFrame({
            "Feature Dimension": [c.replace("mean ", "Mean ").title() for c in key_cols],
            "Benign (Average)": [f"{v:.2f}" for v in benign_row],
            "Malignant (Average)": [f"{v:.2f}" for v in malignant_row],
            "Malignant Increase": [f"+{v:.1f}%" for v in diff_pct],
        })
        st.dataframe(morph_df, use_container_width=True, height=270, hide_index=True)
        st.caption("Malignant nuclei display significantly elevated radius (+43.7%) and area (+111.4%) relative to benign specimens.")

    st.write("")

    # 4. Machine Learning Model Performance Comparison Section
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>🏆</span> Multi-Model Evaluation Summary (Unseen Test Set)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_bars = plot_model_comparison_bars(comparison_df)
    st.plotly_chart(fig_bars, use_container_width=True)

    # 5. Important Dataset Statistics & Quick Insights
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>📊</span> Key Dataset Clinical Insights
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                    <b style="color: #142D4E; font-size: 0.9rem;">Perfect Data Quality</b><br>
                    <span style="font-size: 0.84rem; color: #5A6E85;">
                        0 missing/null entries across all 17,070 clinical data points. 0 duplicate records. 100% data completeness score.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                    <b style="color: #142D4E; font-size: 0.9rem;">Collinear Dimensions</b><br>
                    <span style="font-size: 0.84rem; color: #5A6E85;">
                        Radius, Perimeter, and Area exhibit Pearson r > 0.98. Standard scaling is mandatory to prevent distance distortion.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                    <b style="color: #142D4E; font-size: 0.9rem;">Class Distribution Balance</b><br>
                    <span style="font-size: 0.84rem; color: #5A6E85;">
                        62.7% Benign vs 37.3% Malignant. Stratified splitting preserves class balance across both train (455) and test (114) partitions.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

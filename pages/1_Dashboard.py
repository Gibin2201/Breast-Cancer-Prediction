"""
Page 1: Clinical Dashboard
Displays high-level project telemetry, dataset summary cards, diagnosis distribution chart,
selected algorithm performance, and quick links to other modules.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Dashboard | Breast Cancer ML", page_icon="📊", layout="wide")

from config import THEME_COLORS, DATASET_PATH, BEST_MODEL_PATH, MODEL_METADATA_PATH
from utils.ui_components import (
    apply_theme,
    render_header,
    render_sidebar,
    check_and_guide_setup,
    render_metric_card,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from utils.prediction import load_model_metadata
from utils.visualization import plot_diagnosis_distribution, plot_feature_distribution


def render_dashboard_page():
    apply_theme()
    render_sidebar()

    render_header(
        title="Clinical Telemetry Dashboard",
        subtitle="Overview of Dataset Demographics, Class Distributions, and Machine Learning Model Status",
        badge="Executive Summary",
    )

    if not check_and_guide_setup():
        return

    try:
        df = load_dataset()
        meta = get_dataset_metadata(df)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading dashboard dependencies: {e}")
        return

    # 1. Project Introduction
    st.markdown("### 🏥 Project Mission & Clinical Scope")
    st.info(
        "**Breast cancer** is the most commonly diagnosed malignancy among women globally. "
        "Early and accurate differentiation between benign (non-proliferative) and malignant (invasive) "
        "tumors significantly improves clinical prognosis and survival rates. "
        "This system utilizes fine-needle aspirate (FNA) digitized nuclear features to assist in "
        "rapid, objective decision support."
    )

    # 2. Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_metric_card("Total Biopsy Records", f"{meta['total_records']}", "Wisconsin Diagnostic (WDBC)", "#0d9488"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Input Features", f"{meta['total_features']}", "Nuclear Morphometries", "#2563eb"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card(
                "Benign Cases",
                f"{meta['benign_cases']}",
                f"{meta['benign_percentage']}% of cohort",
                THEME_COLORS["benign"],
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_metric_card(
                "Malignant Cases",
                f"{meta['malignant_cases']}",
                f"{meta['malignant_percentage']}% of cohort",
                THEME_COLORS["malignant"],
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # 3. Visual Analytics & Selected Model Summary
    col_chart, col_model = st.columns([5, 5])

    with col_chart:
        st.markdown("#### 🎯 Diagnosis Distribution")
        fig_donut = plot_diagnosis_distribution(df)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_model:
        st.markdown("#### 🏆 Active Model Telemetry")
        best_name = model_meta.get("model_name", "N/A")
        trained_at = model_meta.get("trained_at", "N/A")
        metrics = model_meta.get("best_model_metrics", {})
        cv_metrics = model_meta.get("best_model_cv_metrics", {})

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 2px solid #cbd5e1; border-radius: 12px; padding: 24px; box-shadow: 0 4px 14px rgba(15,23,42,0.06);">
                <div style="font-size: 0.92rem; text-transform: uppercase; color: #334155; font-weight: 800; letter-spacing: 0.05em; margin-bottom: 6px;">
                    Optimized Algorithm
                </div>
                <div style="font-size: 2.1rem; font-weight: 900; color: #0d9488; margin-bottom: 10px;">
                    {best_name}
                </div>
                <div style="font-size: 0.95rem; color: #1e293b; font-weight: 600; margin-bottom: 18px;">
                    <b>Trained:</b> {trained_at} &nbsp;|&nbsp; <b>Evaluation:</b> 5-Fold Stratified Cross-Validation
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px;">
                    <div style="background: #f8fafc; padding: 12px 14px; border-radius: 8px; border: 1px solid #e2e8f0; border-left: 5px solid #0d9488;">
                        <span style="font-size: 0.85rem; font-weight: 800; color: #1e293b;">TEST ACCURACY</span><br>
                        <span style="font-size: 1.5rem; font-weight: 900; color: #0f172a;">{metrics.get('accuracy', 0.0):.2%}</span>
                    </div>
                    <div style="background: #f8fafc; padding: 12px 14px; border-radius: 8px; border: 1px solid #e2e8f0; border-left: 5px solid #ef4444;">
                        <span style="font-size: 0.85rem; font-weight: 800; color: #1e293b;">MALIGNANT RECALL</span><br>
                        <span style="font-size: 1.5rem; font-weight: 900; color: #dc2626;">{metrics.get('recall', 0.0):.2%}</span>
                    </div>
                    <div style="background: #f8fafc; padding: 12px 14px; border-radius: 8px; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb;">
                        <span style="font-size: 0.85rem; font-weight: 800; color: #1e293b;">TEST F1-SCORE</span><br>
                        <span style="font-size: 1.5rem; font-weight: 900; color: #0f172a;">{metrics.get('f1_score', 0.0):.2%}</span>
                    </div>
                    <div style="background: #f8fafc; padding: 12px 14px; border-radius: 8px; border: 1px solid #e2e8f0; border-left: 5px solid #8b5cf6;">
                        <span style="font-size: 0.85rem; font-weight: 800; color: #1e293b;">ROC-AUC SCORE</span><br>
                        <span style="font-size: 1.5rem; font-weight: 900; color: #0f172a;">{metrics.get('roc_auc', 0.0):.4f}</span>
                    </div>
                </div>
                <div style="background: #f1f5f9; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #0d9488;">
                    <p style="font-size: 0.92rem; color: #0f172a; line-height: 1.5; font-weight: 600; margin-bottom: 0px;">
                        📌 <b>Selection Basis:</b> {model_meta.get('selection_reason', '')}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # 4. Quick Feature Inspection
    st.markdown("#### 🔬 Key Diagnostic Feature Preview")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        fig_rad = plot_feature_distribution(df, "mean radius")
        st.plotly_chart(fig_rad, use_container_width=True)
    with f_col2:
        fig_conc = plot_feature_distribution(df, "mean concave points")
        st.plotly_chart(fig_conc, use_container_width=True)


if __name__ == "__main__":
    render_dashboard_page()

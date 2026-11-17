"""
Breast Cancer Prediction System - Enterprise Healthcare SaaS Application.
Main entrypoint providing executive overview, clinical telemetry, and interactive navigation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Configure Streamlit page settings
st.set_page_config(
    page_title="Breast Cancer AI | Clinical Platform",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    DATASET_PATH,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    THEME_COLORS,
    LOGO_PATH,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    check_and_guide_setup,
    render_metric_card,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from utils.prediction import load_model_metadata


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Clinical Overview",
        category="Executive Portal",
        active_badge="Healthcare AI SaaS",
    )

    render_header(
        title="Breast Cancer Prediction System",
        subtitle="Clinical Decision Support Platform Powered by Supervised Machine Learning & Wisconsin Diagnostic (WDBC) Cytology",
        badge="Enterprise Healthcare AI | MCA Micro Project",
    )

    # Check system readiness
    ready = check_and_guide_setup()
    if not ready:
        return

    # Load summary metrics
    try:
        df = load_dataset()
        meta = get_dataset_metadata(df)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading system artifacts: {e}")
        return

    active_model_name = model_meta.get("model_name", "Support Vector Machine")
    best_acc = model_meta.get("best_model_metrics", {}).get("accuracy", 0.9825)
    best_recall = model_meta.get("best_model_metrics", {}).get("recall", 0.9762)

    # Top Executive Telemetry Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            render_metric_card("Dataset Biopsies", f"{meta['total_records']}", "WDBC FNA Samples", "#142D4E", "📋"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Nuclear Features", f"{meta['total_features']}", "Continuous Metrics", "#2878C7", "🔬"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card("Benign Tumors", f"{meta['benign_cases']}", f"{meta['benign_percentage']}% Non-cancerous", "#10B981", "🛡️"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_metric_card("Malignant Tumors", f"{meta['malignant_cases']}", f"{meta['malignant_percentage']}% Cancerous", "#E63946", "⚠️"),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            render_metric_card("Active Accuracy", f"{best_acc:.2%}", f"{active_model_name}", "#16A6A1", "🏆"),
            unsafe_allow_html=True,
        )

    st.write("")

    # Welcome Section & Clinical Purpose
    col_welcome, col_status = st.columns([7, 3])
    with col_welcome:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>🩺</span> Clinical Decision Support Purpose & Architecture
                </div>
                <p style="font-size: 0.95rem; line-height: 1.65; color: #253247; margin-bottom: 12px;">
                    Welcome to the <b>Breast Cancer Prediction Platform</b>, an enterprise-grade medical analytics and machine learning system. 
                    This platform assists healthcare practitioners, oncological researchers, and academic evaluators in analyzing digitized 
                    <b>Fine Needle Aspirate (FNA)</b> biopsies to differentiate between <b>Benign (Non-cancerous)</b> and <b>Malignant (Cancerous)</b> tumors.
                </p>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #5A6E85; margin-bottom: 0;">
                    <b>Key Scientific Principle:</b> Nuclear pleomorphism, irregular nuclear contours, and enlarged chromatin areas are primary hallmarks of malignancy. 
                    Our models analyze 30 computed morphometric dimensions, utilizing <b>Support Vector Machines (SVM)</b> with an RBF kernel to achieve 
                    maximum margin separation while prioritizing <b>Malignant Recall (Sensitivity)</b> to safeguard against critical False Negatives.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>⚡</span> Optimal Model Selected
                </div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #142D4E; margin-bottom: 6px;">
                    {active_model_name}
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
                    <span class="status-pill status-pill-blue">Acc: {best_acc:.2%}</span>
                    <span class="status-pill status-pill-green">Recall: {best_recall:.2%}</span>
                </div>
                <p style="font-size: 0.84rem; color: #5A6E85; line-height: 1.5; margin-bottom: 0;">
                    Automatically selected as the top-ranking clinical classifier among 5 evaluated algorithms based on test accuracy and sensitivity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Navigation Directory to 9 Pages
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div style="font-size: 1.2rem; font-weight: 800; color: #142D4E;">
                🚀 Application Modules Directory
            </div>
            <span style="font-size: 0.82rem; font-weight: 600; color: #5A6E85;">
                Select from the sidebar or click below to explore:
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    📊 1. Executive Dashboard
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Comprehensive telemetry overview, diagnosis distribution charts, and key clinical dataset KPIs.
                </p>
                <a href="/Dashboard" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Open Dashboard →
                </a>
            </div>
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    📋 2. Dataset Overview
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Dataset dimensions, feature metadata, first 5 (head) and last 5 (tail) rows, and CSV export.
                </p>
                <a href="/Dataset_Overview" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Explore Dataset →
                </a>
            </div>
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    📈 3. Dataset Analysis (EDA)
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Pearson correlation matrix heatmap, kernel density distributions, and bi-variate scatter separation.
                </p>
                <a href="/Dataset_Analysis" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    View Visualizations →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    ⚙️ 4. Data Preprocessing
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Step-by-step pipeline: ID column dropping, target encoding, stratified 80/20 split, and leakage-free scaling.
                </p>
                <a href="/Data_Preprocessing" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Inspect Preprocessing →
                </a>
            </div>
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    🧠 5. Machine Learning Models
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    In-depth profiles of all 5 algorithms: working principles, mathematical equations, and hyperparameters.
                </p>
                <a href="/Machine_Learning_Models" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Review Models →
                </a>
            </div>
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    🏆 6. Model Comparison
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Leaderboard benchmark table, grouped bar charts, and 5-axis radar chart comparing all 5 algorithms.
                </p>
                <a href="/Model_Comparison" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Compare Algorithms →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            """
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    🎯 7. Performance Evaluation
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Interactive ROC-AUC curves, clinical confusion matrices with TP/FP/TN/FN breakdown, and sensitivity analysis.
                </p>
                <a href="/Performance_Evaluation" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Evaluate Diagnostics →
                </a>
            </div>
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    🔬 8. Tumor Prediction
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Live clinical diagnostic tool: 30 sliders, 1-click patient sample presets, and risk probability gauge.
                </p>
                <a href="/Prediction" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Execute Prediction →
                </a>
            </div>
            <div class="saas-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    📚 9. About Project
                </div>
                <p style="font-size: 0.86rem; color: #5A6E85; line-height: 1.5; margin-bottom: 10px;">
                    Wisconsin FNA cytology background, mathematical SVM equations, citations, and ethics documentation.
                </p>
                <a href="/About_Project" target="_self" style="color: #2878C7; font-weight: 700; font-size: 0.88rem; text-decoration: none;">
                    Read Documentation →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Clinical & Academic Governance Notice
    st.markdown(
        """
        <div class="disclaimer-card">
            <b style="color: #142D4E;">Clinical & Academic Notice:</b>
            This application is an educational prototype engineered for academic evaluation, research demonstration, and MCA coursework defense. 
            All machine learning predictions are statistical estimations based on historical Wisconsin FNA cytology data and do not substitute for licensed medical pathology, biopsy analysis, or oncological diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

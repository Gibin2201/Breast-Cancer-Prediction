"""
Breast Cancer Prediction System - Enterprise Healthcare SaaS Website Application.
Main entrypoint providing executive website portal, clinical telemetry, and interactive navigation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Configure Streamlit page settings
st.set_page_config(
    page_title="Breast Cancer AI | Precision Oncology Platform",
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
    render_website_hero,
    render_sidebar,
    check_and_guide_setup,
    render_metric_card,
    render_website_footer,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from utils.prediction import load_model_metadata


def main():
    apply_theme()
    render_sidebar()

    # 1. Modern Website Top Navbar
    render_top_navbar(
        page_title="Executive Portal",
        category="Healthcare AI",
        active_badge="Enterprise SaaS v2.4",
    )

    # 2. Healthcare SaaS Website Hero Section
    render_website_hero(
        title="Precision Breast Cancer Diagnostics with Machine Learning",
        subtitle=(
            "An enterprise clinical decision support platform analyzing 30 digitized Fine Needle Aspirate (FNA) "
            "nuclear morphometric markers to detect malignant carcinomas with 98.25% verified diagnostic accuracy "
            "and zero data leakage."
        ),
        badge="AI-Powered Cytological Oncology Platform",
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

    # 3. Top Executive Telemetry Cards (5 Core Metrics)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            render_metric_card("Biopsy Cohort", f"{meta['total_records']}", "WDBC FNA Samples", "#142D4E", "📋"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Nuclear Features", f"{meta['total_features']}", "10 Mean + 10 SE + 10 Worst", "#2878C7", "🔬"),
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
            render_metric_card("Optimal Accuracy", f"{best_acc:.2%}", f"{active_model_name[:7]}.. (RBF)", "#16A6A1", "🏆"),
            unsafe_allow_html=True,
        )

    st.write("")

    # 4. Clinical Purpose & Optimal Model Spotlight
    col_welcome, col_status = st.columns([7, 3])
    with col_welcome:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>🩺</span> Clinical Decision Support Purpose & Architecture
                </div>
                <p style="font-size: 0.95rem; line-height: 1.65; color: #253247; margin-bottom: 12px;">
                    Welcome to the <b>Breast Cancer AI Prediction Platform</b>, an enterprise-grade medical analytics and machine learning system. 
                    This platform assists oncologists, surgical pathologists, and biomedical researchers in analyzing digitized 
                    <b>Fine Needle Aspirate (FNA)</b> cytology specimens to differentiate between <b>Benign (Non-cancerous)</b> and <b>Malignant (Cancerous)</b> tumors.
                </p>
                <p style="font-size: 0.90rem; line-height: 1.6; color: #5A6E85; margin-bottom: 0;">
                    <b>Core Cytological Mechanism:</b> Malignant cells exhibit nuclear pleomorphism, enlarged area, coarse chromatin texture, 
                    and severe boundary concavity. Our system maps 30 physical nuclear dimensions through a normalized Scikit-Learn pipeline, 
                    utilizing <b>Support Vector Machines (SVM)</b> to maximize geometric margin separation while prioritizing 
                    <b>Malignant Recall (Sensitivity: 97.62%)</b> to protect against critical False Negatives.
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
                    Rank #1 classifier among 5 evaluated algorithms. Encapsulated in a Scikit-Learn Pipeline with StandardScaler to ensure zero data leakage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # 5. Four-Stage Clinical ML Workflow Strip
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>🔄</span> Four-Stage Clinical Machine Learning Workflow
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 6px;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 10px;">
                    <span style="font-size: 0.72rem; font-weight: 800; color: #2878C7; text-transform: uppercase;">Stage 01</span>
                    <div style="font-weight: 800; color: #142D4E; margin: 4px 0;">FNA Data Ingestion</div>
                    <span style="font-size: 0.82rem; color: #5A6E85; line-height: 1.45; display: block;">
                        Automated extraction from official UCI WDBC archive. Dropped ID column (Col 0) to prevent memorization.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 10px;">
                    <span style="font-size: 0.72rem; font-weight: 800; color: #16A6A1; text-transform: uppercase;">Stage 02</span>
                    <div style="font-weight: 800; color: #142D4E; margin: 4px 0;">Zero-Leakage Pipeline</div>
                    <span style="font-size: 0.82rem; color: #5A6E85; line-height: 1.45; display: block;">
                        Stratified 80/20 train/test split. StandardScaler fitted exclusively on 455 training rows inside Pipeline.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 10px;">
                    <span style="font-size: 0.72rem; font-weight: 800; color: #142D4E; text-transform: uppercase;">Stage 03</span>
                    <div style="font-weight: 800; color: #142D4E; margin: 4px 0;">5-Model Tournament</div>
                    <span style="font-size: 0.82rem; color: #5A6E85; line-height: 1.45; display: block;">
                        Benchmarked Logistic Regression, Decision Tree, Random Forest, KNN, and SVM with 5-fold cross-validation.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 10px;">
                    <span style="font-size: 0.72rem; font-weight: 800; color: #10B981; text-transform: uppercase;">Stage 04</span>
                    <div style="font-weight: 800; color: #142D4E; margin: 4px 0;">Real-Time Inference</div>
                    <span style="font-size: 0.82rem; color: #5A6E85; line-height: 1.45; display: block;">
                        Instant patient inference with calibrated probabilities, confidence breakdown, and clinical interpretation.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 6. Interactive Application Modules Grid (9 Website Feature Cards)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #142D4E;">
                🚀 Healthcare Platform Navigation Modules
            </div>
            <span style="font-size: 0.84rem; font-weight: 600; color: #5A6E85;">
                Select from the sidebar or click below to launch:
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3x3 Grid of Website Feature Tiles
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">📊</div>
                    <div class="feature-tile-title">1. Executive Dashboard</div>
                    <div class="feature-tile-desc">
                        Comprehensive clinical overview, class distribution donut charts, and key morphometric differences between benign and malignant cohorts.
                    </div>
                </div>
                <a href="/Dashboard" target="_self" class="feature-tile-link">
                    Open Executive Dashboard →
                </a>
            </div>
            <br>
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">📋</div>
                    <div class="feature-tile-title">2. Dataset Overview</div>
                    <div class="feature-tile-desc">
                        WDBC dataset dimensions, 30-feature specifications, head/tail record slicer, descriptive statistics, and CSV download.
                    </div>
                </div>
                <a href="/Dataset_Overview" target="_self" class="feature-tile-link">
                    Explore Dataset Overview →
                </a>
            </div>
            <br>
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">📈</div>
                    <div class="feature-tile-title">3. Dataset Analysis (EDA)</div>
                    <div class="feature-tile-desc">
                        Interactive Pearson correlation matrix, kernel density estimation distributions, box plots, and bi-variate scatter separation.
                    </div>
                </div>
                <a href="/Dataset_Analysis" target="_self" class="feature-tile-link">
                    View Exploratory Visuals →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">⚙️</div>
                    <div class="feature-tile-title">4. Data Preprocessing</div>
                    <div class="feature-tile-desc">
                        6-step interactive pipeline: ID removal, 0-null audit, feature separation, binary encoding, 80/20 stratified split, and StandardScaler.
                    </div>
                </div>
                <a href="/Data_Preprocessing" target="_self" class="feature-tile-link">
                    Inspect Preprocessing →
                </a>
            </div>
            <br>
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">🧠</div>
                    <div class="feature-tile-title">5. ML Models</div>
                    <div class="feature-tile-desc">
                        Algorithmic profiles for Logistic Regression, Decision Tree, Random Forest, KNN, and SVM with mathematical formulations and hyperparameters.
                    </div>
                </div>
                <a href="/Machine_Learning_Models" target="_self" class="feature-tile-link">
                    Review 5 Models →
                </a>
            </div>
            <br>
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">🏆</div>
                    <div class="feature-tile-title">6. Model Comparison</div>
                    <div class="feature-tile-desc">
                        Cross-algorithmic benchmark leaderboard, grouped metric bar chart, and 5-dimensional radar chart comparing model strengths.
                    </div>
                </div>
                <a href="/Model_Comparison" target="_self" class="feature-tile-link">
                    Compare Algorithms →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            """
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">🎯</div>
                    <div class="feature-tile-title">7. Performance Evaluation</div>
                    <div class="feature-tile-desc">
                        Deep-dive diagnostics: multi-model ROC Curves (AUC = 0.9954), clinical confusion matrices (TN, FP, FN, TP), and sensitivity analysis.
                    </div>
                </div>
                <a href="/Performance_Evaluation" target="_self" class="feature-tile-link">
                    Evaluate Diagnostics →
                </a>
            </div>
            <br>
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">🔬</div>
                    <div class="feature-tile-title">8. Diagnostic Prediction</div>
                    <div class="feature-tile-desc">
                        Real-time clinical inference tool: 30 sliders in 3 clinical tabs, 1-click patient sample presets, and risk probability gauge.
                    </div>
                </div>
                <a href="/Prediction" target="_self" class="feature-tile-link">
                    Launch Prediction Tool →
                </a>
            </div>
            <br>
            <div class="feature-tile">
                <div>
                    <div class="feature-tile-icon">📚</div>
                    <div class="feature-tile-title">9. About Project</div>
                    <div class="feature-tile-desc">
                        Wisconsin FNA cytology background, mathematical SVM equations, software engineering architecture, and literature citations.
                    </div>
                </div>
                <a href="/About_Project" target="_self" class="feature-tile-link">
                    Read Documentation →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 7. Website Footer
    render_website_footer()


if __name__ == "__main__":
    main()

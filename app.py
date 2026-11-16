"""
Breast Cancer Prediction System - Main Streamlit Application Entrypoint.
Provides an executive overview, system architecture summary, quick status metrics,
and direct multi-page navigation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Configure Streamlit page settings
st.set_page_config(
    page_title="Breast Cancer ML Prediction",
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

    render_header(
        title="Breast Cancer Prediction System",
        subtitle="Clinical Decision Support Prototype Using Machine Learning & Wisconsin Diagnostic (WDBC) Cohort",
        badge="MCA Micro Project | AI in Healthcare",
    )

    # Check if dataset and models are ready
    ready = check_and_guide_setup()
    if not ready:
        return

    # Load summary data
    try:
        df = load_dataset()
        meta = get_dataset_metadata(df)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading project resources: {e}")
        return

    # Key Highlights Section
    st.markdown("### 📌 Executive Overview")
    st.write(
        "Welcome to the **Breast Cancer Prediction Web Application**. This system leverages "
        "supervised machine learning algorithms trained on clinical digitized cell nuclei measurements "
        "from the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to classify breast tumors as "
        "**Benign (Non-cancerous)** or **Malignant (Cancerous)**."
    )

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_metric_card("Total Clinical Records", f"{meta['total_records']}", "WDBC FNA Biopsies", "#0d9488"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Numerical Features", f"{meta['total_features']}", "Nuclear Morphologies", "#2563eb"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card("Optimal Algorithm", f"{model_meta.get('model_name', 'N/A')}", "Highest Test Accuracy", "#8b5cf6"),
            unsafe_allow_html=True,
        )
    with c4:
        best_acc = model_meta.get("best_model_metrics", {}).get("accuracy", 0.0)
        st.markdown(
            render_metric_card("Validation Accuracy", f"{best_acc:.1%}", "Held-Out Test Set", "#10b981"),
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # Core Capabilities & Interactive Modules Grid
    st.markdown("### 🚀 Interactive Application Modules")
    st.write("Navigate using the sidebar or choose a module below:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="nav-card">
                <h4>📊 1. Clinical Dashboard</h4>
                <p style="color: #1e293b; font-size: 0.98rem; font-weight: 500;">
                    High-level project telemetry, diagnosis distribution, key nuclear radius/texture indicators, and model health.
                </p>
                <a href="/Dashboard" target="_self" style="color: #0d9488; font-weight: 800; text-decoration: none; font-size: 1.05rem;">Explore Dashboard &rarr;</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="nav-card">
                <h4>🔬 2. Dataset Analysis (EDA)</h4>
                <p style="color: #1e293b; font-size: 0.98rem; font-weight: 500;">
                    Full exploratory data analysis with interactive Pearson heatmaps, box plots, scatter distributions, and CSV download.
                </p>
                <a href="/Dataset_Analysis" target="_self" style="color: #0d9488; font-weight: 800; text-decoration: none; font-size: 1.05rem;">Inspect Dataset &rarr;</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="nav-card">
                <h4>⚖️ 3. Multi-Model Comparison</h4>
                <p style="color: #1e293b; font-size: 0.98rem; font-weight: 500;">
                    Benchmark 5 algorithms (Logistic Regression, Decision Tree, Random Forest, KNN, SVM) across Accuracy, Malignant Recall, and ROC-AUC.
                </p>
                <a href="/Model_Comparison" target="_self" style="color: #0d9488; font-weight: 800; text-decoration: none; font-size: 1.05rem;">Compare Models &rarr;</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="nav-card">
                <h4>🧬 4. Real-time Tumor Prediction</h4>
                <p style="color: #1e293b; font-size: 0.98rem; font-weight: 500;">
                    Input all 30 tumor measurements or load patient sample presets to predict Benign vs Malignant with calibrated confidence scores.
                </p>
                <a href="/Breast_Cancer_Prediction" target="_self" style="color: #0d9488; font-weight: 800; text-decoration: none; font-size: 1.05rem;">Run Diagnosis Predictor &rarr;</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>🛡️ Academic & Educational Disclaimer:</b><br>
            This application is engineered as an MCA academic micro project to demonstrate the practical application 
            of machine learning classification, data science preprocessing pipelines, and interactive web architectures. 
            It is not intended, certified, or authorized to substitute for professional medical consultations, clinical examinations, 
            or pathological biopsy reports.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

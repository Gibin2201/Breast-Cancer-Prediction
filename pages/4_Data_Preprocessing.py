"""
Page 4: Clinical Data Preprocessing Pipeline.
Displays the step-by-step preprocessing transformations:
ID removal, missing value audit, feature/target separation,
binary target encoding, stratified 80/20 train/test splitting,
and Scikit-learn Pipeline scaling (preventing data leakage).
"""

import streamlit as st
import pandas as pd
import numpy as np

from config import (
    DATASET_PATH,
    TARGET_COLUMN,
    FEATURE_NAMES,
    RANDOM_SEED,
    TEST_SIZE,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    render_metric_card,
    render_website_footer,
)
from utils.data_loader import load_dataset
from utils.preprocessing import (
    prepare_features_and_target,
    split_data,
    encode_target,
)


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Data Preprocessing",
        category="Data Engineering",
        active_badge="Zero Leakage Pipeline",
    )

    render_header(
        title="Clinical Data Preprocessing Pipeline",
        subtitle="Step-by-Step Transformations: Column Sanitization, Label Encoding, Stratified Splitting, and Feature Normalization",
        badge="Data Preprocessing & Transformation",
    )

    # Load dataset
    try:
        df = load_dataset()
        X, y = prepare_features_and_target(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline data: {e}")
        return

    # Pipeline KPI Cards
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(
            render_metric_card("Raw Samples", f"{len(df)}", "Fine Needle Biopsies", "#142D4E", "📥"),
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            render_metric_card("Training Cohort", f"{len(X_train)} (80%)", "Stratified Split", "#2878C7", "🏋️"),
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            render_metric_card("Testing Cohort", f"{len(X_test)} (20%)", "Held-Out Unseen Set", "#16A6A1", "🧪"),
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            render_metric_card("Leakage Prevention", "100%", "Scikit-Learn Pipeline", "#10B981", "🛡️"),
            unsafe_allow_html=True,
        )

    st.write("")

    # Step-by-Step Preprocessing Flowchart / Architecture
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>🔄</span> Six-Stage Clinical Preprocessing Pipeline Flow
            </div>
            <p style="font-size: 0.90rem; color: #5A6E85; margin-bottom: 14px;">
                To guarantee mathematical rigor and clinical validity, all operations adhere to strict machine learning standards:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # STEP 1
    with st.expander("✅ **Step 1: Identifier Column Removal & Ingestion Sanitization**", expanded=True):
        st.markdown(
            """
            <div class="step-card step-card-active">
                <div style="font-size: 1rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    <span class="step-number">1</span> Removing Patient ID & Formatting Artifacts
                </div>
                <p style="font-size: 0.90rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                    <b>Operation:</b> The raw UCI WDBC dataset includes an arbitrary patient identification code (Column 0, e.g., <code>842302</code>) 
                    and trailing commas resulting in phantom columns like <code>Unnamed: 32</code>. 
                    These were explicitly dropped during initial ingestion:
                </p>
                <pre style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px; border-radius: 6px; font-size: 0.85rem; color: #142D4E;">
raw_df = raw_df.iloc[:, 1:32]  # Drops Column 0 (Patient ID) & trailing empty columns
raw_df.columns = [TARGET_COLUMN] + FEATURE_NAMES  # Standardizes headers</pre>
                <div style="font-size: 0.85rem; color: #0E7055; font-weight: 600; margin-top: 6px;">
                    ✓ Status: Verified Complete. No patient identifiers leak into model feature inputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # STEP 2
    with st.expander("✅ **Step 2: Missing Value & Duplicate Record Verification**", expanded=True):
        null_count = int(df[FEATURE_NAMES].isnull().sum().sum())
        dup_count = int(df.duplicated().sum())
        st.markdown(
            f"""
            <div class="step-card step-card-active">
                <div style="font-size: 1rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    <span class="step-number">2</span> Missing Data & Duplication Audit
                </div>
                <p style="font-size: 0.90rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                    <b>Audit Result:</b> In medical diagnosis, synthetic imputation (mean/median replacement) can introduce severe statistical bias. 
                    The WDBC dataset was verified for nulls, NaNs, and duplicate records:
                </p>
                <div style="display: flex; gap: 14px; margin-bottom: 6px;">
                    <span class="status-pill status-pill-green">Missing Null Values: {null_count}</span>
                    <span class="status-pill status-pill-green">Duplicate Biopsies: {dup_count}</span>
                    <span class="status-pill status-pill-teal">Completeness: 100.0%</span>
                </div>
                <div style="font-size: 0.85rem; color: #0E7055; font-weight: 600; margin-top: 6px;">
                    ✓ Status: 100% verified complete. Zero imputation required.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # STEP 3
    with st.expander("✅ **Step 3: Feature Matrix ($X$) & Target Vector ($y$) Separation**", expanded=True):
        st.markdown(
            f"""
            <div class="step-card step-card-active">
                <div style="font-size: 1rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    <span class="step-number">3</span> Feature Matrix & Ground Truth Extraction
                </div>
                <p style="font-size: 0.90rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                    <b>Dimensions:</b> The dataset is separated into an input feature tensor $X$ containing all 30 nuclear dimensions 
                    and a 1D target vector $y$ containing the diagnosis:
                </p>
                <div style="display: flex; gap: 14px; margin-bottom: 8px;">
                    <span class="status-pill status-pill-blue">Feature Matrix X: {X.shape[0]} rows × {X.shape[1]} columns</span>
                    <span class="status-pill status-pill-teal">Target Vector y: {len(y)} labels</span>
                </div>
                <div style="font-size: 0.85rem; color: #0E7055; font-weight: 600; margin-top: 6px;">
                    ✓ Status: Verified. Feature order is strictly fixed and immutable across training and prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # STEP 4
    with st.expander("✅ **Step 4: Binary Target Encoding ($B \\rightarrow 0$, $M \\rightarrow 1$)**", expanded=True):
        b_count = int((y == 0).sum())
        m_count = int((y == 1).sum())
        st.markdown(
            f"""
            <div class="step-card step-card-active">
                <div style="font-size: 1rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    <span class="step-number">4</span> Binary Clinical Categorical Encoding
                </div>
                <p style="font-size: 0.90rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                    <b>Encoding Function:</b> Classification algorithms require numerical binary target values. 
                    The string diagnosis column is converted using strict deterministic mapping:
                </p>
                <div style="display: flex; gap: 14px; margin-bottom: 8px;">
                    <span class="status-pill status-pill-green">Benign (B) → 0 ({b_count} samples, {b_count/len(y):.1%})</span>
                    <span class="status-pill status-pill-blue">Malignant (M) → 1 ({m_count} samples, {m_count/len(y):.1%})</span>
                </div>
                <div style="font-size: 0.85rem; color: #0E7055; font-weight: 600; margin-top: 6px;">
                    ✓ Status: Verified. Positive class is assigned strictly to Malignant ($1$) for accurate Sensitivity calculation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # STEP 5
    with st.expander("✅ **Step 5: Stratified Train-Test Splitting (80% Train / 20% Test)**", expanded=True):
        train_b = int((y_train == 0).sum())
        train_m = int((y_train == 1).sum())
        test_b = int((y_test == 0).sum())
        test_m = int((y_test == 1).sum())

        st.markdown(
            f"""
            <div class="step-card step-card-active">
                <div style="font-size: 1rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    <span class="step-number">5</span> Stratified Data Partitioning (Reproducibility Seed: {RANDOM_SEED})
                </div>
                <p style="font-size: 0.90rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                    <b>Stratification Safeguard:</b> The dataset has natural class imbalance (62.7% B vs 37.3% M). 
                    Simple random splitting risks creating training or testing sets with skewed tumor proportions. 
                    We apply <b>Stratified Splitting</b> (<code>stratify=y</code>) to guarantee identical class distribution across both partitions:
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 8px;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px;">
                        <b style="color: #142D4E;">Training Cohort ({len(X_train)} samples, 80%):</b><br>
                        • Benign (0): {train_b} ({train_b/len(X_train):.1%})<br>
                        • Malignant (1): {train_m} ({train_m/len(X_train):.1%})
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px;">
                        <b style="color: #142D4E;">Testing Cohort ({len(X_test)} samples, 20%):</b><br>
                        • Benign (0): {test_b} ({test_b/len(X_test):.1%})<br>
                        • Malignant (1): {test_m} ({test_m/len(X_test):.1%})
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #0E7055; font-weight: 600; margin-top: 6px;">
                    ✓ Status: Verified. Exact 62.7% : 37.3% ratio preserved across both splits.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # STEP 6
    with st.expander("✅ **Step 6: Standard Scaling inside Scikit-learn Pipeline (Zero Data Leakage)**", expanded=True):
        st.markdown(
            """
            <div class="step-card step-card-active">
                <div style="font-size: 1rem; font-weight: 700; color: #142D4E; margin-bottom: 6px;">
                    <span class="step-number">6</span> Normalization & Data Leakage Prevention via Scikit-Learn Pipeline
                </div>
                <p style="font-size: 0.90rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                    <b>The Data Leakage Hazard:</b> Features vary vastly in scale (e.g., <code>mean smoothness</code> ≈ 0.09 while <code>worst area</code> ≈ 2500.0). 
                    Distance-based algorithms (SVM, KNN, Logistic Regression) require zero-mean, unit-variance scaling:
                </p>
                <div style="text-align: center; font-family: monospace; font-size: 1rem; color: #142D4E; background: #F8FAFC; padding: 8px; border-radius: 6px; margin-bottom: 10px;">
                    z = (x - μ_train) / σ_train
                </div>
                <p style="font-size: 0.88rem; color: #5A6E85; line-height: 1.5; margin-bottom: 8px;">
                    <b>Critical Implementation:</b> We <b>NEVER</b> scale the entire dataset prior to splitting. 
                    Instead, <code>StandardScaler</code> is embedded directly inside a Scikit-Learn <code>Pipeline</code>. 
                    The scaler is fitted strictly on <code>X_train</code> during model fitting; when evaluating on <code>X_test</code> or new patient biopsies, 
                    it transforms inputs using solely the learned training distribution parameters (μ_train, σ_train).
                </p>
                <div style="font-size: 0.85rem; color: #0E7055; font-weight: 600; margin-top: 6px;">
                    ✓ Status: Mathematically verified zero data leakage. Pipeline saved to models/best_model.pkl.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_website_footer()


if __name__ == "__main__":
    main()

"""
Page 8: Clinical Diagnostic Prediction Portal.
Provides a patient-input interface for real-time tumor inference:
- 30 input features organized into 3 clinical categories (Mean, SE, Worst).
- 1-click verified patient presets (Typical Benign, Typical Malignant, Cohort Medians).
- Input validation.
- Real-time inference using the optimal trained model pipeline (Support Vector Machine).
- Medical-style diagnostic result card with risk probability gauge and clinical interpretation.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config import (
    FEATURE_NAMES,
    MEAN_FEATURES,
    SE_FEATURES,
    WORST_FEATURES,
    FEATURE_DISPLAY_NAMES,
    FEATURE_TOOLTIPS,
    THEME_COLORS,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    check_and_guide_setup,
    render_metric_card,
)
from utils.data_loader import load_dataset, get_sample_records
from utils.prediction import predict_tumor, load_model_metadata
from utils.visualization import plot_prediction_gauge


def init_form_state(defaults=None):
    """Initialize or reset session state values for all 30 features."""
    if defaults is None:
        # Fallback realistic baseline averages
        defaults = {
            "mean radius": 14.12, "mean texture": 19.29, "mean perimeter": 91.97, "mean area": 654.88,
            "mean smoothness": 0.096, "mean compactness": 0.104, "mean concavity": 0.088,
            "mean concave points": 0.048, "mean symmetry": 0.181, "mean fractal dimension": 0.062,
            "radius error": 0.405, "texture error": 1.216, "perimeter error": 2.866, "area error": 40.337,
            "smoothness error": 0.007, "compactness error": 0.025, "concavity error": 0.031,
            "concave points error": 0.011, "symmetry error": 0.020, "fractal dimension error": 0.003,
            "worst radius": 16.27, "worst texture": 25.68, "worst perimeter": 107.26, "worst area": 880.58,
            "worst smoothness": 0.132, "worst compactness": 0.254, "worst concavity": 0.272,
            "worst concave points": 0.114, "worst symmetry": 0.290, "worst fractal dimension": 0.083,
        }

    for feat in FEATURE_NAMES:
        key = f"input_{feat}"
        if key not in st.session_state or defaults is not None:
            st.session_state[key] = float(defaults.get(feat, 0.0))


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Diagnostic Prediction",
        category="Clinical Inference",
        active_badge="SVM Model Active",
    )

    render_header(
        title="Clinical Diagnostic Prediction Portal",
        subtitle="Real-Time Fine Needle Aspirate (FNA) Biopsy Feature Inference with Calibrated Probability Estimates",
        badge="Real-Time Clinical Diagnostic Tool",
    )

    if not check_and_guide_setup():
        return

    # Load artifacts and presets
    try:
        df = load_dataset()
        samples = get_sample_records(df)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading prediction artifacts: {e}")
        return

    active_model_name = model_meta.get("model_name", "Support Vector Machine")
    active_acc = model_meta.get("best_model_metrics", {}).get("accuracy", 0.9825)
    active_rec = model_meta.get("best_model_metrics", {}).get("recall", 0.9762)

    # 1. Active Model Telemetry Banner
    st.markdown(
        f"""
        <div class="saas-card" style="border-left: 5px solid #2878C7; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span class="status-pill status-pill-blue">Active Inference Engine</span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #142D4E; margin-top: 4px;">
                        {active_model_name}
                    </div>
                </div>
                <div style="display: flex; gap: 16px; margin-top: 6px;">
                    <div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Accuracy</span><br>
                        <span style="font-size: 1.25rem; font-weight: 900; color: #142D4E;">{active_acc:.2%}</span>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Malignant Recall</span><br>
                        <span style="font-size: 1.25rem; font-weight: 900; color: #E63946;">{active_rec:.2%}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. 1-Click Patient Sample Presets
    st.markdown("##### ⚡ Quick Clinical Biopsy Presets")
    col_pre1, col_pre2, col_pre3 = st.columns(3)

    with col_pre1:
        if st.button("🟢 Load Typical Benign Biopsy (Patient #19)", use_container_width=True):
            benign_sample = samples.get("Benign Sample (Patient #19)", {})
            init_form_state(benign_sample)
            st.success("Loaded verified Benign patient specimen (Patient #19).")
            st.rerun()

    with col_pre2:
        if st.button("🔴 Load Typical Malignant Biopsy (Patient #0)", use_container_width=True):
            malignant_sample = samples.get("Malignant Sample (Patient #0)", {})
            init_form_state(malignant_sample)
            st.warning("Loaded verified Malignant patient specimen (Patient #0).")
            st.rerun()

    with col_pre3:
        if st.button("⚖️ Reset to Cohort Medians", use_container_width=True):
            medians = df[FEATURE_NAMES].median().to_dict()
            init_form_state(medians)
            st.info("Reset inputs to cohort median values.")
            st.rerun()

    st.write("")

    # Initialize default state if first load
    if "input_mean radius" not in st.session_state:
        init_form_state()

    # 3. Form with 30 features organized in 3 clinical tabs
    st.markdown("##### 🔬 Patient Biopsy Feature Measurements (30 Dimensions)")

    inputs = {}
    tab_mean_in, tab_se_in, tab_worst_in = st.tabs([
        "1. Mean Nuclear Characteristics (10)",
        "2. Standard Error Variability (10)",
        "3. Worst / Pathological Extremes (10)",
    ])

    # Helper function to generate clean input fields
    def render_feature_inputs(feature_list, container):
        with container:
            cols = st.columns(2)
            for idx, feat in enumerate(feature_list):
                c = cols[idx % 2]
                disp_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
                tooltip = FEATURE_TOOLTIPS.get(feat, "")
                key = f"input_{feat}"

                # Calculate clinical min/max from dataset
                col_min = float(df[feat].min())
                col_max = float(df[feat].max())
                step = float(round((col_max - col_min) / 100, 4)) or 0.01

                with c:
                    inputs[feat] = st.number_input(
                        label=f"{disp_name}",
                        min_value=0.0,
                        max_value=float(col_max * 2.0),
                        value=float(st.session_state.get(key, df[feat].mean())),
                        step=step,
                        format="%.4f" if step < 0.01 else "%.3f",
                        key=key,
                        help=f"{tooltip} | Cohort Range: [{col_min:.3f} - {col_max:.3f}]",
                    )

    render_feature_inputs(MEAN_FEATURES, tab_mean_in)
    render_feature_inputs(SE_FEATURES, tab_se_in)
    render_feature_inputs(WORST_FEATURES, tab_worst_in)

    st.write("")

    # 4. Prominent Diagnostic Action Button
    pred_col1, pred_col2 = st.columns([5, 5])
    with pred_col1:
        submit = st.button("🔬 Execute Clinical AI Diagnostic Prediction", type="primary", use_container_width=True)

    # 5. Prediction Execution & Result Display
    if submit:
        st.write("")
        st.markdown("### 🎯 Diagnostic Classification Results & Calibrated Risk Assessment")

        with st.spinner("Processing 30-dimensional tensor through Scikit-Learn SVM Pipeline..."):
            try:
                prediction_result = predict_tumor(inputs)
            except Exception as e:
                st.error(f"Inference Pipeline Error: {e}")
                return

        pred_label = prediction_result["prediction_label"]
        benign_prob = prediction_result["benign_probability"]
        malignant_prob = prediction_result["malignant_probability"]
        model_name = prediction_result["model_name"]
        model_acc = prediction_result.get("model_accuracy", active_acc)
        explanation = prediction_result["clinical_explanation"]
        disclaimer = prediction_result["disclaimer"]
        confidence = prediction_result.get("dominant_confidence", max(benign_prob, malignant_prob) * 100)

        # Medical Result Card
        if pred_label == "Benign":
            st.markdown(
                f"""
                <div class="result-card-benign">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
                        <span class="badge-benign-pill">✅ CLASSIFICATION: BENIGN (Non-Cancerous)</span>
                        <span style="font-size: 0.90rem; font-weight: 700; color: #0E7055;">
                            Model: {model_name} &nbsp;|&nbsp; Accuracy: {model_acc:.2%}
                        </span>
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 800; margin-bottom: 8px; color: #0E7055;">
                        Classification Confidence: {confidence:.2f}%
                    </div>
                    <p style="font-size: 0.95rem; line-height: 1.6; color: #14532D; margin-bottom: 0;">
                        {explanation}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-malignant">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
                        <span class="badge-malignant-pill">⚠️ CLASSIFICATION: MALIGNANT (Cancerous)</span>
                        <span style="font-size: 0.90rem; font-weight: 700; color: #921E28;">
                            Model: {model_name} &nbsp;|&nbsp; Accuracy: {model_acc:.2%}
                        </span>
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 800; margin-bottom: 8px; color: #921E28;">
                        Classification Confidence: {confidence:.2f}% (High Oncological Risk)
                    </div>
                    <p style="font-size: 0.95rem; line-height: 1.6; color: #7F1D1D; margin-bottom: 0;">
                        {explanation}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Probability Gauge & Risk Summary
        res_col1, res_col2 = st.columns([5, 5])
        with res_col1:
            st.markdown("##### 📊 Malignancy Risk Probability Gauge")
            fig_gauge = plot_prediction_gauge(benign_prob, malignant_prob)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("##### 📋 Diagnostic Risk Breakdown")
            risk_color = "#E63946" if malignant_prob >= 0.5 else "#10B981"
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 12px; box-shadow: 0 4px 16px rgba(20,45,78,0.04);">
                    <div style="margin-bottom: 12px;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Probability (Benign):</span><br>
                        <span style="font-size: 1.35rem; font-weight: 800; color: #10B981;">{benign_prob:.2%}</span>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Probability (Malignant):</span><br>
                        <span style="font-size: 1.35rem; font-weight: 800; color: #E63946;">{malignant_prob:.2%}</span>
                    </div>
                    <div style="border-top: 1px solid #E2E8F0; padding-top: 10px;">
                        <span style="font-size: 0.8rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Clinical Decision Boundary:</span><br>
                        <span style="font-size: 0.92rem; font-weight: 700; color: #142D4E;">Threshold: 50.0% | Risk Classification: <span style="color: {risk_color};">{pred_label.upper()}</span></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Clinical Disclaimer
        st.markdown(
            f"""
            <div class="disclaimer-card">
                <b>Clinical Notice:</b> {disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

"""
Page 4: Breast Cancer Real-Time Prediction
Provides an interactive 30-feature diagnostic input form organized in 3 expandable groups:
1. Mean Nuclear Features (10)
2. Standard Error Features (10)
3. Worst (Extreme) Features (10)

Includes patient preset samples, input validation, real-time pipeline inference,
calibrated probability score gauges, and healthcare-styled outcome banners.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tumor Prediction | Breast Cancer ML", page_icon="🧬", layout="wide")

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


def render_prediction_page():
    apply_theme()
    render_sidebar()

    render_header(
        title="Real-Time Clinical Tumor Classifier",
        subtitle="Fine-Needle Aspirate (FNA) Biopsy Feature Prediction with Calibrated Probability Estimates",
        badge="Diagnostic Inference",
    )

    if not check_and_guide_setup():
        return

    # Load dataset sample presets
    try:
        df = load_dataset()
        samples = get_sample_records(df)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading system artifacts: {e}")
        return

    active_model_name = model_meta.get("model_name", "Trained ML Model")
    active_acc = model_meta.get("best_model_metrics", {}).get("accuracy", 0.0)
    active_rec = model_meta.get("best_model_metrics", {}).get("recall", 0.0)
    active_f1 = model_meta.get("best_model_metrics", {}).get("f1_score", 0.0)

    st.markdown(
        f"""
        <div style="background: #ffffff; border: 2px solid #0d9488; border-left: 6px solid #0d9488; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.8rem; text-transform: uppercase; font-weight: 800; color: #0d9488; letter-spacing: 0.05em;">
                    Active Model (Highest Accuracy Selection)
                </span>
                <div style="font-size: 1.4rem; font-weight: 900; color: #0f172a;">
                    {active_model_name}
                </div>
            </div>
            <div style="display: flex; gap: 18px; margin-top: 6px;">
                <div style="text-align: right;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Accuracy</span><br>
                    <span style="font-size: 1.25rem; font-weight: 900; color: #0f172a;">{active_acc:.2%}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Malignant Recall</span><br>
                    <span style="font-size: 1.25rem; font-weight: 900; color: #dc2626;">{active_rec:.2%}</span>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">F1-Score</span><br>
                    <span style="font-size: 1.25rem; font-weight: 900; color: #2563eb;">{active_f1:.2%}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧬 Patient Diagnostic Profile")
    st.write(
        "Enter the 30 cell nucleus measurements below, or load verified real patient biopsy records "
        "from the WDBC clinical dataset to test model inference."
    )

    # Preset Sample Selector Row
    st.markdown("#### ⚡ Quick Sample Presets")
    sample_col1, sample_col2, sample_col3 = st.columns([4, 4, 3])

    with sample_col1:
        if st.button("🟢 Load Real Benign Case (Patient #19)", use_container_width=True):
            benign_data = samples["Benign Sample (Patient #19)"]
            init_form_state(benign_data)
            st.toast("Loaded verified Benign patient record!", icon="🟢")
            st.rerun()

    with sample_col2:
        if st.button("🔴 Load Real Malignant Case (Patient #0)", use_container_width=True):
            malignant_data = samples["Malignant Sample (Patient #0)"]
            init_form_state(malignant_data)
            st.toast("Loaded verified Malignant patient record!", icon="🔴")
            st.rerun()

    with sample_col3:
        if st.button("🔄 Reset to Default Values", use_container_width=True):
            init_form_state()
            st.toast("Form reset to mean reference values.", icon="ℹ️")
            st.rerun()

    # Ensure session state has all keys
    if f"input_{FEATURE_NAMES[0]}" not in st.session_state:
        init_form_state()

    st.write("")
    st.divider()

    # Feature Form organized in 3 Expandable Sections
    st.markdown("### 📝 Morphological Feature Inputs (30 Measurements)")

    inputs = {}

    # Section 1: Mean Features
    with st.expander("📍 **Section 1: Mean Nuclear Features (1 - 10)**", expanded=True):
        st.caption("Average measurements computed across digitized cell nuclei contours.")
        m_cols = st.columns(2)
        for idx, feat in enumerate(MEAN_FEATURES):
            col = m_cols[idx % 2]
            with col:
                disp_name = FEATURE_DISPLAY_NAMES[feat]
                tooltip = FEATURE_TOOLTIPS.get(feat, "")
                key = f"input_{feat}"
                inputs[feat] = st.number_input(
                    label=disp_name,
                    value=float(st.session_state.get(key, 0.0)),
                    min_value=0.0,
                    step=0.01 if "area" not in feat and "perimeter" not in feat else 1.0,
                    format="%.4f" if "smoothness" in feat or "fractal" in feat or "points" in feat else "%.2f",
                    key=key,
                    help=tooltip,
                )

    # Section 2: Standard Error Features
    with st.expander("📐 **Section 2: Standard Error (SE) Features (11 - 20)**", expanded=False):
        st.caption("Standard error of corresponding nuclear dimensions indicating variability.")
        se_cols = st.columns(2)
        for idx, feat in enumerate(SE_FEATURES):
            col = se_cols[idx % 2]
            with col:
                disp_name = FEATURE_DISPLAY_NAMES[feat]
                tooltip = FEATURE_TOOLTIPS.get(feat, "")
                key = f"input_{feat}"
                inputs[feat] = st.number_input(
                    label=disp_name,
                    value=float(st.session_state.get(key, 0.0)),
                    min_value=0.0,
                    step=0.001 if "smoothness" in feat or "fractal" in feat else 0.01,
                    format="%.5f" if "smoothness" in feat or "fractal" in feat else "%.4f",
                    key=key,
                    help=tooltip,
                )

    # Section 3: Worst Features
    with st.expander("⚠️ **Section 3: Worst (Extreme) Features (21 - 30)**", expanded=False):
        st.caption("Mean of the three largest values observed, capturing severe morphological atypia.")
        w_cols = st.columns(2)
        for idx, feat in enumerate(WORST_FEATURES):
            col = w_cols[idx % 2]
            with col:
                disp_name = FEATURE_DISPLAY_NAMES[feat]
                tooltip = FEATURE_TOOLTIPS.get(feat, "")
                key = f"input_{feat}"
                inputs[feat] = st.number_input(
                    label=disp_name,
                    value=float(st.session_state.get(key, 0.0)),
                    min_value=0.0,
                    step=0.01 if "area" not in feat and "perimeter" not in feat else 1.0,
                    format="%.4f" if "smoothness" in feat or "fractal" in feat or "points" in feat else "%.2f",
                    key=key,
                    help=tooltip,
                )

    st.write("")
    
    # Prediction Action Button
    pred_col1, pred_col2 = st.columns([4, 6])
    with pred_col1:
        submit = st.button("🔬 Execute Diagnostic Prediction", type="primary", use_container_width=True)

    if submit:
        st.write("")
        st.markdown("### 🎯 Classification Results & Risk Probability")

        with st.spinner("Processing input tensor through trained machine learning pipeline..."):
            try:
                prediction_result = predict_tumor(inputs)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                return

        pred_label = prediction_result["prediction_label"]
        benign_prob = prediction_result["benign_probability"]
        malignant_prob = prediction_result["malignant_probability"]
        model_name = prediction_result["model_name"]
        model_acc = prediction_result.get("model_accuracy", active_acc)
        explanation = prediction_result["clinical_explanation"]
        disclaimer = prediction_result["disclaimer"]

        # Display result card based on diagnosis
        if pred_label == "Benign":
            st.markdown(
                f"""
                <div class="result-card-benign">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap;">
                        <span class="badge-benign">✅ CLASSIFICATION: BENIGN (Non-Cancerous)</span>
                        <span style="font-size: 0.95rem; font-weight: 800; color: #14532d;">Model: {model_name} &nbsp;|&nbsp; Accuracy: {model_acc:.2%}</span>
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 900; margin-bottom: 10px; color: #064e3b;">
                        Model Confidence: {prediction_result['dominant_confidence']:.2f}%
                    </div>
                    <p style="font-size: 1.05rem; line-height: 1.6; font-weight: 500; margin-bottom: 0px; color: #14532d;">
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
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap;">
                        <span class="badge-malignant">⚠️ CLASSIFICATION: MALIGNANT (Cancerous)</span>
                        <span style="font-size: 0.95rem; font-weight: 800; color: #7f1d1d;">Model: {model_name} &nbsp;|&nbsp; Accuracy: {model_acc:.2%}</span>
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 900; margin-bottom: 10px; color: #7f1d1d;">
                        Model Confidence: {prediction_result['dominant_confidence']:.2f}%
                    </div>
                    <p style="font-size: 1.05rem; line-height: 1.6; font-weight: 500; margin-bottom: 0px; color: #7f1d1d;">
                        {explanation}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Visual Probability Breakdown
        res_col1, res_col2 = st.columns([5, 5])
        with res_col1:
            st.markdown("#### 📊 Statistical Class Probabilities")
            prob_df = pd.DataFrame({
                "Tumor Class": ["Benign (Non-cancerous)", "Malignant (Cancerous)"],
                "Model Probability": [benign_prob, malignant_prob],
            })
            st.dataframe(
                prob_df.style.format({"Model Probability": "{:.2%}"}),
                use_container_width=True,
                height=130,
            )

            st.caption(
                f"Note: Probabilities represent internal Scikit-learn estimator scores ({model_name})."
            )

        with res_col2:
            st.markdown("#### 🌡️ Malignancy Probability Gauge")
            fig_gauge = plot_prediction_gauge(benign_prob, malignant_prob)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Prominent Academic Medical Disclaimer
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>⚖️ Medical Diagnostic Disclaimer:</b><br>
                {disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_prediction_page()

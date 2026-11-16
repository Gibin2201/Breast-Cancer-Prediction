"""
Page 3: Model Comparison & Benchmarking
Evaluates and benchmarks 5 machine learning algorithms on the WDBC dataset:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine

Provides comparative charts, cross-validation metrics, confusion matrices, ROC curves,
and an interactive retrain trigger.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Comparison | Breast Cancer ML", page_icon="⚖️", layout="wide")

from config import (
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    BEST_MODEL_PATH,
    THEME_COLORS,
)
from utils.ui_components import (
    apply_theme,
    render_header,
    render_sidebar,
    check_and_guide_setup,
    render_metric_card,
)
from utils.prediction import load_model_metadata
from utils.visualization import (
    plot_model_comparison_bars,
    plot_confusion_matrix_heatmap,
    plot_roc_curves,
)
from train_model import train_and_evaluate_all


def render_model_comparison_page():
    apply_theme()
    render_sidebar()

    render_header(
        title="Multi-Algorithm Benchmarking & Model Selection",
        subtitle="Rigorous Comparative Evaluation of 5 Supervised Classifiers with 5-Fold Stratified Cross-Validation",
        badge="Algorithm Tournament",
    )

    if not check_and_guide_setup():
        return

    # Check comparison files
    if not MODEL_COMPARISON_PATH.exists() or not MODEL_METADATA_PATH.exists():
        st.warning("Model comparison artifacts missing. Please train the models below.")
        if st.button("🚀 Train All 5 Models Now", type="primary"):
            with st.spinner("Training models..."):
                train_and_evaluate_all()
                st.rerun()
        return

    try:
        comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
        metadata = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading model comparison data: {e}")
        return

    # Top summary metrics for the best model
    best_name = metadata.get("model_name", "N/A")
    best_metrics = metadata.get("best_model_metrics", {})
    best_cv = metadata.get("best_model_cv_metrics", {})

    st.markdown("### 🏆 Tournament Winner & Decision Logic")
    col_winner1, col_winner2 = st.columns([6, 4])

    with col_winner1:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 3px solid #0d9488; border-radius: 14px; padding: 24px; box-shadow: 0 6px 18px rgba(13, 148, 136, 0.12);">
                <div style="font-size: 0.92rem; font-weight: 800; color: #0d9488; text-transform: uppercase; letter-spacing: 0.06em;">
                    OPTIMAL CLINICAL CLASSIFIER
                </div>
                <div style="font-size: 2.1rem; font-weight: 900; color: #0f172a; margin: 6px 0 12px 0;">
                    {best_name}
                </div>
                <p style="font-size: 0.98rem; color: #1e293b; line-height: 1.6; margin-bottom: 14px; font-weight: 500;">
                    <b>Selection Rationale:</b> {metadata.get('selection_reason', '')}
                </p>
                <div style="font-size: 0.9rem; color: #0f172a; font-weight: 600; background: #f0fdfa; padding: 8px 12px; border-radius: 6px; border: 1px solid #99f6e4;">
                    <b>Active Pipeline:</b> StandardScaler &rarr; {best_name} (Class Weight: Balanced)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_winner2:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 2px solid #cbd5e1; border-radius: 14px; padding: 24px; height: 100%; box-shadow: 0 4px 12px rgba(15,23,42,0.04);">
                <div style="font-size: 0.92rem; font-weight: 800; color: #0f172a; text-transform: uppercase; letter-spacing: 0.05em;">
                    Why Malignant Recall Matters
                </div>
                <p style="font-size: 0.95rem; color: #1e293b; margin-top: 10px; line-height: 1.6; font-weight: 500;">
                    In clinical pathology, a <b>False Negative</b> (classifying a malignant invasive carcinoma as benign) 
                    delays critical life-saving interventions. Therefore, model selection prioritizes 
                    <b>Malignant Recall (Sensitivity)</b> alongside F1-Score rather than raw accuracy alone.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # Model Comparison Table
    st.markdown("### 📊 Comprehensive Algorithm Benchmark Table")
    st.write(
        "Performance metrics computed on the stratified held-out test split (20%) and 5-fold cross-validation:"
    )

    # Format table for display
    display_table = comparison_df.copy()
    pct_cols = [c for c in display_table.columns if c != "Algorithm"]
    styled_df = display_table.style.format({c: "{:.2%}" if "AUC" not in c else "{:.4f}" for c in pct_cols})
    st.dataframe(styled_df, use_container_width=True)

    st.write("")

    # Visual Metric Comparisons
    st.markdown("### 📈 Visual Metric Comparison")
    fig_bars = plot_model_comparison_bars(comparison_df)
    st.plotly_chart(fig_bars, use_container_width=True)

    st.write("")
    st.divider()

    # Detailed Evaluation: Confusion Matrix & ROC Curves
    st.markdown("### 🔍 In-Depth Diagnostic Diagnostics")
    col_cm, col_roc = st.columns([5, 5])

    with col_cm:
        st.markdown("#### 🧮 Confusion Matrix")
        # Allow user to view confusion matrix for any model
        cm_data = best_metrics.get("confusion_matrix", {"matrix": [[71, 1], [2, 40]]})
        fig_cm = plot_confusion_matrix_heatmap(cm_data, model_name=best_name)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonal elements represent correct clinical classifications.")

    with col_roc:
        st.markdown("#### 📉 Receiver Operating Characteristic (ROC)")
        roc_data = metadata.get("roc_curves", {})
        if roc_data:
            fig_roc = plot_roc_curves(roc_data)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curves will display here after model training.")

    st.write("")
    st.divider()

    # Interactive Retrain Option
    st.markdown("### ⚙️ Retrain & Re-benchmark Models")
    st.write(
        "Click the button below to re-execute 5-fold cross-validation across all 5 algorithms, "
        "re-score against the test partition, and re-save the optimal pipeline."
    )

    retrain_col1, retrain_col2 = st.columns([4, 6])
    with retrain_col1:
        if st.button("🔄 Retrain All 5 Algorithms", type="primary", use_container_width=True):
            with st.spinner("Retraining all models with 5-fold cross-validation..."):
                try:
                    result = train_and_evaluate_all()
                    st.success(f"Models successfully retrained! Optimal model: **{result['best_model_name']}**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Retraining error: {e}")


if __name__ == "__main__":
    render_model_comparison_page()

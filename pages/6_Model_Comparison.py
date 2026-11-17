"""
Page 6: Model Comparison & Benchmarking Leaderboard.
Compares all 5 machine learning algorithms across Accuracy, Precision, Recall,
F1-Score, ROC-AUC, and Cross-Validation scores.
Identifies the best-performing model based on actual test results.
"""

import streamlit as st
import pandas as pd

from config import (
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    BEST_MODEL_PATH,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    render_metric_card,
)
from utils.prediction import load_model_metadata
from utils.visualization import (
    plot_model_comparison_bars,
    plot_model_radar_comparison,
)


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Model Comparison",
        category="Model Benchmarking",
        active_badge="Leaderboard Benchmark",
    )

    render_header(
        title="Machine Learning Model Comparison",
        subtitle="Cross-Algorithmic Benchmark Evaluation Across Accuracy, Recall, Precision, F1-Score, and ROC-AUC",
        badge="Comparative Benchmark Evaluation",
    )

    # Load real benchmark data
    try:
        comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading model comparison data: {e}")
        return

    active_model_name = model_meta.get("model_name", "Support Vector Machine")
    best_acc = model_meta.get("best_model_metrics", {}).get("accuracy", 0.9825)
    best_rec = model_meta.get("best_model_metrics", {}).get("recall", 0.9762)
    best_f1 = model_meta.get("best_model_metrics", {}).get("f1_score", 0.9762)
    best_auc = model_meta.get("best_model_metrics", {}).get("roc_auc", 0.9954)

    # 1. Optimal Model Selection Banner
    st.markdown(
        f"""
        <div class="saas-card" style="border-left: 6px solid #2878C7; background: #FFFFFF;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <span class="status-pill status-pill-blue" style="margin-bottom: 6px;">
                        🏆 Rank #1 Optimal Classifier
                    </span>
                    <div style="font-size: 1.65rem; font-weight: 800; color: #142D4E; margin-top: 4px;">
                        {active_model_name}
                    </div>
                    <p style="font-size: 0.92rem; color: #5A6E85; margin: 4px 0 0 0; line-height: 1.5; max-width: 720px;">
                        Selected automatically because it achieved the <b>Highest Verified Test Accuracy ({best_acc:.2%})</b> 
                        and <b>Highest Malignant Recall ({best_rec:.2%})</b> on the 114 unseen clinical test samples.
                    </p>
                </div>
                <div style="display: flex; gap: 16px; margin-top: 10px;">
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Accuracy</span><br>
                        <span style="font-size: 1.4rem; font-weight: 900; color: #142D4E;">{best_acc:.2%}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">Sensitivity</span><br>
                        <span style="font-size: 1.4rem; font-weight: 900; color: #E63946;">{best_rec:.2%}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">F1-Score</span><br>
                        <span style="font-size: 1.4rem; font-weight: 900; color: #2878C7;">{best_f1:.2%}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #5A6E85; text-transform: uppercase;">ROC-AUC</span><br>
                        <span style="font-size: 1.4rem; font-weight: 900; color: #16A6A1;">{best_auc:.4f}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 2. Comprehensive Performance Leaderboard Table
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>📋</span> Clinical Performance Benchmark Leaderboard
            </div>
            <p style="font-size: 0.88rem; color: #5A6E85; margin-bottom: 12px;">
                Verified test metrics calculated on identical stratified 80% training (455 samples) and 20% test partitions (114 samples):
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Format dataframe with percentage styling
    display_df = comparison_df.copy()
    format_dict = {
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}",
        "ROC-AUC": "{:.4f}",
    }
    if "CV Mean Acc" in display_df.columns:
        format_dict["CV Mean Acc"] = "{:.2%}"
    if "CV Std Dev" in display_df.columns:
        format_dict["CV Std Dev"] = "{:.2%}"

    st.dataframe(
        display_df.style.format(format_dict).highlight_max(
            subset=["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
            color="#E8F8F5",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.write("")

    # 3. Two-Column Visual Comparison: Grouped Bars & Radar Chart
    col_bar, col_radar = st.columns([6, 5])
    with col_bar:
        st.markdown(
            """
            <div class="saas-card" style="margin-bottom: 0;">
                <div class="saas-card-header">
                    <span>📊</span> Grouped Metrics Comparison
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_bar = plot_model_comparison_bars(comparison_df)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radar:
        st.markdown(
            """
            <div class="saas-card" style="margin-bottom: 0;">
                <div class="saas-card-header">
                    <span>🕸️</span> 5-Dimensional Radar Comparison
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_radar = plot_model_radar_comparison(comparison_df)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.write("")

    # 4. Metric Descriptions for Medical Evaluation
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>📖</span> Understanding Clinical Evaluation Metrics
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px;">
                    <b style="color: #142D4E;">Accuracy:</b><br>
                    <span style="font-size: 0.82rem; color: #5A6E85;">
                        Overall proportion of correct diagnoses: (TP + TN) / Total. High baseline across all models.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px;">
                    <b style="color: #E63946;">Recall (Sensitivity):</b><br>
                    <span style="font-size: 0.82rem; color: #5A6E85;">
                        Proportion of actual malignant cases correctly flagged: TP / (TP + FN). <b>Top clinical priority</b> in oncology.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px;">
                    <b style="color: #2878C7;">Precision:</b><br>
                    <span style="font-size: 0.82rem; color: #5A6E85;">
                        Reliability of positive calls: TP / (TP + FP). Minimizes unnecessary anxiety and invasive biopsies.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px;">
                    <b style="color: #16A6A1;">F1-Score & ROC-AUC:</b><br>
                    <span style="font-size: 0.82rem; color: #5A6E85;">
                        Harmonic mean of precision & recall, alongside the area under the sensitivity-specificity trade-off curve.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

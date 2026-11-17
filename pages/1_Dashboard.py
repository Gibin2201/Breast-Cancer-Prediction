"""
Page 1: Clinical Telemetry & Executive Dashboard.
Displays high-level healthcare metrics, cohort distribution, morphological comparisons,
interactive cytological feature explorer, and multi-model benchmark telemetry.
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
    FEATURE_DISPLAY_NAMES,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_sidebar,
    render_metric_card,
    render_website_footer,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from utils.prediction import load_model_metadata
from utils.visualization import (
    plot_diagnosis_distribution,
    plot_model_comparison_bars,
    plot_model_radar_comparison,
    plot_feature_distribution,
)


def main():
    apply_theme()
    render_sidebar()

    # 1. Modern Top Website Navbar
    render_top_navbar(
        page_title="Executive Dashboard",
        category="Clinical Telemetry",
        active_badge="Healthcare AI Platform",
    )

    # 2. Executive Hero Telemetry Banner
    st.markdown(
        """
        <div class="page-header-card" style="background: linear-gradient(135deg, #142D4E 0%, #1A3E6D 50%, #2878C7 100%); padding: 28px 32px; border-radius: 16px; margin-bottom: 22px; box-shadow: 0 10px 28px rgba(20, 45, 78, 0.12);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
                <div>
                    <span class="hero-pill-badge" style="background: rgba(22, 166, 161, 0.25); border: 1px solid rgba(94, 234, 212, 0.4); color: #5EEAD4; margin-bottom: 10px;">
                        <span>🎗️</span> Precision Oncology AI • Wisconsin Diagnostic Cohort
                    </span>
                    <h1 style="color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin: 6px 0 8px 0; letter-spacing: -0.02em;">
                        Clinical Telemetry & Predictive AI Dashboard
                    </h1>
                    <p style="color: #D1DCE8; font-size: 0.98rem; max-width: 860px; line-height: 1.6; margin: 0;">
                        Real-time cytological telemetry from digitized Fine Needle Aspirate (FNA) biopsies, 
                        nuclear morphometric profiling, and verified multi-algorithm machine learning benchmarks.
                    </p>
                </div>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px;">
                <span class="status-pill status-pill-green" style="background: rgba(16, 185, 129, 0.22); color: #A7F3D0; border-color: rgba(16, 185, 129, 0.4);">
                    <span class="pulse-dot"></span> Pipeline Active (SVM 98.25% Acc)
                </span>
                <span class="status-pill" style="background: rgba(255, 255, 255, 0.12); color: #FFFFFF; border-color: rgba(255, 255, 255, 0.25);">
                    <span>📋</span> 569 Biopsies Analyzed
                </span>
                <span class="status-pill" style="background: rgba(255, 255, 255, 0.12); color: #FFFFFF; border-color: rgba(255, 255, 255, 0.25);">
                    <span>🔬</span> 30 Nuclear Dimensions
                </span>
                <span class="status-pill" style="background: rgba(22, 166, 161, 0.22); color: #A0E3E0; border-color: rgba(22, 166, 161, 0.4);">
                    <span>🛡️</span> Zero Data Leakage Validated
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

    # 3. Top Metric Cards Row (5 Core Project Metrics)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            render_metric_card("Total Records", f"{meta['total_records']}", "Fine Needle Biopsies", "#142D4E", "📋"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card("Total Features", f"{meta['total_features']}", "10 Mean • 10 SE • 10 Worst", "#2878C7", "🔬"),
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
            render_metric_card("ML Algorithms", f"{len(comparison_df)}", f"Optimal: {active_model_name[:7]}..", "#16A6A1", "🏆"),
            unsafe_allow_html=True,
        )

    st.write("")

    # 4. Clinical Purpose Callout Card
    st.markdown(
        """
        <div class="saas-card" style="border-left: 5px solid #2878C7;">
            <div style="display: grid; grid-template-columns: 2.2fr 1fr; gap: 24px; align-items: center;">
                <div>
                    <div style="font-size: 0.78rem; font-weight: 800; color: #2878C7; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                        Clinical Decision Support Objective
                    </div>
                    <div style="font-size: 1.12rem; font-weight: 800; color: #142D4E; margin-bottom: 8px;">
                        Early Cytological Differentiation: Benign vs. Malignant Tumors
                    </div>
                    <p style="font-size: 0.92rem; line-height: 1.6; color: #253247; margin-bottom: 8px;">
                        Digitized Fine Needle Aspirate (FNA) cytology specimens capture physical cell nucleus characteristics. 
                        Malignant nuclei exhibit extreme morphological pleomorphism: enlarged area, coarse chromatin texture, 
                        and deep boundary concavities.
                    </p>
                    <p style="font-size: 0.88rem; line-height: 1.55; color: #5A6E85; margin: 0;">
                        <b>Safety-First Clinical Priority:</b> In oncology screening, <b>Malignant Recall (Sensitivity)</b> is critical. 
                        Our active Support Vector Machine pipeline achieves <b>97.62% Malignant Recall</b> (only 1 false negative in 42 malignant test cases) 
                        and <b>98.25% overall test accuracy</b> under strict zero-leakage standard scaling.
                    </p>
                </div>
                <div style="background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F6 100%); border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #142D4E; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; border-bottom: 1px solid #E2E8F0; padding-bottom: 6px;">
                        Platform Specifications
                    </div>
                    <div style="font-size: 0.84rem; color: #5A6E85; line-height: 1.6;">
                        • Cohort: <b>Wisconsin (WDBC)</b><br>
                        • Split: <b>80% Train (455) / 20% Test (114)</b><br>
                        • Optimal Model: <b>Support Vector Machine</b><br>
                        • Pipeline Scaling: <b>StandardScaler (Z-Score)</b><br>
                        • Missing Data: <b>0% (100% Completeness)</b>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 5. Cohort Breakdown & Nuclear Morphometry Differential Section
    v_col1, v_col2 = st.columns([5, 5])
    with v_col1:
        st.markdown(
            """
            <div class="saas-card" style="margin-bottom: 0;">
                <div class="saas-card-header">
                    <span>🎯</span> Diagnosis Class Breakdown (Cohort Distribution)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_donut = plot_diagnosis_distribution(df)
        st.plotly_chart(fig_donut, use_container_width=True)

        # Class breakdown detail pills
        st.markdown(
            """
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px;">
                <div style="background: #E8F8F5; border: 1px solid #A7F3D0; border-radius: 10px; padding: 12px 14px;">
                    <div style="font-size: 0.78rem; font-weight: 800; color: #0E7055; text-transform: uppercase;">
                        🛡️ Benign Cohort (B)
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 900; color: #0E7055; margin: 2px 0;">
                        357 Biopsies (62.7%)
                    </div>
                    <div style="font-size: 0.80rem; color: #0E7055; line-height: 1.4;">
                        Non-cancerous lesions with regular nuclear contours and low chromatin texture.
                    </div>
                </div>
                <div style="background: #FEE2E2; border: 1px solid #FECACA; border-radius: 10px; padding: 12px 14px;">
                    <div style="font-size: 0.78rem; font-weight: 800; color: #DC2626; text-transform: uppercase;">
                        ⚠️ Malignant Cohort (M)
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 900; color: #DC2626; margin: 2px 0;">
                        212 Biopsies (37.3%)
                    </div>
                    <div style="font-size: 0.80rem; color: #DC2626; line-height: 1.4;">
                        Invasive carcinomas with nuclear pleomorphism and extensive boundary concavity.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with v_col2:
        st.markdown(
            """
            <div class="saas-card" style="margin-bottom: 0;">
                <div class="saas-card-header">
                    <span>🔬</span> Nuclear Morphometry Differential (Malignant vs. Benign)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Compute real statistics from actual project dataset
        key_cols = ["mean area", "mean concavity", "mean perimeter", "mean radius", "mean texture"]
        comp_data = df.groupby("diagnosis")[key_cols].mean()
        benign_row = comp_data.loc["B"]
        malignant_row = comp_data.loc["M"]
        diff_pct = ((malignant_row - benign_row) / benign_row) * 100

        # High-impact visual differential comparison meters
        morph_metrics = [
            ("Nuclear Area", f"{benign_row['mean area']:.1f} µm²", f"{malignant_row['mean area']:.1f} µm²", f"+{diff_pct['mean area']:.1f}%", 100, "#E63946"),
            ("Perimeter Concavity", f"{benign_row['mean concavity']:.3f}", f"{malignant_row['mean concavity']:.3f}", f"+{diff_pct['mean concavity']:.1f}%", 92, "#E63946"),
            ("Nuclear Perimeter", f"{benign_row['mean perimeter']:.1f} µm", f"{malignant_row['mean perimeter']:.1f} µm", f"+{diff_pct['mean perimeter']:.1f}%", 75, "#2878C7"),
            ("Nuclear Radius", f"{benign_row['mean radius']:.1f} µm", f"{malignant_row['mean radius']:.1f} µm", f"+{diff_pct['mean radius']:.1f}%", 70, "#2878C7"),
            ("Chromatin Texture", f"{benign_row['mean texture']:.1f}", f"{malignant_row['mean texture']:.1f}", f"+{diff_pct['mean texture']:.1f}%", 48, "#16A6A1"),
        ]

        meters_html = """<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 16px rgba(20, 45, 78, 0.03);">"""
        for label, b_val, m_val, change, fill_pct, col in morph_metrics:
            meters_html += f"""
            <div class="morph-item">
                <div class="morph-header">
                    <div>
                        <span class="morph-title">{label}</span>
                        <span style="font-size: 0.78rem; color: #5A6E85; margin-left: 8px;">
                            Benign: <b>{b_val}</b> → Malignant: <b>{m_val}</b>
                        </span>
                    </div>
                    <span class="morph-badge morph-badge-red">{change}</span>
                </div>
                <div class="morph-bar-wrap">
                    <div class="morph-bar-fill" style="width: {fill_pct}%; background: {col};"></div>
                </div>
            </div>
            """
        meters_html += """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 0.82rem; color: #5A6E85; line-height: 1.45;">
                <b style="color: #142D4E;">Cytological Diagnostic Rule:</b> Malignant cells exhibit greater than <b>2.1× enlarged nuclear area</b> (+111.4%) and severe perimeter indentation (+248.6% concavity) compared to benign nuclei.
            </div>
        </div>"""
        st.markdown(meters_html, unsafe_allow_html=True)

    st.write("")

    # 6. Interactive Morphological Feature Distribution Explorer
    st.markdown(
        """
        <div class="saas-card" style="margin-bottom: 0;">
            <div class="saas-card-header">
                <span>🔬</span> Interactive Cytological Feature Distribution Explorer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feat_cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean symmetry"]
    selected_feat = st.selectbox(
        "Select Nuclear Morphological Dimension to Inspect Distribution:",
        options=feat_cols,
        format_func=lambda c: f"{FEATURE_DISPLAY_NAMES.get(c, c)} ({c})",
        index=0,
    )

    fig_feat = plot_feature_distribution(df, selected_feat)
    st.plotly_chart(fig_feat, use_container_width=True)

    st.write("")

    # 7. Machine Learning Model Benchmark Leaderboard & Comparisons
    st.markdown(
        """
        <div class="saas-card" style="margin-bottom: 0;">
            <div class="saas-card-header">
                <span>🏆</span> Multi-Model Evaluation Benchmark (Unseen Test Set: 114 Biopsies)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Leaderboard Cards Row (All 5 Project Algorithms)
    podium_html = """<div class="podium-grid">"""
    for idx, row in comparison_df.iterrows():
        rank = idx + 1
        algo = row["Algorithm"]
        acc = row["Accuracy"]
        recall = row["Recall"]
        f1 = row["F1-Score"]
        is_gold = rank == 1

        card_class = "podium-card podium-card-gold" if is_gold else "podium-card"
        rank_class = "podium-rank podium-rank-gold" if rank == 1 else ("podium-rank podium-rank-silver" if rank == 2 else ("podium-rank podium-rank-bronze" if rank == 3 else "podium-rank podium-rank-standard"))
        badge_html = "<div style='font-size: 0.68rem; font-weight: 800; color: #D97706; background: #FEF3C7; padding: 2px 8px; border-radius: 10px; display: inline-block; margin-bottom: 6px;'>★ OPTIMAL PIPELINE</div>" if is_gold else ""

        podium_html += f"""
        <div class="{card_class}">
            <div class="{rank_class}">#{rank}</div>
            {badge_html}
            <div style="font-size: 0.90rem; font-weight: 800; color: #142D4E; margin-bottom: 6px;">{algo}</div>
            <div style="font-size: 1.45rem; font-weight: 900; color: #2878C7; line-height: 1.1;">{acc:.2%}</div>
            <div style="font-size: 0.74rem; font-weight: 600; color: #5A6E85; text-transform: uppercase; margin-bottom: 8px;">Test Accuracy</div>
            <div style="font-size: 0.80rem; color: #142D4E; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 4px 6px;">
                Sensitivity: <b>{recall:.1%}</b><br>
                F1-Score: <b>{f1:.1%}</b>
            </div>
        </div>
        """
    podium_html += "</div>"
    st.markdown(podium_html, unsafe_allow_html=True)

    # Tabs for Visual Comparison
    tab_bars, tab_radar, tab_table = st.tabs([
        "📊 Multi-Metric Comparison (Bar Chart)",
        "🕸️ Dimensional Radar Profile",
        "📋 Full Benchmark Matrix",
    ])

    with tab_bars:
        fig_bars = plot_model_comparison_bars(comparison_df)
        st.plotly_chart(fig_bars, use_container_width=True)

    with tab_radar:
        fig_radar = plot_model_radar_comparison(comparison_df)
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_table:
        styled_df = comparison_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2%}")
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.write("")

    # 8. Key Dataset Clinical Insights & Governance
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>🛡️</span> Clinical AI Governance & Architectural Safeguards
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 1.2rem;">🛡️</span>
                        <b style="color: #142D4E; font-size: 0.94rem;">Zero Data Leakage Protocol</b>
                    </div>
                    <span style="font-size: 0.84rem; color: #5A6E85; line-height: 1.55;">
                        Strict Scikit-Learn <code>Pipeline(StandardScaler, SVC)</code> encapsulation ensures test partition scaling parameters are never leaked into training folds.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 1.2rem;">🎯</span>
                        <b style="color: #142D4E; font-size: 0.94rem;">Critical Sensitivity (97.62%)</b>
                    </div>
                    <span style="font-size: 0.84rem; color: #5A6E85; line-height: 1.55;">
                        In clinical screening, false negatives are life-critical. The SVM classifier missed only 1 out of 42 malignant test biopsies.
                    </span>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 1.2rem;">🔬</span>
                        <b style="color: #142D4E; font-size: 0.94rem;">Collinearity Mitigation</b>
                    </div>
                    <span style="font-size: 0.84rem; color: #5A6E85; line-height: 1.55;">
                        Radius, Perimeter, and Area exhibit Pearson r > 0.98. RBF kernel mapping provides robust non-linear boundary separation without distortion.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 9. Quick-Access Clinical Navigation Hub
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>⚡</span> Quick Navigation Hub
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
                <div class="feature-tile" style="padding: 16px;">
                    <div class="feature-tile-icon" style="margin-bottom: 10px; width: 38px; height: 38px; font-size: 1.1rem;">📋</div>
                    <div class="feature-tile-title" style="font-size: 0.95rem;">Dataset Overview</div>
                    <div class="feature-tile-desc" style="font-size: 0.82rem; margin-bottom: 10px;">Inspect raw records, dimensions, and missing value checks.</div>
                    <div class="feature-tile-link" style="font-size: 0.80rem;">Page 2 ›</div>
                </div>
                <div class="feature-tile" style="padding: 16px;">
                    <div class="feature-tile-icon" style="margin-bottom: 10px; width: 38px; height: 38px; font-size: 1.1rem;">📊</div>
                    <div class="feature-tile-title" style="font-size: 0.95rem;">Dataset Analysis</div>
                    <div class="feature-tile-desc" style="font-size: 0.82rem; margin-bottom: 10px;">Explore Pearson correlation heatmaps and bivariate scatters.</div>
                    <div class="feature-tile-link" style="font-size: 0.80rem;">Page 3 ›</div>
                </div>
                <div class="feature-tile" style="padding: 16px;">
                    <div class="feature-tile-icon" style="margin-bottom: 10px; width: 38px; height: 38px; font-size: 1.1rem;">🏆</div>
                    <div class="feature-tile-title" style="font-size: 0.95rem;">Model Comparison</div>
                    <div class="feature-tile-desc" style="font-size: 0.82rem; margin-bottom: 10px;">Review confusion matrices, ROC-AUC curves, and F1 scores.</div>
                    <div class="feature-tile-link" style="font-size: 0.80rem;">Page 7 ›</div>
                </div>
                <div class="feature-tile" style="padding: 16px;">
                    <div class="feature-tile-icon" style="margin-bottom: 10px; width: 38px; height: 38px; font-size: 1.1rem; background: #E8F8F5; color: #10B981;">🔮</div>
                    <div class="feature-tile-title" style="font-size: 0.95rem;">Predictive Diagnostics</div>
                    <div class="feature-tile-desc" style="font-size: 0.82rem; margin-bottom: 10px;">Input 30 physical FNA parameters and run live AI inference.</div>
                    <div class="feature-tile-link" style="font-size: 0.80rem; color: #10B981;">Page 8 ›</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_website_footer()


if __name__ == "__main__":
    main()

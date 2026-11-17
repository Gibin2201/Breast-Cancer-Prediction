"""
Page 9: About Project & MCA Academic Documentation.
Comprehensive documentation covering clinical relevance, mathematical formulations,
FNA cytology methodology, software engineering architecture, limitations, and citations.
"""

import streamlit as st

from config import THEME_COLORS
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    render_website_footer,
)


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="About Project",
        category="Technical Brief",
        active_badge="MCA Documentation",
    )

    render_header(
        title="Project Documentation & Academic Defense Brief",
        subtitle="Mathematical Formulations, Fine Needle Aspirate (FNA) Cytology Methodology, and Architecture",
        badge="Academic Documentation & Viva Guide",
    )

    tabs = st.tabs([
        "📌 Clinical Background & Problem Statement",
        "⚙️ Machine Learning Architecture",
        "🤖 Mathematical Formulations (SVM & Kernels)",
        "💻 Software Engineering Stack",
        "📚 Academic Literature & Citations",
    ])

    # TAB 1: Clinical Background
    with tabs[0]:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>🎯</span> Problem Statement & Clinical Significance
                </div>
                <p style="font-size: 0.94rem; line-height: 1.65; color: #253247; margin-bottom: 12px;">
                    Breast cancer remains the most prevalent malignant neoplasm affecting women worldwide. 
                    Accurate differentiation of breast masses into <b>Benign (Non-cancerous, localized, non-invasive)</b> 
                    versus <b>Malignant (Cancerous, invasive, aggressive cellular proliferation)</b> is critical for patient survival. 
                    Manual cytologic review of Fine Needle Aspirates (FNA) is labor-intensive and subject to inter-observer variability.
                </p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-top: 14px;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                        <b style="color: #142D4E;">Early Detection Saves Lives:</b><br>
                        <span style="font-size: 0.85rem; color: #5A6E85;">
                            When localized, the 5-year relative survival rate exceeds 99%. Automated screening expedites clinical triage.
                        </span>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                        <b style="color: #E63946;">Prioritizing Malignant Recall (Sensitivity):</b><br>
                        <span style="font-size: 0.85rem; color: #5A6E85;">
                            A False Negative (calling cancer benign) is a life-threatening catastrophe. The active SVM model optimizes recall to 97.62%.
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TAB 2: ML Architecture
    with tabs[1]:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>🔬</span> End-to-End Machine Learning Pipeline Architecture
                </div>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #253247; margin-bottom: 12px;">
                    The system follows an enterprise-level, disciplined ML lifecycle adhering to Scikit-Learn best practices:
                </p>
                <pre style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; font-size: 0.84rem; color: #142D4E; line-height: 1.5;">
[UCI Machine Learning Repository WDBC Archive]
                      │
                      ▼
[download_dataset.py: Auto Download, Drop ID, Standardize Headers]
                      │
                      ▼
[utils/preprocessing.py: Stratified 80/20 Split (Seed 42)]
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Scikit-Learn Pipeline Encapsulation            │
│  1. StandardScaler (Fit strictly on 455 training samples)   │
│  2. Classifier Estimator (Tuned Hyperparameters)            │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
[5-Fold Stratified Cross-Validation & Test Metric Benchmark]
                      │
                      ▼
[Programmatic Model Selection: Support Vector Machine (98.25% Acc)]
                      │
                      ▼
[Persistence: models/best_model.pkl & models/model_metadata.json]
                      │
                      ▼
[Real-Time Streamlit Healthcare SaaS Interface (pages/8_Prediction.py)]
                </pre>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TAB 3: Mathematical Formulations
    with tabs[2]:
        st.markdown(
            r"""
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>📐</span> Support Vector Machine (SVM) Mathematical Foundation
                </div>
                <p style="font-size: 0.92rem; line-height: 1.6; color: #253247; margin-bottom: 8px;">
                    Given a training set of instance-label pairs $(x_i, y_i)$ where $x_i \in \mathbb{R}^{30}$ and $y_i \in \{-1, +1\}$:
                </p>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; font-family: monospace; font-size: 0.88rem; color: #142D4E; margin-bottom: 12px;">
                    <b>Primal Optimization Objective:</b><br>
                    min_{w, b, ξ}  ½ ||w||² + C ∑ ξ_i<br>
                    subject to: y_i (w · Φ(x_i) + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
                </div>
                <p style="font-size: 0.90rem; line-height: 1.6; color: #253247; margin-bottom: 8px;">
                    <b>Radial Basis Function (RBF) Kernel:</b> To handle non-linear boundaries between benign and malignant cell clusters:
                </p>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px; font-family: monospace; font-size: 0.88rem; color: #142D4E; margin-bottom: 12px;">
                    K(x, x') = exp(-γ ||x - x'||²),  where γ = 1 / (n_features · Var(X))
                </div>
                <p style="font-size: 0.88rem; color: #5A6E85; line-height: 1.5; margin-bottom: 0;">
                    <b>Platt Scaling Probability Calibration:</b> To transform raw distance from the hyperplane $f(x)$ into calibrated probabilities:
                    $P(Y = 1 | x) = \frac{1}{1 + \exp(A f(x) + B)}$, where parameters $A$ and $B$ are fitted via cross-validation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TAB 4: Tech Stack
    with tabs[3]:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>💻</span> Software Engineering Technology Stack
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                        <b style="color: #142D4E;">Frontend / Dashboard:</b><br>
                        <span style="font-size: 0.85rem; color: #5A6E85;">
                            • Streamlit 1.30+<br>
                            • Custom CSS SaaS Styling<br>
                            • Inter Typography
                        </span>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                        <b style="color: #2878C7;">Machine Learning Engine:</b><br>
                        <span style="font-size: 0.85rem; color: #5A6E85;">
                            • Scikit-Learn 1.4+<br>
                            • NumPy 1.26+<br>
                            • Joblib Pipeline Serialization
                        </span>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 8px;">
                        <b style="color: #16A6A1;">Interactive Visualizations:</b><br>
                        <span style="font-size: 0.85rem; color: #5A6E85;">
                            • Plotly Express<br>
                            • Plotly Graph Objects<br>
                            • Seaborn & Matplotlib
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TAB 5: References
    with tabs[4]:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-header">
                    <span>📚</span> Academic Literature Citations
                </div>
                <ol style="font-size: 0.90rem; color: #253247; line-height: 1.7; padding-left: 20px;">
                    <li>
                        <b>Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993).</b> 
                        <i>Nuclear feature extraction for breast tumor diagnosis.</i> 
                        IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870.
                    </li>
                    <li>
                        <b>Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995).</b> 
                        <i>Image analysis and machine learning applied to breast cancer diagnosis and prognosis.</i> 
                        Analytical and Quantitative Cytology and Histology, 17(2), 77-87.
                    </li>
                    <li>
                        <b>Cortes, C., & Vapnik, V. (1995).</b> 
                        <i>Support-vector networks.</i> 
                        Machine Learning, 20(3), 273-297.
                    </li>
                    <li>
                        <b>Pedregosa, F., et al. (2011).</b> 
                        <i>Scikit-learn: Machine Learning in Python.</i> 
                        Journal of Machine Learning Research, 12, 2825-2830.
                    </li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_website_footer()


if __name__ == "__main__":
    main()

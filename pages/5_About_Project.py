"""
Page 5: About Project & Documentation
Comprehensive documentation for MCA academic project defense, covering problem statement,
clinical relevance, ML architecture, mathematical formulations, limitations, and literature references.
"""

import streamlit as st

st.set_page_config(page_title="About Project | Breast Cancer ML", page_icon="📖", layout="wide")

from config import THEME_COLORS
from utils.ui_components import apply_theme, render_header, render_sidebar


def render_about_page():
    apply_theme()
    render_sidebar()

    render_header(
        title="Project Documentation & MCA Defense Brief",
        subtitle="Architectural Specifications, Machine Learning Methodology, and Mathematical Formulations",
        badge="Academic Documentation",
    )

    tabs = st.tabs([
        "📌 Problem Statement & Relevance",
        "⚙️ ML Methodology & Architecture",
        "🤖 Algorithms & Formulations",
        "💻 Technology Stack",
        "🔭 Limitations & Future Scope",
        "📚 References",
    ])

    # TAB 1: Problem Statement
    with tabs[0]:
        st.markdown("### 🎯 Problem Statement")
        st.write(
            "Breast cancer remains the foremost malignant neoplasm affecting women worldwide. "
            "Accurate differentiation of breast masses into **Benign** (non-cancerous, localized, non-invasive) "
            "versus **Malignant** (cancerous, invasive, high mitotic activity) is paramount. "
            "Conventional manual cytologic inspection of fine needle aspirate (FNA) slides is time-intensive "
            "and susceptible to inter-observer variability."
        )

        st.markdown("### 🏥 Clinical Relevance & Need")
        st.markdown(
            """
            - **Early Detection:** When diagnosed at localized stages, breast cancer has a 5-year relative survival rate of over 99%.
            - **Reduction in Unnecessary Biopsies:** Accurate computational screening reduces trauma and cost associated with exploratory excisional surgeries for benign lesions.
            - **Minimizing False Negatives:** In oncology, a False Negative (missed malignancy) is life-threatening. Algorithmic tuning prioritizes **Malignant Recall (Sensitivity)** to protect patient outcomes.
            - **Standardized Objectivity:** Quantitative image feature extraction provides unbiased, reproducible morphological analysis.
            """
        )

        st.markdown("### 🎯 Project Objectives")
        st.markdown(
            """
            1. **Automated Pipeline:** Download and validate the official Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset with zero manual friction.
            2. **Data Leakage Prevention:** Implement Scikit-learn `Pipeline` architectures that fit transformers exclusively on training partitions.
            3. **Multi-Model Benchmark:** Implement, cross-validate, and evaluate 5 distinct ML paradigms (Logistic Regression, Decision Tree, Random Forest, KNN, SVM).
            4. **Clinical Model Selection:** Optimize model selection using a composite objective prioritizing cross-validated Malignant Recall.
            5. **Interactive Interface:** Deliver a Streamlit web application with real-time inference, probability estimation, and EDA telemetry.
            """
        )

    # TAB 2: ML Methodology & Pipeline
    with tabs[1]:
        st.markdown("### 🔬 Machine Learning Lifecycle Architecture")
        st.markdown(
            """
            The project follows an end-to-end disciplined data science lifecycle:
            
            ```
            [Official UCI Repository / Sklearn Fallback]
                                |
                                v
                    [Data Validation & Cleaning]
              (569 rows x 30 features, Drop ID, Encode B=0, M=1)
                                |
                                v
                   [Stratified Train-Test Split]
                         (80% Train, 20% Test)
                                |
                                v
               +---------------------------------+
               |   Scikit-Learn Pipeline Flow   |
               |  - StandardScaler (Fit on Train) |
               |  - Estimator (Classifier)       |
               +---------------------------------+
                                |
                                v
                 [5-Fold Stratified Cross-Validation]
                                |
                                v
                   [Model Benchmark & Selection]
               (Score = 0.45*Recall + 0.35*F1 + 0.20*AUC)
                                |
                                v
               [Artifact Persistence (Joblib & JSON)]
                                |
                                v
             [Real-Time Streamlit Decision Support Web UI]
            ```
            """
        )

        st.markdown("#### 🛡️ Prevention of Data Leakage")
        st.info(
            "A critical pitfall in medical machine learning is applying normalization (StandardScaler) "
            "across the entire dataset before train/test splitting. In this project, all scalers are "
            "encapsulated inside Scikit-learn `Pipeline` objects. The scaler fits strictly on `X_train` "
            "and only transforms `X_test` or incoming user prediction inputs, guaranteeing valid generalizability."
        )

    # TAB 3: Algorithms & Formulations
    with tabs[2]:
        st.markdown("### 🤖 Comparative Supervised Learning Classifiers")

        st.markdown("#### 1. Logistic Regression")
        st.write(
            "Models the posterior log-odds of malignancy as a linear combination of input features: "
            "$$\\log\\left(\\frac{P(Y=1|X)}{1 - P(Y=1|X)}\\right) = \\beta_0 + \\sum_{i=1}^{30} \\beta_i X_i$$ "
            "Trained with L2 ridge regularization and balanced class weighting."
        )

        st.markdown("#### 2. Decision Tree Classifier (CART)")
        st.write(
            "Constructs an orthogonal hierarchical decision tree using Gini Impurity: "
            "$$Gini(t) = 1 - \\sum_{k=0}^{1} p_k^2$$ "
            "Constrained with maximum depth 5 and minimum leaf samples to curb overfitting."
        )

        st.markdown("#### 3. Random Forest Classifier")
        st.write(
            "Ensemble of 100 de-correlated bagged decision trees with random feature subspace sampling: "
            "$$\\hat{y} = \\text{mode}\\left(\\{h_b(x)\\}_{b=1}^{B}\\right)$$ "
            "Substantially reduces variance while maintaining low bias."
        )

        st.markdown("#### 4. K-Nearest Neighbors (KNN)")
        st.write(
            "Non-parametric distance-based instance classifier utilizing Minkowski metric ($p=2$, Euclidean) "
            "with inverse-distance weighting ($w_i = 1 / d(x, x_i)$) so closer neighbors exert greater clinical influence."
        )

        st.markdown("#### 5. Support Vector Machine (SVM)")
        st.write(
            "Maximizes the clinical decision margin in transformed feature space via the Radial Basis Function (RBF) kernel: "
            "$$K(x, x') = \\exp\\left(-\\gamma \\|x - x'\\|^2\\right)$$ "
            "Calibrated using Platt scaling to output continuous class probabilities."
        )

    # TAB 4: Technology Stack
    with tabs[3]:
        st.markdown("### 💻 Technology & Library Stack")
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            st.markdown(
                """
                - **Language:** Python 3.11+
                - **Machine Learning Core:** Scikit-Learn 1.4+
                - **Data Structures:** Pandas 2.2+
                - **Numerical Computing:** NumPy 1.26+
                - **Model Persistence:** Joblib 1.3+
                """
            )
        with tech_col2:
            st.markdown(
                """
                - **Interactive Frontend:** Streamlit 1.32+
                - **Visualizations:** Plotly 5.19+, Seaborn, Matplotlib
                - **Unit Testing Suite:** Pytest 8.0+
                - **Image Processing:** Pillow (PIL) 10.2+
                - **Spreadsheet Interop:** OpenPyXL 3.1+
                """
            )

    # TAB 5: Limitations & Future Enhancements
    with tabs[4]:
        st.markdown("### ⚠️ Project Limitations")
        st.markdown(
            """
            1. **Dataset Scope:** The WDBC cohort consists of 569 patient biopsies collected in the 1990s at the University of Wisconsin Hospital. It lacks demographic diversity across modern global populations.
            2. **Tabular Feature Dependency:** The model relies on pre-computed morphological metrics rather than raw whole-slide histopathology images (WSI).
            3. **Static Thresholds:** Fixed binary decision thresholds (0.50) may need dynamic adjustment based on clinical risk appetite.
            """
        )

        st.markdown("### 🚀 Future Enhancements")
        st.markdown(
            """
            - **Deep Learning Integration:** Implement Convolutional Neural Networks (ResNet, EfficientNet) directly on raw microscopic FNA cytology slides.
            - **Multimodal Prognosis:** Fuse tabular morphometry with patient genomic profiles (BRCA1/BRCA2) and clinical history.
            - **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) and LIME to generate patient-specific nuclear contribution waterfall plots.
            - **DICOM & PACS Interoperability:** Build FHIR-compliant REST endpoints for hospital electronic health record (EHR) integration.
            """
        )

    # TAB 6: References
    with tabs[5]:
        st.markdown("### 📚 Academic & Clinical References")
        st.markdown(
            """
            1. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993).** *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870.
            2. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995).** *Image analysis and machine learning applied to breast cancer diagnosis and prognosis.* Analytical and Quantitative Cytology and Histology, 17(2), 77-87.
            3. **UCI Machine Learning Repository:** *Breast Cancer Wisconsin (Diagnostic) Data Set.* Available: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
            4. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
            5. **American Cancer Society (2024).** *Breast Cancer Facts & Figures 2023-2024.* Atlanta: American Cancer Society, Inc.
            """
        )


if __name__ == "__main__":
    render_about_page()

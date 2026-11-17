"""
Page 5: Machine Learning Models.
Displays detailed algorithmic profiles for all 5 implemented models:
Logistic Regression, Decision Tree, Random Forest, KNN, and Support Vector Machine.
Details working principles, mathematical formulation, hyperparameters, training status,
and evaluation scores.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

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
    render_website_footer,
)
from utils.prediction import load_model_metadata


# Technical algorithmic reference profiles
ALGO_PROFILES = {
    "Support Vector Machine": {
        "badge": "Active Winning Classifier",
        "icon": "🏆",
        "family": "Kernelized Maximum-Margin Classifier",
        "overview": (
            "Support Vector Machines (SVM) find the optimal hyperplane that maximizes the margin "
            "(distance) between the two closest opposing tumor data points (support vectors). "
            "Using the Radial Basis Function (RBF) kernel, it projects the 30-dimensional nuclear "
            "feature space into an infinite-dimensional Hilbert space where non-linear cytological "
            "boundaries become linearly separable."
        ),
        "math": "f(x) = sign(∑ α_i y_i K(x_i, x) + b), where K(x, x') = exp(-γ ||x - x'||^2)",
        "hyperparameters": {
            "Kernel": "RBF (Radial Basis Function)",
            "C (Regularization)": "1.0 (Balanced soft margin penalty)",
            "Gamma": "'scale' (1 / (n_features * X.var()))",
            "Class Weight": "'balanced' (Inversely proportional to class frequencies)",
            "Probability": "True (Platt scaling calibrated sigmoid)",
        },
        "clinical_suitability": (
            "Exceptional performance on high-dimensional biomedical data (p = 30) with moderate sample sizes (n = 569). "
            "Maximizing the geometric margin yields superior generalization and minimizes False Negatives."
        ),
    },
    "Logistic Regression": {
        "badge": "Baseline Parametric Model",
        "icon": "📈",
        "family": "Generalized Linear Model (GLM)",
        "overview": (
            "Logistic Regression models the probability of a binary tumor outcome using the logistic sigmoid function. "
            "It learns a linear combination of the 30 nuclear features and maps the continuous log-odds score into a "
            "calibrated probability range [0, 1]."
        ),
        "math": "P(Y=1|X) = 1 / (1 + exp(- (β_0 + ∑ β_i X_i)))",
        "hyperparameters": {
            "Solver": "'lbfgs' (Limited-memory BFGS quasi-Newton)",
            "Regularization (C)": "1.0 (Inverse regularization strength)",
            "Penalty": "L2 Ridge regularization",
            "Max Iterations": "1000 (Guaranteed convergence)",
            "Class Weight": "'balanced'",
        },
        "clinical_suitability": (
            "Highly interpretable with well-calibrated probabilities. Coefficients directly indicate how changes in "
            "nuclear dimensions impact tumor log-odds. Highly resilient to overfitting when features are properly scaled."
        ),
    },
    "Random Forest": {
        "badge": "Ensemble Bagging Classifier",
        "icon": "🌲",
        "family": "Bootstrapped Decision Forest",
        "overview": (
            "Random Forest is an ensemble meta-estimator that fits 100 decorrelated decision tree classifiers on various "
            "bootstrapped subsamples of the clinical dataset. It combines individual tree predictions through majority "
            "voting, drastically reducing model variance and resisting noise."
        ),
        "math": "H(x) = mode(h_1(x), h_2(x), ..., h_B(x)), where B = 100 bagged trees",
        "hyperparameters": {
            "Number of Trees": "100 estimators",
            "Criterion": "'gini' (Gini impurity minimization)",
            "Max Depth": "None (Fully grown until pure leaves)",
            "Max Features": "'sqrt' (Square root of 30 features per split ≈ 5)",
            "Class Weight": "'balanced'",
        },
        "clinical_suitability": (
            "Captures complex non-linear feature interactions without requiring feature scaling. Provides built-in "
            "Gini feature importance rankings across nuclear morphology metrics."
        ),
    },
    "K-Nearest Neighbors": {
        "badge": "Instance-Based Non-Parametric Model",
        "icon": "📍",
        "family": "Lazy Instance Learner",
        "overview": (
            "K-Nearest Neighbors (KNN) classifies an unknown biopsy sample by locating its k closest neighbors in the "
            "30-dimensional standardized feature space based on Euclidean distance. The biopsy is assigned to the majority "
            "class among its nearest peers."
        ),
        "math": "d(p, q) = sqrt(∑ (p_i - q_i)^2), k = 5 nearest clinical neighbors",
        "hyperparameters": {
            "n_neighbors": "5 (Optimal balance between bias and variance)",
            "Metric": "'euclidean' (L2 distance in scaled space)",
            "Weights": "'uniform' (Equal vote per neighbor)",
            "Algorithm": "'auto' (Standard KD-Tree / Ball-Tree indexing)",
        },
        "clinical_suitability": (
            "Intuitively mirrors clinical case-based reasoning: new tumors are diagnosed by comparing them to the most "
            "morphologically similar historical biopsy records. Strict standard scaling is vital to prevent high-magnitude features from dominating distance."
        ),
    },
    "Decision Tree": {
        "badge": "Hierarchical Rule-Based Model",
        "icon": "🌿",
        "family": "Recursive Partitioning Tree",
        "overview": (
            "Decision Tree recursively partitions the 30-dimensional feature space into orthogonal hyper-rectangles by "
            "selecting the feature and threshold that maximizes Gini impurity reduction at each node. It creates a human-readable "
            "flowchart of clinical diagnostic rules."
        ),
        "math": "Gini(D) = 1 - ∑ p_i^2, where p_i is the proportion of class i at node D",
        "hyperparameters": {
            "Criterion": "'gini' impurity",
            "Splitter": "'best' (Exhaustive search over all thresholds)",
            "Max Depth": "5 (Pruned to prevent catastrophic overfitting)",
            "Min Samples Split": "5",
            "Class Weight": "'balanced'",
        },
        "clinical_suitability": (
            "Maximum interpretability: pathologists can trace the exact logical branch leading to a diagnosis. "
            "However, prone to high variance and unstable decision boundaries, which ensemble methods like Random Forest resolve."
        ),
    },
}


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Machine Learning Models",
        category="Model Architecture",
        active_badge="5 Evaluated Algorithms",
    )

    render_header(
        title="Machine Learning Algorithm Profiles",
        subtitle="Mathematical Foundations, Hyperparameter Configurations, Training Status, and Clinical Evaluation Metrics",
        badge="Supervised Learning Architecture",
    )

    # Load comparison metrics
    try:
        comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
        model_meta = load_model_metadata()
    except Exception as e:
        st.error(f"Error loading model comparison data: {e}")
        return

    # Model Summary KPI Bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            render_metric_card("Evaluated Models", f"{len(comparison_df)}", "Supervised Algorithms", "#142D4E", "🤖"),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_metric_card("Optimal Classifier", "SVM", "Rank #1 Overall", "#2878C7", "🏆"),
            unsafe_allow_html=True,
        )
    with k3:
        best_acc = comparison_df["Accuracy"].max()
        st.markdown(
            render_metric_card("Top Accuracy", f"{best_acc:.2%}", "Unseen Test Set", "#10B981", "🎯"),
            unsafe_allow_html=True,
        )
    with k4:
        best_rec = comparison_df["Recall"].max()
        st.markdown(
            render_metric_card("Top Recall", f"{best_rec:.2%}", "Sensitivity Score", "#E63946", "🩺"),
            unsafe_allow_html=True,
        )

    st.write("")

    # Loop through all 5 algorithms and render attractive profiles
    for _, row in comparison_df.iterrows():
        algo_name = row["Algorithm"]
        profile = ALGO_PROFILES.get(algo_name, {})
        is_best = "Support Vector" in algo_name

        card_border = "#2878C7" if is_best else "#E2E8F0"
        header_badge = (
            "<span class='status-pill status-pill-blue'>🏆 Optimal Clinical Classifier (Rank #1)</span>"
            if is_best
            else f"<span class='status-pill status-pill-teal'>{profile.get('family', 'Classifier')}</span>"
        )

        with st.expander(
            f"{profile.get('icon', '🔹')} **{algo_name}** — Test Accuracy: **{row['Accuracy']:.2%}** | Recall: **{row['Recall']:.2%}**",
            expanded=is_best,
        ):
            st.markdown(
                f"""
                <div style="border-left: 4px solid {card_border}; padding-left: 14px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 6px;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #142D4E;">{algo_name}</span>
                        {header_badge}
                    </div>
                    <p style="font-size: 0.92rem; color: #253247; line-height: 1.6; margin-bottom: 8px;">
                        {profile.get('overview', 'Supervised classification algorithm evaluated on the WDBC cohort.')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metric Score Pills Row
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("Test Accuracy", f"{row['Accuracy']:.2%}")
            with m_col2:
                st.metric("Malignant Recall", f"{row['Recall']:.2%}")
            with m_col3:
                st.metric("Precision", f"{row['Precision']:.2%}")
            with m_col4:
                st.metric("F1-Score", f"{row['F1-Score']:.2%}")
            with m_col5:
                st.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")

            st.write("")

            # Two columns: Hyperparameters & Mathematics
            param_col, math_col = st.columns(2)
            with param_col:
                st.markdown("##### ⚙️ Tuned Hyperparameters")
                param_dict = profile.get("hyperparameters", {})
                param_table = pd.DataFrame(
                    [{"Parameter": k, "Configured Value": v} for k, v in param_dict.items()]
                )
                st.dataframe(param_table, use_container_width=True, hide_index=True)

            with math_col:
                st.markdown("##### 📐 Mathematical Formulation")
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px 14px; border-radius: 8px; font-family: monospace; font-size: 0.86rem; color: #142D4E; margin-bottom: 10px;">
                        {profile.get('math', 'Equation')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("##### 🩺 Clinical Suitability")
                st.caption(profile.get("clinical_suitability", ""))

    render_website_footer()


if __name__ == "__main__":
    main()

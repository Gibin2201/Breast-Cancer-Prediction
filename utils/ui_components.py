"""
UI Components and Theme Styling for the Breast Cancer Prediction Web Application.
Provides healthcare-inspired CSS, responsive metric cards, banners, and setup checks.
"""

import streamlit as st
from typing import Any
from pathlib import Path
from config import (
    THEME_COLORS,
    LOGO_PATH,
    DATASET_PATH,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
)
from download_dataset import download_dataset
from train_model import train_and_evaluate_all


CUSTOM_CSS = """
<style>
    /* Google Fonts & Base Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }

    /* High-Contrast Healthcare Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #091324 0%, #0f2744 50%, #0d9488 100%);
        border-radius: 14px;
        padding: 28px 36px;
        color: #ffffff;
        margin-bottom: 26px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.16);
        border: 2px solid rgba(255, 255, 255, 0.18);
    }
    .header-banner h1 {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 2.15rem !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .header-banner p {
        color: #e2e8f0 !important;
        font-size: 1.05rem !important;
        font-weight: 500;
        margin-bottom: 0px !important;
        line-height: 1.6;
    }

    /* High-Contrast Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 22px 24px;
        border: 2px solid #cbd5e1;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        border-color: #94a3b8;
    }
    .metric-title {
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #334155;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.15;
    }
    .metric-sub {
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 6px;
    }

    /* High-Contrast Result Cards */
    .result-card-benign {
        background-color: #f0fdf4 !important;
        border: 3px solid #16a34a !important;
        border-radius: 14px;
        padding: 26px;
        color: #14532d !important;
        box-shadow: 0 6px 18px rgba(22, 163, 74, 0.18);
    }
    .result-card-malignant {
        background-color: #fef2f2 !important;
        border: 3px solid #dc2626 !important;
        border-radius: 14px;
        padding: 26px;
        color: #7f1d1d !important;
        box-shadow: 0 6px 18px rgba(220, 38, 38, 0.18);
    }
    .badge-benign {
        background-color: #16a34a !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.3);
    }
    .badge-malignant {
        background-color: #dc2626 !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
    }

    /* High-Contrast Clinical Disclaimer Box */
    .disclaimer-box {
        background-color: #f1f5f9;
        border-left: 6px solid #0d9488;
        border-right: 1px solid #cbd5e1;
        border-top: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
        padding: 18px 22px;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #0f172a;
        margin-top: 22px;
        line-height: 1.6;
        font-weight: 500;
    }

    /* Sidebar High-Contrast Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* High-Contrast Navigation CTA Card */
    .nav-card {
        background: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
    }
    .nav-card:hover {
        border-color: #0d9488;
        box-shadow: 0 8px 20px rgba(13, 148, 136, 0.14);
        transform: translateY(-2px);
    }
    .nav-card h4 {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-bottom: 8px !important;
    }
    .nav-card p {
        color: #334155 !important;
        font-size: 0.95rem !important;
        line-height: 1.5;
        font-weight: 500;
    }
    .nav-card a {
        font-size: 1rem !important;
        font-weight: 700 !important;
    }

    /* Headings Enhancement */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    p, span, label {
        color: #1e293b;
    }

    /* Form & Input Enhancements */
    div[data-testid="stExpander"] {
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
        margin-bottom: 14px !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        color: #0f172a !important;
    }

    /* Primary Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #0d9488 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0f766e !important;
        box-shadow: 0 6px 16px rgba(13, 148, 136, 0.45) !important;
    }

    /* Tables & Dataframes */
    div[data-testid="stDataFrame"] {
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
</style>
"""


def apply_theme():
    """Inject custom healthcare styles into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: str = "MCA Micro Project"):
    """Render a uniform top banner across all pages."""
    st.markdown(
        f"""
        <div class="header-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <span style="background: rgba(13, 148, 136, 0.3); border: 1px solid #14b8a6; color: #5eead4; font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.05em;">
                        {badge}
                    </span>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render unified navigation and project context in the Streamlit sidebar."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=110)
        
        st.markdown("### **Breast Cancer ML**")
        st.caption("Wisconsin Diagnostic Breast Cancer (WDBC) Clinical Prediction System")

        st.divider()

        st.markdown("#### **System Status**")
        # Check dataset status
        if DATASET_PATH.exists():
            st.success("Dataset: Loaded (569 records)", icon="✅")
        else:
            st.warning("Dataset: Missing", icon="⚠️")

        # Check model status
        if BEST_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
            st.success("Best Model: Ready", icon="✅")
        else:
            st.warning("Model: Not Trained", icon="⚠️")

        st.divider()
        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">
                <b>Academic Notice:</b><br>
                For educational & MCA project demonstration only. Not for medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )


def check_and_guide_setup() -> bool:
    """
    Check if required dataset and model artifacts exist.
    If missing, display interactive guided setup buttons.
    Returns True if system is ready, False if user action is needed.
    """
    needs_dataset = not DATASET_PATH.exists()
    needs_model = not (BEST_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists())

    if needs_dataset or needs_model:
        st.warning("⚠️ **Initial Setup Required**")
        st.info(
            "Some required project artifacts are not yet detected. You can download the dataset "
            "and train the models directly using the buttons below."
        )

        col1, col2 = st.columns(2)
        if needs_dataset:
            with col1:
                if st.button("📥 Download Official Dataset", type="primary", use_container_width=True):
                    with st.spinner("Downloading and validating WDBC dataset..."):
                        try:
                            download_dataset()
                            st.success("Dataset downloaded & verified successfully! Refreshing...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Download failed: {e}")

        if needs_model and not needs_dataset:
            with col2:
                if st.button("⚙️ Train All 5 Models", type="primary", use_container_width=True):
                    with st.spinner("Training 5 algorithms with 5-fold cross-validation..."):
                        try:
                            train_and_evaluate_all()
                            st.success("Models trained & saved successfully! Refreshing...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Training failed: {e}")

        return False

    return True


def render_metric_card(title: str, value: Any, subtext: str = "", delta_color: str = "#0d9488"):
    """Render a high-contrast HTML metric card with prominent left accent stripe."""
    return f"""
    <div class="metric-card" style="border-left: 6px solid {delta_color};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub" style="color: {delta_color};">{subtext}</div>
    </div>
    """

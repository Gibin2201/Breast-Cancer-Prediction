"""
UI Components and Healthcare SaaS Theme Styling for Breast Cancer Prediction System.
Implements the enterprise clinical AI design system:
- Primary: Deep Navy (#142D4E)
- Secondary: Medical Blue (#2878C7)
- Accent: Teal (#16A6A1)
- Background: Light Grey (#F5F7FA)
- Cards: White (#FFFFFF)
- Text: Dark Slate (#253247)
- Typography: Inter, sans-serif
"""

import streamlit as st
from typing import Any, Optional
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


# -----------------------------------------------------------------------------
# Enterprise Healthcare SaaS CSS Stylesheet
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Google Fonts: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global Typography & Base Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #F5F7FA !important;
        color: #253247 !important;
    }

    /* Main Container Spacing */
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1350px !important;
    }

    /* Top Navigation / Breadcrumb Bar */
    .top-navbar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 22px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 2px 10px rgba(20, 45, 78, 0.04);
    }
    .breadcrumb-text {
        font-size: 0.82rem;
        font-weight: 600;
        color: #5A6E85;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .breadcrumb-active {
        color: #2878C7;
        font-weight: 700;
    }
    .status-badge-container {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 20px;
        background: #F5F7FA;
        color: #142D4E;
        border: 1px solid #E2E8F0;
    }
    .status-pill-green {
        background: #E8F8F5;
        color: #0E7055;
        border: 1px solid #A7F3D0;
    }
    .status-pill-blue {
        background: #EBF4FC;
        color: #1A5FA3;
        border: 1px solid #BCE0FD;
    }
    .status-pill-teal {
        background: #E6F7F7;
        color: #0D7571;
        border: 1px solid #A0E3E0;
    }

    /* Healthcare SaaS Executive Header Banner */
    .executive-header {
        background: linear-gradient(135deg, #142D4E 0%, #1D3B63 60%, #1A4F82 100%);
        border-radius: 14px;
        padding: 26px 32px;
        color: #FFFFFF;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 24px rgba(20, 45, 78, 0.12);
        position: relative;
        overflow: hidden;
    }
    .executive-header::after {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(22, 166, 161, 0.25) 0%, rgba(22, 166, 161, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .executive-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(22, 166, 161, 0.2);
        border: 1px solid rgba(22, 166, 161, 0.45);
        color: #5EEAD4;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .executive-header h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.95rem !important;
        margin: 0 0 8px 0 !important;
        letter-spacing: -0.02em;
    }
    .executive-header p {
        color: #D1DCE8 !important;
        font-size: 0.98rem !important;
        margin: 0 !important;
        line-height: 1.55;
        max-width: 860px;
        font-weight: 400;
    }

    /* Metric Cards - Modern Healthcare SaaS */
    .metric-card-saas {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 4px 16px rgba(20, 45, 78, 0.04);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card-saas:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(20, 45, 78, 0.09);
        border-color: #2878C7;
    }
    .metric-card-stripe {
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    .metric-label-saas {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #5A6E85;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value-saas {
        font-size: 2.15rem;
        font-weight: 800;
        color: #142D4E;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .metric-subtext-saas {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Clinical Result Cards */
    .result-card-benign {
        background-color: #E8F8F5 !important;
        border: 2px solid #10B981 !important;
        border-radius: 14px;
        padding: 24px 28px;
        color: #0E7055 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.12);
        margin-top: 18px;
    }
    .result-card-malignant {
        background-color: #FEE8E8 !important;
        border: 2px solid #E63946 !important;
        border-radius: 14px;
        padding: 24px 28px;
        color: #921E28 !important;
        box-shadow: 0 6px 20px rgba(230, 57, 70, 0.14);
        margin-top: 18px;
    }
    .badge-benign-pill {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.92rem;
        display: inline-block;
        letter-spacing: 0.04em;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25);
    }
    .badge-malignant-pill {
        background-color: #E63946 !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.92rem;
        display: inline-block;
        letter-spacing: 0.04em;
        box-shadow: 0 2px 8px rgba(230, 57, 70, 0.25);
    }

    /* Content Cards & Containers */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(20, 45, 78, 0.04);
    }
    .saas-card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #142D4E;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 8px;
    }

    /* Clinical Disclaimer Box */
    .disclaimer-card {
        background-color: #FFFFFF;
        border-left: 4px solid #16A6A1;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 0.88rem;
        color: #5A6E85;
        margin-top: 24px;
        line-height: 1.55;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #142D4E !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }

    /* Streamlit Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 16px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #5A6E85;
        border: none !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2878C7 !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Primary and Secondary Action Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #2878C7 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(40, 120, 199, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #142D4E !important;
        box-shadow: 0 6px 18px rgba(20, 45, 78, 0.35) !important;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #142D4E !important;
        font-weight: 600 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #2878C7 !important;
        color: #2878C7 !important;
    }

    /* Dataframe Table Enhancements */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(20, 45, 78, 0.02) !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: #142D4E !important;
    }

    /* Radio buttons and inputs */
    div.row-widget.stRadio > div {
        flex-direction: row;
        gap: 16px;
    }

    /* Step Pipeline Progress Card */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: all 0.18s ease;
    }
    .step-card-active {
        border-left: 5px solid #2878C7;
        box-shadow: 0 4px 14px rgba(40, 120, 199, 0.08);
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #2878C7;
        color: #FFFFFF;
        font-size: 0.78rem;
        font-weight: 800;
        margin-right: 8px;
    }
</style>
"""


def apply_theme():
    """Inject custom healthcare styles into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_top_navbar(
    page_title: str,
    category: str = "Clinical AI",
    active_badge: str = "WDBC Validated",
):
    """
    Render a modern enterprise SaaS top navigation bar with breadcrumbs and live system status.
    """
    st.markdown(
        f"""
        <div class="top-navbar">
            <div class="breadcrumb-text">
                <span>Healthcare AI</span>
                <span style="color: #CBD5E1;">›</span>
                <span>{category}</span>
                <span style="color: #CBD5E1;">›</span>
                <span class="breadcrumb-active">{page_title}</span>
            </div>
            <div class="status-badge-container">
                <span class="status-pill status-pill-green">
                    <span style="font-size: 8px;">●</span> Dataset: 569 Cases
                </span>
                <span class="status-pill status-pill-blue">
                    <span style="font-size: 8px;">●</span> Active: SVM (98.25%)
                </span>
                <span class="status-pill status-pill-teal">
                    <span style="font-size: 8px;">●</span> {active_badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    title: str,
    subtitle: str,
    badge: str = "Healthcare SaaS | Clinical Intelligence",
):
    """Render a uniform executive header banner across all pages."""
    st.markdown(
        f"""
        <div class="executive-header">
            <span class="executive-badge">
                <span>🎗️</span> {badge}
            </span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render unified navigation and project context in the Streamlit sidebar."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=100)

        st.markdown(
            """
            <div style="margin-top: 4px; margin-bottom: 12px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #142D4E; letter-spacing: -0.01em;">
                    Breast Cancer AI
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #2878C7; text-transform: uppercase; letter-spacing: 0.05em;">
                    Clinical Prediction Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # System Status Telemetry
        st.markdown("##### **System Status**")
        if DATASET_PATH.exists():
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; font-size: 0.85rem;">
                    <span style="color: #5A6E85;">Dataset (WDBC):</span>
                    <span style="color: #0E7055; font-weight: 700; background: #E8F8F5; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                        ● Loaded (569)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Dataset missing", icon="⚠️")

        if BEST_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; font-size: 0.85rem;">
                    <span style="color: #5A6E85;">Active Classifier:</span>
                    <span style="color: #1A5FA3; font-weight: 700; background: #EBF4FC; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                        ● SVM (98.25%)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Model not ready", icon="⚠️")

        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; font-size: 0.85rem;">
                <span style="color: #5A6E85;">Evaluation Split:</span>
                <span style="color: #142D4E; font-weight: 600; font-size: 0.78rem;">
                    80% Train / 20% Test
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #5A6E85; line-height: 1.45; background: #F8FAFC; padding: 10px 12px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <b style="color: #142D4E;">Academic Clinical Notice:</b><br>
                Engineered for educational and MCA research evaluation. Cytological interpretations require physician validation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def check_and_guide_setup() -> bool:
    """
    Check if required dataset and model artifacts exist.
    If missing, display guided setup buttons.
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


def render_metric_card(
    title: str,
    value: Any,
    subtext: str = "",
    delta_color: str = "#2878C7",
    icon: str = "",
) -> str:
    """Render a premium healthcare SaaS metric card with clean typography and accent stripe."""
    icon_html = f"<span style='font-size: 1.1rem; margin-right: 4px;'>{icon}</span>" if icon else ""
    return f"""
    <div class="metric-card-saas">
        <div class="metric-card-stripe" style="background-color: {delta_color};"></div>
        <div class="metric-label-saas">
            <span>{title}</span>
            {icon_html}
        </div>
        <div class="metric-value-saas">{value}</div>
        <div class="metric-subtext-saas" style="color: {delta_color};">
            {subtext}
        </div>
    </div>
    """

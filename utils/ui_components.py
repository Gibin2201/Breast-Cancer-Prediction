"""
UI Components and Healthcare SaaS Website Theme Styling for Breast Cancer Prediction System.
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
# Enterprise Healthcare SaaS Website CSS Stylesheet
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Google Fonts: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Hide Streamlit Raw Header & Footer to achieve Standalone Website Experience */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 1.5rem !important;
        z-index: 10 !important;
    }
    div[data-testid="stToolbar"] {visibility: hidden !important;}
    div[data-testid="stDecoration"] {visibility: hidden !important;}
    div[data-testid="stStatusWidget"] {visibility: hidden !important;}

    /* Global Typography & Base Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #F5F7FA !important;
        color: #253247 !important;
        letter-spacing: -0.01em;
    }

    /* Main Container Spacing */
    .main .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 1380px !important;
    }

    /* Top Website Navigation Bar */
    .website-navbar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 24px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 4px 20px rgba(20, 45, 78, 0.04);
    }
    .navbar-brand-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .navbar-brand-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: #142D4E;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .navbar-brand-sub {
        font-size: 0.74rem;
        font-weight: 600;
        color: #2878C7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .navbar-breadcrumb {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #5A6E85;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .navbar-breadcrumb-active {
        color: #2878C7;
        font-weight: 700;
    }
    .navbar-status-group {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    /* Status Pills with Pulsing Indicators */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.77rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        background: #F8FAFC;
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
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Clinician Session Badge */
    .clinician-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #142D4E;
    }
    .clinician-avatar {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #2878C7;
        color: #FFFFFF;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.70rem;
        font-weight: 800;
    }

    /* Healthcare SaaS Website Hero Banner */
    .website-hero {
        background: linear-gradient(135deg, #142D4E 0%, #1D3B63 55%, #1A4F82 100%);
        border-radius: 16px;
        padding: 36px 40px;
        color: #FFFFFF;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 30px rgba(20, 45, 78, 0.14);
        position: relative;
        overflow: hidden;
    }
    .website-hero::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(22, 166, 161, 0.28) 0%, rgba(22, 166, 161, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(22, 166, 161, 0.22);
        border: 1px solid rgba(22, 166, 161, 0.50);
        color: #5EEAD4;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
    }
    .website-hero h1 {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 2.25rem !important;
        margin: 0 0 10px 0 !important;
        letter-spacing: -0.025em;
        line-height: 1.25 !important;
    }
    .website-hero p {
        color: #D1DCE8 !important;
        font-size: 1.05rem !important;
        margin: 0 0 18px 0 !important;
        line-height: 1.65;
        max-width: 900px;
        font-weight: 400;
    }

    /* Page Section Header Banners */
    .page-header-card {
        background: linear-gradient(135deg, #142D4E 0%, #1D3B63 70%, #2878C7 100%);
        border-radius: 14px;
        padding: 24px 30px;
        color: #FFFFFF;
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 24px rgba(20, 45, 78, 0.10);
    }
    .page-header-card h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.02em;
    }
    .page-header-card p {
        color: #D1DCE8 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        line-height: 1.55;
    }

    /* Metric Cards - Modern Healthcare SaaS */
    .metric-card-saas {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 4px 18px rgba(20, 45, 78, 0.04);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card-saas:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(20, 45, 78, 0.10);
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
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value-saas {
        font-size: 2.25rem;
        font-weight: 900;
        color: #142D4E;
        line-height: 1.15;
        letter-spacing: -0.025em;
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
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.14);
        margin-top: 18px;
    }
    .result-card-malignant {
        background-color: #FEE8E8 !important;
        border: 2px solid #E63946 !important;
        border-radius: 14px;
        padding: 24px 28px;
        color: #921E28 !important;
        box-shadow: 0 6px 22px rgba(230, 57, 70, 0.16);
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
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(20, 45, 78, 0.04);
        transition: border-color 0.15s ease;
    }
    .saas-card-header {
        font-size: 1.15rem;
        font-weight: 800;
        color: #142D4E;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 10px;
    }

    /* Algorithm Podium / Leaderboard Cards */
    .podium-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    @media (max-width: 992px) {
        .podium-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 576px) {
        .podium-grid {
            grid-template-columns: 1fr;
        }
    }
    .podium-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 14px;
        box-shadow: 0 2px 10px rgba(20, 45, 78, 0.03);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        text-align: center;
    }
    .podium-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 22px rgba(20, 45, 78, 0.09);
        border-color: #2878C7;
    }
    .podium-card-gold {
        border: 2px solid #F59E0B !important;
        background: linear-gradient(180deg, #FFFDF5 0%, #FFFFFF 100%) !important;
        box-shadow: 0 4px 18px rgba(245, 158, 11, 0.14) !important;
    }
    .podium-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    .podium-rank-gold {
        background: #FEF3C7;
        color: #D97706;
    }
    .podium-rank-silver {
        background: #E2E8F0;
        color: #475569;
    }
    .podium-rank-bronze {
        background: #FFEDD5;
        color: #C2410C;
    }
    .podium-rank-standard {
        background: #F1F5F9;
        color: #64748B;
    }

    /* Morphological Differential Comparison Meters */
    .morph-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: border-color 0.15s ease;
    }
    .morph-item:hover {
        border-color: #2878C7;
        background: #FFFFFF;
    }
    .morph-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .morph-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #142D4E;
    }
    .morph-badge {
        font-size: 0.74rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .morph-badge-red {
        background: #FEE2E2;
        color: #DC2626;
    }
    .morph-bar-wrap {
        height: 7px;
        background: #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 6px;
    }
    .morph-bar-fill {
        height: 100%;
        border-radius: 10px;
    }

    /* Website Feature Tile / Module Card */
    .feature-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 4px 16px rgba(20, 45, 78, 0.03);
        transition: all 0.22s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .feature-tile:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(20, 45, 78, 0.08);
        border-color: #2878C7;
    }
    .feature-tile-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: #EBF4FC;
        color: #2878C7;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 14px;
    }
    .feature-tile-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #142D4E;
        margin-bottom: 6px;
    }
    .feature-tile-desc {
        font-size: 0.86rem;
        color: #5A6E85;
        line-height: 1.5;
        margin-bottom: 14px;
    }
    .feature-tile-link {
        font-size: 0.86rem;
        font-weight: 700;
        color: #2878C7;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Website Footer Styling */
    .website-footer {
        background: #FFFFFF;
        border-top: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 36px;
        box-shadow: 0 -2px 14px rgba(20, 45, 78, 0.02);
    }
    .footer-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 16px;
        margin-bottom: 14px;
    }
    .footer-brand {
        font-size: 1rem;
        font-weight: 800;
        color: #142D4E;
    }
    .footer-tag {
        font-size: 0.78rem;
        color: #5A6E85;
    }
    .footer-disclaimer {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.5;
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
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(20, 45, 78, 0.02);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 18px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #5A6E85;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #142D4E;
        background-color: #F8FAFC !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2878C7 !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #2878C7 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(40, 120, 199, 0.28) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #142D4E !important;
        box-shadow: 0 6px 20px rgba(20, 45, 78, 0.35) !important;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #142D4E !important;
        font-weight: 600 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #2878C7 !important;
        color: #2878C7 !important;
    }

    /* Dataframe Table Enhancements */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(20, 45, 78, 0.02) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 10px rgba(20, 45, 78, 0.02) !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: #142D4E !important;
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
    Render a modern website navbar with breadcrumbs, institutional badge, and user session.
    """
    st.markdown(
        f"""
        <div class="website-navbar">
            <div class="navbar-brand-section">
                <span style="font-size: 1.5rem;">🎗️</span>
                <div>
                    <div class="navbar-brand-title">Breast Cancer AI Platform</div>
                    <div class="navbar-brand-sub">Clinical Decision Support System</div>
                </div>
                <div style="height: 24px; width: 1px; background: #E2E8F0; margin: 0 6px;"></div>
                <div class="navbar-breadcrumb">
                    <span>{category}</span>
                    <span style="color: #CBD5E1;">›</span>
                    <span class="navbar-breadcrumb-active">{page_title}</span>
                </div>
            </div>
            <div class="navbar-status-group">
                <span class="status-pill status-pill-green">
                    <span class="pulse-dot"></span> System Online
                </span>
                <span class="status-pill status-pill-blue">
                    <span>⚡</span> SVM 98.25% Acc
                </span>
                <span class="status-pill status-pill-teal">
                    <span>📋</span> 569 Cases
                </span>
                <div class="clinician-badge">
                    <span class="clinician-avatar">Dr</span>
                    <span>Oncologist Session</span>
                </div>
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
        <div class="page-header-card">
            <span class="hero-pill-badge">
                <span>🎗️</span> {badge}
            </span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_website_hero(
    title: str = "Precision Breast Cancer Diagnostics with Machine Learning",
    subtitle: str = "An enterprise clinical decision support system analyzing 30 digitized Fine Needle Aspirate (FNA) nuclear morphometric markers with 98.25% verified diagnostic accuracy.",
    badge: str = "AI-Powered Cytological Oncology Platform",
):
    """Render a website landing hero section."""
    st.markdown(
        f"""
        <div class="website-hero">
            <span class="hero-pill-badge">
                <span>🎗️</span> {badge}
            </span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div style="display: flex; gap: 14px; flex-wrap: wrap;">
                <span class="status-pill status-pill-green" style="background: rgba(16, 185, 129, 0.25); color: #A7F3D0; border-color: rgba(16, 185, 129, 0.5);">
                    <span class="pulse-dot"></span> Model Verified (98.25% Accuracy)
                </span>
                <span class="status-pill" style="background: rgba(40, 120, 199, 0.25); color: #BCE0FD; border-color: rgba(40, 120, 199, 0.5);">
                    <span>🔬</span> Wisconsin Cytology (569 FNA Records)
                </span>
                <span class="status-pill" style="background: rgba(22, 166, 161, 0.25); color: #A0E3E0; border-color: rgba(22, 166, 161, 0.5);">
                    <span>🛡️</span> Zero Data Leakage Pipeline
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render unified navigation and project context in the Streamlit sidebar."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=95)

        st.markdown(
            """
            <div style="margin-top: 4px; margin-bottom: 12px;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #142D4E; letter-spacing: -0.01em;">
                    Breast Cancer AI
                </div>
                <div style="font-size: 0.74rem; font-weight: 600; color: #2878C7; text-transform: uppercase; letter-spacing: 0.05em;">
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
            <div style="font-size: 0.74rem; color: #5A6E85; line-height: 1.45; background: #F8FAFC; padding: 10px 12px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <b style="color: #142D4E;">Academic Clinical Notice:</b><br>
                Engineered for educational and MCA research evaluation. Cytological interpretations require physician validation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_card(
    title: str,
    value: Any,
    subtext: str = "",
    delta_color: str = "#2878C7",
    icon: str = "",
) -> str:
    """Render a premium healthcare SaaS metric card with clean typography and accent stripe."""
    icon_html = (
        f"""<div style="width: 36px; height: 36px; border-radius: 10px; background: {delta_color}18; display: inline-flex; align-items: center; justify-content: center; font-size: 1.15rem;">
            {icon}
        </div>"""
        if icon
        else ""
    )
    return f"""
    <div class="metric-card-saas" style="border-top: 3px solid {delta_color};">
        <div class="metric-label-saas">
            <span>{title}</span>
            {icon_html}
        </div>
        <div class="metric-value-saas">{value}</div>
        <div class="metric-subtext-saas" style="color: {delta_color};">
            {subtext}
        </div>
        <div style="height: 3px; background: #F1F5F9; border-radius: 2px; margin-top: 10px; overflow: hidden;">
            <div style="height: 100%; width: 100%; background: {delta_color}; border-radius: 2px; opacity: 0.85;"></div>
        </div>
    </div>
    """


def render_website_footer():
    """Render a modern healthcare SaaS website footer."""
    st.markdown(
        """
        <div class="website-footer">
            <div class="footer-top">
                <div>
                    <div class="footer-brand">🎗️ Breast Cancer AI Clinical Platform</div>
                    <div class="footer-tag">Wisconsin Diagnostic Breast Cancer (WDBC) Machine Learning Benchmark</div>
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #5A6E85;">
                    Version 2.4.0 • Enterprise Clinical Edition • Python 3.12 & Streamlit
                </div>
            </div>
            <div class="footer-disclaimer">
                <b>Clinical Governance & Academic Disclaimer:</b> This software platform is an educational machine learning prototype 
                engineered for Master of Computer Applications (MCA) technical evaluation and academic research. 
                Diagnostic predictions are statistical estimations based on historical cytological FNA vectors and must not be used 
                as an unverified substitute for professional oncological pathology or clinical medical diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_and_guide_setup() -> bool:
    """Check if required dataset and model artifacts exist."""
    needs_dataset = not DATASET_PATH.exists()
    needs_model = not (BEST_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists())

    if needs_dataset or needs_model:
        st.warning("⚠️ **Initial Setup Required**")
        st.info("Required project artifacts are not yet detected. Download dataset or train models below:")

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

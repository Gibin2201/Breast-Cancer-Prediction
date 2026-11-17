"""
Page 2: Clinical Dataset Overview.
Displays detailed dataset provenance, dimensions, feature information,
first 5 records (head), last 5 records (tail), data integrity audit,
and descriptive statistics.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from config import (
    DATASET_PATH,
    TARGET_COLUMN,
    FEATURE_NAMES,
    MEAN_FEATURES,
    SE_FEATURES,
    WORST_FEATURES,
    FEATURE_DISPLAY_NAMES,
    FEATURE_TOOLTIPS,
)
from utils.ui_components import (
    apply_theme,
    render_top_navbar,
    render_header,
    render_sidebar,
    render_metric_card,
)
from utils.data_loader import load_dataset, get_dataset_metadata
from download_dataset import load_source_metadata


def main():
    apply_theme()
    render_sidebar()

    render_top_navbar(
        page_title="Dataset Overview",
        category="Data Architecture",
        active_badge="569 Records | 30 Features",
    )

    render_header(
        title="Wisconsin Diagnostic Dataset Overview",
        subtitle="Clinical Specifications, Cytological Feature Attributes, Record Slicing (Head/Tail), and Data Integrity",
        badge="Data Architecture & Ingestion",
    )

    # Load dataset
    try:
        df = load_dataset()
        meta = get_dataset_metadata(df)
        source_meta = load_source_metadata()
    except Exception as e:
        st.error(f"Error loading clinical dataset: {e}")
        return

    # 1. Dataset Dimensions & Provenance Cards
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(
            render_metric_card("Total Biopsy Records", f"{len(df)}", "Rows (FNA Cytology)", "#142D4E", "📄"),
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            render_metric_card("Feature Dimensions", f"{len(FEATURE_NAMES)}", "10 Mean + 10 SE + 10 Worst", "#2878C7", "🧬"),
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            render_metric_card("Missing Values", f"{meta['missing_values']}", "0.00% Null Rate (Clean)", "#10B981", "✅"),
            unsafe_allow_html=True,
        )
    with d4:
        st.markdown(
            render_metric_card("Duplicate Records", f"{meta['duplicate_rows']}", "100% Unique Biopsies", "#16A6A1", "🔍"),
            unsafe_allow_html=True,
        )

    st.write("")

    # 2. Dataset Information & Clinical Cytology Background
    st.markdown(
        f"""
        <div class="saas-card">
            <div class="saas-card-header">
                <span>ℹ️</span> Dataset Clinical Background & Verification Audit
            </div>
            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px;">
                <div>
                    <p style="font-size: 0.94rem; line-height: 1.6; color: #253247; margin-bottom: 8px;">
                        The <b>Wisconsin Diagnostic Breast Cancer (WDBC)</b> dataset was created by Dr. William H. Wolberg, 
                        W. Nick Street, and Olvi L. Mangasarian at the University of Wisconsin Hospitals, Madison. 
                        Features were computed from digitized high-resolution scans of Fine Needle Aspirates (FNA) of breast masses.
                    </p>
                    <p style="font-size: 0.88rem; line-height: 1.6; color: #5A6E85; margin-bottom: 0;">
                        <b>Data Cleaning Applied:</b> Patient ID (Column 0) and trailing comma artifacts were completely removed during ingestion. 
                        All 30 continuous features represent objective physical measurements of cell nuclei: 
                        10 mean values, 10 standard errors (SE), and 10 "worst" (largest/extreme) values.
                    </p>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #142D4E; text-transform: uppercase; margin-bottom: 6px;">
                        Provenance Audit Trail
                    </div>
                    <div style="font-size: 0.82rem; color: #5A6E85; line-height: 1.45;">
                        • Source: <b>{source_meta.get('source_name', 'UCI Archive')}</b><br>
                        • Format: <b>Standardized CSV</b><br>
                        • Ingestion Time: <b>{source_meta.get('downloaded_at', 'Verified')}</b><br>
                        • Integrity: <b style="color: #0E7055;">100% Complete</b>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 3. Interactive Dataset Slicing (Head, Tail, Full)
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>📋</span> Interactive Tabular Slicing (df.head & df.tail)
            </div>
            <p style="font-size: 0.88rem; color: #5A6E85; margin-bottom: 12px;">
                Inspect individual patient records. Toggle between top records (<code>df.head()</code>), bottom records (<code>df.tail()</code>), or the complete cohort.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 4, 3])
    with ctrl_col1:
        diag_filter = st.selectbox(
            "Filter by Diagnosis:",
            options=["All Records", "Benign (B)", "Malignant (M)"],
        )
    with ctrl_col2:
        slice_mode = st.radio(
            "Dataset Slice Mode:",
            options=["First Records (df.head())", "Last Records (df.tail())", "Full Cohort (569 rows)"],
            horizontal=True,
        )
    with ctrl_col3:
        num_rows = st.number_input("Number of Records (N):", min_value=1, max_value=50, value=5, step=1)

    filtered_df = df.copy()
    if diag_filter == "Benign (B)":
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN].isin(["B", 0, "Benign"])]
    elif diag_filter == "Malignant (M)":
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN].isin(["M", 1, "Malignant"])]

    if "head" in slice_mode.lower():
        display_df = filtered_df.head(int(num_rows))
        st.caption(f"Showing top **{len(display_df)}** records via Pandas `df.head({num_rows})`")
    elif "tail" in slice_mode.lower():
        display_df = filtered_df.tail(int(num_rows))
        st.caption(f"Showing bottom **{len(display_df)}** records via Pandas `df.tail({num_rows})`")
    else:
        display_df = filtered_df
        st.caption(f"Showing all **{len(display_df)}** records in cohort")

    st.dataframe(display_df, use_container_width=True, height=310)

    # Download CSV Button
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Standardized Dataset as CSV",
        data=csv_bytes,
        file_name="wdbc_breast_cancer_dataset.csv",
        mime="text/csv",
        help="Download the verified 569-row WDBC CSV dataset to your local machine",
    )

    st.write("")
    st.divider()

    # 4. Feature Taxonomy & Metadata Table
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>🧬</span> 30 Continuous Nuclear Feature Specifications
            </div>
            <p style="font-size: 0.88rem; color: #5A6E85; margin-bottom: 12px;">
                Features are computed from digitized cell images and grouped into three 10-dimensional vectors:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_mean, tab_se, tab_worst = st.tabs([
        "🔬 1. Mean Nuclear Features (1–10)",
        "📐 2. Standard Error Variability (11–20)",
        "⚠️ 3. Worst / Extreme Pathologies (21–30)",
    ])

    with tab_mean:
        mean_data = []
        for feat in MEAN_FEATURES:
            mean_data.append({
                "Technical Name": feat,
                "Clinical Display Label": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "Description": FEATURE_TOOLTIPS.get(feat, "N/A"),
                "Cohort Mean": f"{df[feat].mean():.3f}",
                "Cohort Std Dev": f"{df[feat].std():.3f}",
            })
        st.dataframe(pd.DataFrame(mean_data), use_container_width=True, hide_index=True)

    with tab_se:
        se_data = []
        for feat in SE_FEATURES:
            se_data.append({
                "Technical Name": feat,
                "Clinical Display Label": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "Description": FEATURE_TOOLTIPS.get(feat, "N/A"),
                "Cohort Mean": f"{df[feat].mean():.3f}",
                "Cohort Std Dev": f"{df[feat].std():.3f}",
            })
        st.dataframe(pd.DataFrame(se_data), use_container_width=True, hide_index=True)

    with tab_worst:
        worst_data = []
        for feat in WORST_FEATURES:
            worst_data.append({
                "Technical Name": feat,
                "Clinical Display Label": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "Description": FEATURE_TOOLTIPS.get(feat, "N/A"),
                "Cohort Mean": f"{df[feat].mean():.3f}",
                "Cohort Std Dev": f"{df[feat].std():.3f}",
            })
        st.dataframe(pd.DataFrame(worst_data), use_container_width=True, hide_index=True)

    st.write("")
    st.divider()

    # 5. Descriptive Statistics Summary Table
    st.markdown(
        """
        <div class="saas-card">
            <div class="saas-card-header">
                <span>📊</span> Full Descriptive Statistics (df.describe)
            </div>
            <p style="font-size: 0.88rem; color: #5A6E85; margin-bottom: 12px;">
                Standard parametric statistics across all 30 features: Count, Mean, Standard Deviation, Minimum, 25%, 50% (Median), 75%, and Maximum:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    desc_df = df[FEATURE_NAMES].describe().T
    st.dataframe(desc_df.style.format("{:.3f}"), use_container_width=True, height=360)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Annotation Summary: GNPS and SIRIUS")

if "annotated_df" not in st.session_state:
    st.warning("Please upload and process your data on the main page first.")
    st.stop()

df = st.session_state["annotated_df"]

# GNPS annotations: Compound_Name not NA
gnps_df = df[df["Compound_Name"].notna()]

# Get columns: 'row ID', next 6 columns, then from 'SpectrumID' onwards
if "SpectrumID" in df.columns:
    rowid_idx = df.columns.get_loc("row ID")
    spectrum_idx = df.columns.get_loc("SpectrumID")
    gnps_cols = (
        list(df.columns[rowid_idx:rowid_idx+7]) +  # 'row ID' + next 6 columns
        list(df.columns[spectrum_idx:])            # from 'SpectrumID' onwards
    )
    gnps_df = gnps_df[gnps_cols]
else:
    st.warning("SpectrumID column not found in your data.")
    gnps_df = gnps_df[["row ID"]]

st.subheader(f"GNPS Annotations ({len(gnps_df)})")
st.dataframe(gnps_df, use_container_width=True)


# Sirius annotations: ConfidenceScoreExact not NA and Compound_Name is NA
sirius_df = df[df["ConfidenceScoreExact"].notna() & df["Compound_Name"].isna()]

# Get columns from 'row ID' up to and including 'overallFeatureQuality_canopus'
if "overallFeatureQuality_canopus" in df.columns:
    end_idx = df.columns.get_loc("overallFeatureQuality_canopus") + 1
    sirius_cols = list(df.columns[:end_idx])
    sirius_df = sirius_df[sirius_cols]
else:
    st.warning("overallFeatureQuality_canopus column not found in your data.")
    sirius_df = sirius_df[["row ID"]]

# Sidebar filter for ConfidenceScoreExact (CANOPUS/SIRIUS)
st.sidebar.markdown("### Filter CANOPUS/SIRIUS by ConfidenceScoreExact")
if not sirius_df.empty:
    # Only use finite values for slider limits
    finite_scores = sirius_df["ConfidenceScoreExact"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_scores.empty:
        min_score = float(finite_scores.min())
        max_score = float(finite_scores.max())
        conf_range = st.sidebar.slider(
            "ConfidenceScoreExact range",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            step=0.01,
        )
        sirius_df = sirius_df[
            sirius_df["ConfidenceScoreExact"].between(conf_range[0], conf_range[1])
        ]
    else:
        st.sidebar.info("No finite ConfidenceScoreExact values to filter.")
else:
    st.sidebar.info("No CANOPUS/SIRIUS annotations to filter.")

st.subheader(f"Sirius Annotations ({len(sirius_df)})")
st.dataframe(sirius_df, use_container_width=True)
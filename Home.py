import streamlit as st
import pandas as pd
from utils.io import load_uploaded_file
from utils.preprocessing import compute_feature_proportions, merge_feature_annotations

st.set_page_config(page_title="South Africa Foodomics App", layout="wide")
st.title("South Africa Food Explorer App")

st.subheader("Data Upload and Processing")



# Required: quantification + metadata
quant_file = st.file_uploader("Upload quantification table (.csv)", type=["csv"])
metadata_file = st.file_uploader("Upload metadata file (.csv)", type=["csv"])

# Optional: SIRIUS / CANOPUS / GNPS
st.subheader("Optional: Add Annotations")
sirius_file = st.file_uploader("SIRIUS structure_identifications.tsv", type=["tsv"])
canopus_file = st.file_uploader("CANOPUS structure_summary.tsv", type=["tsv"])
gnps_file = st.file_uploader("GNPS annotations.tsv", type=["tsv"])

if quant_file and metadata_file:
    quant_df = load_uploaded_file(quant_file)
    metadata_df = load_uploaded_file(metadata_file)

    st.success("Files loaded. Computing proportions...")
    proportions_df, description_cols = compute_feature_proportions(quant_df, metadata_df)
    st.dataframe(proportions_df.head(), use_container_width=True)
    st.session_state["description_cols"] = description_cols
    # Check for optional annotation merging
    if sirius_file and canopus_file and gnps_file:
        sirius_df = pd.read_csv(sirius_file, sep="\t",engine="python", on_bad_lines='skip')
        canopus_df = pd.read_csv(canopus_file, sep="\t")
        gnps_df = pd.read_csv(gnps_file, sep="\t")

        annotated_df = merge_feature_annotations(proportions_df, sirius_df, canopus_df, gnps_df)
        st.success("Proportions merged with annotations.")
        st.session_state["annotated_df"] = annotated_df
        st.dataframe(annotated_df, use_container_width=True)


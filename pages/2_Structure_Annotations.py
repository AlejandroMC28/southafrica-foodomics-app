import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Optional: For structure images
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_OK = True
except ImportError:
    RDKit_OK = False

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

# --- Feature-level log2FC analysis for filtered CANOPUS/SIRIUS annotations ---
st.markdown("### Feature-Level log2 Fold Change Analysis (CANOPUS/SIRIUS)")

sample_cols = st.session_state["description_cols"]

if not sirius_df.empty and len(sample_cols) >= 2:
    # Sidebar controls for feature-level log2FC
    st.sidebar.markdown("#### Feature log2FC analysis")
    feat_fc_sample1 = st.sidebar.selectbox("Reference Food (feature log2FC)", sample_cols, key="feat_fc_sample1")
    feat_fc_sample2 = st.sidebar.selectbox("Comparison Food (feature log2FC)", sample_cols, key="feat_fc_sample2")
    feat_top_n = st.sidebar.slider("Number of top features to show", 1, 50, 10, key="feat_top_n")
    feat_sort_mode = st.sidebar.radio(
        "Sort by",
        ["Most variable (|log2FC|)", "Top increasing (log2FC)", "Top decreasing (log2FC)"],
        key="feat_sort_mode"
    )
    run_feat_fc = st.sidebar.button("Run Feature log2FC", key="run_feat_fc")

    # --- Feature source selection ---
    st.sidebar.markdown("### Feature Source for log2FC")
    feature_source = st.sidebar.radio(
        "Choose annotation source for log2FC analysis:",
        ("Both (CANOPUS/SIRIUS)", "Only GNPS", "Only SIRIUS"),
        key="feature_source_radio"
    )

    # Prepare the dataframe to use for log2FC
    if feature_source == "Only GNPS":
        fc_df = gnps_df.copy()
    elif feature_source == "Only SIRIUS":
        fc_df = sirius_df.copy()
    else:
        fc_df = pd.concat([gnps_df, sirius_df], ignore_index=True)

    # Only run if button pressed and samples are different
    if run_feat_fc and feat_fc_sample1 != feat_fc_sample2:
        # Calculate log2FC for each feature (row)
        feat_df = fc_df[["row ID", feat_fc_sample1, feat_fc_sample2]].copy()
        feat_df["log2fc"] = np.log2((feat_df[feat_fc_sample2] + 1e-9) / (feat_df[feat_fc_sample1] + 1e-9))
        feat_df["abs_log2fc"] = feat_df["log2fc"].abs()

        # Sort according to user selection
        if feat_sort_mode == "Top increasing (log2FC)":
            feat_df = feat_df.sort_values("log2fc", ascending=False)
        elif feat_sort_mode == "Top decreasing (log2FC)":
            feat_df = feat_df.sort_values("log2fc", ascending=True)
        else:
            feat_df = feat_df.sort_values("abs_log2fc", ascending=False)

        top_feat_df = feat_df.head(feat_top_n)

        st.markdown(f"**Top {feat_top_n} features by {feat_sort_mode}:**")
        st.dataframe(top_feat_df, hide_index=True, use_container_width=True)

        # Barplot of log2FC for top features
        st.markdown("#### log2FC Barplot for Top Features")
        colors = ['#d73027' if val < 0 else '#1a9850' for val in top_feat_df["log2fc"]]
        fig_feat_fc = go.Figure(go.Bar(
            x=top_feat_df["row ID"].astype(str),
            y=top_feat_df["log2fc"],
            marker_color=colors,
            text=[f"{val:.2f}" for val in top_feat_df["log2fc"]],
            textposition='auto',
        ))
        fig_feat_fc.update_layout(
            xaxis_title="row ID",
            yaxis_title="log2 Fold Change",
            title="Top Features by log2FC",
            height=400,
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            template="simple_white",
            xaxis=dict(type="category", categoryorder="array", categoryarray=top_feat_df["row ID"].astype(str).tolist())
        )
        fig_feat_fc.add_shape(
            type="line",
            x0=-0.5,
            x1=len(top_feat_df) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1),
        )
        st.plotly_chart(fig_feat_fc, use_container_width=True)

        # Optional: Show abundance comparison for these features
        st.markdown("#### Abundance Comparison for Top Features")
        fig_feat_comp = go.Figure()
        fig_feat_comp.add_trace(go.Bar(
            x=top_feat_df["row ID"].astype(str),
            y=top_feat_df[feat_fc_sample1],
            name=f"{feat_fc_sample1} (Reference)",
            marker_color='#377eb8',
        ))
        fig_feat_comp.add_trace(go.Bar(
            x=top_feat_df["row ID"].astype(str),
            y=top_feat_df[feat_fc_sample2],
            name=f"{feat_fc_sample2} (Comparison)",
            marker_color='#ff7f00',
        ))
        fig_feat_comp.update_layout(
            xaxis_title="row ID",
            yaxis_title="Abundance",
            barmode='group',
            height=400,
            title="Abundance Comparison for Top Features",
            xaxis=dict(type="category", categoryorder="array", categoryarray=top_feat_df["row ID"].astype(str).tolist())
        )
        st.plotly_chart(fig_feat_comp, use_container_width=True)

st.markdown("### Draw Structure for Selected Feature")

# Source selection buttons
col1, col2 = st.columns(2)
use_sirius = col1.button("Use SIRIUS matches")
use_gnps = col2.button("Use GNPS matches")

# Default to SIRIUS if neither pressed
if "structure_source" not in st.session_state:
    st.session_state["structure_source"] = "sirius"
if use_sirius:
    st.session_state["structure_source"] = "sirius"
if use_gnps:
    st.session_state["structure_source"] = "gnps"

source = st.session_state["structure_source"]

# Text input for row ID
row_id_input = st.text_input("Enter row ID to view structure")

# Select the dataframe and SMILES column
if source == "sirius":
    df_struct = sirius_df
    smiles_col = "smiles"
else:
    df_struct = gnps_df
    smiles_col = "Smiles"

# Try to find the row and show structure and name
if row_id_input:
    match = df_struct[df_struct["row ID"].astype(str) == row_id_input]
    if not match.empty:
        # Get the name column (SIRIUS: 'name', GNPS: 'Compound_Name')
        if source == "sirius":
            name = match.iloc[0].get("name", None)
        else:
            name = match.iloc[0].get("Compound_Name", None)
        if pd.notnull(name) and name:
            st.markdown(f"**Name:** {name}")
        else:
            st.info("No name available for this feature.")

        smiles = match.iloc[0].get(smiles_col, None)
        if pd.notnull(smiles) and smiles:
            st.code(smiles, language="none")
            if RDKit_OK:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.image(Draw.MolToImage(mol, size=(300, 300)), caption=f"Structure for row ID {row_id_input}")
                else:
                    st.info("Could not parse SMILES for image.")
            else:
                st.info("Install RDKit for structure images.")
        else:
            st.info(f"No SMILES found in column '{smiles_col}' for this row.")
    else:
        st.warning("Row ID not found in the selected source.")
else:
    st.info("Enter a row ID above to view its structure.")
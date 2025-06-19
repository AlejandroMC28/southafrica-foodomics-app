import pandas as pd
import streamlit as st

@st.cache_data
def load_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    else:
        raise ValueError("Unsupported file type.")

import pandas as pd

def compute_feature_proportions(quant_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize feature intensities to proportions and average across replicates using metadata description.

    Parameters
    ----------
    quant_df : pd.DataFrame
        The full feature quantification table with 'row ID' and sample columns.
    metadata_df : pd.DataFrame
        Sample metadata containing 'filename' and 'description'.

    Returns
    -------
    pd.DataFrame
        Feature table with 'row ID' and averaged proportions by sample description.
    """
    # Step 1: Identify peak area columns
    peak_area_cols = [col for col in quant_df.columns if col.endswith(" Peak area")]
    
    # Step 2: Strip " Peak area" to get sample filenames
    stripped_cols = {col: col.replace(" Peak area", "") for col in peak_area_cols}
    quant_df = quant_df.rename(columns=stripped_cols)
    
    # Step 3: Normalize each sample's intensities to proportions
    sample_cols = list(stripped_cols.values())
    normalized_df = quant_df.copy()
    normalized_df[sample_cols] = normalized_df[sample_cols].div(
        normalized_df[sample_cols].sum(axis=0), axis=1
    )

    # Step 4: Map sample filename → description
    sample_to_description = metadata_df.set_index("filename")["description"].to_dict()
    renamed = normalized_df[sample_cols].rename(columns=sample_to_description)

    # Step 5: Average replicates by description
    grouped = renamed.groupby(axis=1, level=0).mean()

    # Step 6: Reattach row ID
    grouped_with_id = quant_df[["row ID", "row m/z", "row retention time"]].join(grouped)

    return grouped_with_id

def merge_feature_annotations(
    proportions_df: pd.DataFrame,
    sirius_df: pd.DataFrame,
    canopus_df: pd.DataFrame,
    gnps_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge SIRIUS, CANOPUS, and GNPS annotations with the proportions table.

    Parameters
    ----------
    proportions_df : pd.DataFrame
        Table with row ID and normalized intensity columns.
    sirius_df : pd.DataFrame
        SIRIUS structure identification data (with 'mappingFeatureId').
    canopus_df : pd.DataFrame
        CANOPUS chemical classification data (with 'mappingFeatureId').
    gnps_df : pd.DataFrame
        GNPS annotation table (with '#Scan#').

    Returns
    -------
    pd.DataFrame
        Annotated proportions table with compound metadata.
    """
    # Standardize merge keys
    sirius_df["mappingFeatureId"] = sirius_df["mappingFeatureId"].astype("Int64")
    canopus_df["mappingFeatureId"] = canopus_df["mappingFeatureId"].astype("Int64")
    gnps_df = gnps_df.rename(columns={"#Scan#": "row ID"})
    gnps_df["row ID"] = gnps_df["row ID"].astype("Int64")

    # Merge in order: SIRIUS → CANOPUS → GNPS
    merged = proportions_df.merge(
        sirius_df, how="left", left_on="row ID", right_on="mappingFeatureId", suffixes=("", "_sirius")
    )
    merged = merged.merge(
        canopus_df, how="left", on="mappingFeatureId", suffixes=("", "_canopus")
    )
    merged = merged.merge(
        gnps_df, how="left", on="row ID", suffixes=("", "_gnps")
    )

    return merged

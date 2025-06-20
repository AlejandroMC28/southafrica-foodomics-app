import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Chemical Class-Based Visualizations")

# Add this near the top of your file, after imports
if "update_counter" not in st.session_state:
    st.session_state["update_counter"] = 0

# Load annotated data from session
if "annotated_df" not in st.session_state:
    st.warning("Please upload and process your data on the main page first.")
    st.stop()

df = st.session_state["annotated_df"]

# --- User filters
level = st.selectbox("Select chemical classification level", [
    "ClassyFire#superclass",
    "ClassyFire#class",
    "ClassyFire#subclass",
    "ClassyFire#level 5"
])
if level == "ClassyFire#superclass":
    prob_col = "ClassyFire#superclass probability"
else:
    prob_col = f"{level} Probability"

threshold = st.slider("Minimum classification probability", 0.0, 1.0, 0.7)

sample_cols = st.session_state["description_cols"]

filtered = df[df[prob_col] >= threshold].copy()
filtered["is_gnps"] = filtered["Compound_Name"].notna()

# --- Create sidebar for variable class analysis ---
st.sidebar.title("Find Variable Chemical Classes")

# Select two food samples to compare
fc_sample1 = st.sidebar.selectbox("Reference Food", sample_cols, index=0, key="fc_sample1_sidebar")
fc_sample2 = st.sidebar.selectbox("Comparison Food", sample_cols, index=min(1, len(sample_cols)-1), key="fc_sample2_sidebar")
top_n = st.sidebar.slider("Number of classes to show", 1, 30, 10, key="top_n_sidebar")

# Buttons for different sorting options
col1, col2 = st.sidebar.columns(2)
with col1:
    show_top_increasing = st.button("Top Increasing", key="top_increasing_sidebar")
with col2:
    show_top_decreasing = st.button("Top Decreasing", key="top_decreasing_sidebar")

find_variable_classes = st.sidebar.button("Find Most Variable Classes", key="find_variable_sidebar")

# Initialize session state for variable classes
if "variable_classes" not in st.session_state:
    st.session_state["variable_classes"] = None

# Calculate log2FC when any button is pressed
if (find_variable_classes or show_top_increasing or show_top_decreasing) and fc_sample1 != fc_sample2:
    # Aggregate by class for both samples
    agg = filtered.groupby(level)[[fc_sample1, fc_sample2]].sum().reset_index()
    
    # Calculate log2FC (with small pseudocount to avoid division by zero)
    agg["log2fc"] = np.log2((agg[fc_sample2] + 1e-9) / (agg[fc_sample1] + 1e-9))
    agg["abs_log2fc"] = agg["log2fc"].abs()
    
    # Sort based on which button was clicked
    if show_top_increasing:
        agg = agg.sort_values("log2fc", ascending=False)
        direction = "increasing"
    elif show_top_decreasing:
        agg = agg.sort_values("log2fc", ascending=True)
        direction = "decreasing"
    else:  # Default to abs log2FC (most variable)
        agg = agg.sort_values("abs_log2fc", ascending=False)
        direction = "variable"
        
    # Get top N classes
    top_classes = agg.head(top_n)
    
    # Store in session state
    st.session_state["variable_classes"] = top_classes
    
    # Show results table
    st.sidebar.markdown(f"### Top {top_n} {direction.capitalize()} Classes")
    display_df = top_classes[[level, fc_sample1, fc_sample2, "log2fc"]].copy()
    display_df.columns = [level, "Reference", "Comparison", "log2FC"]
    st.sidebar.dataframe(display_df.style.format({"log2FC": "{:.2f}"}), hide_index=True)
    
    # Button to use these classes for plotting
    # This button now directly sets the st.session_state["selected_class_list"]
    if st.sidebar.button("Use these classes for plotting", key="use_variable_classes"):
        selected_classes_from_sidebar = top_classes[level].tolist()
        st.session_state["selected_class_list"] = selected_classes_from_sidebar
        # Increment counter to force widget refresh
        st.session_state["update_counter"] += 1
        st.rerun()

# --- Select classes to plot (with session state override) ---
available_classes = filtered[level].dropna().unique()
default_classes = available_classes[:2] if len(available_classes) >= 2 else available_classes

# Initialize session state for selected_class_list if not present.
# This ensures that if the sidebar button hasn't been clicked yet,
# the multiselect has a default value.
if "selected_class_list" not in st.session_state:
    st.session_state["selected_class_list"] = available_classes[:5].tolist()

# The crucial part: The default for multiselect now directly uses the session state.
# When st.rerun() is called after the sidebar button, this line will be re-executed
# and the multiselect will reflect the new value from st.session_state.
selected_classes = st.multiselect(
    f"Select {level} to plot",
    options=available_classes,
    default=default_classes,
    key="selected_classes_bar"
)

# It's important to update the session state *after* the multiselect,
# so that user changes in the multiselect are also persisted.
st.session_state["selected_class_list"] = selected_classes

filtered = filtered[filtered[level].isin(selected_classes)]

viz_mode = st.radio("Select plot type", ["Summed Proportions (Barplot)", "Feature Distribution (Boxplot)"])

# === BARPLOT ===
if viz_mode == "Summed Proportions (Barplot)":
    selected_samples = st.multiselect("Select foods (samples) to plot", sample_cols, default=sample_cols[:3])

    # Color pickers for each selected sample
    default_palette = plotly.colors.qualitative.Plotly
    color_map = {}
    st.markdown("**Pick a color for each food (sample):**")
    for i, sample in enumerate(selected_samples):
        default_color = default_palette[i % len(default_palette)]
        color = st.color_picker(f"Color for {sample}", default_color, key=f"color_picker_{sample}_bar") # Added key
        color_map[sample] = color

    plot_data = []
    for chem_class in selected_classes:
        class_df = filtered[filtered[level] == chem_class]
        for sample in selected_samples:
            gnps = class_df[class_df["is_gnps"]][sample].sum()
            unannotated = class_df[~class_df["is_gnps"]][sample].sum()
            plot_data.append({
                "Chemical Class": chem_class,
                "Sample": sample,
                "Type": "GNPS",
                "Proportion": gnps
            })
            plot_data.append({
                "Chemical Class": chem_class,
                "Sample": sample,
                "Type": "Unannotated",
                "Proportion": unannotated
            })

    plot_df = pd.DataFrame(plot_data)

    fig = go.Figure()

    for sample in selected_samples:
        base_color = color_map[sample]
        for type_label, opacity in zip(["Unannotated", "GNPS"], [0.4, 1.0]):
            sub = plot_df[(plot_df["Sample"] == sample) & (plot_df["Type"] == type_label)]
            fig.add_trace(go.Bar(
                x=sub["Chemical Class"],
                y=sub["Proportion"],
                name=f"{sample} - {type_label}",
                opacity=opacity,
                offsetgroup=sample,
                legendgroup=sample,
                marker=dict(color=base_color, line=dict(width=0.5, color='black'))
            ))

    fig.update_layout(
        barmode="stack",
        title="Summed Proportions per Chemical Class (Grouped and Stacked by Food)",
        xaxis_title="Chemical Class",
        yaxis_title="Summed Proportion",
        height=600,
        legend_title="Food (Sample) & Type"
    )

    st.plotly_chart(fig, use_container_width=True)

# === BOXPLOT ===
else:
    # Melt the filtered data to create a long-format dataframe for boxplot
    melted = filtered.melt(
        id_vars=["row ID", level],  # Include row ID for hover info
        value_vars=sample_cols,
        var_name="Sample",
        value_name="Proportion"
    )

    # Create a figure for boxplot with points
    fig = go.Figure()
    
    # Get a color palette for the samples
    default_palette = plotly.colors.qualitative.Plotly
    color_map = {sample: default_palette[i % len(default_palette)] 
                for i, sample in enumerate(sample_cols)}

    # Add a boxplot for each food sample, grouped by chemical class
    for sample in sample_cols:
        fig.add_trace(go.Box(
            x=melted[melted["Sample"] == sample][level],
            y=melted[melted["Sample"] == sample]["Proportion"],
            name=sample,
            boxpoints='all',  # Show all points
            jitter=0.5,       # Add jitter to separate overlapping points
            pointpos=0,       # Center points in the box
            marker=dict(
                color=color_map[sample],
                opacity=0.7,
                size=4,
            ),
            hoverinfo='all',
            text=melted[melted["Sample"] == sample]["row ID"],  # Add row ID as hover text
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Sample: " + sample + "<br>" +
                "Value: %{y:.6f}<br>" +
                "Row ID: %{text}<br>" +
                "<extra></extra>"  # Hide secondary box
            )
        ))

    # Update layout
    fig.update_layout(
        title="Feature-Level Proportions per Food (by Chemical Class)",
        xaxis_title="Chemical Class",
        yaxis_title="Proportion",
        height=600,
        boxmode='group',
        hovermode='closest',
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Optionally add a data table below to allow scrolling through the data
    with st.expander("View Raw Feature Data"):
        # Create a nice display version of the melted data
        display_df = melted[["row ID", level, "Sample", "Proportion"]].copy()
        display_df = display_df.sort_values([level, "Sample", "Proportion"], ascending=[True, True, False])
        st.dataframe(display_df, use_container_width=True)

# === LOG2FC VISUALIZATION ===
# Check if we have log2FC results to display
if "variable_classes" in st.session_state and st.session_state["variable_classes"] is not None:
    st.markdown("---")
    st.header("Log2 Fold Change Analysis")
    
    top_classes = st.session_state["variable_classes"]
    
    # Get direction from data sorting
    if len(top_classes) > 0:
        if top_classes["log2fc"].iloc[0] > 0 and top_classes["log2fc"].iloc[-1] > 0:
            direction = "increasing"
        elif top_classes["log2fc"].iloc[0] < 0 and top_classes["log2fc"].iloc[-1] < 0:
            direction = "decreasing"
        else:
            direction = "variable"
    
    # Display log2FC metric explanation
    st.markdown(f"""
    **Showing log2FC between:**
    - Reference: {fc_sample1}
    - Comparison: {fc_sample2}
    
    *Positive log2FC means higher in {fc_sample2}, negative means higher in {fc_sample1}*
    """)
    
    # Create log2FC barplot
    class_names = top_classes[level].tolist()
    log2fc_values = top_classes["log2fc"].tolist()
    
    # Create a color scale for log2FC bars
    colors = ['#d73027' if val < 0 else '#1a9850' for val in log2fc_values]
    
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(
        x=class_names,
        y=log2fc_values,
        marker_color=colors,
        text=[f"{val:.2f}" for val in log2fc_values],
        textposition='auto',
    ))
    
    fig_fc.update_layout(
        title=f"Top {len(class_names)} {direction.capitalize()} Classes by log2FC",
        xaxis_title="Chemical Class",
        yaxis_title="log2 Fold Change",
        height=500,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        template="simple_white",
    )
    
    # Add a horizontal line at y=0
    fig_fc.add_shape(
        type="line",
        x0=-0.5,
        x1=len(class_names) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="black", width=1),
    )
    
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Add a comparison barplot showing actual values in both samples
    st.subheader("Abundance Comparison")
    
    # Create a grouped barplot comparing the two samples
    fig_comp = go.Figure()
    
    # Add reference sample bars
    fig_comp.add_trace(go.Bar(
        x=class_names,
        y=top_classes[fc_sample1].tolist(),
        name=f"{fc_sample1} (Reference)",
        marker_color='#377eb8',
    ))
    
    # Add comparison sample bars
    fig_comp.add_trace(go.Bar(
        x=class_names,
        y=top_classes[fc_sample2].tolist(),
        name=f"{fc_sample2} (Comparison)",
        marker_color='#ff7f00',
    ))
    
    fig_comp.update_layout(
        title="Abundance Comparison Between Samples",
        xaxis_title="Chemical Class",
        yaxis_title="Summed Proportion",
        barmode='group',
        height=500,
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

# --- Chemical Ontology Hierarchy Distribution ---
st.markdown("### Chemical Ontology Hierarchy Distribution")

# User selects which class level to filter by
level_options = [
    ("ClassyFire#superclass", "ClassyFire#superclass probability"),
    ("ClassyFire#class", "ClassyFire#class Probability"),
    ("ClassyFire#subclass", "ClassyFire#subclass Probability"),
    ("ClassyFire#level 5", "ClassyFire#level 5 Probability"),
]
level_names = [x[0] for x in level_options]
selected_level = st.selectbox("Select ontology level for probability filter", level_names)
prob_col = dict(level_options)[selected_level]

# Probability threshold slider
prob_threshold = st.slider(
    f"Minimum probability for {selected_level}",
    min_value=0.0, max_value=1.0, value=0.7, step=0.01
)

# Prepare columns for hierarchy
hierarchy_cols = [
    "ClassyFire#superclass",
    "ClassyFire#class",
    "ClassyFire#subclass",
    "ClassyFire#level 5"
]

# Filter by probability at the selected level
filtered_hierarchy = filtered[filtered[prob_col] >= prob_threshold]

# Drop rows with all hierarchy levels missing
hierarchy_df = filtered_hierarchy[hierarchy_cols].dropna(how="all")

# Count occurrences for each path in the hierarchy (integer count)
hierarchy_counts = (
    hierarchy_df
    .groupby(hierarchy_cols)
    .size()
    .reset_index(name="count")
)
hierarchy_counts["count"] = hierarchy_counts["count"].astype(int)  # Ensure integer

hierarchy_counts = hierarchy_counts[hierarchy_cols + ["count"]]
hierarchy_counts["count"] = hierarchy_counts["count"].astype(int)

fig = px.sunburst(
    hierarchy_counts,
    path=hierarchy_cols,
    values="count",
    color="ClassyFire#superclass",  # or another hierarchy level
    title="Distribution of Chemical Ontology Classes"
)
fig.update_traces(
    hovertemplate='<b>%{label}</b><br>Parent: %{parent}<br>Count: %{value:d}<extra></extra>'
)
st.plotly_chart(fig, use_container_width=True)
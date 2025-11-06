import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Datanaut: Advanced Data Explorer",
    page_icon="🧑‍🚀",
    layout="wide"
)

# --- Title ---
st.title("🧑‍🚀 Datanaut: Advanced Data Explorer")

# --- Initialize Session State ---
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.individual_dfs = {}
    st.session_state.joined_data = None
    st.session_state.edited_data = None
    st.session_state.column_descriptions = {}

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload your CSV files", 
    type=["csv"],
    accept_multiple_files=True
)

# --- Data Loading Logic ---
if uploaded_files:
    data_frames = {}
    dfs_to_concat = []
    
    try:
        for file in uploaded_files:
            df = pd.read_csv(file)
            data_frames[file.name] = df
            
            df_copy = df.copy()
            df_copy['_source_file'] = file.name
            dfs_to_concat.append(df_copy)
            
        st.session_state.data = pd.concat(dfs_to_concat, ignore_index=True)
        st.session_state.individual_dfs = data_frames
        
        # Initialize column descriptions
        for file_name, df in data_frames.items():
            for col in df.columns:
                key = f"{file_name}::{col}"
                if key not in st.session_state.column_descriptions:
                    st.session_state.column_descriptions[key] = f"Column '{col}' from {file_name}"
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.session_state.data = None

# --- Main App Logic (requires data to be loaded) ---
if st.session_state.data is None:
    st.info("Upload one or more CSV files to get started.")
    st.stop()

# --- Create a copy for filtering ---
df = st.session_state.data.copy()

# --- Sidebar Filters ---
st.sidebar.header("Filter Your Data")

# Get column lists
num_cols = list(df.select_dtypes(include=['number']).columns)
cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)

if '_source_file' in cat_cols:
    cat_cols.remove('_source_file')
    cat_cols.insert(0, '_source_file') 

# Create a filtered dataframe
df_filtered = df.copy()

# --- Categorical Filters ---
st.sidebar.subheader("Categorical Filters")
for col in cat_cols:
    unique_values = ["All"] + sorted(list(map(str, df[col].unique())))
    selected = st.sidebar.multiselect(f"Filter by {col}", unique_values, default=["All"])
    
    if "All" not in selected:
        df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected)]

# --- Numerical Filters (Sliders) ---
st.sidebar.subheader("Numerical Filters")
for col in num_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    
    slider_range = st.sidebar.slider(
        f"Filter by {col}", 
        min_val, 
        max_val, 
        (min_val, max_val)
    )
    
    df_filtered = df_filtered[
        (
            (df_filtered[col] >= slider_range[0]) & 
            (df_filtered[col] <= slider_range[1])
        ) |
        (df_filtered[col].isna())
    ]

# Get columns for the *filtered* data
filtered_num_cols = list(df_filtered.select_dtypes(include=['number']).columns)
filtered_cat_cols = list(df_filtered.select_dtypes(include=['object', 'category']).columns)
all_cols = list(df_filtered.columns)

# --- Enhanced Tabbed Interface ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔄 Join & Compare", 
    "📊 Data Insights", 
    "📈 Visual Analytics",
    "✏️ Data Editor",
    "🔍 Data Explorer"
])

# ============================================
# --- Tab 1: Join & Compare ---
# ============================================
with tab1:
    st.header("🔄 Smart Data Joining & Comparison")
    
    individual_dfs = st.session_state.individual_dfs
    
    if len(individual_dfs) < 2:
        st.info("Upload at least two CSV files to use the join feature.")
    else:
        file_names = list(individual_dfs.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            left_file = st.selectbox("Select Left Table", options=file_names, index=0)
        with col2:
            right_file = st.selectbox("Select Right Table", options=file_names, index=1)
            
        df_left = individual_dfs[left_file]
        df_right = individual_dfs[right_file]
        
        # Auto-detect common columns for joining
        left_cols = set(df_left.columns)
        right_cols = set(df_right.columns)
        common_cols = list(left_cols.intersection(right_cols))
        
        if common_cols:
            st.success(f"🤖 Auto-detected common columns: {', '.join(common_cols)}")
            
            join_col = st.selectbox("Select Join Column", options=common_cols)
            
            # Join type selection with explanations
            st.subheader("Join Type Selection")
            
            join_info = {
                "inner": {
                    "name": "Inner Join",
                    "description": "Returns only records that have matching values in both tables",
                    "result": "Intersection of both tables",
                    "use_case": "Find common records between datasets",
                    "example": "Find customers who have placed orders"
                },
                "left": {
                    "name": "Left Join",
                    "description": "Returns all records from the left table and matched records from the right table",
                    "result": "All left table records + matches from right",
                    "use_case": "Keep all main records and attach additional data where available",
                    "example": "All customers with their orders (if any)"
                },
                "right": {
                    "name": "Right Join", 
                    "description": "Returns all records from the right table and matched records from the left table",
                    "result": "All right table records + matches from left",
                    "use_case": "Keep all reference records and attach main data where available",
                    "example": "All products with customer orders (if any)"
                },
                "outer": {
                    "name": "Full Outer Join",
                    "description": "Returns all records when there's a match in either left or right table",
                    "result": "Union of both tables with NULLs for non-matching fields",
                    "use_case": "See all records from both datasets together",
                    "example": "All customers and all products with their relationships"
                }
            }
            
            join_type = st.selectbox(
                "Select Join Type", 
                options=list(join_info.keys()),
                format_func=lambda x: join_info[x]["name"]
            )
            
            # Display join type explanation
            if join_type in join_info:
                info = join_info[join_type]
                with st.expander(f"ℹ️ About {info['name']}", expanded=True):
                    st.write(f"**How it works**: {info['description']}")
                    st.write(f"**Result**: {info['result']}")
                    st.write(f"**Best for**: {info['use_case']}")
                    st.write(f"**Example**: {info['example']}")
                    
                    # Visual explanation
                    if join_type == "inner":
                        st.info("🔍 **Visual**: Only the overlapping part of both circles in a Venn diagram")
                    elif join_type == "left":
                        st.info("🔍 **Visual**: Entire left circle + overlapping part with right")
                    elif join_type == "right":
                        st.info("🔍 **Visual**: Entire right circle + overlapping part with left")
                    else:
                        st.info("🔍 **Visual**: Both complete circles in a Venn diagram")
            
            if st.button("🔄 Perform Join"):
                try:
                    with st.spinner("Joining datasets..."):
                        joined_df = pd.merge(df_left, df_right, on=join_col, how=join_type, suffixes=('_left', '_right'))
                        st.session_state.joined_data = joined_df
                    
                    st.success(f"✅ Join successful! Result: {joined_df.shape[0]} rows × {joined_df.shape[1]} columns")
                    
                    # Display join insights
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Left Table Rows", len(df_left))
                    col2.metric("Right Table Rows", len(df_right))
                    col3.metric("Joined Rows", len(joined_df))
                    
                    efficiency = (len(joined_df) / max(len(df_left), len(df_right)) * 100) if max(len(df_left), len(df_right)) > 0 else 0
                    col4.metric("Join Efficiency", f"{efficiency:.1f}%")
                    
                    # Show join summary
                    st.subheader("Join Preview")
                    
                    # Create a preview with source information
                    left_only_cols = [col for col in df_left.columns if col != join_col]
                    right_only_cols = [col for col in df_right.columns if col != join_col]
                    
                    # Display source information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**📊 {left_file}**")
                        st.write(f"- Columns: {len(left_only_cols) + 1}")
                        st.write(f"- Rows: {len(df_left):,}")
                        
                    with col2:
                        st.write(f"**📊 {right_file}**")
                        st.write(f"- Columns: {len(right_only_cols) + 1}")
                        st.write(f"- Rows: {len(df_right):,}")
                    
                    st.dataframe(joined_df.head(100), use_container_width=True)
                    
                    # Relationship insights
                    st.subheader("📈 Relationship Insights")
                    
                    with st.spinner("Analyzing relationships..."):
                        # Check for many-to-many, one-to-many relationships
                        left_counts = df_left[join_col].value_counts()
                        right_counts = df_right[join_col].value_counts()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Unique keys in left", left_counts.nunique())
                        col2.metric("Unique keys in right", right_counts.nunique())
                        common_keys = len(set(df_left[join_col]).intersection(set(df_right[join_col])))
                        col3.metric("Common keys", common_keys)
                        
                        # Relationship type detection
                        left_duplicates = left_counts[left_counts > 1].count()
                        right_duplicates = right_counts[right_counts > 1].count()
                        
                        if left_duplicates == 0 and right_duplicates == 0:
                            rel_type = "One-to-One"
                            rel_color = "green"
                            rel_icon = "✅"
                        elif left_duplicates == 0 and right_duplicates > 0:
                            rel_type = "One-to-Many"
                            rel_color = "orange"
                            rel_icon = "⚠️"
                        elif left_duplicates > 0 and right_duplicates == 0:
                            rel_type = "Many-to-One" 
                            rel_color = "orange"
                            rel_icon = "⚠️"
                        else:
                            rel_type = "Many-to-Many"
                            rel_color = "red"
                            rel_icon = "🚨"
                        
                        col4.metric("Relationship Type", f"{rel_icon} {rel_type}")
                        
                        # Detailed relationship analysis
                        with st.expander("🔍 Detailed Relationship Analysis"):
                            st.write(f"**Left Table Analysis:**")
                            st.write(f"- Records with unique {join_col}: {left_counts.nunique() - left_duplicates}")
                            st.write(f"- Records with duplicate {join_col}: {left_duplicates}")
                            
                            st.write(f"**Right Table Analysis:**")
                            st.write(f"- Records with unique {join_col}: {right_counts.nunique() - right_duplicates}")
                            st.write(f"- Records with duplicate {join_col}: {right_duplicates}")
                            
                            if common_keys == 0:
                                st.error("❌ No common keys found - join result will be empty for inner join")
                            elif common_keys < min(left_counts.nunique(), right_counts.nunique()):
                                st.warning("⚠️ Partial match - not all keys from both tables are present")
                            else:
                                st.success("✅ Good key coverage between tables")
                                
                except Exception as e:
                    st.error(f"Join failed: {str(e)}")
        else:
            st.error("❌ No common columns found for joining. Tables need at least one column with the same name.")

# ============================================
# --- Tab 2: Enhanced Data Insights ---
# ============================================
with tab2:
    st.header("📊 Advanced Data Profile & Insights")
    
    # Let user choose which dataset to analyze
    dataset_choice = st.radio("Analyze Dataset:", 
                             ["Combined Data", "Joined Data"], 
                             horizontal=True)
    
    if dataset_choice == "Joined Data" and st.session_state.joined_data is not None:
        analysis_df = st.session_state.joined_data
        st.success("📊 Analyzing joined dataset")
    else:
        analysis_df = df_filtered
        st.info("📊 Analyzing filtered combined dataset")
    
    # High-Level KPIs
    st.subheader("📈 High-Level Metrics")
    total_rows = len(analysis_df)
    total_cols = len(analysis_df.columns)
    total_missing = analysis_df.isna().sum().sum()
    missing_percent = (total_missing / (total_rows * total_cols)) * 100 if total_rows * total_cols > 0 else 0
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Rows", f"{total_rows:,}")
    kpi2.metric("Total Columns", f"{total_cols:,}")
    kpi3.metric("Missing Cells", f"{total_missing:,} ({missing_percent:.2f}%)")
    kpi4.metric("Memory Usage", f"{analysis_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data Quality Assessment
    st.subheader("🔍 Data Quality Assessment")
    
    with st.spinner("Analyzing data quality..."):
        quality_data = []
        for col in analysis_df.columns:
            col_missing = analysis_df[col].isna().sum()
            col_missing_pct = (col_missing / len(analysis_df)) * 100
            unique_count = analysis_df[col].nunique()
            data_type = analysis_df[col].dtype
            
            # Quality score calculation
            completeness_score = 100 - col_missing_pct
            uniqueness_score = min(100, (unique_count / len(analysis_df)) * 100) if len(analysis_df) > 0 else 0
            quality_score = (completeness_score + uniqueness_score) / 2
            
            quality_data.append({
                'Column': col,
                'Data Type': data_type,
                'Missing Values': col_missing,
                'Missing %': f"{col_missing_pct:.1f}%",
                'Unique Values': unique_count,
                'Quality Score': f"{quality_score:.1f}%"
            })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)
    
    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    numeric_df = analysis_df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe(), use_container_width=True)
        
        # Distribution insights
        st.subheader("📋 Distribution Insights")
        dist_col1, dist_col2, dist_col3 = st.columns(3)
        
        with dist_col1:
            skewed_cols = []
            for col in numeric_df.columns:
                skewness = numeric_df[col].skew()
                if abs(skewness) > 1:
                    skewed_cols.append((col, skewness))
            
            if skewed_cols:
                st.write("**Highly Skewed Columns:**")
                for col, skew in skewed_cols[:3]:
                    direction = "right" if skew > 0 else "left"
                    st.write(f"- `{col}`: {skew:.2f} ({direction} skewed)")
        
        with dist_col2:
            # Zero variance columns
            zero_var_cols = [col for col in numeric_df.columns if numeric_df[col].std() == 0]
            if zero_var_cols:
                st.write("**Constant Value Columns:**")
                for col in zero_var_cols[:3]:
                    st.write(f"- `{col}`: constant value")
        
        with dist_col3:
            # Outlier detection
            outlier_cols = []
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
                if outlier_count > 0:
                    outlier_cols.append((col, outlier_count))
            
            if outlier_cols:
                st.write("**Columns with Outliers:**")
                for col, count in outlier_cols[:3]:
                    st.write(f"- `{col}`: {count} outliers")
                    
    else:
        st.info("No numeric columns found for statistical summary")
    
    # Pattern Detection
    st.subheader("🎯 Pattern Detection")
    
    # Detect potential primary keys
    potential_keys = []
    for col in analysis_df.columns:
        uniqueness = analysis_df[col].nunique() / len(analysis_df)
        if uniqueness > 0.95 and analysis_df[col].isna().sum() == 0:
            potential_keys.append((col, uniqueness))
    
    if potential_keys:
        potential_keys.sort(key=lambda x: x[1], reverse=True)
        key_list = [f"`{col}` ({uniqueness*100:.1f}% unique)" for col, uniqueness in potential_keys[:3]]
        st.success(f"🔑 **Potential Primary Keys:** {', '.join(key_list)}")
    
    # Correlation Analysis
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.subheader("📈 Correlation Analysis")
        
        with st.spinner("Calculating correlations..."):
            corr_matrix = analysis_df[numeric_cols].corr()
            
            # Simple correlation display without styling
            st.write("**Correlation Matrix:**")
            st.dataframe(corr_matrix, use_container_width=True)
            
            # Correlation insights
            st.subheader("🔍 Correlation Insights")
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        corr_pairs.append((col1, col2, corr_value))
            
            # Sort by absolute correlation strength
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            if corr_pairs:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strong Positive Correlations (r > 0.7):**")
                    strong_pos = [(c1, c2, r) for c1, c2, r in corr_pairs if r > 0.7]
                    for col1, col2, corr_val in strong_pos[:3]:
                        st.write(f"- `{col1}` ↔ `{col2}`: {corr_val:.3f}")
                    if not strong_pos:
                        st.write("*None found*")
                
                with col2:
                    st.write("**Strong Negative Correlations (r < -0.7):**")
                    strong_neg = [(c1, c2, r) for c1, c2, r in corr_pairs if r < -0.7]
                    for col1, col2, corr_val in strong_neg[:3]:
                        st.write(f"- `{col1}` ↔ `{col2}`: {corr_val:.3f}")
                    if not strong_neg:
                        st.write("*None found*")
            else:
                st.info("No significant correlations found")

# ============================================
# --- Tab 3: Enhanced Visual Analytics ---
# ============================================
with tab3:
    st.header("📈 Advanced Visual Analytics")
    
    # Dataset selection for visualization
    viz_dataset = st.radio("Visualize Dataset:", 
                          ["Combined Data", "Joined Data"], 
                          horizontal=True,
                          key="viz_dataset")
    
    if viz_dataset == "Joined Data" and st.session_state.joined_data is not None:
        viz_df = st.session_state.joined_data
        st.success("📈 Visualizing joined dataset")
    else:
        viz_df = df_filtered
        st.info("📈 Visualizing filtered combined dataset")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Chart Configuration")
        chart_type = st.selectbox("Chart Type", 
                                ["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", 
                                 "Box Plot", "Heatmap", "Pie Chart", "Area Chart", 
                                 "Bubble Chart", "Density Plot", "Violin Plot"])
        
        # Dynamic controls based on chart type
        if chart_type in ["Scatter Plot", "Bar Chart", "Line Chart", "Area Chart", "Bubble Chart"]:
            x_axis = st.selectbox("X-Axis", options=all_cols)
            y_axis = st.selectbox("Y-Axis", options=filtered_num_cols)
            color_by = st.selectbox("Color By", options=["None"] + filtered_cat_cols)
            
            if chart_type == "Bubble Chart":
                size_by = st.selectbox("Size By", options=["None"] + filtered_num_cols)
            
        elif chart_type in ["Histogram", "Density Plot"]:
            hist_col = st.selectbox("Select Column", options=filtered_num_cols)
            bins = st.slider("Number of Bins", 5, 100, 20) if chart_type == "Histogram" else None
            
        elif chart_type in ["Box Plot", "Violin Plot"]:
            box_col = st.selectbox("Select Column", options=filtered_num_cols)
            group_by = st.selectbox("Group By", options=["None"] + filtered_cat_cols)
            
        elif chart_type == "Pie Chart":
            pie_category = st.selectbox("Category Column", options=filtered_cat_cols)
            pie_value = st.selectbox("Value Column", options=filtered_num_cols)
            max_categories = st.slider("Max Categories to Show", 5, 50, 15)
            
        elif chart_type == "Heatmap":
            st.info("Heatmap will show correlation matrix")
    
    with col2:
        st.subheader("Visualization")
        try:
            if chart_type == "Scatter Plot":
                chart = alt.Chart(viz_df).mark_circle(opacity=0.7).encode(
                    x=alt.X(x_axis, title=x_axis),
                    y=alt.Y(y_axis, title=y_axis),
                    color=alt.Color(color_by, title=color_by) if color_by != "None" else alt.value('steelblue'),
                    tooltip=all_cols[:8]  # Limit tooltip columns for performance
                ).properties(
                    title=f"{chart_type}: {x_axis} vs {y_axis}",
                    width=600,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Bar Chart":
                # For bar charts, aggregate if needed
                if viz_df[x_axis].nunique() < 50:  # If reasonable number of categories
                    if color_by != "None":
                        agg_df = viz_df.groupby([x_axis, color_by])[y_axis].mean().reset_index()
                        base_chart = alt.Chart(agg_df)
                    else:
                        agg_df = viz_df.groupby(x_axis)[y_axis].mean().reset_index()
                        base_chart = alt.Chart(agg_df)
                else:
                    base_chart = alt.Chart(viz_df)
                    st.warning(f"Many categories ({viz_df[x_axis].nunique()}) - consider filtering or using a different chart")
                
                chart = base_chart.mark_bar().encode(
                    x=alt.X(x_axis, title=x_axis),
                    y=alt.Y(f"{y_axis}:Q", title=f"Average {y_axis}"),
                    color=alt.Color(color_by, title=color_by) if color_by != "None" else alt.value('steelblue'),
                    tooltip=[x_axis, alt.Tooltip(f"{y_axis}:Q", format=".2f")]
                ).properties(
                    title=f"Bar Chart: {y_axis} by {x_axis}",
                    width=600,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Line Chart":
                # For line charts, ensure x-axis is sortable
                if pd.api.types.is_numeric_dtype(viz_df[x_axis]) or pd.api.types.is_datetime64_any_dtype(viz_df[x_axis]):
                    chart = alt.Chart(viz_df).mark_line().encode(
                        x=alt.X(x_axis, title=x_axis),
                        y=alt.Y(y_axis, title=y_axis),
                        color=alt.Color(color_by, title=color_by) if color_by != "None" else alt.value('steelblue'),
                        tooltip=[x_axis, y_axis]
                    ).properties(
                        title=f"Line Chart: {y_axis} over {x_axis}",
                        width=600,
                        height=400
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error("Line charts require numeric or datetime x-axis. Current type: " + str(viz_df[x_axis].dtype))
                
            elif chart_type == "Histogram":
                chart = alt.Chart(viz_df).mark_bar().encode(
                    x=alt.X(hist_col, bin=alt.Bin(maxbins=bins), title=hist_col),
                    y='count()',
                    tooltip=[alt.Tooltip('count()', title='Frequency')]
                ).properties(
                    title=f"Histogram: Distribution of {hist_col}",
                    width=600,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Box Plot":
                if group_by != "None":
                    chart = alt.Chart(viz_df).mark_boxplot().encode(
                        x=alt.X(group_by, title=group_by),
                        y=alt.Y(box_col, title=box_col),
                        color=alt.Color(group_by, title=group_by)
                    )
                else:
                    chart = alt.Chart(viz_df).mark_boxplot().encode(
                        y=alt.Y(box_col, title=box_col)
                    )
                chart = chart.properties(
                    title=f"Box Plot: {box_col}",
                    width=600,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Violin Plot":
                # Create a violin plot using transformed data
                if group_by != "None":
                    chart = alt.Chart(viz_df).transform_density(
                        box_col,
                        as_=[box_col, 'density'],
                        groupby=[group_by]
                    ).mark_area(orient='horizontal').encode(
                        y=alt.Y(f'{box_col}:Q', title=box_col),
                        x=alt.X('density:Q', title='Density', stack='center'),
                        color=alt.Color(f'{group_by}:N', title=group_by)
                    ).properties(
                        title=f"Violin Plot: {box_col} by {group_by}",
                        width=600,
                        height=400
                    )
                else:
                    chart = alt.Chart(viz_df).transform_density(
                        box_col,
                        as_=[box_col, 'density']
                    ).mark_area(orient='horizontal').encode(
                        y=alt.Y(f'{box_col}:Q', title=box_col),
                        x=alt.X('density:Q', title='Density', stack='center')
                    ).properties(
                        title=f"Violin Plot: {box_col}",
                        width=600,
                        height=400
                    )
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Pie Chart":
                # Prepare data for pie chart
                pie_data = viz_df.groupby(pie_category)[pie_value].sum().reset_index()
                pie_data = pie_data.sort_values(pie_value, ascending=False).head(max_categories)
                
                # Create pie chart
                pie_chart = alt.Chart(pie_data).mark_arc().encode(
                    theta=alt.Theta(f"{pie_value}:Q", title=pie_value),
                    color=alt.Color(f"{pie_category}:N", title=pie_category,
                                  legend=alt.Legend(orient="right")),
                    tooltip=[pie_category, pie_value]
                ).properties(
                    title=f"Pie Chart: {pie_value} by {pie_category}",
                    width=500,
                    height=400
                )
                st.altair_chart(pie_chart, use_container_width=True)
                
            elif chart_type == "Area Chart":
                chart = alt.Chart(viz_df).mark_area(opacity=0.7).encode(
                    x=alt.X(x_axis, title=x_axis),
                    y=alt.Y(y_axis, title=y_axis),
                    color=alt.Color(color_by, title=color_by) if color_by != "None" else alt.value('steelblue'),
                    tooltip=[x_axis, y_axis]
                ).properties(
                    title=f"Area Chart: {y_axis} over {x_axis}",
                    width=600,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Bubble Chart":
                chart = alt.Chart(viz_df).mark_circle(opacity=0.7).encode(
                    x=alt.X(x_axis, title=x_axis),
                    y=alt.Y(y_axis, title=y_axis),
                    size=alt.Size(size_by, title=size_by) if size_by != "None" else alt.SizeValue(100),
                    color=alt.Color(color_by, title=color_by) if color_by != "None" else alt.value('steelblue'),
                    tooltip=all_cols[:6]
                ).properties(
                    title=f"Bubble Chart: {x_axis} vs {y_axis}",
                    width=600,
                    height=400
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Density Plot":
                chart = alt.Chart(viz_df).transform_density(
                    hist_col,
                    as_=[hist_col, 'density']
                ).mark_area().encode(
                    x=alt.X(f"{hist_col}:Q", title=hist_col),
                    y=alt.Y('density:Q', title='Density')
                ).properties(
                    title=f"Density Plot: {hist_col}",
                    width=600,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
                
            elif chart_type == "Heatmap":
                numeric_cols = viz_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_data = viz_df[numeric_cols].corr().reset_index().melt('index')
                    corr_data.columns = ['var1', 'var2', 'value']
                    
                    # Filter out NaN values
                    corr_data = corr_data.dropna()
                    
                    chart = alt.Chart(corr_data).mark_rect().encode(
                        x=alt.X('var2:O', title=''),
                        y=alt.Y('var1:O', title=''),
                        color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue'), title='Correlation'),
                        tooltip=['var1', 'var2', alt.Tooltip('value', format='.3f')]
                    ).properties(
                        title="Correlation Heatmap",
                        width=500,
                        height=500
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for heatmap")
                    
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
            st.info("💡 Try selecting different columns or check your data types")

# ============================================
# --- Tab 4: Data Editor ---
# ============================================
with tab4:
    st.header("✏️ Interactive Data Editor")
    
    edit_dataset = st.radio("Edit Dataset:", 
                           ["Combined Data", "Joined Data"], 
                           horizontal=True,
                           key="edit_dataset")
    
    if edit_dataset == "Joined Data" and st.session_state.joined_data is not None:
        edit_df = st.session_state.joined_data.copy()
    else:
        edit_df = df_filtered.copy()
    
    st.info("💡 Use the table below to edit your data. Changes are saved to session state.")
    
    # Data editing interface
    edited_df = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Edits"):
            st.session_state.edited_data = edited_df
            st.success("Edits saved to session!")
            
    with col2:
        if st.button("🔄 Reset to Original"):
            st.session_state.edited_data = None
            st.rerun()
    
    with col3:
        if st.download_button(
            label="📥 Download Edited Data",
            data=edited_df.to_csv(index=False),
            file_name="edited_data.csv",
            mime="text/csv"
        ):
            st.success("Download ready!")
    
    # Data transformation tools
    st.subheader("🛠️ Data Transformation Tools")
    
    trans_col1, trans_col2, trans_col3 = st.columns(3)
    
    with trans_col1:
        if st.button("🧹 Clean Missing Values"):
            cleaned_df = edited_df.dropna()
            st.session_state.edited_data = cleaned_df
            st.rerun()
    
    with trans_col2:
        if st.button("🔢 Standardize Numeric Columns"):
            numeric_cols = edited_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if edited_df[col].std() != 0:  # Avoid division by zero
                    edited_df[col] = (edited_df[col] - edited_df[col].mean()) / edited_df[col].std()
            st.session_state.edited_data = edited_df
            st.rerun()
    
    with trans_col3:
        new_col_name = st.text_input("New Column Name", "new_column")
        if st.button("➕ Add Calculated Column"):
            if new_col_name and new_col_name not in edited_df.columns:
                edited_df[new_col_name] = np.nan
                st.session_state.edited_data = edited_df
                st.rerun()
            elif new_col_name in edited_df.columns:
                st.error("Column already exists!")

# ============================================
# --- Tab 5: Data Explorer ---
# ============================================
with tab5:
    st.header("🔍 Data Explorer")
    
    explore_dataset = st.radio("Explore Dataset:", 
                              ["Combined Data", "Joined Data", "Edited Data"], 
                              horizontal=True,
                              key="explore_dataset")
    
    if explore_dataset == "Joined Data" and st.session_state.joined_data is not None:
        display_df = st.session_state.joined_data
    elif explore_dataset == "Edited Data" and st.session_state.edited_data is not None:
        display_df = st.session_state.edited_data
    else:
        display_df = df_filtered
    
    st.write(f"Displaying **{len(display_df)}** rows × **{len(display_df.columns)}** columns")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Numeric Columns", len(display_df.select_dtypes(include=[np.number]).columns))
    col2.metric("Text Columns", len(display_df.select_dtypes(include=['object']).columns))
    col3.metric("Total Cells", len(display_df) * len(display_df.columns))
    complete_cells = (1 - display_df.isna().sum().sum() / (len(display_df) * len(display_df.columns))) * 100
    col4.metric("Complete Cells", f"{complete_cells:.1f}%")
    
    # Data preview with configurable number of rows
    rows_to_show = st.slider("Rows to display", 10, 1000, 100)
    
    # Enhanced dataframe display with better formatting
    st.dataframe(display_df.head(rows_to_show), use_container_width=True)
    
    # Column-specific analysis
    st.subheader("🔬 Column Deep Dive")
    selected_col = st.selectbox("Select column for analysis", options=all_cols)
    
    if selected_col:
        col_data = display_df[selected_col]
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Stats**")
            st.write(f"**Data type**: `{col_data.dtype}`")
            st.write(f"**Unique values**: `{col_data.nunique()}`")
            st.write(f"**Missing values**: `{col_data.isna().sum()}` ({col_data.isna().sum()/len(col_data)*100:.1f}%)")
            
            if col_data.dtype in ['object']:
                st.write("**Sample Values:**")
                sample_values = col_data.dropna().unique()[:5]
                for val in sample_values:
                    st.write(f"- `{val}`")
            
        with col2:
            st.write("**Value Distribution**")
            if col_data.dtype in ['object']:
                top_values = col_data.value_counts().head(10)
                st.write("Top 10 values:")
                st.dataframe(top_values, use_container_width=True)
            else:
                st.write(f"**Min**: `{col_data.min()}`")
                st.write(f"**Max**: `{col_data.max()}`")
                st.write(f"**Mean**: `{col_data.mean():.2f}`")
                st.write(f"**Median**: `{col_data.median():.2f}`")
                st.write(f"**Standard Dev**: `{col_data.std():.2f}`")
                
                # Quick distribution visualization
                if not col_data.empty and col_data.nunique() > 1:
                    hist_chart = alt.Chart(display_df).mark_bar().encode(
                        x=alt.X(selected_col, bin=alt.Bin(maxbins=20), title=selected_col),
                        y='count()'
                    ).properties(
                        title=f"Distribution of {selected_col}",
                        width=300,
                        height=200
                    )
                    st.altair_chart(hist_chart, use_container_width=True)

# --- Tooltip and Description Management ---
st.sidebar.header("🔧 Advanced Settings")

with st.sidebar.expander("📝 Column Descriptions"):
    st.write("Manage column descriptions for better tooltips")
    
    if st.session_state.individual_dfs:
        selected_file = st.selectbox("Select File", options=list(st.session_state.individual_dfs.keys()))
        if selected_file:
            df_cols = st.session_state.individual_dfs[selected_file].columns
            selected_col = st.selectbox("Select Column", options=df_cols)
            
            current_desc = st.session_state.column_descriptions.get(
                f"{selected_file}::{selected_col}", 
                f"Column '{selected_col}' from {selected_file}"
            )
            
            new_desc = st.text_area("Column Description", value=current_desc, height=100)
            
            if st.button("💾 Save Description"):
                st.session_state.column_descriptions[f"{selected_file}::{selected_col}"] = new_desc
                st.success("Description saved!")

# --- Footer with helpful information ---
st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Pro Tips:**
- Upload related CSV files (users, orders, products) for join analysis
- Use the Join tab to combine datasets on common columns
- Hover over charts for interactive tooltips
- Edit data directly in the Data Editor tab
- Check data quality in the Insights tab
""")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <small>Crafted with care by 🦉 <a href='https://ramc26.github.io/RamTechSuite' target='_blank'>RamTechSuite</a> | Hosted on Streamlit</small>
    </div>
    """, 
    unsafe_allow_html=True
)
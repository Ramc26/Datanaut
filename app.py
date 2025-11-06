import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="DataVista — Interactive Dashboard", layout="wide")

# ------------------------- Helpers -------------------------
@st.cache_data
def load_dataframe(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif name.endswith('.xls') or name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            # try csv as fallback
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

@st.cache_data
def generate_sample_df():
    # a small sample mixed dataset
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n).to_pydatetime().tolist()
    df = pd.DataFrame({
        'date': dates,
        'category': rng.choice(['A','B','C','D'], size=n, p=[0.4,0.25,0.2,0.15]),
        'region': rng.choice(['North','South','East','West'], size=n),
        'sales': np.round(rng.normal(200, 60, size=n).clip(10), 2),
        'units': rng.integers(1, 50, size=n),
    })
    return df


def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = []
    for c in df.columns:
        if c not in numeric_cols:
            # try to parse as datetime
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().sum() > max(1, int(0.6*len(df))):
                    datetime_cols.append(c)
                    df[c] = parsed
            except Exception:
                pass
    cat_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]
    return numeric_cols, cat_cols, datetime_cols


# ------------------------- UI -------------------------
st.title("DataVista — Interactive Data Dashboard")
st.write("Upload a CSV/Excel file (or use the sample) and explore with filters & charts.")

col1, col2 = st.columns([1,3])
with col1:
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv','xls','xlsx'], accept_multiple_files=False)
    use_sample = st.checkbox("Use sample dataset", value=(uploaded is None))

    st.markdown("**Data options**")
    preview_rows = st.number_input("Preview rows", min_value=5, max_value=1000, value=10)
    st.markdown("---")
    st.markdown("**Export filtered**")
    if st.button("Download filtered CSV"):
        if 'filtered_df' in st.session_state:
            to_download = st.session_state['filtered_df']
            csv = to_download.to_csv(index=False).encode('utf-8')
            st.download_button(label='Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')
        else:
            st.info('No filtered data yet. Apply filters first.')

with col2:
    if uploaded is not None:
        df = load_dataframe(uploaded)
    elif use_sample:
        df = generate_sample_df()
    else:
        df = None

if df is None:
    st.stop()

st.session_state['original_df'] = df

numeric_cols, cat_cols, datetime_cols = detect_column_types(df)

# sidebar filters
st.sidebar.header("Filters & Settings")
with st.sidebar.form(key='filters'):
    chosen_cols = st.sidebar.multiselect("Columns to show (table)", options=df.columns.tolist(), default=df.columns.tolist())

    # dynamic filters
    filters = {}
    st.sidebar.subheader("Column filters")
    # numeric sliders
    for col in numeric_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        step = (col_max - col_min) / 100 if col_max != col_min else 1
        lo, hi = st.sidebar.slider(f"{col}", min_value=col_min, max_value=col_max, value=(col_min, col_max), step=step)
        filters[col] = (lo, hi)

    # categorical multiselect (show top 50 unique values)
    for col in cat_cols:
        vals = df[col].astype(str).value_counts().index.tolist()
        default_sel = vals[:5] if len(vals) > 0 else []
        sel = st.sidebar.multiselect(f"{col}", options=vals, default=default_sel)
        filters[col] = sel

    # datetime range if available
    for col in datetime_cols:
        dmin = df[col].min().date()
        dmax = df[col].max().date()
        start_date, end_date = st.sidebar.date_input(f"{col} range", value=(dmin, dmax))
        filters[col] = (pd.to_datetime(start_date), pd.to_datetime(end_date))

    submitted = st.form_submit_button('Apply Filters')

# Apply filters to produce filtered_df
filtered = df.copy()
for col, cond in filters.items():
    if col in numeric_cols:
        lo, hi = cond
        filtered = filtered[filtered[col].between(lo, hi)]
    elif col in datetime_cols:
        start, end = cond
        filtered = filtered[(filtered[col] >= start) & (filtered[col] <= end)]
    else:
        # categorical
        sel = cond
        if sel:
            filtered = filtered[filtered[col].astype(str).isin(sel)]

# save filtered to session for download
st.session_state['filtered_df'] = filtered

# Main layout: data preview and charts
st.subheader('Data Preview')
st.write(f"Showing **{len(filtered)}** rows (from {len(df)} total).")
st.dataframe(filtered[chosen_cols].head(preview_rows), use_container_width=True)

# Quick stats
c1, c2, c3 = st.columns(3)
with c1:
    st.metric('Rows (filtered)', value=len(filtered))
with c2:
    st.metric('Columns', value=len(filtered.columns))
with c3:
    st.metric('Numeric cols', value=len(numeric_cols))

st.markdown('---')

# Chart controls
st.subheader('Charts')
chart_col1, chart_col2 = st.columns([1,1])
with chart_col1:
    chart_type = st.selectbox('Chart type', ['Line', 'Bar', 'Pie'])
    x_axis = st.selectbox('X axis', options=filtered.columns.tolist(), index=0)
    y_axis = st.selectbox('Y axis (numeric)', options=numeric_cols, index=0 if numeric_cols else None)
    agg = st.selectbox('Aggregation (for Bar)', ['sum', 'mean', 'count'])

with chart_col2:
    color_by = st.selectbox('Color / Group by (optional)', options=[None] + filtered.columns.tolist(), index=0)
    limit_categories = st.number_input('Max categories to show (bar/pie)', value=10, min_value=2, max_value=100)

# build chart
chart = None
try:
    if chart_type == 'Line':
        # if x_axis is datetime or numeric use direct line chart, else try to aggregate by x
        if x_axis in datetime_cols:
            temp = filtered.sort_values(x_axis)
            chart = px.line(temp, x=x_axis, y=y_axis, color=color_by, title=f'Line: {y_axis} vs {x_axis}')
        elif filtered[x_axis].dtype.kind in 'biufc':
            chart = px.line(filtered, x=x_axis, y=y_axis, color=color_by, title=f'Line: {y_axis} vs {x_axis}')
        else:
            # aggregate
            grp = filtered.groupby(x_axis)[y_axis].mean().reset_index()
            chart = px.line(grp, x=x_axis, y=y_axis, title=f'Line (agg mean): {y_axis} by {x_axis}')

    elif chart_type == 'Bar':
        if color_by and color_by != 'None':
            # grouped bar: aggregate by x and color
            grp = filtered.groupby([x_axis, color_by])[y_axis].agg(agg).reset_index()
            chart = px.bar(grp, x=x_axis, y=y_axis, color=color_by, barmode='group', title=f'Bar ({agg}): {y_axis} by {x_axis} and {color_by}')
        else:
            grp = filtered.groupby(x_axis)[y_axis].agg(agg).reset_index().nlargest(limit_categories, y_axis)
            chart = px.bar(grp, x=x_axis, y=y_axis, title=f'Bar ({agg}): {y_axis} by {x_axis}')

    elif chart_type == 'Pie':
        # Pie requires a categorical split; we'll use x_axis as category
        counts = filtered[x_axis].astype(str).value_counts().nlargest(limit_categories).reset_index()
        counts.columns = [x_axis, 'count']
        chart = px.pie(counts, names=x_axis, values='count', title=f'Pie: {x_axis} distribution')

    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)
except Exception as e:
    st.error(f"Failed to build chart: {e}")

st.markdown('---')

# Column summary (simple)
st.subheader('Column summary')
selected_summary_cols = st.multiselect('Pick columns to summarize', options=filtered.columns.tolist(), default=(numeric_cols[:3] if numeric_cols else filtered.columns[:3]))
if selected_summary_cols:
    st.write(filtered[selected_summary_cols].describe(include='all').transpose())

st.markdown('---')

# Tips / Next steps
st.info("Tips: Try using a time/date column as X for line charts. Use aggregation 'count' for bar charts when you want frequency. You can extend this app by adding saved views, more chart types (scatter, histogram), and cross filters between charts.")

# end


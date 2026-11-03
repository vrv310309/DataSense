from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="DataSense | AI Data Analyst", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
    [data-testid="stMetric"] {background: #f6f7f8; border: 1px solid #e8eaed; padding: 14px; border-radius: 12px;}
    .hero {padding: 1.25rem 1.5rem; border: 1px solid #e7e9ec; border-radius: 16px; background: linear-gradient(120deg,#f7f8fa,#ffffff);}
    .muted {color: #68717d;}
</style>
""", unsafe_allow_html=True)

st.title("📊 DataSense")
st.caption("Your data, understood. Upload a dataset to explore patterns, quality and useful insights.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.caption("No file handy?")
    if st.button("Load sample sales dataset", use_container_width=True):
        st.session_state["use_sample"] = True
    st.markdown("---")
    st.caption("**Privacy**")
    st.caption("This MVP processes uploaded data in the active app session. It does not intentionally save your dataset to this repository or send it to an external AI service. Hosting providers may have their own logging and retention policies.")

def load_data(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() or f"column_{i+1}" for i, c in enumerate(df.columns)]
    return df

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def categorical_columns(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

def quality_table(df):
    missing = df.isna().sum()
    result = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(x) for x in df.dtypes],
        "Missing values": [int(missing[c]) for c in df.columns],
        "Missing (%)": [round(float(missing[c] / max(len(df), 1) * 100), 2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return result

def generate_insights(df):
    lines = []
    rows, cols = df.shape
    lines.append(f"Your dataset contains **{rows:,} rows** and **{cols:,} columns**.")
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    lines.append(f"There are **{missing_total:,} missing cells** and **{duplicate_total:,} fully duplicated rows**.")
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    if nums:
        col = nums[0]
        series = df[col].dropna()
        if not series.empty:
            lines.append(f"For **{col}**, the average is **{series.mean():,.2f}**, while the median is **{series.median():,.2f}**.")
            lines.append(f"The observed range for **{col}** is **{series.min():,.2f} to {series.max():,.2f}**.")
    if cats:
        col = cats[0]
        mode = df[col].dropna().mode()
        if not mode.empty:
            count = int((df[col] == mode.iloc[0]).sum())
            lines.append(f"The most frequent value in **{col}** is **{mode.iloc[0]}** ({count:,} rows).")
    if not nums and not cats:
        lines.append("This dataset has no standard numeric or categorical columns to summarize.")
    return lines

if uploaded is not None:
    try:
        df = clean_column_names(load_data(uploaded))
        st.session_state["data"] = df
        st.session_state["data_name"] = uploaded.name
        st.session_state["use_sample"] = False
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.stop()
elif st.session_state.get("use_sample"):
    sample_path = Path(__file__).parent / "sample_data" / "sales.csv"
    df = pd.read_csv(sample_path, parse_dates=["date"])
    st.session_state["data"] = clean_column_names(df)
    st.session_state["data_name"] = "sample_data/sales.csv"
elif "data" in st.session_state:
    df = st.session_state["data"]
else:
    st.markdown('<div class="hero"><h3>Start with a dataset</h3><p class="muted">Upload a CSV or Excel workbook from the sidebar, or load the sample sales dataset to try the app.</p></div>', unsafe_allow_html=True)
    st.subheader("What you can do")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🔎 Understand**\n\nSee dataset size, column types, summary statistics and distributions.")
    c2.markdown("**🧹 Check quality**\n\nFind missing values, duplicate rows and potential numeric outliers.")
    c3.markdown("**📈 Explore**\n\nBuild interactive charts and export a compact analysis report.")
    st.info("This is the first MVP. Its insights are generated with transparent statistical rules; a connected generative AI model is not included yet.")
    st.stop()

if df.empty:
    st.warning("The uploaded dataset has no rows.")
    st.stop()

st.markdown(f"**Current dataset:** `{st.session_state.get('data_name', 'dataset')}`")
rows, cols = df.shape
n_numeric = len(numeric_columns(df))
n_missing = int(df.isna().sum().sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{rows:,}")
m2.metric("Columns", f"{cols:,}")
m3.metric("Numeric columns", n_numeric)
m4.metric("Missing cells", f"{n_missing:,}")

tab_overview, tab_quality, tab_visuals, tab_data = st.tabs(
    ["✨ Overview & insights", "🧹 Data quality", "📈 Visual explorer", "🗂️ Explore data"]
)

with tab_overview:
    st.subheader("Quick read")
    for insight in generate_insights(df):
        st.markdown(f"- {insight}")
    st.subheader("Descriptive statistics")
    if numeric_columns(df):
        st.dataframe(df[numeric_columns(df)].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found for descriptive statistics.")
    st.subheader("Column overview")
    st.dataframe(quality_table(df), use_container_width=True, hide_index=True)

with tab_quality:
    st.subheader("Data quality report")
    q = quality_table(df)
    st.dataframe(q, use_container_width=True, hide_index=True)
    duplicate_count = int(df.duplicated().sum())
    a, b = st.columns(2)
    a.metric("Fully duplicated rows", f"{duplicate_count:,}")
    a.metric("Missing cells", f"{n_missing:,}")
    st.markdown("#### Potential outliers (IQR method)")
    nums = numeric_columns(df)
    if nums:
        records = []
        for col in nums:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((s < low) | (s > high)).sum())
            records.append({"Column": col, "Lower fence": low, "Upper fence": high, "Potential outliers": count})
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
            st.caption("IQR flags are screening indicators, not proof that a value is erroneous.")
        else:
            st.info("Not enough numeric observations to calculate IQR outliers.")
    else:
        st.info("No numeric columns available.")

with tab_visuals:
    st.subheader("Build a chart")
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    chart_type = st.selectbox("Chart type", ["Histogram", "Bar chart", "Scatter plot", "Correlation heatmap"])
    if chart_type == "Histogram":
        if nums:
            x = st.selectbox("Numeric column", nums)
            fig = px.histogram(df, x=x, nbins=30, title=f"Distribution of {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A histogram needs at least one numeric column.")
    elif chart_type == "Bar chart":
        if cats:
            x = st.selectbox("Category column", cats)
            counts = df[x].fillna("(Missing)").astype(str).value_counts().head(25).rename_axis(x).reset_index(name="Count")
            fig = px.bar(counts, x=x, y="Count", title=f"Top values in {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A bar chart needs a categorical column.")
    elif chart_type == "Scatter plot":
        if len(nums) >= 2:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X axis", nums, index=0)
            y_default = 1 if len(nums) > 1 else 0
            y = c2.selectbox("Y axis", nums, index=y_default)
            color = c3.selectbox("Color (optional)", ["None"] + cats)
            fig = px.scatter(df, x=x, y=y, color=None if color == "None" else color, title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A scatter plot needs at least two numeric columns.")
    else:
        if len(nums) >= 2:
            corr = df[nums].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Numeric correlation matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A correlation heatmap needs at least two numeric columns.")

with tab_data:
    st.subheader("Preview and filter")
    filtered = df.copy()
    searchable = st.text_input("Search all text values (case-insensitive)")
    if searchable:
        mask = filtered.astype(str).apply(lambda col: col.str.contains(searchable, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")
    st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv_bytes, file_name="datasense_filtered.csv", mime="text/csv")

st.markdown("---")
st.subheader("Export your analysis")
report_lines = [
    "# DataSense analysis report",
    f"**Dataset:** {st.session_state.get('data_name', 'dataset')}",
    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"**Shape:** {rows:,} rows × {cols:,} columns",
    "",
    "## Key insights",
]
report_lines.extend([f"- {line.replace('**', '')}" for line in generate_insights(df)])
report_lines += ["", "## Data quality", quality_table(df).to_markdown(index=False)]
report = "\n".join(report_lines)
st.download_button("Download report (Markdown)", report, file_name="datasense_report.md", mime="text/markdown")
st.caption("DataSense MVP · Statistical and rule-based analysis · AI/LLM integration is planned, not yet implemented.")

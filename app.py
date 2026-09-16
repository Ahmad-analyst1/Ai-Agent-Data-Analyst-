import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Analyst Agent")
st.caption("Upload a CSV or Excel file to automatically explore, analyze, visualize, and ask questions about your data.")

def load_data(file):
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None

def detect_types(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric + datetime_cols]
    return numeric, categorical, datetime_cols

def outliers(df, numeric):
    results = []
    for col in numeric:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < low) | (s > high)).sum())
        results.append({"Column": col, "Outliers": n, "Outlier %": round(n / len(s) * 100, 2)})
    return pd.DataFrame(results)

def insights(df):
    numeric, categorical, _ = detect_types(df)
    out = [f"Dataset contains **{len(df):,} rows** and **{len(df.columns)} columns**."]
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    out.append(f"Found **{missing:,} missing values**." if missing else "No missing values detected.")
    if duplicates:
        out.append(f"Found **{duplicates:,} duplicate rows**.")
    for col in numeric:
        s = df[col].dropna()
        if len(s):
            mean, median = s.mean(), s.median()
            if mean > median * 1.1:
                out.append(f"**{col}** shows a right-skewed tendency.")
            elif median > mean * 1.1:
                out.append(f"**{col}** shows a left-skewed tendency.")
    return out

def summary(df):
    numeric, categorical, datetime_cols = detect_types(df)
    data = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": datetime_cols,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }
    if numeric:
        data["statistics"] = df[numeric].describe().round(3).to_dict()
        data["correlation"] = df[numeric].corr().round(3).to_dict()
    return data

def ask_ai(df, question):
    if not OPENAI_AVAILABLE:
        return "OpenAI package is not installed."
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API is not configured. Add OPENAI_API_KEY to Streamlit Secrets."
    client = OpenAI(api_key=api_key)
    prompt = f"""You are an expert data analyst.
Dataset summary:
{json.dumps(summary(df), default=str, indent=2)}

User question:
{question}

Answer clearly using only the available dataset information. Do not invent values.
"""
    try:
        response = client.responses.create(model="gpt-5-mini", input=prompt)
        return response.output_text
    except Exception as e:
        return f"AI error: {e}"

with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"])
    show_preview = st.checkbox("Show data preview", True)

if uploaded is None:
    st.info("Upload a CSV or Excel file from the sidebar to begin.")
    st.markdown("### Features")
    c1, c2, c3 = st.columns(3)
    c1.write("🔎 **Profiling**\n\nColumns, types, missing values and duplicates.")
    c2.write("📊 **EDA**\n\nStatistics, correlations, charts and outliers.")
    c3.write("🤖 **AI Q&A**\n\nAsk natural-language questions about your dataset.")
    st.stop()

df = load_data(uploaded)
if df is None or df.empty:
    st.error("The dataset is empty or could not be loaded.")
    st.stop()

numeric, categorical, datetime_cols = detect_types(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Numeric", len(numeric))
c4.metric("Categorical", len(categorical))
c5.metric("Missing", f"{df.isna().sum().sum():,}")

tabs = st.tabs(["📄 Data", "📊 Statistics", "📈 Charts", "⚠️ Quality", "💡 Insights", "🤖 AI Analyst"])

with tabs[0]:
    st.subheader("Dataset Preview")
    if show_preview:
        st.dataframe(df.head(100), use_container_width=True, height=400)
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(df[c].isna().mean()*100, 2) for c in df.columns],
        "Unique": [int(df[c].nunique()) for c in df.columns]
    })
    st.subheader("Column Profile")
    st.dataframe(profile, use_container_width=True)

with tabs[1]:
    st.subheader("Descriptive Statistics")
    if numeric:
        stats = df[numeric].describe().T
        stats["median"] = df[numeric].median()
        stats["skewness"] = df[numeric].skew()
        st.dataframe(stats.round(3), use_container_width=True)
    else:
        st.info("No numerical columns detected.")
    if categorical:
        cat = st.selectbox("Categorical column", categorical)
        counts = df[cat].value_counts(dropna=False).head(20).rename_axis(cat).reset_index(name="Count")
        st.dataframe(counts, use_container_width=True)

with tabs[2]:
    st.subheader("Interactive Visualization")
    chart = st.selectbox("Chart type", ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart", "Correlation Matrix"])
    if chart == "Histogram" and numeric:
        col = st.selectbox("Column", numeric)
        st.plotly_chart(px.histogram(df, x=col, title=f"Distribution of {col}"), use_container_width=True)
    elif chart == "Box Plot" and numeric:
        col = st.selectbox("Column", numeric)
        st.plotly_chart(px.box(df, y=col, title=f"Box Plot: {col}"), use_container_width=True)
    elif chart == "Scatter Plot" and len(numeric) >= 2:
        x = st.selectbox("X axis", numeric)
        y = st.selectbox("Y axis", numeric, index=min(1, len(numeric)-1))
        st.plotly_chart(px.scatter(df, x=x, y=y, title=f"{x} vs {y}"), use_container_width=True)
    elif chart == "Bar Chart" and categorical:
        col = st.selectbox("Category", categorical)
        counts = df[col].value_counts().head(20).reset_index()
        counts.columns = [col, "Count"]
        st.plotly_chart(px.bar(counts, x=col, y="Count", title=f"Top Categories: {col}"), use_container_width=True)
    elif chart == "Line Chart" and numeric:
        x = st.selectbox("X axis", df.columns)
        y = st.selectbox("Y axis", numeric)
        st.plotly_chart(px.line(df, x=x, y=y, title=f"{y} over {x}"), use_container_width=True)
    elif chart == "Correlation Matrix" and len(numeric) >= 2:
        st.plotly_chart(px.imshow(df[numeric].corr(), text_auto=".2f", aspect="auto", title="Correlation Matrix"), use_container_width=True)
    elif chart in ["Histogram", "Box Plot", "Line Chart"] and not numeric:
        st.warning("No numerical columns detected.")
    elif chart == "Scatter Plot" and len(numeric) < 2:
        st.warning("At least two numerical columns are required.")
    elif chart == "Bar Chart" and not categorical:
        st.warning("No categorical columns detected.")
    elif chart == "Correlation Matrix" and len(numeric) < 2:
        st.warning("At least two numerical columns are required.")

with tabs[3]:
    st.subheader("Data Quality Report")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        st.dataframe(pd.DataFrame({
            "Column": missing.index,
            "Missing": missing.values,
            "Missing %": [round(df[c].isna().mean()*100, 2) for c in missing.index]
        }), use_container_width=True)
    else:
        st.success("No missing values detected.")
    dup = int(df.duplicated().sum())
    st.warning(f"{dup:,} duplicate rows detected." if dup else "No duplicate rows detected.")
    st.subheader("Outlier Detection")
    od = outliers(df, numeric)
    if not od.empty:
        st.dataframe(od, use_container_width=True)
    else:
        st.info("Not enough numerical data for outlier analysis.")

with tabs[4]:
    st.subheader("💡 Automatic Insights")
    for item in insights(df):
        st.info(item)
    if len(numeric) >= 2:
        corr = df[numeric].corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                v = corr.iloc[i, j]
                if not pd.isna(v):
                    pairs.append({"Variable 1": corr.columns[i], "Variable 2": corr.columns[j], "Absolute Correlation": round(v, 3)})
        if pairs:
            st.subheader("Strongest Numerical Relationships")
            st.dataframe(pd.DataFrame(pairs).sort_values("Absolute Correlation", ascending=False).head(10), use_container_width=True)

with tabs[5]:
    st.subheader("🤖 Ask the AI Data Analyst")
    question = st.text_area("Ask a question", placeholder="What are the most important findings in this dataset?")
    if st.button("🚀 Analyze with AI", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("AI is analyzing the dataset..."):
                st.write(ask_ai(df, question))

st.sidebar.divider()
st.sidebar.subheader("📥 Export")
st.sidebar.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    "analyzed_dataset.csv",
    "text/csv"
)

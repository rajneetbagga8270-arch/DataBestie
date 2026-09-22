import os
import time
import sqlite3
import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dotenv import load_dotenv
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DataBestie",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# GEMINI SETUP
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash"
]


def call_gemini(prompt, retries=2):

    if client is None:
        return "Gemini API key is not configured."

    last_error = None

    for model in GEMINI_MODELS:

        for attempt in range(retries):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response and response.text:
                    return response.text

            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                if (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "429" in error_text
                    or "resource exhausted" in error_text
                ):
                    time.sleep(3)
                    continue

                break

    return f"Gemini could not complete the request right now.\n\nError: {last_error}"


# =========================================================
# SESSION STATE
# =========================================================

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

if "last_sql_result" not in st.session_state:
    st.session_state.last_sql_result = None

if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""

if "auto_analysis" not in st.session_state:
    st.session_state.auto_analysis = None

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None


# =========================================================
# HEADER
# =========================================================

st.title("📊 DataBestie")

st.subheader(
    "Intelligent data analysis. Without the spreadsheet drama."
)

st.write(
    "Upload any structured dataset and DataBestie automatically "
    "profiles the data, detects column types, checks data quality, "
    "generates visualizations, runs SQL queries and uses AI to "
    "explain your results."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"]
)


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(file):

    try:

        if file.name.lower().endswith(".csv"):

            df = pd.read_csv(file)

        else:

            df = pd.read_excel(file)

        return df

    except Exception as e:

        st.error(f"Could not load the dataset: {e}")
        return None


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_dataframe(df):

    df = df.copy()

    # Remove completely empty columns
    empty_columns = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    if empty_columns:
        df = df.drop(columns=empty_columns)

    for column in df.columns:

        # Clean column names
        df[column] = df[column].replace(
            r"^\s+|\s+$",
            "",
            regex=True
        ) if df[column].dtype == "object" else df[column]

        column_name = str(column).lower()

        # Avoid converting IDs/codes
        if "id" in column_name or "code" in column_name:
            continue

        # Try numeric conversion for object columns
        if df[column].dtype == "object":

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.80:

                df[column] = converted

        # Try date conversion
        if any(
            keyword in column_name
            for keyword in ["date", "time", "timestamp"]
        ):

            converted_date = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            valid_ratio = converted_date.notna().mean()

            if valid_ratio >= 0.70:

                df[column] = converted_date

    return df


# =========================================================
# COLUMN TYPE DETECTION
# =========================================================

def detect_column_types(df):

    numerical = []
    categorical = []
    datetime_cols = []
    id_columns = []
    text_columns = []
    boolean_columns = []
    empty_columns = []

    for column in df.columns:

        series = df[column]

        if series.dropna().empty:

            empty_columns.append(column)

            continue

        if pd.api.types.is_bool_dtype(series):

            boolean_columns.append(column)

            continue

        if pd.api.types.is_datetime64_any_dtype(series):

            datetime_cols.append(column)

            continue

        unique_count = series.nunique(dropna=True)
        row_count = len(series)

        column_name = str(column).lower()

        # ID detection
        if (
            ("id" in column_name or "code" in column_name)
            and unique_count >= max(5, row_count * 0.5)
        ):

            id_columns.append(column)

            continue

        # Numerical
        if pd.api.types.is_numeric_dtype(series):

            numerical.append(column)

            continue

        # Categorical
        if (
            series.dtype == "object"
            and unique_count <= max(20, row_count * 0.10)
        ):

            categorical.append(column)

            continue

        # Everything else
        text_columns.append(column)

    return {
        "Numerical": numerical,
        "Categorical": categorical,
        "Date/Time": datetime_cols,
        "ID": id_columns,
        "Text": text_columns,
        "Boolean": boolean_columns,
        "Empty": empty_columns
    }


# =========================================================
# DATA PROFILER
# =========================================================

def create_profile(df, column_types):

    profile = []

    for column in df.columns:

        series = df[column]

        if column in column_types["Numerical"]:
            detected_type = "Numerical"

        elif column in column_types["Categorical"]:
            detected_type = "Categorical"

        elif column in column_types["Date/Time"]:
            detected_type = "Date/Time"

        elif column in column_types["ID"]:
            detected_type = "ID"

        elif column in column_types["Text"]:
            detected_type = "Text"

        elif column in column_types["Boolean"]:
            detected_type = "Boolean"

        else:
            detected_type = "Empty"

        profile.append({
            "Column": column,
            "Type": detected_type,
            "Data Type": str(series.dtype),
            "Unique Values": int(series.nunique(dropna=True)),
            "Missing Values": int(series.isna().sum()),
            "Missing %": round(
                series.isna().mean() * 100,
                2
            )
        })

    return pd.DataFrame(profile)


# =========================================================
# AUTOMATIC ANALYSIS
# =========================================================

def automatic_analysis(df, column_types):

    result = {}

    result["rows"] = len(df)
    result["columns"] = len(df.columns)
    result["missing_cells"] = int(df.isna().sum().sum())
    result["duplicate_rows"] = int(df.duplicated().sum())

    numerical = column_types["Numerical"]
    categorical = column_types["Categorical"]
    datetime_cols = column_types["Date/Time"]

    result["numerical_stats"] = {}

    for column in numerical[:10]:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        result["numerical_stats"][column] = {
            "mean": round(float(series.mean()), 3)
            if series.notna().any() else None,

            "median": round(float(series.median()), 3)
            if series.notna().any() else None,

            "min": round(float(series.min()), 3)
            if series.notna().any() else None,

            "max": round(float(series.max()), 3)
            if series.notna().any() else None
        }

    result["categorical_summary"] = {}

    for column in categorical[:10]:

        counts = (
            df[column]
            .value_counts(dropna=True)
            .head(5)
            .to_dict()
        )

        result["categorical_summary"][column] = counts

    result["date_ranges"] = {}

    for column in datetime_cols:

        valid_dates = df[column].dropna()

        if not valid_dates.empty:

            result["date_ranges"][column] = {
                "min": str(valid_dates.min()),
                "max": str(valid_dates.max())
            }

    result["correlations"] = {}

    if len(numerical) >= 2:

        correlation_matrix = df[numerical].corr()

        pairs = []

        for i in range(len(numerical)):

            for j in range(i + 1, len(numerical)):

                x = numerical[i]
                y = numerical[j]

                value = correlation_matrix.loc[x, y]

                if pd.notna(value):

                    pairs.append(
                        (x, y, round(float(value), 3))
                    )

        pairs.sort(
            key=lambda x: abs(x[2]),
            reverse=True
        )

        result["correlations"] = pairs[:5]

    return result


# =========================================================
# REPORT CONTEXT
# =========================================================

def build_report_context(df, column_types, auto_result):

    context = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_types": column_types,
        "auto_analysis": auto_result
    }

    return context


# =========================================================
# DATA CLEANING
# =========================================================

def clean_dataset(df):

    cleaned = df.copy()

    before_rows = len(cleaned)
    before_missing = int(cleaned.isna().sum().sum())
    before_duplicates = int(cleaned.duplicated().sum())

    # Strip strings
    for column in cleaned.select_dtypes(
        include=["object", "string"]
    ).columns:

        cleaned[column] = cleaned[column].astype("string").str.strip()

    # Fill numerical missing values with median
    numerical_columns = cleaned.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:

        if cleaned[column].isna().any():

            median_value = cleaned[column].median()

            cleaned[column] = cleaned[column].fillna(
                median_value
            )

    # Fill categorical/text missing values with mode
    object_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in object_columns:

        if cleaned[column].isna().any():

            mode_values = cleaned[column].mode()

            if not mode_values.empty:

                cleaned[column] = cleaned[column].fillna(
                    mode_values.iloc[0]
                )

    # Remove duplicates
    cleaned = cleaned.drop_duplicates()

    after_rows = len(cleaned)
    after_missing = int(cleaned.isna().sum().sum())
    after_duplicates = int(cleaned.duplicated().sum())

    report = {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "before_duplicates": before_duplicates,
        "after_duplicates": after_duplicates
    }

    return cleaned, report


# =========================================================
# SQL SAFETY
# =========================================================

def is_safe_sql(query):

    query = query.strip().lower()

    if not (
        query.startswith("select")
        or query.startswith("with")
    ):

        return False

    forbidden = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "create ",
        "replace ",
        "truncate ",
        "attach ",
        "detach "
    ]

    for word in forbidden:

        if word in query:

            return False

    return True


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel dataset from the sidebar to begin."
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

raw_df = load_dataset(uploaded_file)

if raw_df is None:

    st.stop()


analysis_df = prepare_dataframe(raw_df)

st.session_state.analysis_df = analysis_df


# =========================================================
# COLUMN PROFILING
# =========================================================

column_types = detect_column_types(analysis_df)

profile_df = create_profile(
    analysis_df,
    column_types
)


# =========================================================
# SIDEBAR INFO
# =========================================================

st.sidebar.success("Dataset loaded")

st.sidebar.metric(
    "Rows",
    len(analysis_df)
)

st.sidebar.metric(
    "Columns",
    len(analysis_df.columns)
)

st.sidebar.write("### Detected columns")

for key, values in column_types.items():

    if values:

        st.sidebar.write(
            f"**{key}:** {len(values)}"
        )


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("🔎 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        len(analysis_df)
    )

with col2:
    st.metric(
        "Columns",
        len(analysis_df.columns)
    )

with col3:
    st.metric(
        "Missing Cells",
        int(analysis_df.isna().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(analysis_df.duplicated().sum())
    )


st.subheader("Dataset Preview")

st.dataframe(
    analysis_df.head(20),
    use_container_width=True
)


# =========================================================
# COLUMN PROFILE
# =========================================================

st.header("🧠 Dataset Profiler")

st.dataframe(
    profile_df,
    use_container_width=True
)


# =========================================================
# DATA HEALTH
# =========================================================

st.header("🩺 Data Health")

health_col1, health_col2 = st.columns(2)

with health_col1:

    st.subheader("Missing Values")

    missing_df = (
        analysis_df.isna()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if missing_df.empty:

        st.success(
            "No missing values detected."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True
        )


with health_col2:

    st.subheader("Duplicate Rows")

    duplicate_count = int(
        analysis_df.duplicated().sum()
    )

    if duplicate_count == 0:

        st.success(
            "No duplicate rows detected."
        )

    else:

        st.warning(
            f"{duplicate_count} duplicate rows detected."
        )


# =========================================================
# STATISTICS
# =========================================================

st.header("📈 Statistical Analysis")

numerical_columns = column_types["Numerical"]
categorical_columns = column_types["Categorical"]
datetime_columns = column_types["Date/Time"]


if numerical_columns:

    st.subheader("Numerical Statistics")

    statistics_df = analysis_df[
        numerical_columns
    ].describe().T

    st.dataframe(
        statistics_df,
        use_container_width=True
    )

    st.subheader("Numerical KPIs")

    kpi_columns = numerical_columns[:4]

    kpi_list = st.columns(
        max(1, len(kpi_columns))
    )

    for i, column in enumerate(kpi_columns):

        with kpi_list[i]:

            value = analysis_df[column].mean()

            st.metric(
                f"Average {column}",
                f"{value:,.2f}"
                if pd.notna(value)
                else "N/A"
            )


if categorical_columns:

    st.subheader("Categorical Analysis")

    selected_category = st.selectbox(
        "Select a categorical column",
        categorical_columns,
        key="categorical_analysis_select"
    )

    category_counts = (
        analysis_df[selected_category]
        .value_counts()
        .head(15)
        .reset_index()
    )

    category_counts.columns = [
        selected_category,
        "Count"
    ]

    st.dataframe(
        category_counts,
        use_container_width=True
    )


# =========================================================
# VISUALIZATIONS
# =========================================================

st.header("📊 Visualizations")

# ---------------------------------------------------------
# NUMERICAL DISTRIBUTION
# ---------------------------------------------------------

if numerical_columns:

    st.subheader("Numerical Distribution")

    selected_num = st.selectbox(
        "Select numerical column",
        numerical_columns,
        key="distribution_column"
    )

    fig = px.histogram(
        analysis_df,
        x=selected_num,
        title=f"Distribution of {selected_num}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"distribution_{selected_num}"
    )


# ---------------------------------------------------------
# CATEGORICAL DISTRIBUTION
# ---------------------------------------------------------

if categorical_columns:

    st.subheader("Categorical Distribution")

    selected_cat = st.selectbox(
        "Select categorical column",
        categorical_columns,
        key="categorical_distribution_column"
    )

    cat_data = (
        analysis_df[selected_cat]
        .value_counts()
        .head(15)
        .reset_index()
    )

    cat_data.columns = [
        selected_cat,
        "Count"
    ]

    fig = px.bar(
        cat_data,
        x=selected_cat,
        y="Count",
        title=f"Distribution of {selected_cat}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"categorical_distribution_{selected_cat}"
    )


# ---------------------------------------------------------
# NUMERICAL VS NUMERICAL
# ---------------------------------------------------------

if len(numerical_columns) >= 2:

    st.subheader("Numerical vs Numerical")

    x_column = st.selectbox(
        "X-axis",
        numerical_columns,
        key="numerical_x_axis"
    )

    y_options = [
        column
        for column in numerical_columns
        if column != x_column
    ]

    if y_options:

        y_column = st.selectbox(
            "Y-axis",
            y_options,
            key="numerical_y_axis"
        )

        scatter_df = analysis_df[
            [x_column, y_column]
        ].dropna()

        fig = px.scatter(
            scatter_df,
            x=x_column,
            y=y_column,
            title=f"{x_column} vs {y_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"scatter_{x_column}_{y_column}"
        )


# ---------------------------------------------------------
# CATEGORICAL VS NUMERICAL
# ---------------------------------------------------------

if categorical_columns and numerical_columns:

    st.subheader("Categorical vs Numerical")

    category_column = st.selectbox(
        "Category",
        categorical_columns,
        key="category_vs_num_category"
    )

    value_column = st.selectbox(
        "Numerical value",
        numerical_columns,
        key="category_vs_num_value"
    )

    grouped_data = (
        analysis_df
        .groupby(category_column, dropna=False)[value_column]
        .mean()
        .reset_index()
        .sort_values(
            value_column,
            ascending=False
        )
        .head(20)
    )

    fig = px.bar(
        grouped_data,
        x=category_column,
        y=value_column,
        title=f"Average {value_column} by {category_column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"category_numeric_{category_column}_{value_column}"
    )


# ---------------------------------------------------------
# DATE VS NUMERICAL
# ---------------------------------------------------------

if datetime_columns and numerical_columns:

    st.subheader("Date vs Numerical")

    date_column = st.selectbox(
        "Date column",
        datetime_columns,
        key="date_analysis_column"
    )

    date_value_column = st.selectbox(
        "Numerical value",
        numerical_columns,
        key="date_analysis_value"
    )

    trend_df = analysis_df[
        [date_column, date_value_column]
    ].dropna()

    if not trend_df.empty:

        trend_df = (
            trend_df
            .sort_values(date_column)
            .groupby(date_column)[date_value_column]
            .mean()
            .reset_index()
        )

        fig = px.line(
            trend_df,
            x=date_column,
            y=date_value_column,
            title=f"{date_value_column} over time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"date_numeric_{date_column}_{date_value_column}"
        )


# ---------------------------------------------------------
# CORRELATION
# ---------------------------------------------------------

if len(numerical_columns) >= 2:

    st.subheader("Correlation Matrix")

    correlation_df = analysis_df[
        numerical_columns
    ].corr()

    fig = px.imshow(
        correlation_df,
        text_auto=True,
        aspect="auto",
        title="Numerical Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="correlation_matrix_chart"
    )


# =========================================================
# AUTO ANALYSIS
# =========================================================

st.header("✨ Auto Analysis")

if st.button(
    "🚀 Run Auto Analysis",
    key="run_auto_analysis"
):

    with st.spinner(
        "Analyzing your dataset..."
    ):

        auto_result = automatic_analysis(
            analysis_df,
            column_types
        )

        st.session_state.auto_analysis = auto_result


if st.session_state.auto_analysis:

    result = st.session_state.auto_analysis

    st.subheader("Dataset Summary")

    st.write(
        f"Your dataset contains **{result['rows']:,} rows** "
        f"and **{result['columns']:,} columns**."
    )

    st.write(
        f"It contains **{result['missing_cells']:,} missing cells** "
        f"and **{result['duplicate_rows']:,} duplicate rows**."
    )

    if result["numerical_stats"]:

        st.subheader("Numerical Findings")

        for column, values in result[
            "numerical_stats"
        ].items():

            st.write(
                f"**{column}** — "
                f"Mean: {values['mean']}, "
                f"Median: {values['median']}, "
                f"Min: {values['min']}, "
                f"Max: {values['max']}"
            )

    if result["categorical_summary"]:

        st.subheader("Categorical Findings")

        for column, values in result[
            "categorical_summary"
        ].items():

            st.write(
                f"**{column}**: {values}"
            )

    if result["date_ranges"]:

        st.subheader("Date Ranges")

        for column, values in result[
            "date_ranges"
        ].items():

            st.write(
                f"**{column}**: "
                f"{values['min']} → {values['max']}"
            )

    if result["correlations"]:

        st.subheader("Strongest Numerical Relationships")

        for x, y, correlation in result[
            "correlations"
        ]:

            st.write(
                f"**{x} ↔ {y}: {correlation}**"
            )

    st.subheader("🔍 Recommended Investigations")

    recommendations = []

    if result["missing_cells"] > 0:

        recommendations.append(
            "Investigate columns containing missing values."
        )

    if result["duplicate_rows"] > 0:

        recommendations.append(
            "Investigate duplicate records."
        )

    if len(numerical_columns) >= 2:

        recommendations.append(
            "Explore relationships between numerical variables."
        )

    if categorical_columns:

        recommendations.append(
            "Compare numerical metrics across categories."
        )

    if datetime_columns:

        recommendations.append(
            "Investigate trends over time."
        )

    if not recommendations:

        recommendations.append(
            "Dataset appears suitable for deeper exploratory analysis."
        )

    for recommendation in recommendations:

        st.write(
            f"• {recommendation}"
        )


# =========================================================
# AI DATA ANALYST REPORT
# =========================================================

st.header("🧠 AI Data Analyst Report")

if st.button(
    "🧠 Generate Full AI Report",
    key="generate_ai_report"
):

    if client is None:

        st.error(
            "Gemini API key is not configured."
        )

    elif st.session_state.auto_analysis is None:

        st.warning(
            "Run Auto Analysis first."
        )

    else:

        report_context = build_report_context(
            analysis_df,
            column_types,
            st.session_state.auto_analysis
        )

        prompt = f"""
You are an experienced Data Analyst.

Analyze the following VERIFIED dataset analysis.

DATASET PROFILE:
{report_context}

Create a professional but beginner-friendly Data Analyst report.

Use exactly these sections:

1. Executive Summary
2. Data Quality Assessment
3. Key Findings
4. Important Relationships
5. Recommended Next Analyses
6. Practical Takeaways

IMPORTANT RULES:

- Use only facts contained in the supplied analysis.
- Do not invent statistics.
- Do not create numbers that are not present.
- Do not claim causation unless it is explicitly supported.
- Clearly mention when something cannot be determined.
- Focus on useful analytical observations.
"""

        with st.spinner(
            "Generating AI Data Analyst Report..."
        ):

            report = call_gemini(prompt)

            st.session_state.ai_report = report


if st.session_state.ai_report:

    st.markdown(
        st.session_state.ai_report
    )


# =========================================================
# DATA CLEANING ENGINE
# =========================================================

st.header("🧹 Data Cleaning Engine")

if st.button(
    "🧹 Clean Dataset",
    key="clean_dataset"
):

    with st.spinner(
        "Cleaning dataset..."
    ):

        cleaned_df, cleaning_report = clean_dataset(
            analysis_df
        )

        st.session_state.cleaned_df = cleaned_df

        st.success(
            "Dataset cleaning completed."
        )

        st.subheader("Cleaning Report")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.write("Rows")

            st.write(
                f"{cleaning_report['before_rows']} → "
                f"{cleaning_report['after_rows']}"
            )

        with c2:

            st.write("Missing Values")

            st.write(
                f"{cleaning_report['before_missing']} → "
                f"{cleaning_report['after_missing']}"
            )

        with c3:

            st.write("Duplicates")

            st.write(
                f"{cleaning_report['before_duplicates']} → "
                f"{cleaning_report['after_duplicates']}"
            )

        st.subheader("Cleaned Preview")

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True
        )

        cleaned_csv = cleaned_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Cleaned CSV",
            cleaned_csv,
            file_name="DataBestie_Cleaned.csv",
            mime="text/csv",
            key="download_cleaned_csv"
        )


# =========================================================
# SQL EXPLORER
# =========================================================

st.header("🗄️ SQL Data Explorer")

sql_connection = sqlite3.connect(
    ":memory:",
    check_same_thread=False
)

sql_df = analysis_df.copy()

for column in sql_df.columns:

    if pd.api.types.is_datetime64_any_dtype(
        sql_df[column]
    ):

        sql_df[column] = sql_df[column].astype(str)


sql_df.to_sql(
    "data",
    sql_connection,
    index=False,
    if_exists="replace"
)


st.subheader("Table Schema")

schema_df = pd.read_sql_query(
    "PRAGMA table_info(data);",
    sql_connection
)

st.dataframe(
    schema_df,
    use_container_width=True
)


st.subheader("Example Queries")

st.code(
    """SELECT * FROM data LIMIT 10;""",
    language="sql"
)

st.code(
    """SELECT COUNT(*) AS total_rows
FROM data;""",
    language="sql"
)


# =========================================================
# NATURAL LANGUAGE TO SQL
# =========================================================

st.subheader("💬 Ask Your Data")

natural_question = st.text_input(
    "Ask a question about your dataset",
    placeholder="Example: Which category has the highest average value?",
    key="natural_language_question"
)


if st.button(
    "✨ Generate SQL",
    key="generate_sql_button"
):

    if not natural_question.strip():

        st.warning(
            "Please enter a question."
        )

    elif client is None:

        st.error(
            "Gemini API key is not configured."
        )

    else:

        schema_text = schema_df.to_string(
            index=False
        )

        prompt = f"""
You are a SQL expert.

Convert the user's question into ONE SQLite SQL query.

DATABASE TABLE:
data

SCHEMA:
{schema_text}

USER QUESTION:
{natural_question}

RULES:

- Return only SQL.
- The query must be SELECT or WITH.
- Do not modify the database.
- Do not use INSERT.
- Do not use UPDATE.
- Do not use DELETE.
- Do not use DROP.
- Do not use ALTER.
- Do not use CREATE.
- Do not use ATTACH.
"""

        with st.spinner(
            "Generating SQL..."
        ):

            generated_sql = call_gemini(
                prompt
            )

        generated_sql = generated_sql.strip()

        generated_sql = re.sub(
            r"```sql",
            "",
            generated_sql,
            flags=re.IGNORECASE
        )

        generated_sql = generated_sql.replace(
            "```",
            ""
        ).strip()

        st.session_state.generated_sql = generated_sql


if st.session_state.generated_sql:

    st.subheader("Generated SQL")

    st.code(
        st.session_state.generated_sql,
        language="sql"
    )

    if st.button(
        "▶️ Run Generated SQL",
        key="run_generated_sql"
    ):

        query = st.session_state.generated_sql

        if not is_safe_sql(query):

            st.error(
                "Unsafe SQL query blocked."
            )

        else:

            try:

                result_df = pd.read_sql_query(
                    query,
                    sql_connection
                )

                st.session_state.last_sql_result = (
                    result_df
                )

                st.success(
                    "SQL query executed successfully."
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                result_csv = result_df.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download SQL Result",
                    result_csv,
                    file_name="DataBestie_SQL_Result.csv",
                    mime="text/csv",
                    key="download_generated_sql_result"
                )

            except Exception as e:

                st.error(
                    f"SQL Error: {e}"
                )


# =========================================================
# MANUAL SQL
# =========================================================

st.subheader("⌨️ Manual SQL Explorer")

manual_sql = st.text_area(
    "Write your own SQL query",
    value="SELECT * FROM data LIMIT 10;",
    height=120,
    key="manual_sql_query"
)


if st.button(
    "▶️ Run SQL",
    key="run_manual_sql"
):

    if not is_safe_sql(manual_sql):

        st.error(
            "Only safe SELECT/WITH queries are allowed."
        )

    else:

        try:

            manual_result = pd.read_sql_query(
                manual_sql,
                sql_connection
            )

            st.session_state.last_sql_result = (
                manual_result
            )

            st.success(
                "SQL query executed successfully."
            )

            st.dataframe(
                manual_result,
                use_container_width=True
            )

            manual_csv = manual_result.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download SQL Result",
                manual_csv,
                file_name="DataBestie_SQL_Result.csv",
                mime="text/csv",
                key="download_manual_sql_result"
            )

        except Exception as e:

            st.error(
                f"SQL Error: {e}"
            )


# =========================================================
# AI RESULT EXPLANATION
# =========================================================

st.header("🤖 AI Result Explanation")

if st.session_state.last_sql_result is None:

    st.info(
        "Run a SQL query first, then DataBestie can explain the result."
    )

else:

    if st.button(
        "🧠 Explain These Results",
        key="explain_sql_result"
    ):

        result_for_ai = (
            st.session_state.last_sql_result
            .head(100)
            .to_string(index=False)
        )

        prompt = f"""
You are a Data Analyst.

Explain the following VERIFIED SQL query result.

RESULT:
{result_for_ai}

Give:

1. Most important observation
2. Highest or lowest value if applicable
3. Important patterns or differences
4. One practical next question or action

IMPORTANT:

- Use only the values present in the result.
- Do not invent numbers.
- Do not calculate statistics that are not shown.
- Do not claim causation.
- If the result is too limited to conclude something, say so.
"""

        with st.spinner(
            "Analyzing the SQL result..."
        ):

            explanation = call_gemini(
                prompt
            )

        st.markdown(
            explanation
        )


# =========================================================
# ASK DATABESTIE
# =========================================================

st.header("💬 Ask DataBestie")

general_question = st.text_area(
    "Your question",
    placeholder="Example: What problems should I check in this dataset?",
    key="general_question"
)


if st.button(
    "✨ Ask DataBestie",
    key="ask_databestie"
):

    if not general_question.strip():

        st.warning(
            "Please enter a question."
        )

    elif client is None:

        st.error(
            "Gemini API key is not configured."
        )

    else:

        profile_context = profile_df.to_string(
            index=False
        )

        sample_context = analysis_df.head(
            10
        ).to_string(
            index=False
        )

        prompt = f"""
You are DataBestie, an AI Data Analyst assistant.

DATASET PROFILE:
{profile_context}

SAMPLE DATA:
{sample_context}

USER QUESTION:
{general_question}

Answer clearly and practically.

IMPORTANT:

- Do not invent facts.
- Do not invent statistics.
- Use the provided dataset information.
- If something cannot be determined, say so.
- Explain technical concepts simply.
"""

        with st.spinner(
            "DataBestie is thinking..."
        ):

            answer = call_gemini(
                prompt
            )

        st.markdown(
            answer
        )


# =========================================================
# DOWNLOAD PROFILE
# =========================================================

st.header("📥 Export")

profile_csv = profile_df.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Dataset Profile",
    profile_csv,
    file_name="DataBestie_Profile.csv",
    mime="text/csv",
    key="download_profile"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DataBestie • Intelligent data analysis without the spreadsheet drama."
)
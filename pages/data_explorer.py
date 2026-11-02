
import streamlit as st
import pandas as pd

from src.data_loader import (
    load_dataset,
    validate_dataset,
    normalize_column_names,
    get_dataset_info,
    get_column_information,
    get_dataset_preview,
    get_numerical_summary,
    get_categorical_summary,
    get_missing_value_report,
    get_duplicate_count,
    get_dataset_quality
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Dataset Explorer | AI Studio",
    page_icon="🗂️",
    layout="wide"
)


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}

[data-testid="stMetric"] {
    background: #151f32;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #26354d;
}

[data-testid="stSidebar"] {
    background-color: #111a2d;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #26354d;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.title("🗂️ Dataset Explorer")

st.write(
    "Explore, inspect and manage the dataset powering "
    "your Student Performance AI Studio."
)

st.divider()


# =====================================================
# SESSION STATE
# =====================================================

if "active_dataset" not in st.session_state:
    st.session_state["active_dataset"] = None

if "dataset_source" not in st.session_state:
    st.session_state["dataset_source"] = "Default Dataset"


# =====================================================
# DATASET UPLOAD
# =====================================================

st.subheader("📤 Upload Student Dataset")

st.caption(
    "Upload a CSV file containing student academic information. "
    "The dataset must include the final_score target column."
)

uploaded_file = st.file_uploader(
    "Choose CSV Dataset",
    type=["csv"],
    help="Upload your student performance dataset."
)

upload_col, restore_col = st.columns(2)


with upload_col:

    if st.button(
        "Load Uploaded Dataset",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    ):

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            uploaded_df = normalize_column_names(
                uploaded_df
            )

            validate_dataset(uploaded_df)

            st.session_state["active_dataset"] = (
                uploaded_df
            )

            st.session_state["dataset_source"] = (
                uploaded_file.name
            )

            st.success(
                f"Successfully loaded {uploaded_file.name}"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Dataset upload failed: {error}"
            )


with restore_col:

    if st.button(
        "Restore Default Dataset",
        use_container_width=True
    ):

        try:

            default_df = load_dataset()

            validate_dataset(default_df)

            st.session_state["active_dataset"] = (
                default_df
            )

            st.session_state["dataset_source"] = (
                "Default Dataset"
            )

            st.success(
                "Default dataset restored successfully."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to restore dataset: {error}"
            )


st.divider()


# =====================================================
# ACTIVE DATASET
# =====================================================

try:

    active_df = st.session_state.get(
        "active_dataset"
    )

    if isinstance(active_df, pd.DataFrame):

        df = active_df.copy()

    else:

        df = load_dataset()

        validate_dataset(df)

        st.session_state["active_dataset"] = df

        st.session_state["dataset_source"] = (
            "Default Dataset"
        )

except Exception as error:

    st.error(
        f"Dataset unavailable: {error}"
    )

    st.info(
        "Upload a valid student performance CSV to continue."
    )

    st.stop()


dataset_source = st.session_state.get(
    "dataset_source",
    "Default Dataset"
)

st.markdown(
    f"**Active Dataset:** `{dataset_source}`"
)


# =====================================================
# DATASET OVERVIEW
# =====================================================

st.subheader("📊 Dataset Overview")

info = get_dataset_info(df)

quality = get_dataset_quality(df)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Total Records",
        f"{info['total_rows']:,}"
    )

with metric2:
    st.metric(
        "Total Features",
        info["total_columns"]
    )

with metric3:
    st.metric(
        "Missing Cells",
        f"{info['missing_cells']:,}"
    )

with metric4:
    st.metric(
        "Data Completeness",
        f"{quality['completeness_percentage']:.1f}%"
    )


st.divider()


# =====================================================
# DATASET PREVIEW
# =====================================================

st.subheader("👀 Dataset Preview")

max_rows = min(100, len(df))

preview_rows = st.slider(
    "Records to display",
    min_value=1,
    max_value=max_rows,
    value=min(10, max_rows)
)

st.dataframe(
    get_dataset_preview(
        df,
        rows=preview_rows
    ),
    use_container_width=True,
    hide_index=True
)


st.divider()


# =====================================================
# COLUMN INFORMATION
# =====================================================

st.subheader("🏷️ Column Information")

column_info = get_column_information(df)

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =====================================================
# MISSING VALUE ANALYSIS
# =====================================================

st.subheader("🔍 Missing Value Analysis")

missing_report = get_missing_value_report(df)

missing_columns = missing_report[
    missing_report["Missing Values"] > 0
]

if missing_columns.empty:

    st.success(
        "No missing values detected."
    )

else:

    st.warning(
        f"{len(missing_columns)} columns contain missing values."
    )

    st.dataframe(
        missing_columns,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# DUPLICATE ANALYSIS
# =====================================================

st.subheader("♻️ Duplicate Record Analysis")

duplicate_count = get_duplicate_count(df)

if duplicate_count > 0:

    st.warning(
        f"{duplicate_count} duplicate records detected."
    )

    st.caption(
        "Duplicate records are handled during preprocessing."
    )

else:

    st.success(
        "No duplicate records detected."
    )


st.divider()


# =====================================================
# NUMERICAL SUMMARY
# =====================================================

st.subheader("🔢 Numerical Feature Summary")

numerical_summary = get_numerical_summary(df)

if numerical_summary.empty:

    st.info(
        "No numerical features found."
    )

else:

    st.dataframe(
        numerical_summary,
        use_container_width=True
    )


st.divider()


# =====================================================
# CATEGORICAL SUMMARY
# =====================================================

st.subheader("🔠 Categorical Feature Summary")

categorical_summary = get_categorical_summary(df)

if categorical_summary.empty:

    st.info(
        "No categorical features found."
    )

else:

    st.dataframe(
        categorical_summary,
        use_container_width=True
    )


st.divider()


# =====================================================
# DATASET QUALITY
# =====================================================

st.subheader("🧪 Dataset Quality Report")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:

    st.metric(
        "Completeness",
        f"{quality['completeness_percentage']}%"
    )

with quality_col2:

    st.metric(
        "Duplicate Records",
        quality["duplicate_rows"]
    )

with quality_col3:

    st.metric(
        "Total Features",
        quality["total_features"]
    )


# =====================================================
# DATASET DOWNLOAD
# =====================================================

st.divider()

st.subheader("⬇️ Export Dataset")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Active Dataset",
    data=csv_data,
    file_name="student_performance_dataset.csv",
    mime="text/csv",
    use_container_width=True
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Student Performance AI Studio | Dataset Management"
)

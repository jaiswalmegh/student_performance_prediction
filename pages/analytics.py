
import streamlit as st
import pandas as pd

from src.data_loader import load_dataset, validate_dataset
from src.visualizations import (
    score_distribution_chart,
    study_hours_chart,
    correlation_heatmap,
    attendance_performance_chart,
    performance_category_chart,
    get_performance_statistics
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Analytics Studio | AI Studio",
    page_icon="📊",
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

[data-testid="stSidebar"] {
    background-color: #111a2d;
}

[data-testid="stMetric"] {
    background-color: #151f32;
    padding: 17px;
    border-radius: 12px;
    border: 1px solid #26354d;
}

.analytics-card {
    background: #151f32;
    padding: 20px;
    border: 1px solid #26354d;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.title("📊 Analytics Studio")

st.write(
    "Explore academic trends, discover performance patterns "
    "and understand the factors associated with student outcomes."
)

st.divider()


# =====================================================
# LOAD ACTIVE DATASET
# =====================================================

@st.cache_data
def load_default_data():
    return load_dataset()


def get_active_dataset():

    active_df = st.session_state.get("active_dataset")

    if isinstance(active_df, pd.DataFrame):
        return active_df.copy()

    return load_default_data()


try:

    df = get_active_dataset()

    validate_dataset(df)

except Exception as error:

    st.error(f"Dataset loading failed: {error}")

    st.info(
        "Please upload a valid dataset from Dataset Explorer."
    )

    st.stop()


# =====================================================
# DATA VALIDATION
# =====================================================

if "final_score" not in df.columns:
    st.error("The final_score target column is missing.")
    st.stop()

df["final_score"] = pd.to_numeric(
    df["final_score"],
    errors="coerce"
)

df = df.replace(
    [float("inf"), float("-inf")],
    float("nan")
)

df = df.dropna(
    subset=["final_score"]
)

if df.empty:
    st.error("No valid student scores are available.")
    st.stop()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("🔎 Analytics Filters")

filtered_df = df.copy()

filter_columns = [
    "academic_support",
    "internet_access",
    "extracurricular_activities"
]

for column in filter_columns:

    if column in filtered_df.columns:

        options = sorted(
            filtered_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected = st.sidebar.multiselect(
            column.replace("_", " ").title(),
            options,
            default=options,
            key=f"analytics_filter_{column}"
        )

        if selected:
            filtered_df = filtered_df[
                filtered_df[column]
                .astype(str)
                .isin(selected)
            ]
        else:
            filtered_df = filtered_df.iloc[0:0]


if "study_hours_per_day" in filtered_df.columns:

    study_hours = pd.to_numeric(
        filtered_df["study_hours_per_day"],
        errors="coerce"
    ).dropna()

    if not study_hours.empty:

        minimum_hours = float(study_hours.min())
        maximum_hours = float(study_hours.max())

        if minimum_hours < maximum_hours:

            selected_hours = st.sidebar.slider(
                "Study Hours Per Day",
                min_value=minimum_hours,
                max_value=maximum_hours,
                value=(minimum_hours, maximum_hours)
            )

            filtered_df = filtered_df[
                pd.to_numeric(
                    filtered_df["study_hours_per_day"],
                    errors="coerce"
                ).between(
                    selected_hours[0],
                    selected_hours[1]
                )
            ]


st.sidebar.divider()

if st.sidebar.button(
    "Reset Filters",
    use_container_width=True
):

    for column in filter_columns:
        st.session_state.pop(
            f"analytics_filter_{column}",
            None
        )

    st.rerun()


# =====================================================
# FILTER VALIDATION
# =====================================================

if filtered_df.empty:

    st.warning(
        "No records match the selected filters. "
        "Please adjust your filters."
    )

    st.stop()


# =====================================================
# PERFORMANCE STATISTICS
# =====================================================

statistics = get_performance_statistics(
    filtered_df
)


# =====================================================
# KPI DASHBOARD
# =====================================================

st.subheader("📌 Academic Performance Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Students Analyzed",
        f"{statistics['total_students']:,}"
    )

with kpi2:

    st.metric(
        "Average Score",
        f"{statistics['average_score']:.2f}"
    )

with kpi3:

    st.metric(
        "Highest Score",
        f"{statistics['highest_score']:.2f}"
    )

with kpi4:

    st.metric(
        "Score Deviation",
        f"{statistics['standard_deviation']:.2f}"
    )


# =====================================================
# PERFORMANCE CATEGORIES
# =====================================================

st.divider()

st.subheader("🎯 Performance Distribution")

category1, category2, category3 = st.columns(3)

with category1:

    st.metric(
        "High Performers",
        statistics["high_performers"]
    )

with category2:

    st.metric(
        "Moderate Performers",
        statistics["moderate_performers"]
    )

with category3:

    st.metric(
        "Needs Support",
        statistics["students_needing_support"]
    )


chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.plotly_chart(
        performance_category_chart(filtered_df),
        use_container_width=True,
        key="performance_category_chart"
    )

with chart_col2:

    st.plotly_chart(
        score_distribution_chart(filtered_df),
        use_container_width=True,
        key="score_distribution_chart"
    )


# =====================================================
# STUDY HOURS ANALYSIS
# =====================================================

st.divider()

st.subheader("📚 Study Habits & Academic Performance")

if "study_hours_per_day" in filtered_df.columns:

    st.plotly_chart(
        study_hours_chart(filtered_df),
        use_container_width=True,
        key="study_hours_chart"
    )

    study_data = filtered_df[
        ["study_hours_per_day", "final_score"]
    ].copy()

    study_data["study_hours_per_day"] = pd.to_numeric(
        study_data["study_hours_per_day"],
        errors="coerce"
    )

    study_data["final_score"] = pd.to_numeric(
        study_data["final_score"],
        errors="coerce"
    )

    study_data = study_data.dropna()

    if not study_data.empty:

        study_correlation = study_data[
            "study_hours_per_day"
        ].corr(
            study_data["final_score"]
        )

        if pd.notna(study_correlation):

            st.info(
                f"Correlation between study hours and final score: "
                f"**{study_correlation:.3f}**"
            )

else:

    st.info(
        "Study hours information is not available in this dataset."
    )


# =====================================================
# ATTENDANCE ANALYSIS
# =====================================================

st.divider()

st.subheader("📅 Attendance & Performance Analysis")

if "attendance_percentage" in filtered_df.columns:

    st.plotly_chart(
        attendance_performance_chart(filtered_df),
        use_container_width=True,
        key="attendance_chart"
    )

else:

    st.info(
        "Attendance information is not available in this dataset."
    )


# =====================================================
# CORRELATION ANALYSIS
# =====================================================

st.divider()

st.subheader("🔗 Feature Correlation Analysis")

st.write(
    "Explore relationships between numerical academic "
    "and behavioural features."
)

st.plotly_chart(
    correlation_heatmap(filtered_df),
    use_container_width=True,
    key="correlation_heatmap"
)


# =====================================================
# ADDITIONAL FEATURE ANALYSIS
# =====================================================

st.divider()

st.subheader("📈 Additional Academic Insights")

available_features = [
    column
    for column in [
        "previous_score",
        "assignment_average",
        "sleep_hours",
        "class_participation"
    ]
    if column in filtered_df.columns
]

if available_features:

    selected_feature = st.selectbox(
        "Select a feature to analyze",
        available_features,
        format_func=lambda value: value.replace(
            "_", " "
        ).title()
    )

    feature_data = filtered_df[
        [selected_feature, "final_score"]
    ].copy()

    feature_data[selected_feature] = pd.to_numeric(
        feature_data[selected_feature],
        errors="coerce"
    )

    feature_data["final_score"] = pd.to_numeric(
        feature_data["final_score"],
        errors="coerce"
    )

    feature_data = feature_data.dropna()

    if not feature_data.empty:

        import plotly.express as px

        fig = px.scatter(
            feature_data,
            x=selected_feature,
            y="final_score",
            color="final_score",
            color_continuous_scale="Viridis",
            opacity=0.8,
            labels={
                selected_feature: selected_feature.replace(
                    "_", " "
                ).title(),
                "final_score": "Final Score"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#151f32",
            font_color="#e2e8f0",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="additional_feature_chart"
        )

    else:

        st.info("No valid values available for this feature.")

else:

    st.info("No additional numerical features are available.")


# =====================================================
# DATASET QUALITY
# =====================================================

st.divider()

st.subheader("🧪 Dataset Quality Snapshot")

quality1, quality2, quality3 = st.columns(3)

with quality1:

    st.metric(
        "Missing Values",
        int(filtered_df.isnull().sum().sum())
    )

with quality2:

    st.metric(
        "Duplicate Records",
        int(filtered_df.duplicated().sum())
    )

with quality3:

    st.metric(
        "Available Features",
        len(filtered_df.columns)
    )


# =====================================================
# DOWNLOAD FILTERED DATASET
# =====================================================

st.divider()

st.subheader("⬇️ Export Analytics Dataset")

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Filtered Dataset",
    data=csv_data,
    file_name="student_performance_analytics.csv",
    mime="text/csv",
    use_container_width=True
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Student Performance AI Studio | Analytics Studio"
)

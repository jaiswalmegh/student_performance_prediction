
import os
import sys
from pathlib import Path


# ==================================================
# APPLICATION LAUNCHER
# Direct VS Code Run Button Support
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


def is_streamlit_running():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__" and not is_streamlit_running():

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve())
    ]

    stcli.main()
    sys.exit(0)


# ==================================================
# IMPORTS
# ==================================================

import streamlit as st
import pandas as pd

from src.data_loader import (
    load_dataset,
    validate_dataset,
    get_dataset_info,
    get_dataset_quality
)

from src.visualizations import (
    score_distribution_chart,
    performance_category_chart,
    get_performance_statistics
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance AI Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# PATH CONFIGURATION
# ==================================================

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "best_model.joblib"


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0f19;
}

[data-testid="stSidebar"] {
    background-color: #101827;
    border-right: 1px solid #263248;
}

.hero-section {
    padding: 35px;
    border-radius: 20px;
    background:
        radial-gradient(
            circle at top right,
            rgba(99, 102, 241, 0.30),
            transparent 40%
        ),
        linear-gradient(
            120deg,
            #172554,
            #1e1b4b,
            #111827
        );
    border: 1px solid #3730a3;
    margin-bottom: 25px;
}

.hero-section h1 {
    font-size: 36px;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero-section p {
    color: #c7d2fe;
    font-size: 16px;
    line-height: 1.7;
}

.section-card {
    background: #151d2c;
    border: 1px solid #29364c;
    padding: 22px;
    border-radius: 15px;
    min-height: 150px;
}

.section-card h3 {
    color: #a5b4fc;
    margin-bottom: 10px;
}

.section-card p {
    color: #cbd5e1;
    font-size: 14px;
}

.status-card {
    background: #151d2c;
    border: 1px solid #29364c;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetric"] {
    background: #151d2c;
    border: 1px solid #29364c;
    padding: 16px;
    border-radius: 13px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None

if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = "default"


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown("## 🎓 Student AI Studio")

    st.caption("Machine Learning Dashboard")

    st.divider()

    st.markdown("### Navigation")

    st.page_link(
        "main.py",
        label="🏠 Dashboard",
        icon="🏠"
    )

    st.page_link(
        "pages/data_explorer.py",
        label="📂 Data Explorer",
        icon="📂"
    )

    st.page_link(
        "pages/analytics.py",
        label="📊 Analytics",
        icon="📊"
    )

    st.page_link(
        "pages/prediction_lab.py",
        label="🎯 Prediction Lab",
        icon="🎯"
    )

    st.page_link(
        "pages/model_intelligence.py",
        label="🧠 Model Intelligence",
        icon="🧠"
    )

    st.divider()

    st.markdown("### System Status")

    if MODEL_PATH.exists():
        st.success("AI Model: Available")
    else:
        st.warning("AI Model: Not Trained")

    if DATA_DIR.exists():
        st.success("Data Directory: Available")
    else:
        st.error("Data Directory: Missing")

    st.divider()

    st.caption("Student Performance AI Studio")
    st.caption("Built with Python & Machine Learning")


# ==================================================
# DATASET LOADING
# ==================================================

@st.cache_data
def get_default_dataset():
    return load_dataset()


try:

    if (
        st.session_state.active_dataset is not None
        and st.session_state.dataset_source == "uploaded"
    ):

        df = st.session_state.active_dataset.copy()

    else:

        df = get_default_dataset()

    validate_dataset(df)

    df["final_score"] = pd.to_numeric(
        df["final_score"],
        errors="coerce"
    )

    df = df.dropna(subset=["final_score"])

    dataset_available = True

except Exception as error:

    dataset_available = False
    df = pd.DataFrame()

    st.error(f"Dataset loading failed: {error}")


# ==================================================
# HERO SECTION
# FIXED HTML RENDERING
# ==================================================

hero_html = """
<div class="hero-section">
<h1>🎓 Student Performance AI Studio</h1>
<p>
An intelligent machine learning platform designed to explore
student academic data, analyze performance patterns and predict
future examination scores.
</p>
<p>
Explore your dataset. Understand academic trends.
Train predictive models. Generate intelligent insights.
</p>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)


# ==================================================
# DASHBOARD OVERVIEW
# ==================================================

st.subheader("📌 Dashboard Overview")

if dataset_available:

    info = get_dataset_info(df)
    quality = get_dataset_quality(df)
    statistics = get_performance_statistics(df)

    total_students = len(df)
    total_features = len(df.columns)
    average_score = statistics["average_score"]
    missing_values = int(df.isnull().sum().sum())

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Total Students",
            f"{total_students:,}"
        )

    with metric2:
        st.metric(
            "Dataset Features",
            total_features
        )

    with metric3:
        st.metric(
            "Average Final Score",
            f"{average_score:.2f}"
        )

    with metric4:
        st.metric(
            "Missing Values",
            missing_values
        )

    st.write("")

    # ----------------------------------------------
    # DATASET QUALITY
    # ----------------------------------------------

    st.subheader("📈 Dataset Quality")

    quality_col1, quality_col2 = st.columns([2, 1])

    with quality_col1:

        completeness = quality.get(
            "completeness_percentage",
            0
        )

        st.write("Data Completeness")

        st.progress(
            max(0.0, min(1.0, completeness / 100))
        )

        st.caption(
            f"{completeness:.2f}% of dataset cells are complete."
        )

    with quality_col2:

        st.metric(
            "Duplicate Records",
            quality.get("duplicate_rows", 0)
        )

    # ----------------------------------------------
    # PERFORMANCE INSIGHTS
    # ----------------------------------------------

    st.divider()

    st.subheader("📊 Academic Performance Insights")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("#### Final Score Distribution")

        distribution_chart = score_distribution_chart(df)

        st.plotly_chart(
            distribution_chart,
            use_container_width=True
        )

    with chart_col2:

        st.markdown("#### Student Performance Categories")

        category_chart = performance_category_chart(df)

        st.plotly_chart(
            category_chart,
            use_container_width=True
        )

    # ----------------------------------------------
    # PERFORMANCE SUMMARY
    # ----------------------------------------------

    st.divider()

    st.subheader("🎯 Performance Summary")

    summary1, summary2, summary3 = st.columns(3)

    with summary1:

        st.metric(
            "Highest Score",
            f"{statistics['highest_score']:.2f}"
        )

        st.metric(
            "High Performers",
            statistics["high_performers"]
        )

    with summary2:

        st.metric(
            "Median Score",
            f"{statistics['median_score']:.2f}"
        )

        st.metric(
            "Moderate Performers",
            statistics["moderate_performers"]
        )

    with summary3:

        st.metric(
            "Lowest Score",
            f"{statistics['lowest_score']:.2f}"
        )

        st.metric(
            "Students Needing Support",
            statistics["students_needing_support"]
        )

    # ----------------------------------------------
    # DATASET PREVIEW
    # ----------------------------------------------

    st.divider()

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        df.head(8),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Showing 8 of {total_students:,} student records."
    )

else:

    st.warning(
        "Dashboard analytics are unavailable because the dataset "
        "could not be loaded. Visit Data Explorer to inspect the dataset."
    )


# ==================================================
# QUICK ACCESS MODULES
# ==================================================

st.divider()

st.subheader("🚀 Explore Platform Modules")

card1, card2 = st.columns(2)

with card1:

    st.markdown("""
    <div class="section-card">
        <h3>📂 Data Explorer</h3>
        <p>
            Explore student records, inspect columns, identify missing
            values and understand dataset quality.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/data_explorer.py",
        label="Open Data Explorer →"
    )

with card2:

    st.markdown("""
    <div class="section-card">
        <h3>📊 Analytics Studio</h3>
        <p>
            Discover academic trends, score distributions,
            relationships and performance patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/analytics.py",
        label="Explore Analytics →"
    )

card3, card4 = st.columns(2)

with card3:

    st.markdown("""
    <div class="section-card">
        <h3>🎯 Prediction Lab</h3>
        <p>
            Enter student information and generate machine learning
            based academic performance predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/prediction_lab.py",
        label="Launch Prediction Lab →"
    )

with card4:

    st.markdown("""
    <div class="section-card">
        <h3>🧠 Model Intelligence</h3>
        <p>
            Train algorithms, compare model performance,
            evaluate predictions and analyze feature importance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/model_intelligence.py",
        label="Open Model Intelligence →"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div class="footer">
<p>Student Performance AI Studio</p>
<p>
Powered by Python · Pandas · Scikit-learn · Streamlit · Plotly
</p>
<p>Educational Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)

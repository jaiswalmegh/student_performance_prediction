
import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.data_loader import load_dataset, validate_dataset
from src.train import train_models
from src.predict import load_trained_model

from src.preprocessing import (
    clean_dataset,
    separate_features_target,
    split_dataset
)

from src.visualizations import (
    model_comparison_chart,
    actual_vs_predicted_chart,
    feature_importance_chart
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Intelligence | Student Performance AI",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

METRICS_PATH = OUTPUT_DIR / "metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "comparison.csv"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .intel-header {
        padding: 25px;
        border-radius: 16px;
        background: linear-gradient(
            120deg,
            #172554,
            #312e81,
            #1e1b4b
        );
        border: 1px solid #3730a3;
        margin-bottom: 25px;
    }

    .intel-header h1 {
        color: white;
        margin-bottom: 8px;
    }

    .intel-header p {
        color: #c7d2fe;
        font-size: 15px;
    }

    .model-card {
        background: #171c29;
        border: 1px solid #30384b;
        padding: 20px;
        border-radius: 14px;
    }

    .model-card h3 {
        color: #a5b4fc;
    }

    .section-note {
        color: #a1a1aa;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background: #171c29;
        border: 1px solid #30384b;
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "training_results" not in st.session_state:
    st.session_state.training_results = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="intel-header">
    <h1>🧠 Model Intelligence Center</h1>
    <p>
        Train machine learning models, evaluate their performance,
        compare algorithms and identify the best predictive model.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

@st.cache_data
def get_dataset():
    return load_dataset()


try:
    df = get_dataset()
    validate_dataset(df)

    st.success(
        f"Dataset loaded successfully: {len(df):,} records"
    )

except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()


# --------------------------------------------------
# TRAINING CONFIGURATION
# --------------------------------------------------

st.subheader("⚙️ Model Training")

st.markdown(
    "Train multiple regression algorithms on the selected dataset "
    "and compare their evaluation results."
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="model-card">
        <h3>Available Algorithms</h3>
        <ul>
            <li>Linear Regression</li>
            <li>Decision Tree Regressor</li>
            <li>Random Forest Regressor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <h3>Training Setup</h3>
        <p>Train-Test Split: 80:20</p>
        <p>Cross Validation: Adaptive K-Fold</p>
        <p>Target Variable: final_score</p>
    </div>
    """, unsafe_allow_html=True)


st.write("")

train_button = st.button(
    "🚀 Train All Machine Learning Models",
    use_container_width=True,
    type="primary"
)


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

if train_button:

    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("Preparing dataset...")
        progress.progress(15)

        training_results = train_models(df)

        progress.progress(100)
        status.success("All models trained and evaluated successfully!")

        st.session_state.training_results = training_results

        # Refresh cached prediction model.
        try:
            load_trained_model.clear()
        except Exception:
            pass

        st.rerun()

    except Exception as error:
        progress.empty()
        status.empty()

        st.error(f"Model training failed: {error}")
        st.exception(error)


# --------------------------------------------------
# LOAD TRAINING RESULTS
# --------------------------------------------------

results = st.session_state.training_results

comparison = None
best_metrics = None
best_model_name = None
training_report = None

if results is not None:

    comparison = results.get("comparison")
    best_metrics = results.get("best_metrics", {})
    best_model_name = results.get("best_model_name")
    training_report = results.get("training_report")

else:

    # Load previously saved evaluation results.
    if COMPARISON_PATH.exists():

        try:
            comparison = pd.read_csv(COMPARISON_PATH)
        except Exception:
            comparison = None

    if METRICS_PATH.exists():

        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                saved_metrics = json.load(file)

            best_metrics = saved_metrics.get(
                "best_metrics",
                saved_metrics.get("test_metrics", {})
            )

            best_model_name = saved_metrics.get(
                "best_model_name",
                saved_metrics.get("best_model")
            )

            training_report = saved_metrics.get(
                "training_report",
                saved_metrics
            )

        except Exception:
            best_metrics = None


# --------------------------------------------------
# BEST MODEL OVERVIEW
# --------------------------------------------------

if best_model_name:

    st.divider()
    st.subheader("🏆 Best Performing Model")

    st.markdown(
        f"""
        <div class="model-card">
            <h3>Selected Algorithm</h3>
            <h1 style="color:#818cf8;">{best_model_name}</h1>
            <p>
                Selected using cross-validation performance
                during model training.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

if best_metrics:

    st.markdown("### 📊 Model Evaluation Metrics")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "R² Score",
            f"{best_metrics.get('R2', 0):.4f}"
        )

    with metric2:
        st.metric(
            "Mean Absolute Error",
            f"{best_metrics.get('MAE', 0):.4f}"
        )

    with metric3:
        st.metric(
            "Root Mean Squared Error",
            f"{best_metrics.get('RMSE', 0):.4f}"
        )

    st.caption(
        "Higher R² and lower MAE/RMSE generally indicate better "
        "performance on the evaluation dataset."
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("⚖️ Algorithm Comparison")

if comparison is not None and not comparison.empty:

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    if {"Model", "R2"}.issubset(comparison.columns):

        comparison_chart = model_comparison_chart(comparison)

        st.plotly_chart(
            comparison_chart,
            use_container_width=True
        )

else:

    st.info(
        "No model comparison results available yet. "
        "Train the models to view the comparison."
    )


# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.divider()
st.subheader("🎯 Actual vs Predicted Performance")

if results is not None:

    try:
        best_model = results.get("best_model")
        X_test = results.get("X_test")
        y_test = results.get("y_test")

        if best_model is not None and X_test is not None and y_test is not None:

            predictions = best_model.predict(X_test)

            prediction_chart = actual_vs_predicted_chart(
                y_test,
                predictions
            )

            st.plotly_chart(
                prediction_chart,
                use_container_width=True
            )

            st.caption(
                "Points closer to the diagonal line indicate predictions "
                "closer to actual examination scores."
            )

        else:
            st.info(
                "Actual vs Predicted visualization is unavailable "
                "for the saved model results."
            )

    except Exception as error:
        st.warning(
            f"Could not generate prediction visualization: {error}"
        )

else:

    st.info(
        "Train models in this session to view the actual vs predicted chart."
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()
st.subheader("🔍 Feature Importance Analysis")

st.markdown(
    "Identify which student-related features contribute most "
    "to the selected model's predictions."
)

if results is not None:

    try:
        best_model = results.get("best_model")
        X_test = results.get("X_test")
        y_test = results.get("y_test")

        if best_model is not None and X_test is not None and y_test is not None:

            importance_chart = feature_importance_chart(
                best_model,
                X_test,
                y_test
            )

            st.plotly_chart(
                importance_chart,
                use_container_width=True
            )

            st.caption(
                "Feature importance is estimated using permutation "
                "importance on the test dataset."
            )

        else:
            st.info("Feature importance data is unavailable.")

    except Exception as error:
        st.warning(
            f"Feature importance could not be generated: {error}"
        )

else:

    st.info(
        "Train models in this session to generate feature importance."
    )


# --------------------------------------------------
# TRAINING REPORT
# --------------------------------------------------

if training_report:

    st.divider()
    st.subheader("📄 Training Report")

    with st.expander("View Detailed Training Information"):

        st.json(training_report)


# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

st.divider()
st.subheader("💾 Saved Model Information")

model_files = list(MODEL_DIR.glob("*")) if MODEL_DIR.exists() else []

if model_files:

    for model_file in model_files:

        if model_file.is_file():

            file_col1, file_col2 = st.columns([3, 1])

            with file_col1:
                st.write(f"📦 {model_file.name}")

            with file_col2:
                st.caption(
                    f"{model_file.stat().st_size / 1024:.2f} KB"
                )

else:

    st.info("No saved model files found yet.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance AI Studio | Model Intelligence Module"
)

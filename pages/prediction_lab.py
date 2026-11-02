
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.predict import (
    load_trained_model,
    predict_student_performance
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Prediction Lab | Student Performance AI",
    page_icon="🎯",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .prediction-header {
        padding: 24px;
        border-radius: 15px;
        background: linear-gradient(120deg, #172554, #312e81, #1e1b4b);
        border: 1px solid #3730a3;
        margin-bottom: 25px;
    }

    .prediction-header h1 {
        color: #ffffff;
        margin-bottom: 8px;
    }

    .prediction-header p {
        color: #c7d2fe;
        font-size: 15px;
    }

    .result-card {
        padding: 20px;
        border-radius: 14px;
        background: #171c29;
        border: 1px solid #30384b;
        text-align: center;
    }

    .result-score {
        font-size: 42px;
        font-weight: 800;
        color: #818cf8;
    }

    .suggestion-card {
        padding: 14px 18px;
        border-radius: 10px;
        background: #171c29;
        border-left: 4px solid #818cf8;
        margin-bottom: 10px;
    }

    div[data-testid="stForm"] {
        border: 1px solid #30384b;
        border-radius: 14px;
        padding: 22px;
        background: #121722;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_trained_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="prediction-header">
    <h1>🎯 Student Performance Prediction Lab</h1>
    <p>
        Predict academic performance using machine learning.
        Enter student details and generate an AI-powered performance report.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

try:
    model = get_model()
    model_available = True

    st.success("AI Model Loaded Successfully. Prediction system is ready.")

except FileNotFoundError:
    model_available = False
    model = None

    st.warning(
        "Trained model not found. Please visit Model Intelligence "
        "and train the model before making predictions."
    )

except Exception as error:
    model_available = False
    model = None

    st.error(f"Unable to load the trained model: {error}")


# --------------------------------------------------
# STUDENT INPUT FORM
# --------------------------------------------------

st.subheader("👨‍🎓 Student Information")

st.write(
    "Enter the student's academic and personal learning details below."
)

with st.form("student_prediction_form"):

    st.markdown("### 📚 Academic Information")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "Daily Study Hours",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.5
        )

        attendance = st.slider(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

        previous_score = st.slider(
            "Previous Examination Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with col2:
        assignment_average = st.slider(
            "Assignment Average",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        class_participation = st.slider(
            "Class Participation Level",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Very Low, 5 = Excellent"
        )

        sleep_hours = st.slider(
            "Daily Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    st.divider()

    st.markdown("### 🌐 Learning Environment")

    col3, col4, col5 = st.columns(3)

    with col3:
        internet_access = st.selectbox(
            "Internet Access",
            options=["Yes", "No"]
        )

    with col4:
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=["Yes", "No"]
        )

    with col5:
        academic_support = st.selectbox(
            "Academic Support",
            options=["High", "Medium", "Low"]
        )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Student Performance",
        use_container_width=True,
        type="primary"
    )


# --------------------------------------------------
# GENERATE PREDICTION
# --------------------------------------------------

if submitted:

    if not model_available:
        st.error(
            "Prediction unavailable. Train a model first "
            "from the Model Intelligence page."
        )

    else:

        student_data = {
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "previous_score": previous_score,
            "assignment_average": assignment_average,
            "sleep_hours": sleep_hours,
            "class_participation": class_participation,
            "internet_access": internet_access,
            "extracurricular_activities": extracurricular,
            "academic_support": academic_support
        }

        try:
            result = predict_student_performance(
                student_data,
                model=model
            )

            st.session_state.latest_prediction = {
                "result": result,
                "student_data": student_data
            }

            history_record = {
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Study Hours": study_hours,
                "Attendance": attendance,
                "Previous Score": previous_score,
                "Assignment Average": assignment_average,
                "Predicted Score": round(
                    float(result["predicted_score"]), 2
                ),
                "Performance": result["performance"]
            }

            st.session_state.prediction_history.append(history_record)

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# DISPLAY PREDICTION RESULTS
# --------------------------------------------------

latest = st.session_state.latest_prediction

if latest is not None:

    result = latest["result"]
    student_data = latest["student_data"]

    predicted_score = float(result["predicted_score"])
    performance = result["performance"]
    description = result["description"]
    suggestions = result["suggestions"]

    st.divider()

    st.subheader("📊 Prediction Results")

    score_col, category_col, status_col = st.columns(3)

    with score_col:
        st.markdown(
            f"""
            <div class="result-card">
                <p>Predicted Final Score</p>
                <div class="result-score">
                    {predicted_score:.2f}
                </div>
                <p>Out of 100</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with category_col:
        st.markdown(
            f"""
            <div class="result-card">
                <p>Performance Category</p>
                <h2 style="color:#a5b4fc;">
                    {performance}
                </h2>
                <p>AI Assessment</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with status_col:
        status = result.get("status", "Prediction Generated")

        st.markdown(
            f"""
            <div class="result-card">
                <p>Prediction Status</p>
                <h3 style="color:#34d399;">
                    {status}
                </h3>
                <p>Analysis Complete</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # SCORE GAUGE

    st.markdown("### 🎯 Predicted Score Visualization")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(100, predicted_score)),
            number={
                "suffix": "%",
                "font": {"size": 36}
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#818cf8"
                },
                "steps": [
                    {
                        "range": [0, 60],
                        "color": "#3f2028"
                    },
                    {
                        "range": [60, 80],
                        "color": "#493b20"
                    },
                    {
                        "range": [80, 100],
                        "color": "#173d35"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # DESCRIPTION

    st.markdown("### 📝 AI Performance Assessment")

    st.info(description)

    # SUGGESTIONS

    st.markdown("### 💡 Personalized Improvement Suggestions")

    if isinstance(suggestions, str):
        suggestions = [suggestions]

    if suggestions:
        for index, suggestion in enumerate(suggestions, start=1):

            st.markdown(
                f"""
                <div class="suggestion-card">
                    <strong>Suggestion {index}</strong><br>
                    {suggestion}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("No additional suggestions available.")

    # INPUT SUMMARY

    with st.expander("View Student Input Details"):

        input_df = pd.DataFrame(
            list(student_data.items()),
            columns=["Feature", "Input Value"]
        )

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()

st.subheader("📁 Prediction History")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Total Predictions",
            len(history_df)
        )

    with metric2:
        st.metric(
            "Average Predicted Score",
            f"{history_df['Predicted Score'].mean():.2f}"
        )

    with metric3:
        st.metric(
            "Highest Predicted Score",
            f"{history_df['Predicted Score'].max():.2f}"
        )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="student_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.session_state.latest_prediction = None
        st.rerun()

else:
    st.info(
        "No predictions generated yet. Submit student details "
        "to create your first prediction."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance AI Studio | Machine Learning Prediction Module"
)

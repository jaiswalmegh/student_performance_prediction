
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

TARGET_COLUMN = "final_score"


# =====================================================
# REQUIRED INPUT FEATURES
# =====================================================

REQUIRED_FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "previous_score",
    "assignment_average",
    "sleep_hours",
    "class_participation",
    "internet_access",
    "extracurricular_activities",
    "academic_support"
]


# =====================================================
# VALIDATION RULES
# =====================================================

NUMERIC_RANGES = {
    "study_hours_per_day": (0, 24),
    "attendance_percentage": (0, 100),
    "previous_score": (0, 100),
    "assignment_average": (0, 100),
    "sleep_hours": (0, 24),
    "class_participation": (1, 5)
}

CATEGORICAL_VALUES = {
    "internet_access": ["Yes", "No"],
    "extracurricular_activities": ["Yes", "No"],
    "academic_support": ["High", "Medium", "Low"]
}


# =====================================================
# MODEL LOADING
# =====================================================

def load_trained_model(model_path=None):
    """Load the saved machine learning pipeline."""

    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Please train a model from Model Intelligence first."
        )

    try:
        model = joblib.load(path)

        if not hasattr(model, "predict"):
            raise ValueError(
                "The saved file does not contain a valid prediction model."
            )

        return model

    except Exception as error:
        raise RuntimeError(
            f"Unable to load trained model: {error}"
        )


# =====================================================
# INPUT VALIDATION
# =====================================================

def validate_student_data(student_data):
    """Validate an individual student's input information."""

    if not isinstance(student_data, dict):
        raise TypeError(
            "Student information must be provided as a dictionary."
        )

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in student_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    validated_data = {}

    # Validate numerical features
    for feature, limits in NUMERIC_RANGES.items():

        try:
            value = float(student_data[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a valid numerical value."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        minimum, maximum = limits

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} must be between {minimum} and {maximum}."
            )

        validated_data[feature] = value

    # Validate categorical features
    for feature, allowed_values in CATEGORICAL_VALUES.items():

        value = str(student_data[feature]).strip()

        matched_value = next(
            (
                allowed
                for allowed in allowed_values
                if allowed.lower() == value.lower()
            ),
            None
        )

        if matched_value is None:
            raise ValueError(
                f"{feature} must be one of {allowed_values}."
            )

        validated_data[feature] = matched_value

    return validated_data


# =====================================================
# PERFORMANCE CLASSIFICATION
# =====================================================

def classify_performance(score):
    """Convert predicted score into a performance category."""

    score = float(score)

    if score >= 80:
        return {
            "category": "High Performer",
            "description": (
                "The predicted performance is strong. "
                "Continue maintaining consistent academic habits."
            ),
            "status": "Excellent"
        }

    elif score >= 60:
        return {
            "category": "Moderate Performer",
            "description": (
                "The predicted performance is moderate. "
                "Focused improvement can help achieve stronger results."
            ),
            "status": "Moderate"
        }

    else:
        return {
            "category": "Needs Support",
            "description": (
                "The predicted performance indicates that "
                "additional academic support may be beneficial."
            ),
            "status": "Needs Attention"
        }


# =====================================================
# ACADEMIC SUGGESTIONS
# =====================================================

def generate_academic_suggestions(student_data, predicted_score):
    """Generate rule-based suggestions from student inputs."""

    suggestions = []

    if student_data["study_hours_per_day"] < 2:
        suggestions.append(
            "Build a consistent daily study routine."
        )

    if student_data["attendance_percentage"] < 75:
        suggestions.append(
            "Improve class attendance and regular participation."
        )

    if student_data["assignment_average"] < 60:
        suggestions.append(
            "Focus on completing assignments and revising concepts."
        )

    if student_data["class_participation"] < 3:
        suggestions.append(
            "Participate more actively in classroom activities."
        )

    if student_data["sleep_hours"] < 6:
        suggestions.append(
            "Maintain a healthier sleep schedule."
        )

    if student_data["internet_access"] == "No":
        suggestions.append(
            "Explore accessible offline learning resources."
        )

    if student_data["academic_support"] == "Low":
        suggestions.append(
            "Consider seeking additional academic guidance."
        )

    if predicted_score < 60:
        suggestions.append(
            "Create a focused improvement plan with regular progress reviews."
        )

    if not suggestions:
        suggestions.append(
            "Maintain your current learning habits and review progress regularly."
        )

    return suggestions


# =====================================================
# SINGLE STUDENT PREDICTION
# =====================================================

def predict_student_performance(student_data, model=None):
    """
    Predict estimated final score for one student.

    A previously loaded model can be supplied to avoid
    loading the model repeatedly.
    """

    validated_data = validate_student_data(student_data)

    if model is None:
        model = load_trained_model()

    input_df = pd.DataFrame(
        [validated_data],
        columns=REQUIRED_FEATURES
    )

    try:
        prediction = float(
            np.asarray(model.predict(input_df)).reshape(-1)[0]
        )

    except Exception as error:
        raise RuntimeError(
            f"Prediction failed: {error}"
        )

    if not np.isfinite(prediction):
        raise RuntimeError(
            "Model returned an invalid prediction."
        )

    # Scores are restricted to the expected 0-100 range.
    predicted_score = float(
        np.clip(prediction, 0, 100)
    )

    performance = classify_performance(
        predicted_score
    )

    suggestions = generate_academic_suggestions(
        validated_data,
        predicted_score
    )

    return {
        "predicted_score": round(predicted_score, 2),
        "performance": performance["category"],
        "category": performance["category"],
        "description": performance["description"],
        "status": performance["status"],
        "suggestions": suggestions,
        "student_data": validated_data
    }


# =====================================================
# MULTIPLE STUDENT PREDICTIONS
# =====================================================

def predict_multiple_students(data, model=None):
    """Generate predictions for multiple student records."""

    if isinstance(data, pd.DataFrame):
        records = data.to_dict(orient="records")

    elif isinstance(data, list):
        records = data

    else:
        raise TypeError(
            "Input must be a DataFrame or list of dictionaries."
        )

    if not records:
        raise ValueError(
            "No student records provided."
        )

    if model is None:
        model = load_trained_model()

    validated_records = [
        validate_student_data(record)
        for record in records
    ]

    input_df = pd.DataFrame(
        validated_records,
        columns=REQUIRED_FEATURES
    )

    try:
        predictions = np.asarray(
            model.predict(input_df)
        ).reshape(-1)

    except Exception as error:
        raise RuntimeError(
            f"Batch prediction failed: {error}"
        )

    if len(predictions) != len(validated_records):
        raise RuntimeError(
            "Prediction count does not match input record count."
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Model generated invalid batch predictions."
        )

    predictions = np.clip(
        predictions,
        0,
        100
    )

    results = pd.DataFrame(validated_records)

    results["predicted_score"] = np.round(
        predictions,
        2
    )

    results["performance_category"] = [
        classify_performance(score)["category"]
        for score in predictions
    ]

    return results


# =====================================================
# PREDICTION SUMMARY
# =====================================================

def get_prediction_summary(predictions):
    """Generate summary statistics for prediction results."""

    if isinstance(predictions, pd.DataFrame):

        if "predicted_score" not in predictions.columns:
            raise ValueError(
                "predicted_score column is missing."
            )

        scores = pd.to_numeric(
            predictions["predicted_score"],
            errors="coerce"
        )

    else:
        scores = pd.to_numeric(
            pd.Series(predictions),
            errors="coerce"
        )

    scores = scores.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if scores.empty:
        raise ValueError(
            "No valid prediction scores available."
        )

    return {
        "total_predictions": int(len(scores)),
        "average_score": round(float(scores.mean()), 2),
        "highest_score": round(float(scores.max()), 2),
        "lowest_score": round(float(scores.min()), 2),
        "high_performers": int((scores >= 80).sum()),
        "moderate_performers": int(
            ((scores >= 60) & (scores < 80)).sum()
        ),
        "students_needing_support": int((scores < 60).sum())
    }


import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import (
    clean_dataset,
    separate_features_target,
    detect_feature_types,
    build_preprocessor,
    split_dataset
)


# =====================================================
# PROJECT PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"


# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_STATE = 42

MODEL_CONFIGURATIONS = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
}


# =====================================================
# MODEL EVALUATION
# =====================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained regression pipeline.

    Returns MAE, RMSE and R2.
    """

    if len(X_test) == 0:
        raise ValueError("Testing dataset is empty.")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    ) if len(y_test) >= 2 else 0.0

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }


# =====================================================
# PIPELINE CREATION
# =====================================================

def create_model_pipeline(
    algorithm,
    numerical_features,
    categorical_features
):
    """Create a fresh preprocessing and model pipeline."""

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(algorithm))
    ])


# =====================================================
# MODEL TRAINING
# =====================================================

def train_models(df):
    """
    Train and compare regression algorithms.

    Best model is selected using cross-validation R2.
    The selected pipeline is saved for future predictions.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Training input must be a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot train models on an empty dataset."
        )

    print("\nStarting Student Performance AI Training...")

    # -------------------------------------------------
    # STEP 1: DATA CLEANING
    # -------------------------------------------------

    cleaned_data = clean_dataset(df)

    if len(cleaned_data) < 10:
        raise ValueError(
            "At least 10 valid records are recommended "
            "for reliable model training."
        )

    # -------------------------------------------------
    # STEP 2: FEATURE AND TARGET SEPARATION
    # -------------------------------------------------

    X, y = separate_features_target(cleaned_data)

    numerical_features, categorical_features = (
        detect_feature_types(X)
    )

    # -------------------------------------------------
    # STEP 3: TRAIN-TEST SPLIT
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    # Adaptive cross-validation
    cv_folds = min(5, len(X_train))

    if cv_folds < 2:
        raise ValueError(
            "Not enough training records for cross-validation."
        )

    results = []
    trained_models = {}

    best_model = None
    best_model_name = None
    best_score = -float("inf")

    # -------------------------------------------------
    # STEP 4: TRAIN EACH ALGORITHM
    # -------------------------------------------------

    for model_name, algorithm in MODEL_CONFIGURATIONS.items():

        print(f"\nTraining {model_name}...")

        pipeline = create_model_pipeline(
            algorithm,
            numerical_features,
            categorical_features
        )

        try:

            # Cross-validation on training data only
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="r2",
                n_jobs=1,
                error_score="raise"
            )

            valid_scores = cv_scores[
                np.isfinite(cv_scores)
            ]

            if len(valid_scores) == 0:
                raise ValueError(
                    "Cross-validation returned invalid scores."
                )

            cv_mean = float(np.mean(valid_scores))
            cv_std = float(np.std(valid_scores))

            # Fit final model on complete training split
            pipeline.fit(
                X_train,
                y_train
            )

            # Evaluate on untouched test split
            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            result = {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "CV_R2_Mean": round(cv_mean, 4),
                "CV_R2_Std": round(cv_std, 4)
            }

            results.append(result)

            trained_models[model_name] = pipeline

            print("Completed:", result)

            # Select best model based on CV R2
            if cv_mean > best_score:

                best_score = cv_mean
                best_model = pipeline
                best_model_name = model_name

        except Exception as error:

            print(
                f"Training failed for {model_name}: {error}"
            )

    # -------------------------------------------------
    # STEP 5: CHECK TRAINING RESULTS
    # -------------------------------------------------

    if best_model is None:
        raise RuntimeError(
            "All machine learning algorithms failed. "
            "Please inspect the dataset and training errors."
        )

    # -------------------------------------------------
    # STEP 6: SAVE BEST MODEL
    # -------------------------------------------------

    joblib.dump(
        best_model,
        BEST_MODEL_PATH
    )

    # -------------------------------------------------
    # STEP 7: FINAL EVALUATION
    # -------------------------------------------------

    best_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="CV_R2_Mean",
        ascending=False
    ).reset_index(drop=True)

    comparison.to_csv(
        COMPARISON_PATH,
        index=False
    )

    # -------------------------------------------------
    # STEP 8: SAVE TRAINING REPORT
    # -------------------------------------------------

    training_report = {
        "best_model": best_model_name,

        "best_model_name": best_model_name,

        "selection_metric": (
            f"{cv_folds}-fold cross-validation R2"
        ),

        "cross_validation_r2": round(
            float(best_score),
            4
        ),

        "test_metrics": best_metrics,

        "dataset_rows": int(len(cleaned_data)),

        "training_rows": int(len(X_train)),

        "testing_rows": int(len(X_test)),

        "features": X.columns.tolist(),

        "target": "final_score",

        "cv_folds": cv_folds,

        "models": results
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            training_report,
            file,
            indent=4
        )

    # -------------------------------------------------
    # STEP 9: RETURN RESULTS
    # -------------------------------------------------

    print("\n====================================")
    print("MODEL TRAINING COMPLETED")
    print("====================================")

    print("Best Model:", best_model_name)
    print("Cross Validation R2:", best_score)
    print("Test Metrics:", best_metrics)

    print("Saved Model:", BEST_MODEL_PATH)
    print("Saved Report:", METRICS_PATH)

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "comparison": comparison,
        "trained_models": trained_models,
        "X_test": X_test,
        "y_test": y_test,
        "training_report": training_report
    }


# =====================================================
# STANDALONE EXECUTION
# =====================================================

if __name__ == "__main__":

    from src.data_loader import load_dataset

    dataset = load_dataset()

    results = train_models(dataset)

    print("\nModel Comparison:")
    print(results["comparison"])

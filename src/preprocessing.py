
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# =====================================================
# CONFIGURATION
# =====================================================

TARGET_COLUMN = "final_score"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =====================================================
# DATA CLEANING
# =====================================================

def clean_dataset(df):
    """
    Clean student performance dataset.

    Operations:
    - Standardize column names
    - Remove duplicate records
    - Convert target to numeric
    - Remove records with missing target
    - Handle infinite values
    - Restrict target score to valid range
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    data = df.copy()

    # Standardize column names
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    if data.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found after normalization."
        )

    # Check target
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    # Remove duplicate records
    data = data.drop_duplicates()

    # Replace infinite numerical values
    numerical_columns = data.select_dtypes(
        include=np.number
    ).columns

    data[numerical_columns] = data[numerical_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert target to numeric
    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce"
    )

    # Remove rows where target is missing
    data = data.dropna(subset=[TARGET_COLUMN])

    # Keep scores within 0-100
    data = data[
        data[TARGET_COLUMN].between(0, 100)
    ]

    if data.empty:
        raise ValueError(
            "No valid records remain after cleaning."
        )

    return data.reset_index(drop=True)


# =====================================================
# FEATURE AND TARGET SEPARATION
# =====================================================

def separate_features_target(df):
    """Separate independent features and target variable."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    if X.empty or X.shape[1] == 0:
        raise ValueError("No input features available.")

    return X, y


# =====================================================
# FEATURE TYPE DETECTION
# =====================================================

def detect_feature_types(X):
    """Identify numerical and categorical features."""

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return numerical_features, categorical_features


# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

def build_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Build preprocessing pipeline.

    Numerical:
        Median imputation + Standard Scaling

    Categorical:
        Most frequent imputation + One Hot Encoding
    """

    transformers = []

    # Numerical pipeline
    if numerical_features:

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            )
        )

    # Categorical pipeline
    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    if not transformers:
        raise ValueError(
            "No valid numerical or categorical features found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# =====================================================
# TRAIN-TEST SPLIT
# =====================================================

def split_dataset(X, y):
    """Split data into training and testing sets."""

    if len(X) != len(y):
        raise ValueError(
            "Features and target have different record counts."
        )

    if len(X) < 5:
        raise ValueError(
            "At least 5 valid records are required for splitting."
        )

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


# =====================================================
# COMPLETE DATA PREPARATION
# =====================================================

def prepare_data(df):
    """
    Complete preprocessing workflow.

    Returns raw and transformed data along with
    the fitted preprocessing pipeline.
    """

    cleaned_df = clean_dataset(df)

    X, y = separate_features_target(cleaned_df)

    numerical_features, categorical_features = (
        detect_feature_types(X)
    )

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    # Fit only on training data to prevent data leakage
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    return {
        "cleaned_data": cleaned_df,

        "X": X,
        "y": y,

        "X_train": X_train,
        "X_test": X_test,

        "y_train": y_train,
        "y_test": y_test,

        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,

        "preprocessor": preprocessor,

        "numerical_features": numerical_features,
        "categorical_features": categorical_features,

        "feature_names": (
            preprocessor.get_feature_names_out().tolist()
        )
    }


# =====================================================
# PREPROCESSING SUMMARY
# =====================================================

def get_preprocessing_summary(original_df, cleaned_df):
    """Generate a summary of the cleaning process."""

    original_rows = len(original_df)
    cleaned_rows = len(cleaned_df)

    removed_rows = original_rows - cleaned_rows

    return {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "removed_rows": removed_rows,
        "original_columns": len(original_df.columns),
        "cleaned_columns": len(cleaned_df.columns),
        "missing_values_after_cleaning": int(
            cleaned_df.isnull().sum().sum()
        )
    }


from pathlib import Path
import re
import pandas as pd


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATASET = DATA_DIR / "student_performance.csv"

TARGET_COLUMN = "final_score"


# =====================================================
# COLUMN NORMALIZATION
# =====================================================

def normalize_column_names(df):
    """Standardize column names for consistent processing."""

    data = df.copy()

    normalized_columns = []

    for column in data.columns:
        name = str(column).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = name.strip("_")
        normalized_columns.append(name)

    data.columns = normalized_columns

    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names found after normalization: {duplicates}"
        )

    return data


# =====================================================
# DATASET LOADER
# =====================================================

def load_dataset(file_path=None):
    """Load the default or a custom CSV dataset."""

    path = Path(file_path) if file_path else DEFAULT_DATASET

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    try:
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("The dataset is empty.")

        df = normalize_column_names(df)

        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file contains no data.")

    except pd.errors.ParserError:
        raise ValueError(
            "Unable to read the CSV. Please check its format."
        )

    except UnicodeDecodeError:
        raise ValueError(
            "Unable to decode the CSV. Save it using UTF-8 encoding."
        )


# =====================================================
# DATASET VALIDATION
# =====================================================

def validate_dataset(df):
    """Validate the structure and target column of a dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset contains no records.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing."
        )

    numeric_target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    if numeric_target.notna().sum() == 0:
        raise ValueError(
            "The final_score column contains no valid numeric values."
        )

    if len(df.columns) < 2:
        raise ValueError(
            "Dataset must contain at least one feature and one target column."
        )

    return True


# =====================================================
# DATASET INFORMATION
# =====================================================

def get_dataset_info(df):
    """Return dataset statistics using consistent and compatible keys."""

    rows = int(df.shape[0])
    columns = int(df.shape[1])
    duplicates = int(df.duplicated().sum())
    missing = int(df.isnull().sum().sum())

    memory = round(
        df.memory_usage(deep=True).sum() / 1024,
        2
    )

    return {
        # Current descriptive keys
        "total_rows": rows,
        "total_columns": columns,
        "column_names": df.columns.tolist(),
        "duplicate_rows": duplicates,
        "missing_cells": missing,
        "memory_usage_kb": memory,

        # Compatibility keys for existing application pages
        "rows": rows,
        "columns": columns,
        "duplicates": duplicates
    }


# =====================================================
# COLUMN INFORMATION
# =====================================================

def get_column_information(df):
    """Generate a detailed column profile."""

    information = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    return information


# =====================================================
# DATASET PREVIEW
# =====================================================

def get_dataset_preview(df, rows=10):
    """Return the first requested number of records."""

    rows = max(1, int(rows))

    return df.head(rows).copy()


# =====================================================
# NUMERICAL SUMMARY
# =====================================================

def get_numerical_summary(df):
    """Generate descriptive statistics for numerical columns."""

    numerical_data = df.select_dtypes(include="number")

    if numerical_data.empty:
        return pd.DataFrame()

    return numerical_data.describe().round(2)


# =====================================================
# CATEGORICAL SUMMARY
# =====================================================

def get_categorical_summary(df):
    """Generate descriptive statistics for categorical columns."""

    categorical_data = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    if categorical_data.empty:
        return pd.DataFrame()

    return categorical_data.describe().T


# =====================================================
# MISSING VALUE ANALYSIS
# =====================================================

def get_missing_value_report(df):
    """Return missing-value counts and percentages."""

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    return (
        report.sort_values(
            by="Missing Percentage",
            ascending=False
        )
        .reset_index(drop=True)
    )


# =====================================================
# DUPLICATE ANALYSIS
# =====================================================

def get_duplicate_count(df):
    """Count duplicate records."""

    return int(df.duplicated().sum())


# =====================================================
# DATASET QUALITY SUMMARY
# =====================================================

def get_dataset_quality(df):
    """Return a compact dataset quality report."""

    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isnull().sum().sum())

    completeness = (
        ((total_cells - missing_cells) / total_cells) * 100
        if total_cells > 0
        else 0
    )

    return {
        "completeness_percentage": round(completeness, 2),
        "missing_cells": missing_cells,
        "duplicate_rows": get_duplicate_count(df),
        "total_records": int(len(df)),
        "total_features": int(len(df.columns))
    }


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.inspection import permutation_importance


# =====================================================
# THEME CONFIGURATION
# =====================================================

COLORS = {
    "background": "#0b1120",
    "card": "#151f32",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "blue": "#60a5fa",
    "cyan": "#22d3ee",
    "purple": "#a78bfa",
    "green": "#4ade80",
    "orange": "#fb923c",
    "red": "#f87171"
}

TARGET_COLUMN = "final_score"


# =====================================================
# COMMON CHART THEME
# =====================================================

def apply_chart_theme(fig, title=None, height=420):

    fig.update_layout(
        title={
            "text": title or "",
            "font": {
                "size": 18,
                "color": COLORS["text"]
            }
        },
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        font={
            "color": COLORS["text"],
            "family": "Arial"
        },
        height=height,
        margin=dict(l=35, r=25, t=65, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"])
        ),
        hoverlabel=dict(
            bgcolor=COLORS["card"],
            font_color=COLORS["text"]
        )
    )

    fig.update_xaxes(
        gridcolor="#26354d",
        zerolinecolor="#26354d"
    )

    fig.update_yaxes(
        gridcolor="#26354d",
        zerolinecolor="#26354d"
    )

    return fig


# =====================================================
# DATA CLEANING HELPER
# =====================================================

def _prepare_chart_data(df, required_columns):

    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    data = df[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    return data


# =====================================================
# EMPTY CHART
# =====================================================

def empty_chart(message="Not enough valid data to display this chart."):

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=15,
            color=COLORS["muted"]
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


# =====================================================
# SCORE DISTRIBUTION
# =====================================================

def score_distribution_chart(df):

    data = _prepare_chart_data(
        df,
        [TARGET_COLUMN]
    )

    if data.empty:
        return empty_chart()

    fig = px.histogram(
        data,
        x=TARGET_COLUMN,
        nbins=25,
        color_discrete_sequence=[COLORS["blue"]],
        labels={
            TARGET_COLUMN: "Final Score",
            "count": "Number of Students"
        }
    )

    fig.update_traces(
        marker_line_color=COLORS["cyan"],
        marker_line_width=1
    )

    fig.update_layout(
        xaxis_title="Final Score",
        yaxis_title="Number of Students"
    )

    return apply_chart_theme(
        fig,
        "Student Score Distribution"
    )


# =====================================================
# STUDY HOURS VS PERFORMANCE
# =====================================================

def study_hours_chart(df):

    data = _prepare_chart_data(
        df,
        ["study_hours_per_day", TARGET_COLUMN]
    )

    if data.empty:
        return empty_chart()

    # Marker size is always finite and positive.
    # Missing values are removed before chart creation.

    data["marker_size"] = 10.0

    if "attendance_percentage" in df.columns:

        attendance = pd.to_numeric(
            df.loc[data.index, "attendance_percentage"],
            errors="coerce"
        )

        attendance = attendance.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data["marker_size"] = (
            attendance.fillna(0).clip(0, 100) / 10
        ) + 8

    fig = px.scatter(
        data,
        x="study_hours_per_day",
        y=TARGET_COLUMN,
        size="marker_size",
        size_max=22,
        color=TARGET_COLUMN,
        color_continuous_scale="Blues",
        opacity=0.8,
        labels={
            "study_hours_per_day": "Study Hours Per Day",
            TARGET_COLUMN: "Final Score",
            "marker_size": "Marker Size"
        }
    )

    fig.update_traces(
        marker=dict(
            line=dict(
                width=1,
                color=COLORS["cyan"]
            )
        )
    )

    return apply_chart_theme(
        fig,
        "Study Hours vs Academic Performance"
    )


# =====================================================
# CORRELATION HEATMAP
# =====================================================

def correlation_heatmap(df):

    if df is None or df.empty:
        return empty_chart()

    numerical_df = df.select_dtypes(
        include="number"
    )

    if numerical_df.shape[1] < 2:
        return empty_chart(
            "At least two numerical columns are required."
        )

    numerical_df = numerical_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    correlation = numerical_df.corr()

    if correlation.empty:
        return empty_chart()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(
            color="Correlation"
        )
    )

    return apply_chart_theme(
        fig,
        "Feature Correlation Heatmap",
        height=520
    )


# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

def actual_vs_predicted_chart(y_true, y_pred):

    actual = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)

    if len(actual) != len(predicted):
        return empty_chart(
            "Actual and predicted values have different lengths."
        )

    data = pd.DataFrame({
        "Actual Score": actual,
        "Predicted Score": predicted
    })

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if data.empty:
        return empty_chart()

    fig = px.scatter(
        data,
        x="Actual Score",
        y="Predicted Score",
        color_discrete_sequence=[COLORS["cyan"]],
        opacity=0.8
    )

    minimum = min(
        data["Actual Score"].min(),
        data["Predicted Score"].min()
    )

    maximum = max(
        data["Actual Score"].max(),
        data["Predicted Score"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect Prediction",
            line=dict(
                color=COLORS["orange"],
                dash="dash",
                width=2
            )
        )
    )

    return apply_chart_theme(
        fig,
        "Actual vs Predicted Performance"
    )


# =====================================================
# MODEL COMPARISON
# =====================================================

def model_comparison_chart(comparison_df):

    if comparison_df is None or comparison_df.empty:
        return empty_chart(
            "Train models to view the comparison."
        )

    required = ["Model", "R2"]

    if not all(
        column in comparison_df.columns
        for column in required
    ):
        return empty_chart(
            "Model comparison data is incomplete."
        )

    data = comparison_df.copy()

    data["R2"] = pd.to_numeric(
        data["R2"],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(subset=["R2"])

    if data.empty:
        return empty_chart()

    fig = px.bar(
        data,
        x="Model",
        y="R2",
        color="Model",
        text_auto=".3f",
        color_discrete_sequence=[
            COLORS["blue"],
            COLORS["purple"],
            COLORS["cyan"]
        ]
    )

    fig.update_layout(
        xaxis_title="Machine Learning Model",
        yaxis_title="R² Score"
    )

    return apply_chart_theme(
        fig,
        "Model Performance Comparison"
    )


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def feature_importance_chart(
    model,
    X_test,
    y_test,
    top_n=10
):

    if model is None or X_test is None or y_test is None:
        return empty_chart()

    if len(X_test) == 0:
        return empty_chart()

    try:

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="r2",
            n_jobs=1
        )

        importance = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": result.importances_mean
        })

        importance = importance.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        importance = importance.sort_values(
            "Importance",
            ascending=True
        ).tail(top_n)

        if importance.empty:
            return empty_chart()

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        return apply_chart_theme(
            fig,
            "Feature Importance",
            height=450
        )

    except Exception:
        return empty_chart(
            "Feature importance could not be calculated."
        )


# =====================================================
# ATTENDANCE VS PERFORMANCE
# =====================================================

def attendance_performance_chart(df):

    data = _prepare_chart_data(
        df,
        ["attendance_percentage", TARGET_COLUMN]
    )

    if data.empty:
        return empty_chart()

    fig = px.scatter(
        data,
        x="attendance_percentage",
        y=TARGET_COLUMN,
        color=TARGET_COLUMN,
        color_continuous_scale="Viridis",
        opacity=0.8,
        labels={
            "attendance_percentage": "Attendance Percentage",
            TARGET_COLUMN: "Final Score"
        }
    )

    return apply_chart_theme(
        fig,
        "Attendance vs Academic Performance"
    )


# =====================================================
# PERFORMANCE CATEGORY DISTRIBUTION
# =====================================================

def performance_category_chart(df):

    data = _prepare_chart_data(
        df,
        [TARGET_COLUMN]
    )

    if data.empty:
        return empty_chart()

    data["Performance Category"] = pd.cut(
        data[TARGET_COLUMN],
        bins=[-np.inf, 60, 80, np.inf],
        labels=[
            "Needs Support",
            "Moderate",
            "High"
        ],
        right=False
    )

    category_counts = (
        data["Performance Category"]
        .value_counts()
        .reindex(
            ["Needs Support", "Moderate", "High"],
            fill_value=0
        )
        .reset_index()
    )

    category_counts.columns = [
        "Performance Category",
        "Students"
    ]

    fig = px.pie(
        category_counts,
        names="Performance Category",
        values="Students",
        hole=0.55,
        color="Performance Category",
        color_discrete_map={
            "Needs Support": COLORS["red"],
            "Moderate": COLORS["orange"],
            "High": COLORS["green"]
        }
    )

    return apply_chart_theme(
        fig,
        "Student Performance Categories"
    )


# =====================================================
# PERFORMANCE STATISTICS
# =====================================================

def get_performance_statistics(df):

    data = _prepare_chart_data(
        df,
        [TARGET_COLUMN]
    )

    if data.empty:
        return {
            "average_score": 0.0,
            "median_score": 0.0,
            "highest_score": 0.0,
            "lowest_score": 0.0,
            "standard_deviation": 0.0,
            "total_students": 0,
            "high_performers": 0,
            "moderate_performers": 0,
            "students_needing_support": 0
        }

    scores = data[TARGET_COLUMN]

    return {
        "average_score": round(float(scores.mean()), 2),
        "median_score": round(float(scores.median()), 2),
        "highest_score": round(float(scores.max()), 2),
        "lowest_score": round(float(scores.min()), 2),
        "standard_deviation": round(float(scores.std()), 2),
        "total_students": int(len(scores)),
        "high_performers": int((scores >= 80).sum()),
        "moderate_performers": int(
            ((scores >= 60) & (scores < 80)).sum()
        ),
        "students_needing_support": int((scores < 60).sum())
    }

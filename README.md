# 🎓 Student Performance AI Studio

An interactive Machine Learning application that analyzes student academic data, discovers performance patterns, and predicts final examination scores using regression algorithms.

## 📌 Project Overview

Student Performance AI Studio is a data-driven application developed using Python and Machine Learning.

The platform allows users to explore student datasets, analyze academic performance, train and compare machine learning models, and generate predictions based on student-related parameters.

The project demonstrates an end-to-end Machine Learning workflow, from data preprocessing to model evaluation and prediction.

## ✨ Key Features

### 📂 1. Data Explorer
- Upload and explore student datasets.
- View dataset structure and statistical summaries.
- Identify missing values and duplicate records.
- Analyze numerical and categorical features.
- Download dataset information.

### 📊 2. Analytics Studio
- Student performance distribution.
- Academic performance categories.
- Study hours and performance analysis.
- Attendance and academic score relationships.
- Correlation analysis.
- Interactive visualizations.

### 🎯 3. Prediction Lab
- Enter student academic information.
- Predict final examination scores.
- Generate performance categories.
- View personalized academic improvement suggestions.
- Maintain prediction history.
- Export prediction history as CSV.

### 🧠 4. Model Intelligence
- Train multiple regression algorithms.
- Compare model performance.
- Evaluate models using R², MAE, and RMSE.
- Visualize actual vs predicted scores.
- Analyze feature importance.
- Save the best-performing trained model.

## 🤖 Machine Learning Algorithms

The application implements three regression algorithms:

| Algorithm | Purpose |
|---|---|
| Linear Regression | Establishes a linear relationship between features and final scores |
| Decision Tree Regressor | Learns decision-based patterns from student data |
| Random Forest Regressor | Combines multiple decision trees for prediction |

The best model is selected using cross-validation performance.

## 🛠️ Technology Stack

**Programming Language**
- Python

**Data Processing**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- Joblib

**Visualization**
- Plotly
- Matplotlib
- Seaborn

**Application Interface**
- Streamlit

## 📁 Project Structure

```text
student_performance_prediction/
│
├── data/
│   └── student_performance.csv
│
├── models/
│   └── best_model.joblib
│
├── outputs/
│   ├── metrics.json
│   └── comparison.csv
│
├── pages/
│   ├── analytics.py
│   ├── data_explorer.py
│   ├── prediction_lab.py
│   └── model_intelligence.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── visualizations.py
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── run.bat
```

*Note: The models and outputs directories are populated when model training is performed.*

## 📊 Dataset Information

The project uses a student performance dataset containing academic, behavioral, and learning-environment information.

The dataset includes the following features:

| Feature | Description |
|---|---|
| study_hours_per_day | Daily study duration |
| attendance_percentage | Student attendance percentage |
| previous_score | Previous examination score |
| assignment_average | Average assignment performance |
| sleep_hours | Daily sleep duration |
| class_participation | Classroom participation level |
| internet_access | Availability of internet access |
| extracurricular_activities | Participation in extracurricular activities |
| academic_support | Level of academic support |
| final_score | Final examination score (target variable) |

**Target Variable:** `final_score`

## ⚙️ Machine Learning Workflow

The project follows these major steps:

1. Dataset loading and inspection.
2. Data cleaning and preprocessing.
3. Handling missing values and duplicate records.
4. Exploratory Data Analysis (EDA).
5. Feature and target separation.
6. Train-test splitting.
7. Feature transformation and encoding.
8. Training multiple regression models.
9. Cross-validation and model evaluation.
10. Best-model selection and model saving.
11. Student performance prediction.
12. Visualization and reporting.

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project

Open the project folder in Visual Studio Code.

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the Application

Open `main.py` in Visual Studio Code and click the **Run Python File ▶** button.

The application launcher starts the Streamlit application automatically.

You do not need to manually enter the Streamlit launch command.

## 📈 Model Evaluation

The application evaluates models using:

- **R² Score:** Measures how well the model explains variations in the target variable.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving larger errors greater weight.

The Model Intelligence module displays the evaluation results and compares the trained algorithms.

## 📸 Application Modules

The application includes four dedicated modules:

- Data Explorer
- Analytics Studio
- Prediction Lab
- Model Intelligence

Add application screenshots here as the final UI is completed.

## ⚠️ Limitations

- Predictions depend on the quality and characteristics of the training dataset.
- The dataset used in this project is synthetic and intended for educational demonstration.
- Model performance on synthetic data does not establish real-world predictive accuracy.
- Predictions should not be treated as guaranteed academic outcomes.

## 🔮 Future Enhancements

- Integration with larger real-world academic datasets.
- Advanced regression algorithms and hyperparameter optimization.
- Student performance trend forecasting.
- More detailed explainable AI visualizations.
- User authentication and role-based dashboards.
- Database integration for persistent prediction history.

## 🎯 Project Objective

The primary objective is to demonstrate how Machine Learning can be used to analyze academic data, identify performance patterns, and build an interactive student performance prediction system.

## 👨‍💻 Author

**Developed as a Machine Learning Project**

## 📄 License

This project is intended for educational and learning purposes. Add a suitable open-source license if you plan to distribute it publicly.
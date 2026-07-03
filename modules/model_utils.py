import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR   = BASE_DIR / 'data'

# ── Load models once at startup (not per-request) ──────────────────────────────
logreg = joblib.load(MODELS_DIR / 'logreg.pkl')
rf     = joblib.load(MODELS_DIR / 'random_forest.pkl')
nn     = tf.keras.models.load_model(MODELS_DIR / 'neural_network.keras')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

# ── Load feature column names from saved CSVs ─────────────────────────────────
X_test_scaled   = pd.read_csv(DATA_DIR / 'X_test_scaled.csv')
X_test_unscaled = pd.read_csv(DATA_DIR / 'X_test_unscaled.csv')
y_test          = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()

SCALED_FEATURES   = X_test_scaled.columns.tolist()
UNSCALED_FEATURES = X_test_unscaled.columns.tolist()

# ── Categorical columns — must match 01_data_prep exactly ─────────────────────
CATEGORICAL_COLS = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

# ── Valid category values (must match training data strings exactly) ───────────
CATEGORY_VALUES = {
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'Department':     ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing',
                       'Medical', 'Other', 'Technical Degree'],
    'Gender':         ['Female', 'Male'],
    'JobRole':        ['Healthcare Representative', 'Human Resources',
                       'Laboratory Technician', 'Manager',
                       'Manufacturing Director', 'Research Director',
                       'Research Scientist', 'Sales Executive',
                       'Sales Representative'],
    'MaritalStatus':  ['Divorced', 'Married', 'Single'],
    'OverTime':       ['No', 'Yes'],
}


def preprocess_input(form_data):
    """
    Convert raw HTML form values into scaled and unscaled DataFrames
    that match the exact column structure the models were trained on.
    Returns: (df_scaled, df_unscaled)
    """
    raw = {
        'Age':                      int(form_data.get('age', 35)),
        'BusinessTravel':           form_data.get('business_travel', 'Travel_Rarely'),
        'DailyRate':                int(form_data.get('daily_rate', 800)),
        'Department':               form_data.get('department', 'Research & Development'),
        'DistanceFromHome':         int(form_data.get('distance_from_home', 5)),
        'Education':                int(form_data.get('education', 3)),
        'EducationField':           form_data.get('education_field', 'Life Sciences'),
        'EnvironmentSatisfaction':  int(form_data.get('environment_satisfaction', 3)),
        'Gender':                   form_data.get('gender', 'Male'),
        'HourlyRate':               int(form_data.get('hourly_rate', 66)),
        'JobInvolvement':           int(form_data.get('job_involvement', 3)),
        'JobLevel':                 int(form_data.get('job_level', 2)),
        'JobRole':                  form_data.get('job_role', 'Research Scientist'),
        'JobSatisfaction':          int(form_data.get('job_satisfaction', 3)),
        'MaritalStatus':            form_data.get('marital_status', 'Married'),
        'MonthlyIncome':            int(form_data.get('monthly_income', 5000)),
        'MonthlyRate':              int(form_data.get('monthly_rate', 14000)),
        'NumCompaniesWorked':       int(form_data.get('num_companies_worked', 2)),
        'OverTime':                 form_data.get('overtime', 'No'),
        'PercentSalaryHike':        int(form_data.get('percent_salary_hike', 13)),
        'PerformanceRating':        int(form_data.get('performance_rating', 3)),
        'RelationshipSatisfaction': int(form_data.get('relationship_satisfaction', 3)),
        'StockOptionLevel':         int(form_data.get('stock_option_level', 1)),
        'TotalWorkingYears':        int(form_data.get('total_working_years', 8)),
        'TrainingTimesLastYear':    int(form_data.get('training_times_last_year', 3)),
        'WorkLifeBalance':          int(form_data.get('work_life_balance', 3)),
        'YearsAtCompany':           int(form_data.get('years_at_company', 5)),
        'YearsInCurrentRole':       int(form_data.get('years_in_current_role', 3)),
        'YearsSinceLastPromotion':  int(form_data.get('years_since_last_promotion', 1)),
        'YearsWithCurrManager':     int(form_data.get('years_with_curr_manager', 3)),
    }

    df = pd.DataFrame([raw])

    # One-hot encode — identical to training pipeline
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align columns with training data exactly (fills unseen dummies with 0)
    df_unscaled = df_encoded.reindex(columns=UNSCALED_FEATURES, fill_value=0)

    # Scale for LogReg and NN (RF uses unscaled)
    scaled_arr = scaler.transform(df_unscaled)
    df_scaled  = pd.DataFrame(scaled_arr, columns=SCALED_FEATURES)

    return df_scaled, df_unscaled


def predict(model_name, form_data):
    """
    Run prediction for a given model and form inputs.
    Returns: (risk_probability_float, df_scaled, df_unscaled)
    """
    df_scaled, df_unscaled = preprocess_input(form_data)

    if model_name == 'Logistic Regression':
        proba = logreg.predict_proba(df_scaled)[0][1]
    elif model_name == 'Random Forest':
        proba = rf.predict_proba(df_unscaled)[0][1]
    elif model_name == 'Neural Network':
        proba = float(nn.predict(df_scaled, verbose=0)[0][0])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return float(proba), df_scaled, df_unscaled
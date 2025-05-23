import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import streamlit as st

MODEL_DIR = "models"
SCORED_PATH = "model_scored.csv"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
TRAINED_COLUMNS_PATH = os.path.join(MODEL_DIR, "trained_columns.pkl")
DATA_PATH = "AI-data-cl.csv"

dtype_spec = {
    'time': 'string',
    'client_ip': 'string', 'country': 'category', 'method': 'category',
    'uri_stem': 'category', 'status_code': 'int16', 'user_agent': 'string',
    'response_time': 'float32', 'bytes_transferred': 'int32',
    'device_type': 'category', 'os': 'category', 'screen_resolution': 'string',
    'browser_version': 'string', 'network_type': 'category',
    'sales_rep': 'category', 'product': 'category', 'cost': 'float64',
    'deal_size': 'category', 'revenue_impact': 'float64',
    'customer_lifetime_value': 'float64', 'time_to_convert': 'float32',
    'job_type': 'category', 'is_job_submission': 'boolean',
    'is_demo_request': 'boolean', 'is_assistant_request': 'boolean',
    'is_contact': 'boolean', 'conversion': 'boolean', 'bounce_rate': 'boolean',
    'data_complete': 'boolean', 'customer_type': 'category', 'industry': 'category',
    'company_size': 'category', 'age_group': 'category',
    'preferred_contact_method': 'category', 'session_duration': 'int32',
    'pages_per_visit': 'int16', 'lead_source': 'category',
    'funnel_stage': 'category', 'ai_interaction_depth': 'float32',
    'ai_success_rate': 'float32', 'ai_query_type': 'category',
    'feedback_rating': 'Int8'
}

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH, dtype=dtype_spec, parse_dates=["timestamp", "date"])

@st.cache_data(show_spinner=False)
def model_load_data():
    return pd.read_csv('model_use_data.csv', dtype=dtype_spec, parse_dates=["timestamp", "date"])

def train_and_score_models():
    df = load_data()
    df['conversion'] = df['conversion'].astype(int)
    df['feedback_rating'] = pd.to_numeric(df['feedback_rating'], errors='coerce')
    df['bounce_rate'] = pd.to_numeric(df['bounce_rate'], errors='coerce')

    numeric_features = ['session_duration', 'pages_per_visit', 'ai_success_rate',
                        'bounce_rate', 'feedback_rating', 'time_to_convert']
    categorical_features = ['sales_rep','lead_source', 'device_type', 'customer_type', 'funnel_stage']
    all_features = numeric_features + categorical_features

    df = df[all_features + ['conversion']].dropna()
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    X = df.drop(columns='conversion')
    y = df['conversion']

    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Save scaler and columns
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns.tolist(), TRAINED_COLUMNS_PATH)

    # Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    # ---------- Random Forest ----------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X)[:, 1]

    joblib.dump(rf, RF_MODEL_PATH)

    print("=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
    print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

    # ---------- XGBoost ----------
    xgb = XGBClassifier(
        use_label_encoder=False,
        learning_rate=0.05,
        n_estimators=1000,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X)[:, 1]

    joblib.dump(xgb, XGB_MODEL_PATH)

    print("=== XGBoost ===")
    print("Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))
    print("ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

    # ---------- Score and Save ----------
    df['conversion_proba'] = xgb_probs
    df.to_csv(SCORED_PATH, index=False)

    # Ensure all dashboard-critical columns are preserved in df_raw
    df_scored = load_data()

    # Drop rows missing model features
    df_features = df_scored.dropna()
    df_features = pd.get_dummies(df_features[numeric_features + categorical_features], columns=categorical_features, drop_first=True)

    # Align columns
    model_cols = joblib.load(TRAINED_COLUMNS_PATH)
    for col in model_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[model_cols]

    # Apply model prediction
    df_scored = df_scored.loc[df_features.index]  # match index
    df_scored['conversion_proba'] = xgb.predict_proba(df_features)[:, 1]

    # Save full structure + scored output
    df_scored.to_csv(SCORED_PATH, index=False)
    return df_scored, rf, xgb


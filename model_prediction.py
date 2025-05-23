import streamlit as st
import pandas as pd
import plotly.express as px

def plot_feature_importance(model, features):
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(
        importance_df,
        x='Importance', y='Feature',
        orientation='h',
        title='🔍 Feature Importance for Conversion Prediction'
    )
    fig.update_layout(yaxis=dict(categoryorder='total ascending'))
    return fig

def filter_by_prediction_threshold(df, threshold=0.6):
    if 'conversion_proba' not in df.columns:
        raise ValueError("conversion_proba column not found. Ensure model has been applied.")
    return df[df['conversion_proba'] >= threshold]

def assign_score_band(prob):
    if prob >= 0.75:
        return "🔥 Hot"
    elif prob >= 0.5:
        return "🌤 Warm"
    else:
        return "❄️ Cold"

def add_score_bands(prob):
    if prob >= 0.9:
        return 'Very High'
    elif prob >= 0.7:
        return 'High'
    elif prob >= 0.5:
        return 'Medium'
    elif prob >= 0.3:
        return 'Low'
    else:
        return 'Very Low'


def render_lead_scoring_table(df, threshold=0.6):
    st.markdown("### 🎯 Filter Leads by Confidence")
    confidence_threshold = st.slider("Set prediction threshold:", 0.0, 1.0, threshold, 0.05)
    top_leads = df[df['conversion_proba'] >= confidence_threshold]
    st.dataframe(top_leads[[  # Adjust columns as needed
        'sales_rep', 'product', 'industry', 'session_duration',
        'conversion_proba', 'conversion'
    ]].sort_values(by='conversion_proba', ascending=False).head(20))


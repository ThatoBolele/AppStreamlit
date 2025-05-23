import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, roc_curve
)

def render_statistical_summary_tab(df): 

    tab1, tab2 = st.tabs(["Summary Dashboard", "Model Accuracy & Evaluation"])

    with tab1:

        # ---------- SUMMARY METRICS ----------
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())
            col4.metric("Duplicate Rows", df.duplicated().sum())

        # ---------- PRIMARY CHARTS ----------
        with st.container():
            co_1, co_2, co_3 = st.columns(3, border=True)

            # --- Chart 1: Top Industries ---
            with co_1:
                st.markdown("""
                    <div style="font-size:15px;">
                        <strong>Top 10 Industries by Lead Volume</strong>
                    </div>
                """, unsafe_allow_html=True)

                if 'industry' in df.columns:
                    industry_dist = df['industry'].value_counts().reset_index()
                    industry_dist.columns = ['Industry', 'Count']
                    fig_industry = px.bar(
                        industry_dist.head(10),
                        x='Industry', y='Count', color='Count',
                        color_continuous_scale='Blues', height=120
                    )
                    fig_industry.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_industry, use_container_width=True)
                else:
                    st.info("No industry column found.")

            # --- Chart 2: Session Duration Distribution ---
            with co_2:
                st.markdown("""
                    <div style="font-size:15px;">
                        <strong>Session Duration Distribution – Converted vs Not Converted</strong>
                    </div>
                """, unsafe_allow_html=True)

                if 'session_duration' in df.columns and 'conversion' in df.columns:
                    fig_sd = px.histogram(
                        df,
                        x='session_duration',
                        color='conversion',
                        barmode='overlay',
                        nbins=30,
                        height=120
                    )
                    fig_sd.update_layout(
                        xaxis_title="Session Duration (seconds)",
                        yaxis_title="Frequency",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_sd, use_container_width=True)
                else:
                    st.info("Missing columns to generate session histogram.")

            # --- Chart 3: Correlation Matrix ---
            with co_3:
                st.markdown("""
                    <div style="font-size:15px;">
                        🧬 <strong>Numeric Correlation Matrix</strong>
                    </div>
                """, unsafe_allow_html=True)

                numeric_cols = df.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    corr = numeric_cols.corr().round(2)
                    fig_corr = px.imshow(
                        corr, text_auto=True, color_continuous_scale='RdBu_r', height=120
                    )
                    fig_corr.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("No numeric columns available for correlation.")

        # ---------- DESCRIPTIVE STATISTICS ----------
        with st.container():
            with st.expander("📐 Descriptive Statistics (Numeric Variables)", expanded=False):
                numeric_df = df.select_dtypes(include='number')
                if numeric_df.empty:
                    st.info("No numeric data available for summary statistics.")
                else:
                    desc_df = numeric_df.describe().transpose().reset_index()
                    desc_df.columns = ['Column', 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
                    st.dataframe(desc_df.style.format({
                        'Mean': '{:.2f}', 'Std Dev': '{:.2f}',
                        'Min': '{:.2f}', '25%': '{:.2f}', 'Median': '{:.2f}',
                        '75%': '{:.2f}', 'Max': '{:.2f}'
                    }), use_container_width=True)

        # ---------- MISSING VALUES + OUTLIERS ----------
        with st.container():
            col1, col2 = st.columns((4, 6))

            # --- Missing Values ---
            with col1:
                with st.container(border=True):
                    st.markdown("""
                        <div style="font-size:15px;">
                            <strong>Missing Values per Column</strong>
                        </div>
                    """, unsafe_allow_html=True)

                    missing = df.isnull().sum()
                    missing = missing[missing > 0].sort_values(ascending=False)
                    if not missing.empty:
                        missing_df = missing.reset_index()
                        missing_df.columns = ['Column', 'Missing Count']
                        fig_missing = px.bar(
                            missing_df, x='Missing Count', y='Column',
                            orientation='h', color='Missing Count',
                            color_continuous_scale='Reds', height=120
                        )
                        fig_missing.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=0, b=0)
                        )
                        st.plotly_chart(fig_missing, use_container_width=True)
                    else:
                        st.success("✅ No missing values found in the dataset.")

            # --- Outlier Detection (2 columns) ---
            with col2:
                col_1, col_2 = st.columns(2, border=True)

                with col_1:
                    st.markdown("""
                        <div style="font-size:15px;">
                            🔍 <strong>Outlier Detection: Revenue Impact</strong>
                        </div>
                    """, unsafe_allow_html=True)

                    if 'revenue_impact' in df.columns:
                        fig = px.box(df, y='revenue_impact', height=120)
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Column 'revenue_impact' not found.")

                with col_2:
                    st.markdown("""
                        <div style="font-size:15px;">
                            🔍 <strong>Outlier Detection: Session Duration</strong>
                        </div>
                    """, unsafe_allow_html=True)

                    if 'session_duration' in df.columns:
                        fig = px.box(df, y='session_duration', height=120)
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Column 'session_duration' not found.")


    with tab2:

        # Prepare data
        required = ['conversion', 'session_duration', 'pages_per_visit', 'ai_success_rate', 'bounce_rate', 'feedback_rating']
        df_model = df.dropna(subset=required).copy()
        df_model['conversion'] = df_model['conversion'].astype(int)

        X = df_model[['session_duration', 'pages_per_visit', 'ai_success_rate', 'bounce_rate', 'feedback_rating']]
        y = df_model['conversion']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)

        # Top row: message + metrics
        top_left, top_right = st.columns((6, 4))

        with top_left:
            st.markdown("""
                <div style="font-size:14px;">
                    ✅ <strong>Model Evaluation Key:</strong><br>
                    This model is considered <strong>acceptable</strong> if accuracy exceeds <em>70%</em> and ROC AUC surpasses <em>0.75</em>.
                    These thresholds ensure the model is better than random and reliable for lead prioritisation.<br><br>
                    Predictions are based on engagement, feedback, and session behaviour — perfect for forecasting conversions.
                </div>
            """, unsafe_allow_html=True)

        with top_right:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("ROC AUC", f"{auc:.2f}")

        # Bottom row: prediction plot + threshold view
        bottom_left, bottom_right = st.columns(2)

        # --- Line chart: prediction vs actual
        with bottom_left:
            st.markdown("#### Prediction vs Actual (sorted by probability)")

            vis_df = pd.DataFrame({
                'Predicted Probability': y_probs,
                'Actual Conversion': y_test.reset_index(drop=True)
            }).sort_values(by='Predicted Probability').reset_index(drop=True)

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                y=vis_df['Predicted Probability'],
                mode='lines',
                name='Prediction',
                line=dict(color='blue')
            ))
            fig_line.add_trace(go.Scatter(
                y=vis_df['Actual Conversion'],
                mode='lines',
                name='Actual',
                line=dict(color='green', dash='dot')
            ))

            fig_line.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title="Probability / Label",
                xaxis_title="Sorted Observations"
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # --- Chart 2: Optional – threshold sensitivity
        with bottom_right:
            st.markdown("#### Prediction Probability Distribution")

            prob_df = pd.DataFrame({'Proba': y_probs})
            fig_dist = px.histogram(prob_df, x='Proba', nbins=30, title="", height=220)
            fig_dist.update_layout(
                xaxis_title="Predicted Probability",
                yaxis_title="Count",
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                session_duration = st.number_input("Session Duration (seconds)", min_value=0, value=300)
                feedback_rating = st.slider("Feedback Rating", min_value=1, max_value=5, value=3)
            with col2:
                pages_per_visit = st.slider("Pages per Visit", min_value=1, max_value=20, value=5)
                bounce_rate = st.slider("Bounce Rate (%)", min_value=0, max_value=100, value=35)
            with col3:
                ai_success_rate = st.slider("AI Success Rate (%)", min_value=0, max_value=100, value=70)

            submitted = st.form_submit_button("🔍 Predict Conversion")

        if submitted:
            # Prepare input for prediction
            input_data = pd.DataFrame([[
                session_duration,
                pages_per_visit,
                ai_success_rate / 100,  # Convert to 0–1 scale if needed
                bounce_rate / 100,      # Convert to 0–1 scale if needed
                feedback_rating
            ]], columns=['session_duration', 'pages_per_visit', 'ai_success_rate', 'bounce_rate', 'feedback_rating'])

            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.success(f"🔍 Prediction: {'Will Convert ✅' if prediction else 'Will Not Convert ❌'}")
            st.metric("Predicted Conversion Probability", f"{probability:.2%}")

        


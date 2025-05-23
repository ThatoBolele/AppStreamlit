import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from model_prediction import add_score_bands
from model_ai import train_and_score_models

def format_number(num):
        if num >= 1_000_000:
            if num % 1_000_000 == 0:
                return f'{int(num // 1_000_000)} M'
            return f'{round(num / 1_000_000, 1)} M'
        elif num >= 1_000:
            if num % 1_000 == 0:
                return f'{int(num // 1_000)} K'
            return f'{round(num / 1_000, 1)} K'
        else:
            return f'{round(num)}'

def render_executive_tab(df_general):

    col_1, col_2 = st.columns((4.5, 5.5))

    with col_1:
        co_1, co_2 =st.columns((7,3))

        with co_1:
            with st.container(border=True):
                # Use conversion rate as health proxy
                total_visits = len(df_general)
                total_conversions = df_general['conversion'].sum()
                conversion_rate = (total_conversions / total_visits) * 100 if total_visits > 0 else 0
                health_score = conversion_rate

                # Gauge Chart with Embedded Threshold Labels
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Health Score (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red", 'name': 'Poor'},
                            {'range': [50, 75], 'color': "yellow", 'name': 'OK'},
                            {'range': [75, 100], 'color': "green", 'name': 'Good'},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 6},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))

                # Add custom annotations (keys) to chart layout
                fig.update_layout(
                    height=180,
                    margin=dict(l=0, r=0, t=50, b=10),
                    annotations=[
                        dict(x=0.25, y=0.0, text="<b style='color:red; font-size:8px;'>Poor<br>0–49%</b>", showarrow=False),
                        dict(x=0.45, y=0.0, text="<b style='color:orange; font-size:8px;'>OK<br>50–74%</b>", showarrow=False),
                        dict(x=0.57, y=0.0, text="<b style='color:green; font-size:8px;'>Good<br>75–100%</b>", showarrow=False),
                        dict(x=0.77, y=0.0, text="<b style='color:blue; font-size:8px;'>Current<br>Performance</b>", showarrow=False)
                    ]
                )

                st.plotly_chart(fig, use_container_width=True)

        with co_2:

            with st.container(border=True):
                    current_revenue = df_general['revenue_impact'].sum()
                    previous_revenue = current_revenue * 0.85  # example previous
                    delta_revenue = current_revenue - previous_revenue
                    st.metric("Total Revenue",  f"P{format_number(current_revenue)}", f"P{format_number(delta_revenue)}")

        with st.container():
            col1_1, col1_2 = st.columns((3.2,6.8))

            with col1_1:

                
                    
                with st.container(border=True):
                    # Conversion Rate Metric
                    total_visits = len(df_general)
                    total_conversions = df_general['conversion'].sum()
                    conversion_rate = (total_conversions / total_visits) * 100 if total_visits > 0 else 0
                    previous_conversion_rate = conversion_rate * 0.9
                    st.metric("Conversion Rate", f"{conversion_rate:.1f}%", f"{conversion_rate - previous_conversion_rate:.1f}%")

                with st.container(border=True):
                    # Rep Performance Metric (Average Revenue per Rep)
                    rep_performance = df_general.groupby('sales_rep')['revenue_impact'].sum().mean()
                    prev_rep_performance = rep_performance * 0.88
                    delta = rep_performance - prev_rep_performance
                    st.metric("Rep Performance", f"P{format_number(rep_performance)}", f"P{format_number(delta)}")

            with col1_2:
                with st.container(border=True):
                    st.markdown("""
                        <div style="font-size:13px;">
                            <strong>💼 Rep Effectiveness:</strong> Predicted Conversion vs Lead Source Handled
                        </div>
                        """, unsafe_allow_html=True)

                    # Load components
                    model = joblib.load("models/xgboost_model.pkl")
                    scaler = joblib.load("models/scaler.pkl")
                    trained_cols = joblib.load("models/trained_columns.pkl")

                    # Features
                    numeric_features = ['session_duration', 'pages_per_visit', 'ai_success_rate',
                                        'bounce_rate', 'feedback_rating', 'time_to_convert']
                    categorical_features = ['lead_source', 'device_type', 'customer_type', 'funnel_stage']
                    all_features = numeric_features + categorical_features

                    # Drop NA rows required for prediction
                    df_model = df_general.dropna(subset=all_features + ['sales_rep'])

                    # Encode + align
                    df_encoded = pd.get_dummies(df_model[all_features], drop_first=True)
                    for col in trained_cols:
                        if col not in df_encoded.columns:
                            df_encoded[col] = 0
                    df_encoded = df_encoded[trained_cols]
                    df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])

                    # Predict
                    df_model['conversion_proba'] = model.predict_proba(df_encoded)[:, 1]

                    # Plot logic
                    metric_field = 'lead_source'
                    y_title = 'Leads Handled'

                    # Ensure conversion column exists
                    if 'conversion' not in df_model.columns:
                        df_model['conversion'] = 1  # fallback count

                    # Group and plot
                    rep_summary = df_model.groupby('sales_rep').agg({
                        'conversion_proba': 'mean',
                        metric_field: 'count'
                    }).reset_index()
                    rep_summary.rename(columns={metric_field: y_title}, inplace=True)

                    fig = px.scatter(
                        rep_summary,
                        x='conversion_proba',
                        y=y_title,
                        size=y_title,
                        color='sales_rep',
                        labels={
                            'conversion_proba': 'Avg Predicted Conversion',
                            y_title: y_title
                        }
                    )
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250)
                    st.plotly_chart(fig, use_container_width=True)


    with col_2:
        with st.container(border=True):
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Revenue Trend & Rep Performance Matrix</strong>
                </div>
                """, unsafe_allow_html=True)
            

            df_trend = df_general.copy()
            df_trend['month'] = df_trend['date'].dt.strftime('%b %Y')

            trend_df = df_trend.groupby('month').agg({
                'revenue_impact': 'sum'
            }).reset_index().sort_values(by='month')

            trend_df['forecast'] = trend_df['revenue_impact'].rolling(window=2, min_periods=1).mean() * 1.05

            fig_trend = go.Figure()
            fig_trend.add_bar(x=trend_df['month'], y=trend_df['revenue_impact'], name='Actual Revenue')
            fig_trend.add_trace(go.Scatter(x=trend_df['month'], y=trend_df['forecast'], mode='lines+markers', name='Forecast'))

            fig_trend.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue (P)",
                barmode='group',
                height=160,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
                

        # Revenue by Region Map
        with st.container(border=True):
            if 'country_code' in df_general.columns:
                map_df = df_general.groupby('country_code')['revenue_impact'].sum().reset_index()
                choropleth = px.choropleth(
                    map_df,
                    locations="country_code",
                    locationmode="country names",
                    color="revenue_impact",
                    color_continuous_scale="Blues",
                    range_color=(0, max(map_df["revenue_impact"])),
                    scope="world",
                    labels={'revenue': 'Total Revenue'}
                )
                choropleth.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=260
                )
                st.plotly_chart(choropleth, use_container_width=True)
            else:
                st.info("Country code data is missing for map generation.")


def render_traffic_tab(df_general):

    # Ensure 'quarter' and 'year' exist
    df_general['quarter'] = df_general['date'].dt.to_period('Q').astype(str)
    df_general['year'] = df_general['date'].dt.year

    # ---------- CONTAINER 1 ----------
    with st.container():
        
        col11_1, col11_2, col11_3 =st.columns((2.5,3.8,3.7))

        with col11_1:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Traffic Summary by Country</strong>
                </div>
                """, unsafe_allow_html=True)
            
            traffic_df = df_general.groupby(['country_code']).agg(
                users=('client_ip', 'nunique')
            ).reset_index().sort_values(by='users', ascending=False)

            st.dataframe(traffic_df,
                        column_order=("country_code", "users"),
                        hide_index=True,
                        width=None,
                        column_config={
                            "country_code": st.column_config.TextColumn(
                                "Country",
                            ),
                            "users": st.column_config.ProgressColumn(
                                "Users",
                                format="%f",
                                min_value=0,
                                max_value=max(traffic_df.users),
                            )},
                            height=300
                        )
            
        with col11_2:
            with st.container(border=True):
                fig_map = px.choropleth(
                    traffic_df,
                    locations="country_code",
                    locationmode="country names",
                    color="users",
                    color_continuous_scale="Viridis"
                )

                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    coloraxis_colorbar=dict(
                        orientation="h",           # Horizontal orientation
                        yanchor="bottom",
                        y=-0.3,                    # Push below plot
                        xanchor="center",
                        x=0.5,                     # Center it
                        title="Users"             # Optional label
                    )
                )

                st.plotly_chart(fig_map, use_container_width=True)
            

        with col11_3:
            current_quarter = df_general['quarter'].mode().iloc[0]
            current_df = df_general[df_general['quarter'] == current_quarter]

            quarters_sorted = sorted(df_general['quarter'].unique())
            current_index = quarters_sorted.index(current_quarter)
            if current_index > 0:
                previous_quarter = quarters_sorted[current_index - 1]
                previous_df = df_general[df_general['quarter'] == previous_quarter]
            else:
                previous_df = pd.DataFrame()

            curr_users = current_df['client_ip'].nunique()
            prev_users = previous_df['client_ip'].nunique() if not previous_df.empty else 0
            delta_users = curr_users - prev_users

            curr_sessions = current_df['session_duration'].count()
            prev_sessions = previous_df['session_duration'].count() if not previous_df.empty else 0
            delta_sessions = curr_sessions - prev_sessions

            with st.container():
                col_1, col_2 =st.columns(2, border=True)

                with col_1:
                    st.markdown(f"""
                        <div style="font-size:12px;">
                            <strong>Weekly Unique Users - {current_quarter}"</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    # Filter for the selected quarter
                    quarter_df = df_general[df_general['quarter'] == current_quarter].copy()

                    # Get week ending (Sunday) for each date
                    quarter_df['week_ending'] = quarter_df['date'] + pd.to_timedelta(6 - quarter_df['date'].dt.weekday, unit='d')
                    weekly_users = quarter_df.groupby('week_ending')['client_ip'].nunique().reset_index()
                    weekly_users.columns = ['Week Ending', 'Users']
                    

                    fig_week_users = px.bar(
                        weekly_users,
                        x='Week Ending',
                        y='Users',
                        labels={'Week Ending': 'Week Ending Date'}
                    )
                    fig_week_users.update_layout(xaxis_tickformat='%d %b', 
                                                height=150,
                                                plot_bgcolor='rgba(0, 0, 0, 0)',
                                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                                margin=dict(l=0, r=0, t=0, b=0))

                    st.plotly_chart(fig_week_users, use_container_width=True)

                with col_2:
                    st.markdown(f"""
                        <div style="font-size:12px;">
                            <strong>Weekly Sissions - {current_quarter}"</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    weekly_sessions = quarter_df.groupby('week_ending')['session_duration'].count().reset_index()
                    weekly_sessions.columns = ['Week Ending', 'Sessions']

                    fig_week_sessions = px.bar(
                        weekly_sessions,
                        x='Week Ending',
                        y='Sessions',
                        labels={'Week Ending': 'Week Ending Date'}
                    )
                    fig_week_sessions.update_layout(xaxis_tickformat='%d %b', 
                                                    height=150,
                                                    plot_bgcolor='rgba(0, 0, 0, 0)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)',
                                                    margin=dict(l=0, r=0, t=0, b=0))

                    st.plotly_chart(fig_week_sessions, use_container_width=True)
            
            with st.container():
                
                co1, co2 =st.columns(2, border=True)

                with co1:
                    st.metric("Users (This Quarter)", format_number(curr_users), f"P{format_number(delta_users)}")

                with co2:
                    st.metric("Sessions (This Quarter)", format_number(curr_sessions), f"P{format_number(delta_sessions)}")

                    
                


    # ---------- CONTAINER 2 ----------
    with st.container():

        col1, col2, col3, col4 = st.columns(4, border=True)

        # 1. Engagement by Country
        with col1:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Avg Session Duration by Country</strong>
                </div>
                """, unsafe_allow_html=True)
            
            country_duration = df_general.groupby('country')['session_duration'].mean().reset_index()
            country_duration.columns = ['Country', 'Avg Duration (s)']
            country_duration = country_duration.sort_values(by='Avg Duration (s)', ascending=False).head(10)

            fig_duration = px.bar(
                country_duration,
                x='Country',
                y='Avg Duration (s)',
                height=150,
                text_auto='.2s'
            )
            fig_duration.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_duration, use_container_width=True)


        # 2. Device Type Donut
        with col2:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Devices Used</strong>
                </div>
                """, unsafe_allow_html=True)
            device_df = df_general['device_type'].value_counts().reset_index()
            device_df.columns = ['Device Type', 'Count']

            fig_device = px.pie(
                device_df,
                names='Device Type',
                values='Count',
                hole=0.5, 
                height=150
            )

            fig_device.update_traces(
                textposition='inside',
                textinfo='percent+label'  # Shows both name and percentage inside each wedge
            )

            fig_device.update_layout(
                showlegend=False,  # Hides the external legend
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig_device, use_container_width=True)



        # 3. Lead Source (Social Networks Proxy)
        with col3:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Traffic Sources</strong>
                </div>
                """, unsafe_allow_html=True)
            source_df = df_general['lead_source'].value_counts().reset_index()
            source_df.columns = ['Lead Source', 'Count']
            fig_source = px.pie(source_df, names='Lead Source', values='Count', hole=0.5, height=150)
            fig_source.update_traces(
                textposition='inside',
                textinfo='percent+label'  # Shows both name and percentage inside each wedge
            )

            fig_source.update_layout(
                showlegend=False,  # Hides the external legend
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=0, r=0, t=0, b=0)
            )

            st.plotly_chart(fig_source, use_container_width=True)

        # 4. Bounce Rate by Device Type
        with col4:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Bounce Rate by Device</strong>
                </div>
                """, unsafe_allow_html=True)
            
            bounce_device = df_general.groupby('device_type')['bounce_rate'].mean().reset_index()
            bounce_device.columns = ['Device Type', 'Avg Bounce Rate']
            fig_bounce = px.bar(
                bounce_device,
                x='Device Type',
                y='Avg Bounce Rate',
                height=150,
                text_auto='.1%',
                color='Avg Bounce Rate',
                color_continuous_scale='Reds'
            )
            fig_bounce.update_layout(
                yaxis_ticksuffix='%',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_bounce, use_container_width=True)





def render_sales_tab(df_sales, time_filter):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import joblib
    from model_prediction import add_score_bands, assign_score_band

    # ---------- Container 1 ----------
    with st.container():
        co1, co2, co3 = st.columns((4, 3, 3))

        # === LEFT: Product Conversion Forecast ===
        with co1:
            with st.container(border=True):
                st.markdown(f"""
                    <div style="font-size:15px;">
                        <strong>Predicted Conversion by Product with Forecast ({time_filter})</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Load model
                model = joblib.load("models/xgboost_model.pkl")
                scaler = joblib.load("models/scaler.pkl")
                trained_cols = joblib.load("models/trained_columns.pkl")

                numeric_features = ['session_duration', 'pages_per_visit', 'ai_success_rate',
                                    'bounce_rate', 'feedback_rating', 'time_to_convert']
                categorical_features = ['lead_source', 'device_type', 'customer_type', 'funnel_stage']
                all_features = numeric_features + categorical_features

                df = df_sales.dropna(subset=all_features + ['product', 'date']).copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                df_encoded = pd.get_dummies(df[all_features], drop_first=True)
                for col in trained_cols:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_encoded = df_encoded[trained_cols]
                df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])

                df['conversion_proba'] = model.predict_proba(df_encoded)[:, 1]

                grouped = df.groupby('product')['conversion_proba'].mean()
                expected_products = df['product'].dropna().unique().tolist()
                trend_df = grouped.reindex(expected_products).reset_index()
                trend_df.columns = ['product', 'conversion_proba']
                trend_df['conversion_proba'] = trend_df['conversion_proba'].fillna(0)
                trend_df['forecast'] = trend_df['conversion_proba'].rolling(window=2, min_periods=1).mean() * 1.05

                fig = go.Figure()
                fig.add_bar(
                    x=trend_df['conversion_proba'],
                    y=trend_df['product'],
                    name='Avg Predicted Conversion',
                    marker_color='teal',
                    orientation='h'
                )
                fig.add_trace(go.Scatter(
                    x=trend_df['forecast'],
                    y=trend_df['product'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='orange', dash='dot')
                ))
                fig.update_layout(
                    xaxis_title="Avg Conversion Probability",
                    yaxis_title="Product",
                    barmode='group',
                    height=200,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        # === CENTER: Revenue Summary Table ===
        with co2:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Individual Performance Overview</strong>
                </div>
                """, unsafe_allow_html=True)

            individual_perf = df_sales.groupby('product').agg({
                'revenue_impact': 'sum'
            }).reset_index().sort_values(by='revenue_impact', ascending=False)

            st.dataframe(
                individual_perf,
                column_order=("product", "revenue_impact"),
                hide_index=True,
                width=None,
                column_config={
                    "product": st.column_config.TextColumn("Product"),
                    "revenue_impact": st.column_config.ProgressColumn(
                        "Total Revenue", format="euro", min_value=0,
                        max_value=max(individual_perf.revenue_impact)
                    )
                }
            )

        # === RIGHT: Toggleable AI Score Table ===
        with co3:
            st.markdown("""
                <div style="font-size:15px;">
                    <strong>Top AI Scores by Segment</strong>
                </div>
                """, unsafe_allow_html=True)

            group_by = st.radio("Group by:", ["Industry", "Product"], horizontal=True)

            df = df_sales.dropna(subset=all_features + ['product', 'industry'])

            df_encoded = pd.get_dummies(df[all_features], drop_first=True)
            for col in trained_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[trained_cols]
            df_encoded[numeric_features] = scaler.transform(df_encoded[numeric_features])
            df['conversion_proba'] = model.predict_proba(df_encoded)[:, 1]
            df['score_band'] = df['conversion_proba'].apply(assign_score_band)

            high_conf = df[df['conversion_proba'] >= 0.6]

            if group_by == "Industry":
                summary = high_conf.groupby('industry').agg({
                    'conversion_proba': 'mean',
                    'session_duration': 'mean'
                }).reset_index().sort_values(by='conversion_proba', ascending=False).head(10)
            else:
                summary = high_conf.groupby('product').agg({
                    'conversion_proba': 'mean',
                    'session_duration': 'mean'
                }).reset_index().sort_values(by='conversion_proba', ascending=False).head(10)

            st.dataframe(summary.style.format({
                'conversion_proba': '{:.2%}',
                'session_duration': '{:.0f}'
            }), hide_index=True, height=150)

    # ---------- Container 2 ----------
    with st.container():
        col1, col2, col3 = st.columns((4.5, 1.5, 4))

        # Funnel Chart
        with col1:
            with st.container(border=True):
                st.markdown("""
                    <div style="font-size:15px;">
                        <strong>Lead Progression Through Funnel Stages</strong>
                    </div>
                    """, unsafe_allow_html=True)

                funnel_df = df_sales['funnel_stage'].value_counts().reset_index()
                funnel_df.columns = ['Stage', 'Count']
                funnel_df = funnel_df.sort_values(by='Stage', ascending=False)

                fig_funnel = go.Figure(go.Funnel(
                    y=funnel_df['Stage'],
                    x=funnel_df['Count'],
                    textinfo="value+percent initial"
                ))
                fig_funnel.update_layout(
                    height=200,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_funnel, use_container_width=True)

        # Metrics
        with col2:
            with st.container(border=True):
                total_deals = df_sales['conversion'].sum()
                prev_deals = total_deals * 0.9
                st.metric("Deals Closed", format_number(int(total_deals)), f"+{format_number(int(total_deals - prev_deals))}")

            with st.container(border=True):
                avg_revenue = df_sales['revenue_impact'].mean()
                benchmark = 1000
                delta = avg_revenue - benchmark
                st.metric("Avg Revenue per Deal", f"P{format_number(avg_revenue)}", f"P{format_number(delta)}")

        # Gauge
        with col3:
            with st.container(border=True):
                st.markdown("""
                    <div style="font-size:15px;">
                        <strong>Sales Effectiveness (%)</strong>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                    <div style="font-size:13px; text-align:center; margin-bottom: 6px;">
                        <em>Percentage of total leads that converted</em>
                    </div>
                    """, unsafe_allow_html=True)

                effectiveness = (df_sales['conversion'].sum() / len(df_sales)) * 100

                fig_speed = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=effectiveness,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 6},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                fig_speed.update_layout(
                    height=150,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=20, b=5)
                )
                st.plotly_chart(fig_speed, use_container_width=True)

                # Legend
                st.markdown("""
                    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 10px;">
                        <div><span style="display:inline-block; width:20px; height:20px; background-color:green; border-radius:3px;"></span> Good (75%+)</div>
                        <div><span style="display:inline-block; width:20px; height:20px; background-color:yellow; border-radius:3px;"></span> OK (50–74%)</div>
                        <div><span style="display:inline-block; width:20px; height:20px; background-color:red; border-radius:3px;"></span> Poor (<50%)</div>
                        <div><span style="display:inline-block; width:20px; height:20px; background-color:blue; border-radius:3px;"></span> Current Performance</div>
                            
                    </div>
                    """, unsafe_allow_html=True)

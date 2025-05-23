import streamlit as st
import pandas as pd
from charts import render_executive_tab, render_sales_tab, render_traffic_tab
from metrics import render_statistical_summary_tab
from model_ai import load_data, model_load_data
import pdfkit

path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)


def export_summary_to_pdf(df):
    summary_html = df.describe().to_html(classes="table table-striped", border=0)

    html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 30px; }}
                h1 {{ color: #2E86C1; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table td, .table th {{ border: 1px solid #ddd; padding: 8px; }}
            </style>
        </head>
        <body>
            <h1>Dataset Summary Report</h1>
            <h3>Total Rows: {df.shape[0]}</h3>
            <h3>Total Columns: {df.shape[1]}</h3>
            <h3>Missing Values: {df.isnull().sum().sum()}</h3>
            {summary_html}
        </body>
    </html>
    """

    pdf_file = "summary_report.pdf"
    pdfkit.from_string(html, pdf_file, configuration=config)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="⬇ Download Summary as PDF",
            data=f,
            file_name="summary_report.pdf",
            mime="application/pdf"
        )



# Page Setup
st.set_page_config(
    page_title="AI-Solutions Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
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

# df = load_data()
df = pd.read_csv("AI-data-cl.csv", dtype=dtype_spec, parse_dates=["timestamp", "date"])
# df = pd.read_csv("dira_jang.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['month'] = df['date'].dt.to_period('M').astype(str)


# Sidebar
with st.sidebar:
    st.title("AI-Solutions Dashboard")
    st.markdown("""
        Welcome to the AI-Solutions Analytics Dashboard.
    """)
    print(df.columns)

    st.download_button(
        label="⬇ Download Raw Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ai_solutions_data.csv",
        mime="text/csv"
    )

    # Near the end of the summary tab function
    with st.container():
        export_summary_to_pdf(df)


    selected_tab = st.radio(
    label="Navigate to Section:",
    options=["Executive", "Sales Team", "Traffic", "Statistical Summary"],
    index=0
    )
    


    # ---------- Filters ----------
    df_sales = df.copy()
    df_general = df.copy()

    if selected_tab == "Sales Team":
        # Sales Rep Filter
        sales_reps = df['sales_rep'].dropna().unique()
        selected_rep = st.selectbox("Select Sales Representative:", options=sales_reps)

        # Time Filter Type
        time_filter = st.selectbox("Select Time Period:", options=["Monthly", "Quarterly", "Yearly"])

        # Monthly
        if time_filter == "Monthly":
            df['month'] = pd.to_datetime(df['month'], errors='coerce')
            month_options = df['month'].dt.strftime('%B').unique()
            selected_month = st.selectbox("Select Month:", sorted(month_options, key=lambda x: pd.to_datetime(x, format='%B').month))
            df_sales = df_sales[df_sales['timestamp'].dt.strftime('%B') == selected_month]

        # Quarterly
        elif time_filter == "Quarterly":
            df_sales['quarter'] = df_sales['date'].dt.to_period('Q').astype(str)
            quarter_options = df_sales['quarter'].unique()
            selected_quarter = st.selectbox("Select Quarter:", sorted(quarter_options))
            df_sales = df_sales[df_sales['quarter'] == selected_quarter]

        # Yearly
        elif time_filter == "Yearly":
            year_options = df['date'].dt.year.unique()
            selected_year = st.selectbox("Select Year:", sorted(year_options))
            df_sales = df_sales[df_sales['date'].dt.year == selected_year]

        # Rep filter last to maintain logic
        df_sales = df_sales[df_sales['sales_rep'] == selected_rep]

    elif selected_tab in ["Executive", "Traffic"]:
        # Country & Customer Type Filters
        selected_countries = st.multiselect("Filter by Country", sorted(df['country'].dropna().unique()))
        if 'customer_type' in df.columns:
            selected_customer_types = st.multiselect("Filter by Customer Type", sorted(df['customer_type'].dropna().unique()))
        else:
            st.warning("'customer_type' column not found in the dataset.")

        # selected_customer_types = st.multiselect("Filter by Customer Type", sorted(df['customer_type'].dropna().unique()))

        # Quarter Filter
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')  # Ensure date is datetime
        df['quarter'] = df['date'].dt.to_period('Q').astype(str)
        quarter_options = sorted(df['quarter'].dropna().unique())
        selected_quarter = st.selectbox("Select Quarter:", options=quarter_options)

        # Apply Filters
        df_general = df[
            df['country'].isin(selected_countries if selected_countries else df['country'].unique()) &
            df['customer_type'].isin(selected_customer_types if selected_customer_types else df['customer_type'].unique()) &
            (df['quarter'] == selected_quarter)
        ]

       



# --------------- Dashboard Content ---------------
if selected_tab == "Executive":
    render_executive_tab(df_general)

elif selected_tab == "Sales Team":
    render_sales_tab(df_sales, time_filter)

elif selected_tab == "Traffic":
    render_traffic_tab(df_general)

elif selected_tab == "Statistical Summary":
    render_statistical_summary_tab(df_general)



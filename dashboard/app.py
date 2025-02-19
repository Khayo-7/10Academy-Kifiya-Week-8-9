from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from flask import Flask

# Initialize Flask app
server = Flask(__name__)

# Load data
fraud_data = pd.read_parquet("data/processed/fraud_data_processed.parquet")
credit_data = pd.read_parquet("data/processed/creditcard_processed.parquet")

# Initialize Dash app
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Fraud Detection Dashboard"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Summary Statistics"),
            html.Div(id="summary-stats")
        ], width=4),
        dbc.Col([
            html.H3("Fraud Cases Over Time"),
            dcc.Graph(id="fraud-over-time")
        ], width=8)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Fraud by Country"),
            dcc.Graph(id="fraud-by-country")
        ], width=6),
        dbc.Col([
            html.H3("Fraud by Device/Browser"),
            dcc.Graph(id="fraud-by-device-browser")
        ], width=6)
    ]),
    dcc.Interval(id="interval-component", interval=60*1000, n_intervals=0)  # Auto-refresh every 60 seconds
])

# Callbacks for dynamic updates
@app.callback(
    [Output("summary-stats", "children"),
     Output("fraud-over-time", "figure"),
     Output("fraud-by-country", "figure"),
     Output("fraud-by-device-browser", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    # Summary statistics
    total_transactions = len(fraud_data) + len(credit_data)
    fraud_cases = fraud_data["class"].sum() + credit_data["Class"].sum()
    fraud_percentage = (fraud_cases / total_transactions) * 100
    
    summary_stats = [
        html.P(f"Total Transactions: {total_transactions}"),
        html.P(f"Fraud Cases: {fraud_cases}"),
        html.P(f"Fraud Percentage: {fraud_percentage:.2f}%")
    ]
    
    # Fraud cases over time
    fraud_over_time = fraud_data.groupby(fraud_data["purchase_time"].dt.date)["class"].sum().reset_index()
    fraud_over_time_fig = px.line(fraud_over_time, x="purchase_time", y="class", 
                                    title="Fraud Cases Over Time", labels={"purchase_time": "Date", "class": "Fraud Cases"})
    
    # Fraud by country
    fraud_by_country = fraud_data.groupby("country")["class"].sum().reset_index()
    fraud_by_country_fig = px.choropleth(fraud_by_country, locations="country", locationmode="country names",
                                        color="class", title="Fraud by Country")
    
    # Fraud by device/browser
    fraud_by_device_browser = fraud_data.groupby(["device_id", "browser"])["class"].sum().reset_index()
    fraud_by_device_browser_fig = px.bar(fraud_by_device_browser, x="device_id", y="class", color="browser",
                                        title="Fraud by Device and Browser", barmode="group")
    
    return summary_stats, fraud_over_time_fig, fraud_by_country_fig, fraud_by_device_browser_fig

# Run the app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
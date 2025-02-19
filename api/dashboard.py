import dash
from dash import dcc, html
import requests
import plotly.express as px

# Integrate Dash with Flask
app_dash = dash.Dash(__name__, server=app, url_base_pathname="/dashboard/")

# Fetch data from Flask API
def fetch_data(endpoint):
    """Helper function to get data from Flask API"""
    url = f"http://localhost:5000/api/{endpoint}"
    return requests.get(url).json()

# Fetch initial data
summary = fetch_data("summary")
fraud_trends = fetch_data("fraud_trends")
fraud_by_location = fetch_data("fraud_by_location")
fraud_by_device = fetch_data("fraud_by_device")

# Convert fraud trends to DataFrame
df_trends = pd.DataFrame(list(fraud_trends.items()), columns=["Date", "Fraud Cases"])

# Convert fraud location data to DataFrame
df_location = pd.DataFrame(list(fraud_by_location.items()), columns=["Location", "Count"])

# Convert fraud by device data to DataFrame
df_device = pd.read_json(fraud_by_device)

# Layout
app_dash.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={"text-align": "center"}),

    # Summary Boxes
    html.Div([
        html.Div([html.H3("Total Transactions"), html.P(summary["total_transactions"])], className="summary-box"),
        html.Div([html.H3("Fraud Cases"), html.P(summary["fraud_cases"])], className="summary-box"),
        html.Div([html.H3("Fraud Percentage"), html.P(f"{summary['fraud_percentage']}%")], className="summary-box"),
    ], className="summary-container"),

    # Fraud Trends Over Time
    html.Div([
        dcc.Graph(
            figure=px.line(df_trends, x="Date", y="Fraud Cases", title="Fraud Cases Over Time")
        )
    ]),

    # Fraud by Location
    html.Div([
        dcc.Graph(
            figure=px.bar(df_location, x="Location", y="Count", title="Fraud Cases by Location")
        )
    ]),

    # Fraud by Device & Browser
    html.Div([
        dcc.Graph(
            figure=px.bar(df_device, x="device", y="count", color="browser", title="Fraud Cases by Device & Browser")
        )
    ])
])

# Run Dash
if __name__ == "__main__":
    app.run(debug=True, port=5000)

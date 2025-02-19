from flask import Flask, jsonify
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load fraud data
DATA_FILE = "data/fraud_data.csv"

def load_data():
    """Load fraud data from CSV"""
    return pd.read_csv(DATA_FILE)

@app.route("/api/summary")
def get_summary():
    """API endpoint to return fraud summary statistics"""
    df = load_data()
    total_transactions = len(df)
    fraud_cases = df[df["is_fraud"] == 1].shape[0]
    fraud_percentage = round((fraud_cases / total_transactions) * 100, 2)

    return jsonify({
        "total_transactions": total_transactions,
        "fraud_cases": fraud_cases,
        "fraud_percentage": fraud_percentage
    })

@app.route("/api/fraud_trends")
def get_fraud_trends():
    """API endpoint to return fraud trends over time"""
    df = load_data()
    df["date"] = pd.to_datetime(df["transaction_date"])
    trends = df[df["is_fraud"] == 1].groupby(df["date"].dt.date).size().to_dict()
    
    return jsonify(trends)

@app.route("/api/fraud_by_location")
def get_fraud_by_location():
    """API endpoint to return fraud cases by location"""
    df = load_data()
    fraud_by_location = df[df["is_fraud"] == 1].groupby("location").size().to_dict()
    
    return jsonify(fraud_by_location)

@app.route("/api/fraud_by_device")
def get_fraud_by_device():
    """API endpoint to return fraud cases by device and browser"""
    df = load_data()
    fraud_by_device = df[df["is_fraud"] == 1].groupby(["device", "browser"]).size().reset_index(name="count")
    
    return fraud_by_device.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True, port=5000)

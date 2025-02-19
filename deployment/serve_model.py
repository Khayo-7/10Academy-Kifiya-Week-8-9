from msilib import add_data
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from logging.handlers import RotatingFileHandler
from prometheus_flask_exporter import PrometheusMetrics
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Add a custom metric for fraud predictions
fraud_counter = metrics.counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    labels={'model': 'fraud_detection'}
)

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
handler = RotatingFileHandler(log_dir / "fraud_detection_api.log", maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Load the best model
MODEL_PATH = "models/best_model/best_model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}
    
@app.route("/predict", methods=["POST"])
@fraud_counter
def predict():
    """
    Endpoint for fraud detection predictions.
    Expects JSON input with transaction data.
    """
    try:
        # Parse input data
        data = request.json
        app.logger.info(f"Received prediction request: {data}")
        
        # Convert to DataFrame
        input_data = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        # Prepare response
        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        app.logger.info(f"Prediction response: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/api/fraud-data", methods=["GET"])
def get_fraud_data():
    """Serve fraud data to the dashboard"""
    try:
        data = add_data.to_dict(orient="records")
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# RotatingFileHandler - manage log file size
# curl http://localhost:5000/health
# curl http://localhost:5000/metrics
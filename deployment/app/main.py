from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from utils.inference import predict_fraud
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI, Request
import time
from api.caching import cache_result, get_cached_result
from api.tasks import process_fraud_detection

# Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint"])
RESPONSE_TIME = Histogram("response_time_seconds", "Response time in seconds", ["endpoint"])

app = FastAPI(title="Fraud Detection API", version="1.0")

class TransactionInput(BaseModel):
    """Defines input schema for the API."""
    features: list

@app.post("/predict/")
def predict_transaction(data: TransactionInput, model_type: str = "ml"):
    """API endpoint to predict fraud using ML or DL model."""
    input_data = np.array(data.features).reshape(1, -1)
    try:
        result = predict_fraud(input_data, model_type=model_type)
        return {"fraud": result["prediction"], "probability": result["probability"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()

    # task = process_fraud_detection.delay(data)
    # return {"task_id": task.id, "status": "Processing"}

    cache_key = f"fraud:{json.dumps(data)}"
    
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Perform fraud detection here...
    result = {"fraud": False, "confidence": 0.92}

    # Cache the result
    cache_result(cache_key, result)
    return result

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track API response time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    RESPONSE_TIME.labels(endpoint=endpoint).observe(process_time)
    return response

@app.get("/metrics")
def metrics():
    """Expose metrics for Prometheus."""
    return generate_latest()

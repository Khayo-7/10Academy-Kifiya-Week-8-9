import torch
import numpy as np
import joblib
from models.nn_models import LSTMModel  # Adjust based on your DL model
from utils.data_prep import preprocess_data  # Assumes a preprocessing function exists

def predict_fraud(input_data, model_type="ml"):
    """Runs inference using the best ML or DL model."""
    processed_data = preprocess_data(input_data)

    if model_type == "ml":
        model = joblib.load("models/saved_models/best_ml_model.pkl")
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1] if hasattr(model, "predict_proba") else None

    elif model_type == "dl":
        model = LSTMModel(input_size=processed_data.shape[1])
        model = load_dl_model(model, "models/saved_models/best_dl_model.pth")
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(tensor_data).squeeze().numpy()
            probability = prediction  # Sigmoid output

    return {"prediction": int(prediction > 0.5), "probability": float(probability)}

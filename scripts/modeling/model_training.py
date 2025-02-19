import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, log_loss

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a model and evaluate performance."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC-ROC: {auc:.4f}")

    joblib.dump(model, f"../resources/models/{model_name}.pkl")  # Save model
    return model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.ml_models import get_ml_model
from utils.metrics import compute_metrics
from utils.experiment import log_experiment
from utils.preprocessing import load_data
from utils.config import ML_TRAIN_PARAMS

def train_ml_model(X: np.ndarray, y: np.ndarray, model_name: str, params: dict = ML_TRAIN_PARAMS):
    """Train an ML model and return trained model along with evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_ml_model(model_name, params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    log_experiment(model_name, params, metrics)

    return model, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from models.ml_models import get_ml_model
from utils.metrics import compute_metrics
from utils.experiment import log_experiment
from utils.preprocessing import load_data
from utils.feature_selection import select_features
from utils.config import ML_TRAIN_PARAMS

def train_ml_model(X: np.ndarray, y: np.ndarray, model_name: str, params: dict = ML_TRAIN_PARAMS):
    """Train an ML model with feature selection and return trained model."""
    X_selected = select_features(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    model = get_ml_model(model_name, params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    log_experiment(model_name, params, metrics)

    return model, metrics

def get_ml_model(model_name: str, params: Dict[str, Any]):
    """Factory method to return an ML model with specified parameters."""
    models = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier,
        "catboost": cb.CatBoostClassifier
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    return models[model_name](**params)

def train_ml_models(X_train, y_train, X_test, y_test, dataset_name):
    """Train multiple models on a given dataset."""
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        model_name = f"{dataset_name}_{name}"
        trained_models[model_name] = train_model(model, X_train, y_train, X_test, y_test, model_name)

    return trained_models
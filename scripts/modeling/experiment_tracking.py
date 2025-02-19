import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

def log_experiment(model, model_name, X_test, y_test, model_type="ml"):
    """Log model performance and parameters to MLflow."""

    with mlflow.start_run():
        # Log model to MLflow
        if model_type == "ml":
            mlflow.sklearn.log_model(model, model_name)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        elif model_type == "deep":
            mlflow.tensorflow.log_model(model, model_name)
            y_prob = model.predict(X_test).flatten()  # Deep learning models return probabilities

        # Compute evaluation metrics
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        accuracy = accuracy_score(y_test, np.round(y_prob)) if y_prob is not None else None
        loss = log_loss(y_test, y_prob) if y_prob is not None else None

        # Log metrics
        mlflow.log_metric("AUC-ROC", auc)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Log Loss", loss)
        mlflow.log_param("Model Type", model_type)

        print(f"Logged {model_name} to MLflow with AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")


def log_experiment(model_name, params, metrics):
    """Log ML/DL experiment details into MLflow."""
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_name", model_name)
from sklearn.model_selection import GridSearchCV
from models.ml_models import get_ml_model

def tune_hyperparameters(model_name, X, y, param_grid):
    """Performs hyperparameter tuning using GridSearchCV."""
    model = get_ml_model(model_name, {})
    grid_search = GridSearchCV(model, param_grid, scoring="f1", cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=10):
    """Select top k best features based on ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

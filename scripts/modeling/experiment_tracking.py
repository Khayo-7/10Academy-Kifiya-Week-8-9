import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score

def log_experiment(model, model_name, X_test, y_test):
    """Log model performance and parameters to MLflow."""
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        mlflow.log_metric("AUC-ROC", auc)
        mlflow.log_param("model_type", model_name)

        print(f"Logged {model_name} to MLflow.")

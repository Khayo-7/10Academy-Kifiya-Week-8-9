import mlflow

def configure_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_tracking_uri("postgresql://user:password@localhost/mlflowdb")
    mlflow.set_experiment("Fraud_Detection")
    # Enable automatic logging for sklearn and keras
    mlflow.sklearn.autolog()
    mlflow.keras.autolog()
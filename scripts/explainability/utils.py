import mlflow
import numpy as np
import gradio as gr
import wasserstein_distance

def monitor_explanations():
    # Compare SHAP values between training and production
    training_shap = np.load("reports/explanations/shap_training.npy")
    prod_shap = np.load("reports/explanations/shap_prod.npy")
    
    # Calculate distribution drift
    drift_score = wasserstein_distance(training_shap, prod_shap)
    mlflow.log_metric("shap_drift_score", drift_score)

def create_demo(explainer, X_test):
    interface = gr.Interface(
        fn=explainer.explain_instance,
        inputs=gr.Dataframe(),
        outputs=gr.Image(),
        examples=X_test.head().values.tolist()
    )
    interface.launch()
import shap
import lime
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from typing import Union, Tuple
from pathlib import Path
from tensorflow.keras.models import load_model
from diskcache import Cache

cache = Cache("cache/explanations")

@cache.memoize()
def get_shap_values(explainer, data):
    return explainer.shap_values(data)


class ModelExplainer:
    """Handles model explainability using SHAP and LIME"""
    
    def __init__(self, model: Union[object, str], X_train: pd.DataFrame, model_type: str = "sklearn"):
        """
        Args:
            model: Pre-trained model object or path to saved model
            X_train: Training data for explainer background
            model_type: "sklearn", "keras", or "pytorch"
        """
        self.model = self._load_model(model, model_type)
        self.X_train = X_train
        self.model_type = model_type
        self.feature_names = X_train.columns.tolist()
        
        # Set up directories
        self.report_dir = Path("reports/explanations")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, model: Union[object, str], model_type: str) -> object:
        """Load model from path if needed"""
        if isinstance(model, str):
            if model_type == "sklearn":
                return joblib.load(model)
            elif model_type == "keras":
                return load_model(model)
        return model

    def explain_with_shap(self, X_explain: pd.DataFrame, 
                        n_samples: int = 1000) -> None:
        """Generate global and local SHAP explanations"""
        # Sample data for efficiency
        background = shap.sample(self.X_train, 100)
        X_sampled = shap.sample(X_explain, n_samples)

        # Initialize appropriate explainer
        if self.model_type == "tree":
            explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "keras":
            explainer = shap.DeepExplainer(self.model, background.values)
        else:
            explainer = shap.KernelExplainer(self.model.predict, background)

        # Calculate SHAP values
        shap_values = get_shap_values(explainer, X_sampled)

        # Generate and save plots
        self._save_shap_plots(explainer, shap_values, X_sampled)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_artifacts(str(self.report_dir / "shap"))

    def _save_shap_plots(self, explainer, shap_values, X_sampled) -> None:
        """Generate and save SHAP visualization plots"""
        plt.figure()
        
        # Summary plot
        shap.summary_plot(shap_values, X_sampled, 
                        feature_names=self.feature_names, 
                        show=False)
        plt.savefig(self.report_dir / "shap_summary.png", bbox_inches="tight")
        plt.close()

        # Dependence plot for top feature
        top_feature = np.abs(shap_values).mean(0).argmax()
        shap.dependence_plot(top_feature, shap_values, X_sampled,
                            feature_names=self.feature_names,
                            show=False)
        plt.savefig(self.report_dir / "shap_dependence.png")
        plt.close()

        # Force plot for first sample
        plt.figure()
        shap.force_plot(explainer.expected_value, shap_values[0,:], 
                        X_sampled.iloc[0,:], 
                        feature_names=self.feature_names,
                        matplotlib=True, show=False)
        plt.savefig(self.report_dir / "shap_force_plot.png", bbox_inches="tight")
        plt.close()

    def explain_with_lime(self, X_explain: pd.DataFrame, 
                        num_features: int = 5,
                        num_samples: int = 5000) -> None:
        """Generate LIME explanations for random samples"""
        # Initialize LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            mode="classification",
            discretize_continuous=True
        )

        # Explain random instances
        for idx in np.random.choice(X_explain.index, size=5, replace=False):
            explanation = explainer.explain_instance(
                X_explain.loc[idx].values,
                self.model.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Save explanation visualization
            fig = explanation.as_pyplot_figure()
            plt.title(f"LIME Explanation - Instance {idx}")
            plt.savefig(self.report_dir / f"lime_explanation_{idx}.png")
            plt.close()

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_artifacts(str(self.report_dir / "lime"))

    def generate_report(self, X_test: pd.DataFrame) -> None:
        """Full explainability pipeline"""
        # Sample data for explanations
        X_sample = X_test.sample(n=500, random_state=42)
        
        # SHAP explanations
        self.explain_with_shap(X_sample)
        
        # LIME explanations
        self.explain_with_lime(X_sample)
        
        print(f"Explainability report saved to {self.report_dir}")

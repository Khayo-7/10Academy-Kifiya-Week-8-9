import shap
import numpy as np
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, X_train):
        """
        Initialize SHAP explainer.
        :param model: Trained model (tree-based or deep learning)
        :param X_train: Training data (used to compute SHAP values)
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self):
        """
        Compute SHAP values based on the model type.
        """
        model_type = type(self.model).__name__.lower()
        
        if "tree" in model_type or "forest" in model_type or "boosting" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, self.X_train[:100])  # Subset for efficiency

        self.shap_values = self.explainer.shap_values(self.X_train)
    
    def plot_summary(self):
        """
        SHAP Summary Plot: Global Feature Importance
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        shap.summary_plot(self.shap_values, self.X_train)
    
    def plot_force(self, instance_idx=0):
        """
        SHAP Force Plot: Single Prediction Explanation
        :param instance_idx: Index of the instance to explain
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        shap.initjs()
        return shap.force_plot(self.explainer.expected_value, self.shap_values[instance_idx], self.X_train.iloc[instance_idx])
    
    def plot_dependence(self, feature_name):
        """
        SHAP Dependence Plot: Relationship between a feature and model predictions
        :param feature_name: Name of the feature to analyze
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        shap.dependence_plot(feature_name, self.shap_values, self.X_train)

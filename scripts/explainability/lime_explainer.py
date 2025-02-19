import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

class LIMEExplainer:
    def __init__(self, model, X_train, class_names=["Non-Fraud", "Fraud"]):
        """
        Initialize LIME explainer.
        :param model: Trained model
        :param X_train: Training data
        :param class_names: Labels for fraud classification
        """
        self.model = model
        self.X_train = X_train
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=class_names,
            mode="classification"
        )
    
    def explain_instance(self, instance_idx):
        """
        Explain a single instance.
        :param instance_idx: Index of the instance to explain
        :return: LIME explanation object
        """
        instance = self.X_train.iloc[instance_idx].values
        explanation = self.explainer.explain_instance(instance, self.model.predict_proba)
        return explanation
    
    def plot_instance_explanation(self, instance_idx):
        """
        Generate and display feature importance for a single instance.
        :param instance_idx: Index of the instance to explain
        """
        explanation = self.explain_instance(instance_idx)
        explanation.show_in_notebook()

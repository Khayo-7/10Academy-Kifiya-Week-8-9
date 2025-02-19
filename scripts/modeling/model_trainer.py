import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

class ModelTrainer:
    """Orchestrates model training and evaluation for fraud detection"""
    
    def __init__(self, X_train, X_test, y_train, y_test, name):
        self.experiment_name = "Fraud_Detection_" + name
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.class_weights = self._compute_class_weights()
        self.input_shape = (self.X_train.shape[1],)
        self.best_score = -np.inf
        self.best_model = None
        self.best_model_name = None
        mlflow.set_experiment(self.experiment_name)

    def _compute_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets"""
        classes = np.unique(self.y_train)
        weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
        return {cls: weight for cls, weight in zip(classes, weights)}

    def train_sklearn_model(self, model: Any, model_name: str) -> None:
        """Train and evaluate scikit-learn models"""
        with mlflow.start_run(run_name=f"{model_name}_{self.experiment_name}"):
            # Model training
            model.fit(self.X_train, self.y_train)
            
            # Inference
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Model evaluation
            roc_auc = roc_auc_score(self.y_test, y_proba)
            # mlflow.log_params(params)
            # mlflow.set_tag("model", model_name)
            self._log_metrics(y_pred, y_proba)
            self._log_model(model, model_name, "sklearn")

            # Check if this is the best model
            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_model = model
                self.best_model_name = model_name
                self._save_best_model(model, model_name)

    def train_keras_model(self, model_builder: callable, 
                        model_name: str, 
                        epochs: int = 50,
                        batch_size: int = 256) -> None:
        """Train and evaluate Keras models"""
        with mlflow.start_run(run_name=f"{model_name}_{self.experiment_name}"):
            # Model compilation
            model = model_builder(self.input_shape)
            early_stop = EarlyStopping(monitor='val_loss', patience=5)
            
            # Model training
            history = model.fit(
                self.X_train.values if 'pandas' in str(type(self.X_train)) else self.X_train,
                self.y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=self.class_weights,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Inference
            y_proba = model.predict(self.X_test).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            
            # Model evaluation
            roc_auc = roc_auc_score(self.y_test, y_proba)
            # mlflow.log_params(params)
            # mlflow.set_tag("model", model_name)
            self._log_metrics(y_pred, y_proba)
            self._log_model(model, model_name, "keras")
            
            # Check if this is the best model
            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_model = model
                self.best_model_name = model_name
                self._save_best_model(model, model_name)
            
    def _save_best_model(self, model: Any, model_name: str) -> None:
        """Save the best model to disk"""
        model_dir = "models/best_model"
        os.makedirs(model_dir, exist_ok=True)
        
        if model_name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
            import joblib
            joblib.dump(model, f"{model_dir}/best_model.pkl")
        elif model_name in ["MLP", "CNN", "LSTM"]:
            model.save(f"{model_dir}/best_model.h5")
        
        print(f"New best model saved: {model_name} with ROC-AUC: {self.best_score:.4f}")

    def _log_metrics(self, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
        """Log evaluation metrics to MLflow"""
        report = classification_report(self.y_test, y_pred, 
                                    output_dict=True, 
                                    target_names=['Legit', 'Fraud'])
        
        # Precision-Recall Curve metrics
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        mlflow.log_metrics({
            'roc_auc': roc_auc_score(self.y_test, y_proba),
            'pr_auc': pr_auc,
            'precision_0': report['Legit']['precision'],
            'recall_0': report['Legit']['recall'],
            'f1_0': report['Legit']['f1-score'],
            'precision_1': report['Fraud']['precision'],
            'recall_1': report['Fraud']['recall'],
            'f1_1': report['Fraud']['f1-score'],
            'accuracy': report['accuracy']
        })

    def _log_model(self, model: Any, model_name: str, flavor: str) -> None:
        """Log model to MLflow registry"""
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
            # mlflow.register_model(
            #     "runs:/<run_id>/model",
            #     "Fraud_Detection_Model"
            # )
        elif flavor == "keras":
            mlflow.keras.log_model(model, model_name)
        else:
            raise ValueError("Unsupported model flavor")

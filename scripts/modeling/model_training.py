from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

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

def train_all_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return trained models."""
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_model(model, X_train, y_train, X_test, y_test, name)

    return trained_models

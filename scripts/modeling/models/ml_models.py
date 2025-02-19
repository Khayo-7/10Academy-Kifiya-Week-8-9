from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_ml_model(model_name: str, params: Dict[str, Any]):
    """Factory method to return an ML model with specified parameters."""
    models = {
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC(probability=True),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "lightgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier()
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    return models[model_name](**params)


def get_ml_models():
    """Returns a dictionary of ML models with default hyperparameters."""
    return {
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC(probability=True),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "lightgbm": LGBMClassifier(),
        "catboost": cb.CatBoostClassifier
    }

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5):
    """Performs hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

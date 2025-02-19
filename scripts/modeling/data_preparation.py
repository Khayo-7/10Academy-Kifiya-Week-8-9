import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def preprocess_data(data, target, drop_columns=[]):
    """Prepare Data dataset: separate features and target."""
    X = data.drop(columns=[target]+drop_columns)
    y = data[target]
    return X, y

def stratified_split(X, y, test_size=0.2, random_state=42):
    """Perform stratified train-test split."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

def data_split(X, y, test_size=0.2, random_state=42):
    """Performs stratified train-test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def feature_selection(X_train, y_train, k=20):
    """Selects top K features using SelectKBest."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    return X_train_selected, selector

# def prepare_datasets(fraud_data_path, credit_card_path):
#     """Load, preprocess, and split datasets."""
#     fraud_data, credit_data = load_data(fraud_data_path)
#     fraud_data, credit_data = load_data(credit_card_path)

#     X_fraud, y_fraud = preprocess_data(fraud_data, target)
#     X_credit, y_credit = preprocess_data(credit_data, target)

#     X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = stratified_split(X_fraud, y_fraud)
#     X_credit_train, X_credit_test, y_credit_train, y_credit_test = stratified_split(X_credit, y_credit)

#     logger.info("Data preparation complete.")

#     return (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, 
#             X_credit_train, X_credit_test, y_credit_train, y_credit_test)


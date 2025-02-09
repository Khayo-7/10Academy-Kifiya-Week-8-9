import os, sys
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Setup logger for data_transformation
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("data_transformation")

def normalize_features(data, numeric_columns):
    """Normalize numerical features."""
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def encode_categorical_features(data, categorical_columns):
    """Encode categorical variables."""
    for col in categorical_columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    return data

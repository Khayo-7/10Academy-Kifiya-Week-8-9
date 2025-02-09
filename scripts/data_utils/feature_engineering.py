import os, sys

import pandas as pd

# Setup logger for feature_engineering
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

def create_time_features(data):
    """Extract time-based features from purchase timestamps."""
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    return data

def calculate_transaction_velocity(data):
    """Compute transaction frequency per user within short time windows."""
    data.sort_values(by=['user_id', 'purchase_time'], inplace=True)
    data['time_diff'] = data.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0)
    data['transaction_velocity'] = data.groupby('user_id')['time_diff'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    return data

def compute_signup_to_purchase_delay(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the time difference between signup and purchase in hours."""
    data["signup_to_purchase_hours"] = (data["purchase_time"] - data["signup_time"]).dt.total_seconds() / 3600
    return data

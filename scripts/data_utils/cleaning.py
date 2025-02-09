import os, sys
import pandas as pd
import numpy as np

# Setup logger for cleaning
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("cleaning")

def handle_missing_values(data):
    """Handles missing values by either imputing or dropping columns."""
    missing_info = data.isnull().sum()
    logger.info(f'Missing values before handling:\n{missing_info}')
    logger.info(f"Missing values:\n{missing_info[missing_info > 0]}")
    
    # Drop columns with too many missing values (e.g., >50%)
    threshold = 0.5 * len(data)
    data = data.dropna(thresh=threshold, axis=1)
    
    # Impute missing values for numerical columns
    # Fill missing numerical values with median
    # for col in data.select_dtypes(include=['number']).columns:
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Fill missing categorical values with mode
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Drop rows with missing categorical values
    data.dropna(inplace=True)

    logger.info("Handled missing values.")
    logger.info(f'Missing values after handling:\n{data.isnull().sum()}')
    return data

def clean_data(data):
    """Remove duplicates and correct data types."""
    initial_shape = data.shape
    data.drop_duplicates(inplace=True)
    logger.info(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")

    # Convert timestamps to datetime
    if 'signup_time' in data.columns and 'purchase_time' in data.columns:
        data['signup_time'] = pd.to_datetime(data['signup_time'])
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])

    logger.info('Data cleaned: duplicates removed, data types corrected.')
    return data

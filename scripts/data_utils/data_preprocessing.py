import os, sys
import pandas as pd
import numpy as np

# Setup logger for data_preprocessing
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.data_utils.cleaning import handle_missing_values, clean_data
from scripts.data_utils.feature_engineering import merge_geolocation
from scripts.data_utils.loaders import load_data
from scripts.utils.logger import setup_logger

logger = setup_logger("data_preprocessing")

def preprocess_data(fraud_data_path, creditcard_data_path, ip_data_path):
    """Main function to run all preprocessing steps."""
    fraud_data = load_data(fraud_data_path)
    credit_data = load_data(creditcard_data_path)
    ip_data = load_data(ip_data_path)
    
    fraud_data = handle_missing_values(fraud_data)
    credit_data = handle_missing_values(credit_data)
    
    fraud_data = clean_data(fraud_data)
    credit_data = clean_data(credit_data)
    
    fraud_data = merge_geolocation(fraud_data, ip_data)
    
    return fraud_data, credit_data

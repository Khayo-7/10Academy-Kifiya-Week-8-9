import os, sys, ipaddress
import numpy as np
import pandas as pd
from io import StringIO

# Setup logger for utils
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("utils")

def summarize_data(data: pd.DataFrame):
    """
    Prints summary statistics and detailed information about the dataset, including duplications.
    
    Args:
        data (pd.DataFrame): The dataset to be summarized.
    """
    try:
        logger.info("\n--- Data Summary ---")
        summary_stats = data.describe(include="all").transpose()
        summary_stats['unique'] = data.nunique()
        summary_stats['dtype'] = data.dtypes
        logger.info(summary_stats)
        
        logger.info("\n--- Data Info ---")
        buffer = StringIO() # Use StringIO to capture output
        data.info(buf=buffer)
        logger.info(buffer.getvalue())
        
        logger.info("\n--- Missing Values Analysis ---")
        missing_summary = data.isnull().sum().reset_index()
        missing_summary.columns = ['Column', 'MissingCount']
        missing_summary['MissingPercentage'] = (missing_summary['MissingCount'] / len(data)) * 100
        missing_summary = missing_summary[missing_summary['MissingCount'] > 0]
        
        if not missing_summary.empty:
            logger.info("Columns with missing values:")
            logger.info(missing_summary)
        else:
            logger.info("No missing values found in the dataset.")
        
        logger.info("\n--- Total columns with Missing Values ---")
        logger.info((data.isnull().sum() > 0).sum())
        
        # Check for duplicate rows
        logger.info("\n--- Duplication Analysis ---")
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            logger.info(f"Number of duplicate rows: {duplicate_rows}")
            logger.info("Sample of duplicate rows:")
            logger.info(data[data.duplicated(keep=False)].head())  # Show a sample of duplicate rows
        else:
            logger.info("No duplicate rows found.")
        
        # # Check for duplicate columns
        # duplicate_columns = data.T.duplicated().sum()
        # if duplicate_columns > 0:
        #     logger.info(f"Number of duplicate columns: {duplicate_columns}")
        #     logger.info("Duplicate columns:")
        #     logger.info(data.columns[data.T.duplicated()].tolist())
        # else:
        #     logger.info("No duplicate columns found.")
        
        logger.info("\n--- Data Skewness ---")
        logger.info(data.skew(numeric_only=True))
        
        logger.info("\n--- Data Kurtosis ---")
        logger.info(data.kurtosis(numeric_only=True))
        
        logger.info("\n--- Correlation Matrix ---")
        logger.info(data.corr(numeric_only=True))
        
        logger.info("\n--- Data Head ---")
        logger.info(data.head())
        
        logger.info("\n--- Data Tail ---")
        logger.info(data.tail())
        
        logger.info("Data summary and analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Error summarizing data: {e}", exc_info=True)
        raise

def ip_to_int(ip):
    """Converts an IP address to an integer format."""
    try:
        return int(ipaddress.IPv4Address(ip))
        # octets = list(map(int, ip.split('.')))
        # return (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
    except:
        return np.nan

def int_to_ip(ip_int):
    return str(ipaddress.IPv4Address(ip_int))

def map_merge_geolocation(transactions: pd.DataFrame, ip_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges fraud transactions dataset with geolocation dataset based on IP address
    to map IP addresses to country using interval-based lookup.
    """
    # Validate input columns
    required_transactions_columns = {"ip_address"}
    required_ip_data_columns = {"lower_bound_ip_address", "upper_bound_ip_address", "country"}
    
    if not required_transactions_columns.issubset(transactions.columns):
        raise ValueError(f"Transactions DataFrame must contain columns: {required_transactions_columns}")
    if not required_ip_data_columns.issubset(ip_data.columns):
        raise ValueError(f"IP data DataFrame must contain columns: {required_ip_data_columns}")

    # Ensure the IP columns are of type int64 for accurate comparison
    transactions["ip_address"] = transactions["ip_address"].astype("int64")
    ip_data["lower_bound_ip_address"] = ip_data["lower_bound_ip_address"].astype("int64")
    ip_data["upper_bound_ip_address"] = ip_data["upper_bound_ip_address"].astype("int64")

    # Sort IP data by lower_bound_ip_address for merge_asof
    ip_data = ip_data.sort_values(by="lower_bound_ip_address")

    # Perform a merge_asof to map IPs to their corresponding ranges
    transactions = pd.merge_asof(
        transactions.sort_values(by="ip_address"),
        ip_data,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    # Assign "Unknown" to IPs that don't fall within any range
    transactions["country"] = transactions["country"].where(
        transactions["ip_address"] <= transactions["upper_bound_ip_address"], "Unknown"
    )

    # Drop unnecessary columns
    transactions = transactions.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])

    # Log the result
    logger.info(f"Merged fraud transactions dataset with geolocation data. Resulting shape: {transactions.shape}")

    return transactions
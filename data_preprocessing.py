"""
Data preprocessing module for the Flipkart Customer Support Analysis project.
"""
import pandas as pd
import numpy as np
from utils import print_section_header

def load_data(file_path):
    """
    Load the customer support dataset from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the data
    
    Returns:
    --------
    df : pandas DataFrame
        The loaded dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def analyze_data_quality(df):
    """
    Analyze the quality of the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyze
    
    Returns:
    --------
    quality_stats : dict
        Dictionary containing data quality statistics
    """
    print_section_header("Data Quality Analysis")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print("Missing values per column:")
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Basic statistics for numerical columns
    print("\nBasic statistics for numerical columns:")
    print(df.describe().T)
    
    # Value counts for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    print("\nValue counts for top categorical columns:")
    for col in cat_cols[:5]:  # Show only first 5 categorical columns
        print(f"\n{col}:")
        print(df[col].value_counts().head(5))
    
    # Collect statistics
    quality_stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': missing_values.to_dict(),
        'missing_percent': missing_percent.to_dict(),
        'duplicates': duplicates,
        'duplicate_percent': duplicates/len(df)*100
    }
    
    return quality_stats

def convert_to_datetime(df, col):
    """
    Convert a column to datetime format
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset containing the column
    col : str
        The name of the column to convert
    
    Returns:
    --------
    success : bool
        True if conversion was successful, False otherwise
    """
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        return True
    except Exception as e:
        print(f"Warning: Could not convert {col} to datetime. Error: {e}")
        return False

def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicates, and data type conversions
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to clean
    
    Returns:
    --------
    df : pandas DataFrame
        The cleaned dataset
    cleaning_stats : dict
        Dictionary containing statistics about the cleaning process
    """
    print_section_header("Data Cleaning")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    cleaning_stats = {}
    
    # 1. Convert datetime columns
    datetime_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
    for col in datetime_cols:
        if col in df.columns:
            success = convert_to_datetime(df, col)
            if success:
                print(f"Converted {col} to datetime format")
    
    # 2. Calculate response time if possible
    if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
        try:
            df['response_time_minutes'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
            print(f"Created 'response_time_minutes' feature. Mean response time: {df['response_time_minutes'].mean():.2f} minutes")
            cleaning_stats['mean_response_time'] = df['response_time_minutes'].mean()
        except Exception as e:
            print(f"Warning: Could not calculate response time. Error: {e}")
    
    # 3. Remove duplicates if any
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates_before} duplicate rows")
        cleaning_stats['duplicates_removed'] = duplicates_before
    
    # 4. Handle missing values based on column type and importance
    print("\nHandling missing values...")
    
    # 4.1 For categorical columns, fill with the most frequent value
    cat_cols_low_missing = ['channel_name', 'category', 'Sub-category', 'Tenure Bucket', 'Agent Shift']
    for col in cat_cols_low_missing:
        if col in df.columns and df[col].isnull().sum() > 0:
            missing_before = df[col].isnull().sum()
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled {missing_before} missing values in '{col}' with mode: {mode_value}")
            cleaning_stats[f'{col}_filled'] = missing_before
    
    # 4.2 For Order_id: Generate unique placeholders
    if 'Order_id' in df.columns and df['Order_id'].isnull().sum() > 0:
        missing_order_count = df['Order_id'].isnull().sum()
        df.loc[df['Order_id'].isnull(), 'Order_id'] = [f'generated_order_{i}' for i in range(missing_order_count)]
        print(f"Generated {missing_order_count} placeholder Order_ids")
        cleaning_stats['order_id_generated'] = missing_order_count
    
    # 4.3 For numerical columns, fill with median
    num_cols = ['Item_price', 'connected_handling_time']
    for col in num_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            missing_before = df[col].isnull().sum()
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled {missing_before} missing values in '{col}' with median: {median_value}")
            cleaning_stats[f'{col}_filled'] = missing_before
    
    # 4.4 For text columns, fill with 'Unknown'
    text_cols = ['Customer_City', 'Product_category', 'Customer Remarks']
    for col in text_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            missing_before = df[col].isnull().sum()
            df[col].fillna('Unknown', inplace=True)
            print(f"Filled {missing_before} missing values in '{col}' with 'Unknown'")
            cleaning_stats[f'{col}_filled'] = missing_before
    
    # 4.5 For order_date_time, fill with Issue_reported at time if available
    if 'order_date_time' in df.columns and 'Issue_reported at' in df.columns and df['order_date_time'].isnull().sum() > 0:
        missing_before = df['order_date_time'].isnull().sum()
        df['order_date_time'] = df['order_date_time'].fillna(df['Issue_reported at'])
        print(f"Filled {missing_before} missing values in 'order_date_time' with 'Issue_reported at' time")
        cleaning_stats['order_date_time_filled'] = missing_before
    
    # 5. Check remaining missing values
    remaining_missing = df.isnull().sum().sum()
    print(f"\nRemaining missing values after cleaning: {remaining_missing}")
    cleaning_stats['remaining_missing'] = remaining_missing
    
    return df, cleaning_stats

def preprocess_data(df, target_col=None):
    """
    Preprocess the data for model training
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to preprocess
    target_col : str, optional
        The name of the target column
    
    Returns:
    --------
    df : pandas DataFrame
        The preprocessed dataset
    """
    print_section_header("Data Preprocessing")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # 1. Handle outliers in numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        # Skip id columns and target column
        if col.lower().endswith('id') or col == target_col:
            continue
        
        # Calculate Q1, Q3, and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            # Cap outliers
            df[col] = df[col].clip(lower_bound, upper_bound)
            print(f"Capped {outliers} outliers in column '{col}'")
    
    # 2. Normalize text fields if any
    text_cols = ['Customer Remarks']
    for col in text_cols:
        if col in df.columns:
            # Convert to string and lowercase
            df[col] = df[col].astype(str).str.lower()
            print(f"Normalized text in column '{col}'")
    
    return df

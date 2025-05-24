"""
Feature engineering module for the Flipkart Customer Support Analysis project.
"""
import pandas as pd
import numpy as np
from utils import print_section_header
from sklearn.preprocessing import LabelEncoder

def extract_datetime_features(df):
    """
    Extract features from datetime columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset containing datetime columns
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with added datetime features
    """
    datetime_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
    df = df.copy()
    
    for col in datetime_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            # Extract useful components
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_day_of_week'] = df[col].dt.dayofweek
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Business hours flag (9 AM to 6 PM)
            if f'{col}_hour' in df.columns:
                df[f'{col}_is_business_hours'] = ((df[f'{col}_hour'] >= 9) & 
                                                 (df[f'{col}_hour'] < 18)).astype(int)
            
            print(f"Created datetime features from '{col}'")
    
    return df

def create_time_based_features(df):
    """
    Create features based on time differences
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with datetime columns
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with added time-based features
    """
    df = df.copy()
    
    # Calculate ticket age (time between order and issue reported)
    if 'order_date_time' in df.columns and 'Issue_reported at' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['order_date_time']) and pd.api.types.is_datetime64_any_dtype(df['Issue_reported at']):
            df['ticket_age_days'] = (df['Issue_reported at'] - df['order_date_time']).dt.total_seconds() / (60*60*24)
            df['ticket_age_days'] = df['ticket_age_days'].clip(lower=0)  # Ensure non-negative values
            print("Created 'ticket_age_days' feature")
    
    # Calculate resolution time (if available)
    if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['Issue_reported at']) and pd.api.types.is_datetime64_any_dtype(df['issue_responded']):
            df['resolution_time_hours'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 3600
            df['resolution_time_hours'] = df['resolution_time_hours'].clip(lower=0)  # Ensure non-negative values
            print("Created 'resolution_time_hours' feature")
    
    return df

def create_category_features(df):
    """
    Create features based on categories and issue types
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with category columns
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with added category-based features
    """
    df = df.copy()
    
    # Create a combined category feature
    if 'category' in df.columns and 'Sub-category' in df.columns:
        df['combined_category'] = df['category'].astype(str) + " - " + df['Sub-category'].astype(str)
        print("Created 'combined_category' feature")
    
    # Create complexity indicator based on categories
    if 'category' in df.columns:
        # Map categories to complexity levels (example implementation)
        complexity_map = {
            'Technical': 3,
            'Billing': 2,
            'Shipping': 2,
            'Product': 1,
            'Account': 1
        }
        
        # Assign complexity score based on category
        df['issue_complexity'] = df['category'].map(lambda x: complexity_map.get(x, 1))
        print("Created 'issue_complexity' feature")
    
    return df

def create_text_based_features(df):
    """
    Create features based on text fields
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with text columns
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with added text-based features
    """
    df = df.copy()
    
    # Create features from Customer Remarks if available
    if 'Customer Remarks' in df.columns:
        # Length of customer remarks
        df['remarks_length'] = df['Customer Remarks'].astype(str).apply(len)
        
        # Check for urgent keywords
        urgent_keywords = ['urgent', 'immediately', 'asap', 'emergency', 'critical', 'important']
        df['has_urgent_keywords'] = df['Customer Remarks'].astype(str).apply(
            lambda x: any(keyword in x.lower() for keyword in urgent_keywords)
        ).astype(int)
        
        print("Created features from 'Customer Remarks'")
    
    return df

def encode_categorical_features(df, target_col=None):
    """
    Encode categorical features for model training
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with categorical columns
    target_col : str, optional
        The target column name
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with encoded categorical features
    categorical_columns : list
        List of categorical column names
    numerical_columns : list
        List of numerical column names
    """
    df = df.copy()
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from categorical columns if it's there
    if target_col in categorical_columns:
        categorical_columns.remove(target_col)
    
    # Exclude specific columns we don't want to encode
    exclude_cols = ['Customer Remarks', 'Unique id', 'Order_id']
    categorical_columns = [col for col in categorical_columns if col not in exclude_cols]
    
    # Use label encoding for categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}' using label encoding")
    
    # Identify numerical columns (excluding encoded ones)
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target column from numerical columns if it's there
    if target_col in numerical_columns:
        numerical_columns.remove(target_col)
    
    # Get only the original numerical columns, not the encoded ones
    numerical_columns = [col for col in numerical_columns if not col.endswith('_encoded')]
    
    return df, categorical_columns, numerical_columns

def create_target_variable(df, target_col='is_delayed', threshold_hours=24):
    """
    Create a binary target variable for modeling
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with time-related columns
    target_col : str
        The name for the target column
    threshold_hours : float
        The threshold in hours to consider a ticket as delayed
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with the target variable added
    target_col : str
        The name of the target column
    """
    df = df.copy()
    
    if 'resolution_time_hours' in df.columns:
        # Create binary target: 1 for delayed, 0 for on-time
        df[target_col] = (df['resolution_time_hours'] > threshold_hours).astype(int)
        print(f"Created target variable '{target_col}' using {threshold_hours} hours threshold")
        print(f"Delayed tickets: {df[target_col].sum()} ({df[target_col].mean()*100:.2f}%)")
    else:
        # If resolution time is not available, try to use a provided target column
        print("Warning: 'resolution_time_hours' not available for creating target variable")
        if target_col in df.columns:
            print(f"Using existing '{target_col}' as target variable")
        else:
            raise ValueError("No suitable target variable found")
    
    return df, target_col

def engineer_features(df, create_target=True, target_col='is_delayed', threshold_hours=24):
    """
    Main function to perform feature engineering
    
    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed dataset
    create_target : bool
        Whether to create a target variable
    target_col : str
        The name for the target column
    threshold_hours : float
        The threshold in hours to consider a ticket as delayed
    
    Returns:
    --------
    df : pandas DataFrame
        The dataset with engineered features
    feature_info : dict
        Dictionary with information about the features
    """
    print_section_header("Feature Engineering")
    
    # Extract features from datetime columns
    df = extract_datetime_features(df)
    
    # Create time-based features
    df = create_time_based_features(df)
    
    # Create category-based features
    df = create_category_features(df)
    
    # Create text-based features
    df = create_text_based_features(df)
    
    # Create target variable if requested
    if create_target:
        df, target_col = create_target_variable(df, target_col, threshold_hours)
    
    # Encode categorical features
    df, categorical_columns, numerical_columns = encode_categorical_features(df, target_col)
    
    # Collect feature information
    feature_info = {
        'target_column': target_col,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'feature_columns': numerical_columns + [f'{col}_encoded' for col in categorical_columns],
        'total_features': len(numerical_columns) + len(categorical_columns)
    }
    
    print(f"\nFeature engineering complete. Created {feature_info['total_features']} features for modeling.")
    
    return df, feature_info

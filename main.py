"""
Main execution script for the Flipkart Customer Support Analysis project.
This script orchestrates the entire workflow, from data loading to model training
and visualization.
"""
import os
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

# Import project modules
from data_preprocessing import load_data, analyze_data_quality, clean_data, preprocess_data
from feature_engineering import engineer_features
from model_training import prepare_training_data, train_logistic_regression, train_random_forest, train_xgboost, compare_models
from visualization import visualize_data
from utils import print_section_header, create_output_dirs

def main():
    """
    Main function to execute the entire analysis pipeline
    """
    # Record start time
    start_time = time.time()
    
    # Print welcome message
    print("\n" + "=" * 80)
    print(" FLIPKART CUSTOMER SUPPORT ANALYSIS ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Create necessary output directories
    create_output_dirs()
    
    # Step 1: Load data
    print_section_header("1. Loading Data")
    data_path = 'Customer_support_data.csv'
    df = load_data(data_path)
    
    # Step 2: Analyze data quality
    print_section_header("2. Data Quality Analysis")
    quality_stats = analyze_data_quality(df)
    
    # Step 3: Clean data
    print_section_header("3. Data Cleaning")
    df_clean, cleaning_stats = clean_data(df)
    
    # Step 4: Preprocess data
    print_section_header("4. Data Preprocessing")
    df_preprocessed = preprocess_data(df_clean)
    
    # Step 5: Feature engineering
    print_section_header("5. Feature Engineering")
    df_featured, feature_info = engineer_features(df_preprocessed, create_target=True, target_col='is_delayed', threshold_hours=24)
    
    # Save processed dataset
    processed_data_path = 'results/processed_data.csv'
    df_featured.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    # Step 6: Prepare training data
    print_section_header("6. Preparing Model Training Data")
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(df_featured, feature_info)
    
    # Step 7: Train and evaluate models
    print_section_header("7. Model Training and Evaluation")
    
    # 7.1 Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test, preprocessor)
    
    # 7.2 Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, preprocessor)
    
    # 7.3 XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, preprocessor)
    
    # Step 8: Compare models
    print_section_header("8. Model Comparison")
    models_metrics = {
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }
    best_model_name, comparison_df = compare_models(models_metrics)
    
    # Save model comparison results
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print(f"Model comparison results saved to results/model_comparison.csv")
    
    # Step 9: Data visualization
    print_section_header("9. Data Visualization")
    dashboard_figures = visualize_data(df_featured, feature_info)
    
    # Step 10: Save project metadata
    print_section_header("10. Saving Project Metadata")
    
    # Create metadata
    metadata = {
        'project_name': 'Flipkart Customer Support Analysis',
        'execution_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': data_path,
        'data_shape': df.shape,
        'cleaned_data_shape': df_clean.shape,
        'total_features': feature_info['total_features'],
        'target_column': feature_info['target_column'],
        'best_model': best_model_name,
        'best_model_accuracy': models_metrics[best_model_name]['accuracy'],
        'execution_time_seconds': round(time.time() - start_time, 2)
    }
    
    # Save metadata
    with open('results/project_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Project metadata saved to results/project_metadata.json")
    
    # Print completion message
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80)
    print(f"Execution time: {metadata['execution_time_seconds']} seconds")
    print(f"Best model: {best_model_name} with accuracy {models_metrics[best_model_name]['accuracy']:.4f}")
    print("\nRecommended next steps:")
    print("1. Run 'python dashboard.py' to launch the interactive dashboard")
    print("2. Review visualizations in the 'plots' directory")
    print("3. Check model comparison results in 'results/model_comparison.csv'")
    
    return metadata

if __name__ == "__main__":
    main()

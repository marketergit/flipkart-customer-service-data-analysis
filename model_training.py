"""
Model training module for the Flipkart Customer Support Analysis project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import print_section_header, save_model, plot_confusion_matrix, plot_roc_curve, get_model_metrics
import os

def prepare_training_data(df, feature_info, test_size=0.25, random_state=42):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with engineered features
    feature_info : dict
        Dictionary with information about the features
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train : pandas DataFrame
        Features for training
    X_test : pandas DataFrame
        Features for testing
    y_train : pandas Series
        Target for training
    y_test : pandas Series
        Target for testing
    preprocessor : ColumnTransformer
        Preprocessor for the features
    """
    print_section_header("Preparing Training Data")
    
    # Extract feature information
    target_col = feature_info['target_column']
    categorical_columns = feature_info['categorical_columns']
    numerical_columns = feature_info['numerical_columns']
    
    # Create encoded feature columns list
    cat_encoded_cols = [f'{col}_encoded' for col in categorical_columns]
    
    # Select features (excluding unique identifiers and dates)
    exclude_cols = ['Unique id', 'Order_id', 'order_date_time', 'Issue_reported at', 
                    'issue_responded', 'Survey_response_Date', target_col, 'Customer Remarks'] + categorical_columns
    
    feature_cols = [col for col in numerical_columns + cat_encoded_cols if col not in exclude_cols]
    print(f"Selected {len(feature_cols)} features for modeling")
    
    # Split the data into features and target
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Feature set shape: {X.shape}")
    print(f"Target set shape: {y.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Separate features into numerical and categorical
    numerical_features = [col for col in X.columns if not col.endswith('_encoded')]
    categorical_features = [col for col in X.columns if col.endswith('_encoded')]
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', 'passthrough', categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_logistic_regression(X_train, y_train, X_test, y_test, preprocessor, cv=5, perform_tuning=True):
    """
    Train a Logistic Regression model
    
    Parameters:
    -----------
    X_train, y_train : pandas DataFrame/Series
        Training data
    X_test, y_test : pandas DataFrame/Series
        Testing data
    preprocessor : ColumnTransformer
        Preprocessor for the features
    cv : int
        Number of cross-validation folds
    perform_tuning : bool
        Whether to perform hyperparameter tuning
    
    Returns:
    --------
    model : Pipeline
        Trained model
    metrics : dict
        Model performance metrics
    """
    print_section_header("Training Logistic Regression Model")
    
    # Create pipeline with preprocessor and classifier
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Hyperparameter tuning if requested
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
        
        lr_grid = GridSearchCV(lr_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        
        print(f"Best parameters: {lr_grid.best_params_}")
        print(f"Best CV score: {lr_grid.best_score_:.4f}")
        
        # Use the best model
        lr_pipeline = lr_grid.best_estimator_
    else:
        # Train the model without tuning
        lr_pipeline.fit(X_train, y_train)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Make predictions on test set
    y_pred = lr_pipeline.predict(X_test)
    y_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = get_model_metrics(y_test, y_pred, y_prob)
    
    # Print metrics
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['auc']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['On-time', 'Delayed'], 'Logistic Regression',
                         save_path='plots/logistic_regression_confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob, 'Logistic Regression',
                  save_path='plots/logistic_regression_roc_curve.png')
    
    # Get feature importance if possible
    try:
        # Get feature names after preprocessing
        feature_names = X_train.columns.tolist()
        
        # Get coefficients
        coefficients = lr_pipeline.named_steps['classifier'].coef_[0]
        
        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients)
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Feature Importance - Logistic Regression')
        plt.tight_layout()
        plt.savefig('plots/logistic_regression_feature_importance.png')
        plt.close()
        
        print("Feature importance plot saved to 'plots/logistic_regression_feature_importance.png'")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    # Save model
    model_path = save_model(lr_pipeline, 'logistic_regression')
    
    return lr_pipeline, metrics

def train_random_forest(X_train, y_train, X_test, y_test, preprocessor, cv=5, perform_tuning=True):
    """
    Train a Random Forest model
    
    Parameters:
    -----------
    X_train, y_train : pandas DataFrame/Series
        Training data
    X_test, y_test : pandas DataFrame/Series
        Testing data
    preprocessor : ColumnTransformer
        Preprocessor for the features
    cv : int
        Number of cross-validation folds
    perform_tuning : bool
        Whether to perform hyperparameter tuning
    
    Returns:
    --------
    model : Pipeline
        Trained model
    metrics : dict
        Model performance metrics
    """
    print_section_header("Training Random Forest Model")
    
    # Create pipeline with preprocessor and classifier
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Hyperparameter tuning if requested
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"Best CV score: {rf_grid.best_score_:.4f}")
        
        # Use the best model
        rf_pipeline = rf_grid.best_estimator_
    else:
        # Train the model without tuning
        rf_pipeline.fit(X_train, y_train)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Make predictions on test set
    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = get_model_metrics(y_test, y_pred, y_prob)
    
    # Print metrics
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['auc']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['On-time', 'Delayed'], 'Random Forest',
                         save_path='plots/random_forest_confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob, 'Random Forest',
                  save_path='plots/random_forest_roc_curve.png')
    
    # Get feature importance
    try:
        # Get feature names after preprocessing
        feature_names = X_train.columns.tolist()
        
        # Get feature importances
        importances = rf_pipeline.named_steps['classifier'].feature_importances_
        
        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        plt.savefig('plots/random_forest_feature_importance.png')
        plt.close()
        
        print("Feature importance plot saved to 'plots/random_forest_feature_importance.png'")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    # Save model
    model_path = save_model(rf_pipeline, 'random_forest')
    
    return rf_pipeline, metrics

def train_xgboost(X_train, y_train, X_test, y_test, preprocessor, cv=5, perform_tuning=True):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train, y_train : pandas DataFrame/Series
        Training data
    X_test, y_test : pandas DataFrame/Series
        Testing data
    preprocessor : ColumnTransformer
        Preprocessor for the features
    cv : int
        Number of cross-validation folds
    perform_tuning : bool
        Whether to perform hyperparameter tuning
    
    Returns:
    --------
    model : Pipeline
        Trained model
    metrics : dict
        Model performance metrics
    """
    print_section_header("Training XGBoost Model")
    
    # Create pipeline with preprocessor and classifier
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    
    # Hyperparameter tuning if requested
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
        
        xgb_grid = GridSearchCV(xgb_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        
        print(f"Best parameters: {xgb_grid.best_params_}")
        print(f"Best CV score: {xgb_grid.best_score_:.4f}")
        
        # Use the best model
        xgb_pipeline = xgb_grid.best_estimator_
    else:
        # Train the model without tuning
        xgb_pipeline.fit(X_train, y_train)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(xgb_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Make predictions on test set
    y_pred = xgb_pipeline.predict(X_test)
    y_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = get_model_metrics(y_test, y_pred, y_prob)
    
    # Print metrics
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['auc']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['On-time', 'Delayed'], 'XGBoost',
                         save_path='plots/xgboost_confusion_matrix.png')
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob, 'XGBoost',
                  save_path='plots/xgboost_roc_curve.png')
    
    # Get feature importance
    try:
        # Get feature names after preprocessing
        feature_names = X_train.columns.tolist()
        
        # Get feature importances
        importances = xgb_pipeline.named_steps['classifier'].feature_importances_
        
        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Feature Importance - XGBoost')
        plt.tight_layout()
        plt.savefig('plots/xgboost_feature_importance.png')
        plt.close()
        
        print("Feature importance plot saved to 'plots/xgboost_feature_importance.png'")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    # Save model
    model_path = save_model(xgb_pipeline, 'xgboost')
    
    return xgb_pipeline, metrics

def compare_models(models_metrics):
    """
    Compare multiple models based on their performance metrics
    
    Parameters:
    -----------
    models_metrics : dict
        Dictionary mapping model names to their metrics
    
    Returns:
    --------
    best_model_name : str
        Name of the best performing model
    comparison_df : pandas DataFrame
        DataFrame containing the performance metrics for each model
    """
    print_section_header("Model Comparison")
    
    # Create DataFrame for comparison
    model_names = list(models_metrics.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    comparison_data = []
    for model_name in model_names:
        metrics = models_metrics[model_name]
        comparison_data.append([model_name] + [metrics.get(metric, None) for metric in metrics_names])
    
    comparison_df = pd.DataFrame(comparison_data, columns=['Model'] + metrics_names)
    
    # Print comparison table
    print(comparison_df.to_string(index=False))
    
    # Find best model based on AUC (or accuracy if AUC is not available)
    if all(comparison_df['auc'].notna()):
        best_metric = 'auc'
    else:
        best_metric = 'accuracy'
    
    best_model_index = comparison_df[best_metric].idxmax()
    best_model_name = comparison_df.loc[best_model_index, 'Model']
    best_model_score = comparison_df.loc[best_model_index, best_metric]
    
    print(f"\nBest performing model based on {best_metric}: {best_model_name} with {best_metric} = {best_model_score:.4f}")
    
    # Create bar plot for comparison
    plt.figure(figsize=(12, 8))
    comparison_df_melted = pd.melt(comparison_df, id_vars=['Model'], value_vars=metrics_names,
                                   var_name='Metric', value_name='Score')
    
    sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_df_melted)
    plt.title('Model Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    print("Model comparison plot saved to 'plots/model_comparison.png'")
    
    return best_model_name, comparison_df

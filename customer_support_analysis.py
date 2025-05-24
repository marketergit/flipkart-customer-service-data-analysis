import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("Loading the Flipkart customer support dataset...")
# Load the dataset
df = pd.read_csv('Customer_support_data.csv')

# Display basic information
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_df)

# Data cleaning and preprocessing
print("\n==== Data Cleaning and Preprocessing ====")

# Function to handle datetime columns
def convert_to_datetime(df, col):
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        return True
    except:
        print(f"Warning: Could not convert {col} to datetime.")
        return False

# Convert datetime columns
datetime_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
for col in datetime_cols:
    if col in df.columns:
        convert_to_datetime(df, col)

# Calculate response time if possible (time between issue reported and responded)
if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
    try:
        df['response_time_minutes'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
        print(f"Created 'response_time_minutes' feature. Mean response time: {df['response_time_minutes'].mean():.2f} minutes")
    except:
        print("Warning: Could not calculate response time.")

# Handle missing values based on column type and importance
print("\nHandling missing values...")

# 1. For categorical columns with low missing percentages, fill with the most frequent value
cat_cols_low_missing = ['channel_name', 'category', 'Sub-category', 'Tenure Bucket', 'Agent Shift']
for col in cat_cols_low_missing:
    if col in df.columns and df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"Filled '{col}' missing values with mode: {mode_value}")

# 2. For Order_id: Since it's an identifier, we'll generate unique placeholders
if 'Order_id' in df.columns and df['Order_id'].isnull().sum() > 0:
    missing_order_count = df['Order_id'].isnull().sum()
    df.loc[df['Order_id'].isnull(), 'Order_id'] = [f'generated_order_{i}' for i in range(missing_order_count)]
    print(f"Generated {missing_order_count} placeholder Order_ids")

# 3. For numerical columns, fill with median
num_cols = ['Item_price', 'connected_handling_time']
for col in num_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"Filled '{col}' missing values with median: {median_value}")

# 4. For Customer_City and Product_category, fill with 'Unknown'
text_cols = ['Customer_City', 'Product_category', 'Customer Remarks']
for col in text_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna('Unknown', inplace=True)
        print(f"Filled '{col}' missing values with 'Unknown'")

# 5. For order_date_time, we'll fill with a placeholder date if it's important for analysis
if 'order_date_time' in df.columns and df['order_date_time'].isnull().sum() > 0:
    # Use the Issue_reported at as a proxy when order_date_time is missing
    df['order_date_time'] = df['order_date_time'].fillna(df['Issue_reported at'])
    print("Filled missing order_date_time with Issue_reported at time")

# Check if we've handled all missing values
remaining_missing = df.isnull().sum().sum()
print(f"\nRemaining missing values after cleaning: {remaining_missing}")

# Feature engineering
print("\n==== Feature Engineering ====")

# Create day of week feature for reported issues
if 'Issue_reported at' in df.columns:
    df['issue_day_of_week'] = df['Issue_reported at'].dt.day_name()
    print("Created 'issue_day_of_week' feature")

# Create hour of day feature for reported issues
if 'Issue_reported at' in df.columns:
    df['issue_hour'] = df['Issue_reported at'].dt.hour
    print("Created 'issue_hour' feature")

# Create binary flag for customer remarks
if 'Customer Remarks' in df.columns:
    df['has_customer_remarks'] = df['Customer Remarks'].apply(lambda x: 0 if x == 'Unknown' else 1)
    print("Created 'has_customer_remarks' binary feature")

# Prepare data for modeling
# For this example, we'll predict CSAT Score using the available features
print("\n==== Preparing Data for Modeling ====")

# Define target variable
target = 'CSAT Score'
print(f"Target variable: {target}")

# Display distribution of target variable
print("\nTarget variable distribution:")
print(df[target].value_counts())

# Select features (excluding unique identifiers and dates)
exclude_cols = ['Unique id', 'Order_id', 'order_date_time', 'Issue_reported at', 
                'issue_responded', 'Survey_response_Date', target, 'Customer Remarks']
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\nSelected {len(feature_cols)} features for modeling")

# Split the data into features and target
X = df[feature_cols]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features ({len(categorical_features)}): {categorical_features[:5]}...")
print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Train and evaluate three ML models
print("\n==== Training and Evaluating Models ====")

# 1. Logistic Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 3. Gradient Boosting
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Function to evaluate models
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cm

# Evaluate each model
results = {}
models = [
    (lr_pipeline, "Logistic Regression"),
    (rf_pipeline, "Random Forest"),
    (gb_pipeline, "Gradient Boosting")
]

for model, name in models:
    trained_model, accuracy, cm = evaluate_model(model, name, X_train, X_test, y_train, y_test)
    results[name] = {
        'model': trained_model,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

# Compare model performance
print("\n==== Model Comparison ====")
accuracies = {name: results[name]['accuracy'] for name in results}
for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

best_model_name = max(accuracies, key=accuracies.get)
print(f"\nBest performing model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")

# Feature importance for the best model (if it's Random Forest or Gradient Boosting)
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    best_model = results[best_model_name]['model']
    
    # Try to get feature importances
    try:
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'cat':
                # Get the one-hot encoded feature names
                encoder = transformer.named_steps['onehot']
                encoded_features = encoder.get_feature_names_out(columns)
                feature_names.extend(encoded_features)
            else:
                feature_names.extend(columns)
        
        # Get feature importances
        importances = best_model.named_steps['classifier'].feature_importances_
        
        # Only show top 15 features if there are many
        if len(feature_names) > 15:
            indices = np.argsort(importances)[-15:]
            plt.figure(figsize=(10, 8))
            plt.title(f'Top 15 Feature Importances - {best_model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("\nFeature importance plot saved as 'feature_importance.png'")
        
    except Exception as e:
        print(f"Could not extract feature importances: {e}")

print("\nAnalysis complete! The dataset has been cleaned and three ML models have been trained.")

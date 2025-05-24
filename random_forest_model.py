import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RANDOM FOREST MODEL FOR FLIPKART CUSTOMER SUPPORT DATA")
print("="*80)

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('Customer_support_data.csv')
print(f"Dataset shape: {df.shape}")

# Display missing values
print("\nChecking missing values...")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_df[missing_df['Missing Values'] > 0])

# Data cleaning
print("\nCleaning the dataset...")

# Convert datetime columns
datetime_cols = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
for col in datetime_cols:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            print(f"Warning: Could not convert {col} to datetime.")

# Calculate response time
if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
    try:
        df['response_time_minutes'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
        print(f"Created 'response_time_minutes' feature. Mean response time: {df['response_time_minutes'].mean():.2f} minutes")
    except:
        print("Warning: Could not calculate response time.")

# Handle missing values
# 1. Fill categorical columns with mode
cat_cols = ['channel_name', 'category', 'Sub-category', 'Tenure Bucket', 'Agent Shift']
for col in cat_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)

# 2. Fill Order_id with unique placeholders
if 'Order_id' in df.columns and df['Order_id'].isnull().sum() > 0:
    missing_order_count = df['Order_id'].isnull().sum()
    df.loc[df['Order_id'].isnull(), 'Order_id'] = [f'generated_order_{i}' for i in range(missing_order_count)]

# 3. Fill numerical columns with median
num_cols = ['Item_price', 'connected_handling_time']
for col in num_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

# 4. Fill text columns with 'Unknown'
text_cols = ['Customer_City', 'Product_category', 'Customer Remarks']
for col in text_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna('Unknown', inplace=True)

# 5. Fill order_date_time with Issue_reported at as proxy
if 'order_date_time' in df.columns and df['order_date_time'].isnull().sum() > 0:
    df['order_date_time'] = df['order_date_time'].fillna(df['Issue_reported at'])

# Feature engineering
print("\nPerforming feature engineering...")

# Create day of week feature for reported issues
if 'Issue_reported at' in df.columns:
    df['issue_day_of_week'] = df['Issue_reported at'].dt.day_name()

# Create hour of day feature for reported issues
if 'Issue_reported at' in df.columns:
    df['issue_hour'] = df['Issue_reported at'].dt.hour

# Create binary flag for customer remarks
if 'Customer Remarks' in df.columns:
    df['has_customer_remarks'] = df['Customer Remarks'].apply(lambda x: 0 if x == 'Unknown' else 1)

# Verify no missing values remain
remaining_missing = df.isnull().sum().sum()
print(f"Remaining missing values after cleaning: {remaining_missing}")

# Prepare data for modeling
print("\nPreparing data for Random Forest model...")

# Define target variable (CSAT Score)
target = 'CSAT Score'
print(f"Target variable: {target}")
print(f"Target variable distribution:\n{df[target].value_counts()}")

# Select features (excluding identifiers and dates)
exclude_cols = ['Unique id', 'Order_id', 'order_date_time', 'Issue_reported at', 
                'issue_responded', 'Survey_response_Date', target, 'Customer Remarks']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Split the data
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

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

# Build Random Forest model pipeline
print("\nTraining Random Forest model...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating Random Forest model...")
y_pred = rf_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('random_forest_confusion_matrix.png')
print("Confusion matrix saved as 'random_forest_confusion_matrix.png'")

# Get feature importance
try:
    # Get feature names
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            # Get the one-hot encoded feature names
            encoder = transformer.named_steps['onehot']
            encoded_features = encoder.get_feature_names_out(columns)
            feature_names.extend(encoded_features)
        else:
            feature_names.extend(columns)
    
    # Get feature importances from Random Forest
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    
    # Create a DataFrame of feature importance
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Display top 15 features
    top_features = feature_importance.head(15)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['Importance'], align='center')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.title('Top 15 Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')
    print("Feature importance plot saved as 'random_forest_feature_importance.png'")
    
except Exception as e:
    print(f"Could not extract feature importance: {e}")

print("\nRandom Forest model training and evaluation complete!")

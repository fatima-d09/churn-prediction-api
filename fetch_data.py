import psycopg2
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Database Connection
engine = create_engine("postgresql+psycopg2://fatoumatadembele:@localhost:5432/customer_churn")
query = "SELECT * FROM customers;"
df = pd.read_sql(query, engine)

# Convert Dates to Numeric Features
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
df['customer_tenure'] = (datetime.today() - df['signup_date']).dt.days
df['days_since_last_purchase'] = (datetime.today() - df['last_purchase_date']).dt.days
df.drop(columns=['signup_date', 'last_purchase_date'], inplace=True)

# Convert Boolean Columns
df['subscription_status'] = df['subscription_status'].astype(int)
df['churn_label'] = df['churn_label'].astype(int)  # Fix target column

# Handle Missing Values
df.fillna(0, inplace=True)

# Feature Engineering
df['avg_purchase_value'] = df['total_spent'] / df['purchase_frequency']
df['recent_engagement'] = 1 / (df['days_since_last_purchase'] + 1)

df.fillna(0, inplace=True)  # Handle missing values

# Define Features & Target
X = df.drop(columns=['customer_id', 'churn_label'])
y = df['churn_label']

# Adjust Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Train Random Forest Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = clf.predict(X_test)

# Convert accuracy to percentage
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Print formatted metrics
for label in ['0', '1']:
    precision = report[label]['precision'] * 100
    recall = report[label]['recall'] * 100
    f1 = report[label]['f1-score'] * 100
    support = report[label]['support']
    print(f"\nClass {label} → Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1:.2f}%, Support: {support}")

# Print overall performance
macro_avg = report['macro avg']['f1-score'] * 100
weighted_avg = report['weighted avg']['f1-score'] * 100
print(f"\nMacro Avg F1-Score: {macro_avg:.2f}%")
print(f"Weighted Avg F1-Score: {weighted_avg:.2f}%\n")

# Train XGBoost Model
clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [5, 10, 20],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Minimum samples per split
    'min_samples_leaf': [1, 2, 4]  # Minimum samples per leaf
}

# Run Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Random Forest Parameters: {grid_search.best_params_}")

# Train best model
best_rf = grid_search.best_estimator_

# Show Processed Data
print(df)
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt

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
df['churn_label'] = df['churn_label'].astype(int)

# Feature Engineering
df['avg_purchase_value'] = df['total_spent'] / df['purchase_frequency']
df['recent_engagement'] = 1 / (df['days_since_last_purchase'] + 1)
df.fillna(0, inplace=True)

# Define Features & Target
X = df.drop(columns=['customer_id', 'churn_label'])
y = df['churn_label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define Random Forest Parameter Grid
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20]
}

# Train Random Forest
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Define XGBoost Parameter Grid
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Train XGBoost
grid_search_xgb = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=3, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_

# Save Best XGBoost Model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

# 📊 Churn Visualization
churn_counts = df['churn_label'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(['Not Churned', 'Churned'], churn_counts, color=['green', 'red'])
plt.xlabel('Customer Status')
plt.ylabel('Count')
plt.title('Customer Churn Distribution')
plt.savefig('churn_distribution.png')
plt.show()

print("Best XGBoost Model Saved!")

# 🔥 Model Performance Comparison
rf_accuracy = accuracy_score(y_test, best_rf.predict(X_test)) * 100
xgb_accuracy = accuracy_score(y_test, best_xgb.predict(X_test)) * 100

print(f"Random Forest Accuracy: {rf_accuracy:.2f}%")
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}%")

# Print best hyperparameters
print("Best Random Forest Parameters:", grid_search.best_params_)
print("Best XGBoost Parameters:", grid_search_xgb.best_params_)

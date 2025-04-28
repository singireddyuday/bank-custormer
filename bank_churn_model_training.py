# Credit Card Churn Prediction Project (Fixed for deployment with Flask frontend)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
import joblib
import os

# Load dataset
dataset_path = r"Bank Customer Churn Prediction.csv"
data = pd.read_csv(dataset_path)

# Check columns in dataset
print("Columns in dataset:", data.columns.tolist())

# Drop unnecessary columns if they exist
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'customer_id', 'unused_variable']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True, errors='ignore')

# Rename target column for clarity
if 'churn' in data.columns:
    data.rename(columns={'churn': 'Churn'}, inplace=True)
elif 'Exited' in data.columns:
    data.rename(columns={'Exited': 'Churn'}, inplace=True)

# Encode categorical variables and drop original columns
if 'country' in data.columns:
    data['Geography'] = data['country'].astype('category').cat.codes
    data.drop('country', axis=1, inplace=True)
if 'gender' in data.columns:
    data['Gender'] = data['gender'].astype('category').cat.codes
    data.drop('gender', axis=1, inplace=True)

# Rename other columns to expected names
column_renames = {
    'products_number': 'NumOfProducts',
    'credit_card': 'HasCrCard',
    'active_member': 'IsActiveMember',
    'estimated_salary': 'EstimatedSalary',
    'credit_score': 'CreditScore',
    'age': 'Age',
    'tenure': 'Tenure',
    'balance': 'Balance'
}
data.rename(columns=column_renames, inplace=True)

# Prepare features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Keep only numeric features
dtypes = X.dtypes
numeric_cols = dtypes[dtypes.apply(lambda x: np.issubdtype(x, np.number))].index.tolist()
X = X[numeric_cols]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and SHAP explainer (no heavy SHAP computation here)
explainer = shap.TreeExplainer(model)
joblib.dump(model, 'rf_churn_model.joblib')
joblib.dump(explainer, 'shap_explainer.joblib')

print("Model and explainer saved successfully.")

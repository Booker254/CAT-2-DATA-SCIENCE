import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import numpy as np

# Load the merged data
merged_data = pd.read_csv('merged_data.csv')

# Define features (X) and target variable (y)
features = ['Vibration', 'Temperature', 'Pressure', 'UsageHours']
target = 'MaintenanceType'

X = merged_data[features]
y = merged_data[target]

# Check for missing values in the features
if X.isnull().values.any():
    # Handle missing data using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
else:
    print("No missing values in the features.")

# Check for missing values in the target variable
if y.isnull().values.any():
    # Drop rows with missing target values
    merged_data.dropna(subset=[target], inplace=True)
    X = merged_data[features]
    y = merged_data[target]
    print("Missing values in the target variable have been removed.")
else:
    print("No missing values in the target variable.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in the training data after the split
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
y_train = y_train.dropna()  # Drop rows with missing target values

# Check for missing values in the training data after the imputation
if np.isnan(X_train).any() or y_train.isnull().any():
    raise ValueError("Training data contains missing values.")

# Create and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# Make predictions
X_test_imputed = imputer.transform(X_test)
y_pred = rf_model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Save the trained model (optional)
import joblib
joblib.dump(rf_model, 'predictive_maintenance_model.joblib')

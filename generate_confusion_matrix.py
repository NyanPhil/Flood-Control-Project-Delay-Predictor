"""
Generate confusion matrix for the Random Forest model
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model and preprocessors
print("Loading model and preprocessors...")
model = joblib.load('random_forest_model.joblib')
imputer_numeric = joblib.load('imputer_numeric.joblib')
encoder = joblib.load('encoder.joblib')

# Load training data
print("Loading training data...")
df = pd.read_csv('data/dpwh_flood_control_projects.csv')

# Define features
numerical_features = [
    'ApprovedBudgetForContract', 'ContractCost', 'ProjectDuration',
    'AverageDuration_Region', 'AverageDuration_TypeOfWork', 'ContractorCount',
    'FundingYear', 'ProjectLatitude', 'ProjectLongitude',
    'ProvincialCapitalLatitude', 'ProvincialCapitalLongitude'
]

categorical_features = [
    'MainIsland', 'Region', 'Province', 'LegislativeDistrict',
    'Municipality', 'DistrictEngineeringOffice', 'TypeOfWork', 'ProvincialCapital'
]

# Prepare data
print("Preparing data...")
df['StartDate'] = pd.to_datetime(df['StartDate'])
df['ActualCompletionDate'] = pd.to_datetime(df['ActualCompletionDate'])
df['ProjectDuration'] = (df['ActualCompletionDate'] - df['StartDate']).dt.days

# Calculate averages for feature engineering
avg_by_region = df.groupby('Region')['ProjectDuration'].mean().to_dict()
avg_by_work_type = df.groupby('TypeOfWork')['ProjectDuration'].mean().to_dict()
global_avg = 250.0

# Engineer features
df['AverageDuration_Region'] = df['Region'].map(avg_by_region).fillna(global_avg)
df['AverageDuration_TypeOfWork'] = df['TypeOfWork'].map(avg_by_work_type).fillna(global_avg)

# Create target variable (1 = On Time, 0 = Delayed)
median_duration = df['ProjectDuration'].median()
df['OnTime'] = (df['ProjectDuration'] <= median_duration).astype(int)

# Select features for model
X = df[numerical_features + categorical_features].copy()
y = df['OnTime'].copy()

# Handle missing values in numerical features
X[numerical_features] = imputer_numeric.transform(X[numerical_features])

# One-hot encode categorical features
X_encoded = encoder.transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

# Combine numerical and encoded features
X_final = pd.concat([X[numerical_features].reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_final)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives (TN):  {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP):  {cm[1,1]}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y, y_pred, target_names=['Delayed', 'On Time']))

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
            xticklabels=['Delayed', 'On Time'],
            yticklabels=['Delayed', 'On Time'])
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', cbar=True, ax=axes[1],
            xticklabels=['Delayed', 'On Time'],
            yticklabels=['Delayed', 'On Time'])
axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix visualization saved as 'confusion_matrix.png'")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print("\n" + "="*60)
print("ADDITIONAL METRICS")
print("="*60)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"F1-Score:             {f1:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"Out of {len(y)} projects:")
print(f"  - {cm[0,0]} delayed projects correctly identified")
print(f"  - {cm[0,1]} delayed projects incorrectly predicted as on-time")
print(f"  - {cm[1,0]} on-time projects incorrectly predicted as delayed")
print(f"  - {cm[1,1]} on-time projects correctly identified")

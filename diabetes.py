import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")
print("✅ Data loaded successfully!")
print(f"\nShape: {data.shape}\n")

# Check for missing values
print("Missing values:")
print(data.isnull().sum())

# Remove duplicates
duplicates = data.duplicated().sum()
print(f"\nDuplicates: {duplicates}")
data = data.drop_duplicates()
print(f"✅ Data cleaned. Shape: {data.shape}\n")

# Encode categorical variables
data["gender"] = data["gender"].map({"Male": 1, "Female": 0, "Other": 2})
data["smoking_history"] = data["smoking_history"].map({
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
})

# Define features and target
selected_features = ["HbA1c_level", "blood_glucose_level", "bmi", "age", "smoking_history"]
x = data[selected_features]
y = data["diabetes"]

# Split data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: ({x_train.shape[0]}, {x_train.shape[1]}), Val: ({x_val.shape[0]}, {x_val.shape[1]}), Test: ({x_test.shape[0]}, {x_test.shape[1]})")
print(f"Class distribution in training set: {Counter(y_train)}")

# Scale numeric features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(x_train_scaled, y_train)
print("✅ Model trained successfully!\n")

# Predictions and evaluation
y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("📊 Model Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}\n")
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Save pipeline
pipeline = {
    "model": model,
    "scaler": scaler,
    "features": selected_features
}
joblib.dump(pipeline, "diabetes_rf_pipeline.joblib")

meta = {
    "threshold": 0.85,
    "features": selected_features,
    "model_type": "RandomForestClassifier",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}
with open("diabetes_meta.json", "w") as f:
    json.dump(meta, f, indent=4)

print("💾 Saved: diabetes_rf_pipeline.joblib and diabetes_meta.json")

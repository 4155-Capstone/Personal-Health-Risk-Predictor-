import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("heart.csv")  

# -----------------------
# Identify categorical and numeric columns
# -----------------------
# Columns from your CSV
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['target']]

# -----------------------
# Encode categorical variables
# -----------------------
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Save encoders for later use in Streamlit

# -----------------------
# Features and target
# -----------------------
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Scale numeric features
# -----------------------
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -----------------------
# Train RandomForest model
# -----------------------
model = RandomForestClassifier(
    n_estimators=300, max_depth=15,
    min_samples_split=10, min_samples_leaf=4,
    random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)

# -----------------------
# Evaluate model
# -----------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# Save pipeline and metadata
# -----------------------
pipeline = {
    "model": model,
    "scaler": scaler,
    "label_encoders": le_dict,
    "features": list(X.columns)
}
joblib.dump(pipeline, "heart_pipeline.joblib")

meta = {
    "features": list(X.columns),
    "threshold": 0.5  # Set threshold for high-risk
}
with open("heart_meta.json", "w") as f:
    json.dump(meta, f)

print("Heart Disease model training complete and artifacts saved.")

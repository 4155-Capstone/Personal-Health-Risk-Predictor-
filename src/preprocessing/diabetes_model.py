import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# Load Data
# -----------------------
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        print("Data loaded. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print("File not found!")
        exit()

# -----------------------
# Clean Data
# -----------------------
def clean_data(df):
    df = df.drop_duplicates()
    return df

# -----------------------
# Preprocess & Use ALL Features
# -----------------------
def preprocess_data(df):
    df = df.copy()

    le_gender = LabelEncoder()
    le_smoke = LabelEncoder()

    df["gender"] = le_gender.fit_transform(df["gender"])
    df["smoking_history"] = le_smoke.fit_transform(df["smoking_history"])

    selected_features = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender",
        "smoking_history",
        "hypertension",
        "heart_disease"
    ]

    X = df[selected_features]
    y = df["diabetes"]

    print("Using ALL features:", selected_features)
    return X, y, selected_features, le_gender, le_smoke

# -----------------------
# Split Data
# -----------------------
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("Class balance:", Counter(y_train))
    return X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------
# Scale Data
# -----------------------
def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        scaler
    )

# -----------------------
# Train Model
# -----------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model trained")
    return model

# -----------------------
# Evaluate Model
# -----------------------
def evaluate_model(model, X_test, y_test, threshold):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print("\nModel Performance")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1-score :", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, y_probs), 4))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

# -----------------------
# Save Artifacts
# -----------------------
def save_artifacts(model, scaler, selected_features, threshold):
    pipeline = {
        "model": model,
        "scaler": scaler,
        "features": selected_features
    }
    joblib.dump(pipeline, "diabetes_rf_pipeline.joblib")

    meta = {"features": selected_features, "threshold": threshold}
    with open("diabetes_meta.json", "w") as f:
        json.dump(meta, f)

    print("Saved diabetes_rf_pipeline.joblib & diabetes_meta.json")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    df = load_data("diabetes_prediction_dataset.csv")
    df = clean_data(df)

    X, y, feats, le_gender, le_smoke = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

    model = train_model(X_train_s, y_train)

    threshold = 0.85
    evaluate_model(model, X_test_s, y_test, threshold)

    save_artifacts(model, scaler, feats, threshold)

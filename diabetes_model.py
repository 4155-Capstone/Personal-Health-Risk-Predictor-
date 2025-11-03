import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# Load and Inspect Data
# -----------------------
def load_data(filename):
    """Load the dataset and show basic information."""
    try:
        df = pd.read_csv(filename)
        print("✅ Data loaded successfully!\n")
        print("Shape:", df.shape)
        print("\nMissing values:\n", df.isnull().sum())
        print("\nDuplicates:", df.duplicated().sum())
        return df
    except FileNotFoundError:
        print("❌ File not found! Please check the path.")
        exit()

# -----------------------
# Data Cleaning
# -----------------------
def clean_data(df):
    """Remove duplicates and format categorical columns."""
    df = df.drop_duplicates()
    for col in ['gender', 'smoking_history']:
        df[col] = df[col].astype('category')
    print("✅ Data cleaned. Shape:", df.shape)
    return df

# -----------------------
# Feature Engineering
# -----------------------
def preprocess_data(df):
    """Encode categorical columns and select top features."""
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in ['gender', 'smoking_history']:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected_features = importances.head(5).index.tolist()

    print("\nSelected Features:", selected_features)
    X_selected = X[selected_features]
    return X_selected, y, selected_features

# -----------------------
# Split Data
# -----------------------
def split_data(X, y):
    """Split data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("Class distribution in training set:", Counter(y_train))
    return X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------
# Scale Data
# -----------------------
def scale_data(X_train, X_val, X_test):
    """Standardize numeric data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# -----------------------
# Train Model
# -----------------------
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        min_samples_split=10, min_samples_leaf=4,
        random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    return model

# -----------------------
# Evaluate Model
# -----------------------
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and print metrics."""
    y_probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.85
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)

    print("\n📊 Model Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # Optional: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return acc

# -----------------------
# Save Model
# -----------------------
def save_model(model, scaler, filename="diabetes_rf_model.pkl"):
    """Save trained model and scaler using joblib."""
    joblib.dump({"model": model, "scaler": scaler}, filename)
    print(f"💾 Model and scaler saved as '{filename}'")

# -----------------------
# Predict New Input
# -----------------------
def predict_input(model, scaler, input_data):
    """Predict diabetes for new input data."""
    np_array = np.asarray(input_data).reshape(1, -1)
    scaled = scaler.transform(np_array)
    prediction = model.predict(scaled)[0]
    return "🩸 Diabetic" if prediction == 1 else "💚 Non-Diabetic"

# -----------------------
# Main Program
# -----------------------
if __name__ == "__main__":
    # Load and prepare data
    data = load_data("diabetes_prediction_dataset.csv")
    data = clean_data(data)

    X, y, selected_features = preprocess_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(X_train, X_val, X_test)

    # Train and evaluate model
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)

    # Save trained model
    save_model(model, scaler)

    # Example prediction (using selected features order)
    example = [45, 27.3, 145, 1, 0]  # age, bmi, glucose, gender, smoke
    print("\nExample Prediction:", predict_input(model, scaler, example))

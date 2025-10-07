import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -----------------------
# Load and Inspect Data
# -----------------------
def load_data(filename):
    try:
        data = pd.read_csv(filename)
        print("Data loaded successfully!")
        print("\nDataset Info:")
        print(data.info())
        print("\nMissing values:\n", data.isnull().sum())
        return data
    except FileNotFoundError:
        print("File not found! Please check the path.")
        exit()

# -----------------------
# Preprocess & Split
# -----------------------
def preprocess_split(data):
    X = data.drop(columns='target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2, stratify=y
    )
    return X_train, X_test, y_train, y_test

# -----------------------
# Train Model
# -----------------------
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)  # increase iterations
    model.fit(X_train, y_train)
    return model

# -----------------------
# Evaluate Model
# -----------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")

# -----------------------
# Predict New Data
# -----------------------
def predict_input(model, input_data):
    np_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(np_array)[0]

    if prediction == 1:
        return "Affected by Heart Disease"
    else:
        return " Healthy Heart"

# -----------------------
# Save Model
# -----------------------
def save_model(model, filename="heart_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# -----------------------
# Main Program
# -----------------------
if __name__ == "__main__":
    data = load_data("heart1.csv")
    X_train, X_test, y_train, y_test = preprocess_split(data)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # Example prediction
    input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)
    print("\nPrediction for sample input:", predict_input(model, input_data))

    # Save trained model
    save_model(model)

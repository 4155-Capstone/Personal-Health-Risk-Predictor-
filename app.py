import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    return model, scaler, X.columns.tolist(), acc

df = load_data()
model, scaler, cols, acc = train_model(df)

st.title("Heart Disease Predictor")
st.caption(f"Trained on uploaded dataset • Test accuracy: {acc:.3f}")

with st.form("inputs"):
    age = st.number_input("Age", min_value=1, max_value=120, value=40, step=1)
    sex = st.selectbox("Sex", ["M", "F"])
    cp = st.selectbox("Chest Pain Type", sorted(df["ChestPainType"].unique().tolist()))
    rbp = st.number_input("RestingBP", min_value=0, max_value=300, value=120, step=1)
    chol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=200, step=1)
    fbs = st.selectbox("FastingBS", [0,1])
    recg = st.selectbox("RestingECG", sorted(df["RestingECG"].unique().tolist()))
    maxhr = st.number_input("MaxHR", min_value=0, max_value=300, value=150, step=1)
    exang = st.selectbox("ExerciseAngina", ["N","Y"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    st_slope = st.selectbox("ST_Slope", sorted(df["ST_Slope"].unique().tolist()))
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingBP": rbp,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "RestingECG": recg,
        "MaxHR": maxhr,
        "ExerciseAngina": exang,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }
    X_input = pd.DataFrame([row])
    X_input = pd.get_dummies(X_input, drop_first=True)
    for c in cols:
        if c not in X_input.columns:
            X_input[c] = 0
    X_input = X_input[cols]
    Xs = scaler.transform(X_input)
    proba = model.predict_proba(Xs)[0][1]
    pred = int(proba >= 0.5)
    st.subheader("Result")
    st.metric(label="Predicted Probability of Heart Disease", value=f"{proba:.1%}")
    st.write("Prediction:", "Positive" if pred==1 else "Negative")
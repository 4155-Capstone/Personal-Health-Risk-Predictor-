from cryptography.fernet import Fernet
import base64
import os
import joblib

# --------------------
# Password for encryption
# --------------------
password = "Diabetes1!"  # Keep secret
key = base64.urlsafe_b64encode(password.encode().ljust(32, b"\0"))
cipher = Fernet(key)

# --------------------
# Paths
# --------------------
input_file = "models/diabetes/diabetes_rf_pipeline.joblib"
output_file = "models/diabetes/diabetes_rf_pipeline.enc"

# Encrypt
with open(input_file, "rb") as f:
    data = f.read()

encrypted_data = cipher.encrypt(data)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "wb") as f:
    f.write(encrypted_data)

print("Diabetes model encrypted successfully:", output_file)

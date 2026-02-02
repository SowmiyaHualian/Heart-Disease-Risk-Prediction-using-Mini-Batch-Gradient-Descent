import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os


# -----------------------------
# Load dataset
# -----------------------------
def load_data(path):
    return pd.read_csv(path)


# -----------------------------
# Detect target column
# -----------------------------
def detect_target_column(df):
    for col in ["target", "num", "output", "label"]:
        if col in df.columns:
            print(f"🎯 Target column detected: '{col}'")
            return col
    raise ValueError("❌ No target column found")


# -----------------------------
# Clean dataset
# -----------------------------
def clean_dataset(df):
    print("🧹 Cleaning dataset...")

    # Rename typo if present
    if "thalch" in df.columns:
        df = df.rename(columns={"thalch": "thalach"})

    # Drop non-feature columns
    drop_cols = []
    for col in ["id", "dataset"]:
        if col in df.columns:
            drop_cols.append(col)

    df = df.drop(columns=drop_cols)

    return df


# -----------------------------
# Split features and target
# -----------------------------
def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# -----------------------------
# Encode categorical features
# -----------------------------
def encode_categorical_features(X):
    categorical_cols = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "thal"
    ]

    encoders = {}

    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    return X, encoders


# -----------------------------
# Scale features
# -----------------------------
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# -----------------------------
# Train-test split
# -----------------------------
def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


# -----------------------------
# Save processed data
# -----------------------------
def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)


# -----------------------------
# Complete preprocessing pipeline
# -----------------------------
def preprocess_pipeline():
    RAW_DATA_PATH = r"D:\Heart Disease prediction\data\raw\heart_disease.csv"
    PROCESSED_DATA_PATH = r"D:\Heart Disease prediction\data\processed"

    print("📥 Loading data...")
    df = load_data(RAW_DATA_PATH)

    print("📊 Columns found:")
    print(df.columns)

    target_col = detect_target_column(df)

    df = clean_dataset(df)

    print("✂️ Splitting features and target...")
    X, y = split_features_target(df, target_col)

    print("🔤 Encoding categorical features...")
    X_encoded, encoders = encode_categorical_features(X)

    print("📏 Scaling features...")
    X_scaled, scaler = scale_features(X_encoded)

    print("🔀 Train-test split...")
    X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)

    print("💾 Saving processed data...")
    save_processed_data(
        X_train, X_test, y_train, y_test,
        PROCESSED_DATA_PATH
    )

    print("✅ Preprocessing completed successfully.")
    return X_train, X_test, y_train, y_test, encoders, scaler


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    preprocess_pipeline()

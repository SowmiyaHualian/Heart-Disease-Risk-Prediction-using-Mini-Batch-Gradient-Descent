print("🔥 train.py FILE LOADED")

import numpy as np
import os
from model import LogisticRegressionScratch
from mbgd import MiniBatchGradientDescent
from sklearn.metrics import accuracy_score

print("📦 Imports successful")

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = r"D:\Heart Disease prediction\data\processed"
MODEL_DIR = r"D:\Heart Disease prediction\models"


# -----------------------------
# Helper functions
# -----------------------------
def binarize_labels(y):
    # 0 → no disease, 1–4 → disease
    return np.where(y == 0, 0, 1)


def remove_nan_rows(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


# -----------------------------
# Main training
# -----------------------------
def main():
    print("🚀 main() function ENTERED")
    print("📁 Loading data from:", DATA_DIR)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print("✅ Data loaded")

    # Binarize labels
    y_train = binarize_labels(y_train)
    y_test = binarize_labels(y_test)

    # Remove NaN rows
    X_train, y_train = remove_nan_rows(X_train, y_train)
    X_test, y_test = remove_nan_rows(X_test, y_test)

    print("🧹 NaN rows removed")
    print("🔢 Clean X_train shape:", X_train.shape)
    print("🔢 Clean y_train values:", np.unique(y_train))

    # Initialize model
    model = LogisticRegressionScratch(learning_rate=0.01)

    optimizer = MiniBatchGradientDescent(
        model=model,
        learning_rate=0.01,
        batch_size=32,
        epochs=20
    )

    print("\n🚀 TRAINING STARTED\n")
    optimizer.fit(X_train, y_train)

    print("\n📊 EVALUATION")

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:  {test_acc * 100:.2f}%")

    # -----------------------------
    # Save trained model
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    np.save(os.path.join(MODEL_DIR, "weights.npy"), model.weights)
    np.save(os.path.join(MODEL_DIR, "bias.npy"), model.bias)

    print("💾 Model saved successfully")


# -----------------------------
# Entry point
# -----------------------------
main()

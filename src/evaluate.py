import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from model import LogisticRegressionScratch

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = r"D:\Heart Disease prediction\data\processed"
MODEL_DIR = r"D:\Heart Disease prediction\models"
RESULTS_DIR = r"D:\Heart Disease prediction\results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------
def binarize_labels(y):
    return np.where(y == 0, 0, 1)


def remove_nan_rows(X, y):
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


# -----------------------------
# Load data
# -----------------------------
def load_data():
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_test, y_test


# -----------------------------
# Load trained model
# -----------------------------
def load_model():
    model = LogisticRegressionScratch(learning_rate=0.01)
    model.weights = np.load(os.path.join(MODEL_DIR, "weights.npy"))
    model.bias = np.load(os.path.join(MODEL_DIR, "bias.npy"))
    return model


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    print("📊 Evaluation started...")

    X_test, y_test = load_data()
    y_test = binarize_labels(y_test)
    X_test, y_test = remove_nan_rows(X_test, y_test)

    model = load_model()

    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    print("✅ Evaluation completed")
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall   : {rec:.2f}")
    print(f"F1-score : {f1:.2f}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()

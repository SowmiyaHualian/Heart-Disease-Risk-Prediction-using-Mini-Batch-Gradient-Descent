import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    # -----------------------------
    # Sigmoid function
    # -----------------------------
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # -----------------------------
    # Forward pass
    # -----------------------------
    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    # -----------------------------
    # Binary prediction
    # -----------------------------
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    # -----------------------------
    # Loss (Binary Cross Entropy)
    # -----------------------------
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9  # avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    # -----------------------------
    # Initialize parameters
    # -----------------------------
    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0

import numpy as np


class MiniBatchGradientDescent:
    def __init__(self, model, learning_rate=0.01, batch_size=32, epochs=20):
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.model.initialize_parameters(n_features)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = self.model.predict_proba(X_batch)

                dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / len(y_batch)) * np.sum(y_pred - y_batch)

                self.model.weights -= self.lr * dw
                self.model.bias -= self.lr * db

            loss = self.model.compute_loss(
                y,
                self.model.predict_proba(X)
            )
            print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {loss:.4f}")

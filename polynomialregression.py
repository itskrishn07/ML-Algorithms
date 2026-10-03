import numpy as np

class PolynomialRegressionGD:

    def __init__(self, degree=2, learning_rate=0.01, epochs=1000):
        self.degree = degree
        self.lr = learning_rate
        self.epochs = epochs

        self.coef_ = None
        self.intercept_ = None
        self.cost_history = []

    def _transform(self, X):
        X = np.asarray(X)

        features = []

        for power in range(1, self.degree + 1):
            features.append(X ** power)

        return np.hstack(features)

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        X_poly = self._transform(X)

        n_samples, n_features = X_poly.shape

        self.coef_ = np.zeros((n_features, 1))
        self.intercept_ = 0.0

        for epoch in range(self.epochs):

            # Prediction
            y_pred = X_poly @ self.coef_ + self.intercept_

            # Error
            error = y_pred - y

            # Cost
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)

            self.cost_history.append(cost)

            # Gradients
            dw = (1 / n_samples) * (X_poly.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

        self.coef_ = self.coef_.flatten()
        self.intercept_ = float(self.intercept_)

        return self

    def predict(self, X):

        X = np.asarray(X)

        if self.coef_ is None:
            raise RuntimeError("Model must be fitted first.")

        X_poly = self._transform(X)

        return X_poly @ self.coef_ + self.intercept_
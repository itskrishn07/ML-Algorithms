import numpy as np


class LogisticRegressionGD:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs

        self.coef_ = None
        self.intercept_ = None
        self.cost_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.coef_ = np.zeros((n_features, 1))
        self.intercept_ = 0.0

        for epoch in range(self.epochs):

            # 1. Linear calculation
            z = X @ self.coef_ + self.intercept_

            # 2. Convert to probability
            y_pred = self._sigmoid(z)

            # 3. Error
            error = y_pred - y

            # 4. Log Loss
            epsilon = 1e-15

            y_pred_clipped = np.clip(
                y_pred,
                epsilon,
                1 - epsilon
            )

            cost = -(1 / n_samples) * np.sum(
                y * np.log(y_pred_clipped)
                +
                (1 - y) * np.log(1 - y_pred_clipped)
            )

            self.cost_history.append(cost)

            # 5. Gradients
            dw = (1 / n_samples) * (X.T @ error)

            db = (1 / n_samples) * np.sum(error)

            # 6. Update parameters
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

        self.coef_ = self.coef_.flatten()
        self.intercept_ = float(self.intercept_)

        return self

    def predict_proba(self, X):

        X = np.asarray(X)

        z = X @ self.coef_ + self.intercept_

        return self._sigmoid(z)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)
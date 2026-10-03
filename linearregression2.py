import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """Initializes hyperparameters, weights, and tracking attributes."""
        self.lr = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.cost_history = []  # To track error reduction over time

    def fit(self, X, y):
        """Fits the model using Batch Gradient Descent."""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1) # Ensure y is a column vector
        print(X[:10])
        print(y[:10])
        
        n_samples, n_features = X.shape
        
        # 1. Initialize weights (coefficients) and intercept to zero
        self.coef_ = np.zeros((n_features, 1))
        self.intercept_ = 0.0
        
        # 2. Gradient Descent Loop
        for epoch in range(self.epochs):
            # Compute current predictions: y_hat = XW + b
            y_pred = (X @ self.coef_) + self.intercept_
            
            
            # Compute the Mean Squared Error (Cost)
            error = y_pred - y
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # Calculate gradients (derivatives of the cost function)
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update parameters by moving opposite to the gradient
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db
            
        # Flatten coef_ back to a 1D array to match sklearn format
        self.coef_ = self.coef_.flatten()
        # Convert intercept to a standard float scalar
        self.intercept_ = float(self.intercept_)
        
        return self

    def predict(self, X):
        """Predicts target values using the learned parameters."""
        X = np.asarray(X)
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        return (X @ self.coef_) + self.intercept_

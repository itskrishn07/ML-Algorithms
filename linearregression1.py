import numpy as np

class LinearRegressionFromScratch:
    def __init__(self):
        """Initializes the model weights and intercept attributes."""
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fits the linear model using the Closed-Form Normal Equation.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) or (n_samples, 1)
        """
        # Ensure inputs are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # 1. Add a column of ones to X to represent the intercept/bias term
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]

        # 2. Calculate weights using the Normal Equation: (X^T * X)^-1 * X^T * y
        # We use np.linalg.pinv (pseudo-inverse) to avoid errors if X^T * X is singular
        weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

        # 3. Extract the intercept (first element) and coefficients (remaining elements)
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        
        return self

    def predict(self, X):
        """
        Predicts target values using the linear model.
        X: numpy array of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("This LinearRegression instance is not fitted yet. Call 'fit' first.")
            
        # Formula: y = X * coefficients + intercept
        return X @ self.coef_ + self.intercept_
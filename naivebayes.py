import numpy as np


class GaussianNaiveBayes:

    def __init__(self):
        self.classes_ = None
        self.means_ = {}
        self.variances_ = {}
        self.priors_ = {}

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for c in self.classes_:

            # Select samples belonging to class c
            X_c = X[y == c]

            # Prior probability
            self.priors_[c] = len(X_c) / len(X)

            # Mean and variance for each feature
            self.means_[c] = np.mean(
                X_c,
                axis=0
            )

            self.variances_[c] = np.var(
                X_c,
                axis=0
            )

        return self

    def _gaussian_probability(
        self,
        x,
        mean,
        variance
    ):

        epsilon = 1e-9

        variance = variance + epsilon

        numerator = np.exp(
            -((x - mean) ** 2)
            / (2 * variance)
        )

        denominator = np.sqrt(
            2 * np.pi * variance
        )

        return numerator / denominator

    def predict(self, X):

        X = np.asarray(X)

        predictions = []

        for x in X:

            class_scores = {}

            for c in self.classes_:

                prior = self.priors_[c]

                probabilities = self._gaussian_probability(
                    x,
                    self.means_[c],
                    self.variances_[c]
                )

                likelihood = np.prod(probabilities)

                class_scores[c] = prior * likelihood

            prediction = max(
                class_scores,
                key=class_scores.get
            )

            predictions.append(prediction)

        return np.array(predictions)
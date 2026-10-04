import numpy as np


class KNNClassifier:

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.X_train = X
        self.y_train = y

        return self

    def _euclidean_distance(self, x1, x2):

        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):

        X = np.asarray(X)

        predictions = []

        for x in X:

            # Calculate distance from x to every training point
            distances = []

            for x_train in self.X_train:

                distance = self._euclidean_distance(
                    x,
                    x_train
                )

                distances.append(distance)

            distances = np.array(distances)

            # Get indices of K nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]

            # Get their labels
            nearest_labels = self.y_train[nearest_indices]

            # Majority vote
            values, counts = np.unique(
                nearest_labels,
                return_counts=True
            )

            prediction = values[np.argmax(counts)]

            predictions.append(prediction)

        return np.array(predictions)
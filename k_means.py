import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):

        X = np.asarray(X, dtype=float)

        # Random generator
        rng = np.random.default_rng(self.random_state)

        # --------------------------------
        # Step 1: Initialize centroids
        # --------------------------------

        random_indices = rng.choice(
            len(X),
            size=self.k,
            replace=False
        )

        self.centroids = X[random_indices].copy()

        # --------------------------------
        # Repeat assignment/update
        # --------------------------------

        for _ in range(self.max_iters):

            # Step 2: Calculate distances
            distances = np.zeros(
                (len(X), self.k)
            )

            for i in range(self.k):
                distances[:, i] = np.sqrt(
                    np.sum(
                        (X - self.centroids[i]) ** 2,
                        axis=1
                    )
                )

            # Step 3: Assign points
            labels = np.argmin(
                distances,
                axis=1
            )

            # --------------------------------
            # Step 4: Calculate new centroids
            # --------------------------------

            new_centroids = np.zeros_like(
                self.centroids
            )

            for i in range(self.k):

                cluster_points = X[labels == i]

                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(
                        cluster_points,
                        axis=0
                    )
                else:
                    # Keep old centroid if cluster empty
                    new_centroids[i] = self.centroids[i]

            # --------------------------------
            # Step 5: Check convergence
            # --------------------------------

            if np.allclose(
                self.centroids,
                new_centroids
            ):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        self.labels_ = labels

        # --------------------------------
        # Calculate inertia
        # --------------------------------

        self.inertia_ = 0

        for i in range(self.k):

            cluster_points = X[self.labels_ == i]

            self.inertia_ += np.sum(
                (cluster_points - self.centroids[i]) ** 2
            )

        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        distances = np.zeros(
            (len(X), self.k)
        )

        for i in range(self.k):

            distances[:, i] = np.sqrt(
                np.sum(
                    (X - self.centroids[i]) ** 2,
                    axis=1
                )
            )

        return np.argmin(
            distances,
            axis=1
        )
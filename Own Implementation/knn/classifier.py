import numpy as np

class knn_classifier:

    def __init__(self, n_neighbours: int, distance_metric=None, weights = "uniform"):
        """
            Initialize the knn_classifier class.

            Parameters
            ----------
            n_neighbours : int
                Number of neighbors to use for classification.

            distance_metric : {'euclidean', 'manhattan'}, default=None
                Distance metric used to compute distances between samples.

            weights : {'uniform', 'distance'}, default='uniform'
                Weighting scheme used in voting. If 'uniform', all neighbors
                contribute equally. If 'distance', closer neighbors have
                greater influence.
        """
        self.n_neighbours = n_neighbours
        self.distance_metric = distance_metric
        self.weights = weights

        self._fitted = False
        
        self.classes_ = None
        self._class_to_index = None
        self._index_to_class = None

    def _validate_X(self, X):
        """
            Validate input feature data.

            Ensures that the input array is two-dimensional, non-empty,
            and array-like.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input feature matrix.

            Returns
            -------
            X : ndarray of shape (n_samples, n_features)
                Validated input data.

            Raises
            ------
            ValueError
                If the input array is empty or has invalid shape.
            TypeError
                If the input cannot be converted to an array.
        """
        if X is None:
            raise ValueError("X cannot be None")
        
        try:
            X = np.asarray(X)
        except Exception as e:
            raise TypeError("X must be array-like!") from e

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        
        if X.shape[0]==0:
            raise ValueError("X must contain at least one sample")

    def _validate_y(self, y):
        """
            Validate target labels.

            Ensures that the target array is one-dimensional, non-empty,
            and array-like.

            Parameters
            ----------
            y : array-like of shape (n_samples,)
                Target labels.

            Returns
            -------
            y : ndarray of shape (n_samples,)
                Validated target labels.

            Raises
            ------
            ValueError
                If the target array is empty or has invalid shape.
            TypeError
                If the target cannot be converted to an array.
        """
        if y is None:
            raise ValueError("y cannot be None")
        
        try:
            y = np.asarray(y)
        except Exception as e:
            raise TypeError("y must be array-like!") from e

        if y.ndim != 1:
            raise ValueError("y must be a 2-dimensional array")
        
        if y.shape[0]==0:
            raise ValueError("y must contain at least one label")

    def _check_is_fitted(self):
        """
            Verify that the estimator has been fitted.

            Raises
            ------
            RuntimeError
                If the estimator has not been fitted yet.
        """
        if not getattr(self, "_fitted", False):
            raise RuntimeError(
                "This KNNClassifier instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

    def _validate_predict_input(self, X):
        """
            Validate input data for prediction.

            Ensures that the estimator is fitted and that the input array
            has the correct shape, dimensionality, and numeric type expected
            by the model.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples to validate.

            Returns
            -------
            X : ndarray of shape (n_samples, n_features)
                Validated input data.

            Raises
            ------
            ValueError
                If the input does not have the expected shape or is empty.
            TypeError
                If the input data is not numeric.
        """
        self._check_is_fitted()

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, "
                f"got {X.shape[1]}"
            )
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must contain numeric features")
        
        return X
    
    def _compute_distances(self, x):
        """
            Compute distances between a query sample and all training samples.

            The distance metric is determined by the ``distance_metric`` parameter
            and can be either Euclidean or Manhattan distance.

            Parameters
            ----------
            x : ndarray of shape (n_features,)
                Query sample.

            Returns
            -------
            distances : ndarray of shape (n_train_samples,)
                Distances from the query sample to each training sample.

            Raises
            ------
            ValueError
                If an unsupported distance metric is specified.
        """
        if self.distance_metric.lower() == "euclidean":
            return np.linalg.norm(self._X_train - x, axis=1)
        elif self.distance_metric.lower() == "manhattan":
            return np.sum(np.abs(self._X_train - x), axis=1)
        else:
            raise ValueError("Unknown distance metric")

    def _get_k_neighbour_indices(self, distances):
        """
            Identify indices of the k nearest neighbors.

            Parameters
            ----------
            distances : ndarray of shape (n_train_samples,)
                Distances from a query sample to each training sample.

            Returns
            -------
            neighbour_idx : ndarray of shape (k,)
                Indices of the k nearest neighbors.

            Raises
            ------
            ValueError
                If ``n_neighbours`` is greater than the number of training samples.
        """
        n_train = distances.shape[0]

        if self.n_neighbours > n_train:
            raise ValueError(
                f"The value of n_neighbours ({self.n_neighbours}) cannot be greater than number of trianing samples ({n_train})"
            )
        
        neighbour_idx = np.argpartition(distances, self.n_neighbours-1)[:self.n_neighbours]

        return neighbour_idx

    def _majority_vote(self, neighbour_labels):
        """
            Compute unweighted (majority) vote counts for each class.

            Each neighbor contributes equally to the vote, and the count of
            neighbors per class is returned.

            Parameters
            ----------
            neighbour_labels : ndarray of shape (k,)
                Class labels of the k nearest neighbors.

            Returns
            -------
            vote_counts : ndarray of shape (n_classes,)
                Unweighted vote counts for each class.
        """
        return np.bincount(
            neighbour_labels,
            minlength=self._n_classes
        ).astype(float)

    def _weighted_vote(self, neighbour_labels, neighbour_distances):
        """
            Compute distance-weighted vote counts for each class.

            Each neighbor contributes a weight inversely proportional to its
            distance, so that closer neighbors have a larger influence on
            the final vote.

            Parameters
            ----------
            neighbour_labels : ndarray of shape (k,)
                Class labels of the k nearest neighbors.

            neighbour_distances : ndarray of shape (k,)
                Distances corresponding to the k nearest neighbors.

            Returns
            -------
            vote_counts : ndarray of shape (n_classes,)
                Distance-weighted vote counts for each class.
        """
        vote_counts = np.zeros(self._n_classes, dtype=float)

        for label, dist in zip(neighbour_labels, neighbour_distances):
            vote_counts[label] += 1.0 / (dist + 1e-6)

        return vote_counts

    def _decode_label(self, index_):
        """
            Map an internal class index to the corresponding original label.

            Parameters
            ----------
            index_ : int
                Internal class index.

            Returns
            -------
            label : object
                Original class label corresponding to the given index.

            Raises
            ------
            ValueError
                If the index is outside the valid range.
        """
        if not 0 <= index_ < len(self.classes_):
            raise ValueError(f"Invalid class index: {index_}")
        return self.classes_[index_]

    def _resolve_ties(self, vote_counts, neighbour_labels, neighbour_distances):
        """
            Resolve ties between classes with equal vote counts.

            Ties are broken deterministically using successive criteria:
            (1) smallest total distance to neighbors of the tied classes,
            (2) smallest distance to the closest neighbor,
            and finally a stable class-order fallback.
            A small numerical offset is added to the selected class to
            ensure a unique maximum without affecting normalization.

            Parameters
            ----------
            vote_counts : array-like of shape (n_classes,)
                Raw vote counts for each class.

            neighbour_labels : ndarray of shape (k,)
                Class labels of the k nearest neighbors.

            neighbour_distances : ndarray of shape (k,)
                Distances corresponding to the k nearest neighbors.

            Returns
            -------
            vote_counts : ndarray of shape (n_classes,)
                Vote counts with ties resolved.
        """
        vote_counts = np.asarray(vote_counts, dtype=float).copy()

        max_vote = np.max(vote_counts)
        tied_classes = np.flatnonzero(vote_counts == max_vote)

        if tied_classes.size == 1:
            return vote_counts

        # ---- total distance tie-break ----
        total_dist = []

        for class_ in tied_classes:
            class_mask = neighbour_labels == class_
            total_dist.append(neighbour_distances[class_mask].sum())

        total_dist = np.asarray(total_dist)
        min_distance = np.min(total_dist)
        best = tied_classes[total_dist == min_distance]

        if best.size == 1:
            vote_counts[best[0]] += 1e-12
            return vote_counts

        # ---- closest neighbour tie-break ----
        closest_dist = []

        for class_ in best:
            class_mask = neighbour_labels == class_
            closest_dist.append(neighbour_distances[class_mask].min())

        closest_dist = np.asarray(closest_dist)
        min_closest = np.min(closest_dist)
        best = best[closest_dist == min_closest]

        if best.size == 1:
            vote_counts[best[0]] += 1e-12
            return vote_counts

        # ---- deterministic fallback ----
        vote_counts[np.sort(best)[0]] += 1e-12

        return vote_counts

    def fit(self, X, y):
        """
            Store training data and prepare internal structures for KNN classification.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training feature matrix.
            y : array-like of shape (n_samples,)
                Target labels.

            Returns
            -------
            self : object
                Fitted estimator.
        """
        
        # Validate inputs
        self._validate_X(X)
        self._validate_y(y)

        # Convert to internal format
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Store training data
        self._X_train = X
        self._y_train = y
        self._n_samples = n_samples
        self._n_features = n_features

        # Initialise metadata
        classes = np.unique(y)

        self.classes_ = classes
        self._n_classes = len(classes)

        self._class_to_index = {
            label: idx for idx, label in enumerate(classes)
        }

        self._index_to_class = {
            idx: label for label, idx in self._class_to_index.items()
        }

        # Encode labels for internal use
        self._y_train = np.array(
            [self._class_to_index[label] for label in y],
            dtype=int
        )

        # Mark model as fitted
        self._fitted = True

        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = self._validate_predict_input(X)

        probs = self.predict_prob(X)
        class_indices = np.argmax(probs, axis=1)

        return np.asarray([self._decode_label(idx) for idx in class_indices])

    def predict_prob(self, X):

        """
        Estimate class probabilities for samples in X.

        The class probabilities are computed based on the votes of the
        k nearest neighbors of each sample. Voting can be uniform or
        distance-weighted, depending on the ``weights`` parameter.
        Ties are resolved before normalizing vote counts into probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        prob : ndarray of shape (n_samples, n_classes)
            Estimated class probabilities for each sample.
        """
         
        self._check_is_fitted()
        X = self._validate_predict_input(X)

        n_samples = X.shape[0]
        n_classes = self._n_classes

        prob = np.zeros((n_samples, n_classes), dtype=float)

        for i, x in enumerate(X):
            distances = self._compute_distances(x)

            neighbour_indices = self._get_k_neighbour_indices(distances)
            neighbour_labels = self._y_train[neighbour_indices]
            neighbour_distances = distances[neighbour_indices]

            # ---- raw vote counting ----
            
            if self.weights.lower() == "uniform":
                vote_counts = self._majority_vote(neighbour_labels)

            elif self.weights.lower() == "distance":
                vote_counts = self._weighted_vote(neighbour_labels, neighbour_distances)

            else:
                raise ValueError(f"Unknown weighting scheme: {self.weights}")

            # ---- tie resolution on RAW votes ----

            vote_counts = self._resolve_ties(
                vote_counts=vote_counts,
                neighbour_labels=neighbour_labels,
                neighbour_distances=neighbour_distances,
            )

            # ---- normalize exactly once ----

            total = vote_counts.sum()
            if total == 0:
                raise RuntimeError("Vote counts sum to zero after tie resolution")

            prob[i] = vote_counts / total

        return prob
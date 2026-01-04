from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(
    n_neighbours=5,
    weights="uniform",
    distance_metric="euclidean",
):
    """
    Factory function for a sklearn KNN classifier.

    Parameters
    ----------
    n_neighbours : int, default=5
        Number of nearest neighbors
    weights : {"uniform", "distance"}, default="uniform"
        Weight function used in prediction
    distance_metric : str, default="euclidean"
        Distance metric for neighbor search

    Returns
    -------
    KNeighborsClassifier
        Unfitted sklearn KNN classifier
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbours,
        weights=weights,
        metric=distance_metric,
    )

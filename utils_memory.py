import os
import psutil

def measure_fit_ram_mb(model, X_train, y_train):
    """
        Measure memory usage during model fitting.

        Estimates the additional RAM consumed by the ``fit`` method
        by comparing the process memory usage before and after fitting.

        Parameters
        ----------
        model : estimator
            Unfitted model implementing a ``fit`` method.

        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.

        y_train : array-like of shape (n_samples,)
            Training target labels.

        Returns
        -------
        delta_memory : float
            Estimated increase in memory usage (in megabytes) during fitting.
    """
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / 1024**2

    model.fit(X_train, y_train)

    mem_after = process.memory_info().rss / 1024**2

    return mem_after - mem_before
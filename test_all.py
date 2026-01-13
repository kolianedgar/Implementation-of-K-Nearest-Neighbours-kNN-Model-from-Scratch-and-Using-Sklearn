import time

from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils_memory import measure_fit_ram_mb
from knn.data_loader import load_dataset
from knn.classifier import knn_classifier
from knn.utils import cross_validate_knn
from knn.metrics import (
    macro_f1,
    macro_recall,
    macro_sensitivity,
    macro_roc_auc,
    categorical_cross_entropy,
    multiclass_brier_score,
    expected_calibration_error,
)
from knn.reporting import print_cv_results, print_test_results

# -------------------------------------------------
# Dataset selection
# -------------------------------------------------
DATASETS = [
    {"source": "builtin", "name": "iris"},
    # {"source": "builtin", "name": "wine"},
    # {"source": "builtin", "name": "digits"},
]

# -------------------------------------------------
# Hyperparameter grid (NO TUNING)
# -------------------------------------------------
PARAM_GRID = {
    "n_neighbours": list(range(3, 22, 2)),
    "weights": ["uniform", "distance"],
    "distance_metric": ["euclidean", "manhattan"],
}

# -------------------------------------------------
# Storage for later export
# -------------------------------------------------
all_results = []

for ds in DATASETS:
    print("\n" + "#" * 80)
    print("DATASET:", ds)
    print("#" * 80)

    # -------------------------------
    # Dataset loading (TIMED)
    # -------------------------------

    t0 = time.perf_counter()

    X, y = load_dataset(**ds)

    dataset_load_time = time.perf_counter() - t0

    # -------------------------------
    # Train / test split
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # -------------------------------
    # Scaling
    # -------------------------------

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # Exhaustive hyperparameter loop
    # -------------------------------

    for n_neighbors, weights, metric in product(
        PARAM_GRID["n_neighbours"],
        PARAM_GRID["weights"],
        PARAM_GRID["distance_metric"],
    ):
        print("\n" + "-" * 80)
        print(
            f"PARAMETERS → "
            f"n_neighbours={n_neighbors}, "
            f"weights={weights}, "
            f"distance_metric={metric}"
        )
        print("-" * 80)

        # ---------------------------
        # Cross-validation (TIMED)
        # ---------------------------

        def model_factory():
            return knn_classifier(
                n_neighbours=n_neighbors,
                weights=weights,
                distance_metric=metric,
            )

        t0 = time.perf_counter()
        cv_results = cross_validate_knn(
            X=X_train,
            y=y_train,
            model_factory=model_factory,
            cv=5,
        )
        cv_time = time.perf_counter() - t0

        print_cv_results(cv_results, title="CROSS-VALIDATION RESULTS")

        # ---------------------------
        # Training (TIMED + RAM)
        # ---------------------------
        model = model_factory()

        t0 = time.perf_counter()

        fit_ram_mb = measure_fit_ram_mb(model, X_train, y_train)
        
        fit_time = time.perf_counter() - t0

        # ---------------------------
        # Testing (TIMED)
        # ---------------------------

        t0 = time.perf_counter()

        y_pred = model.predict(X_test)
        y_prob = model.predict_prob(X_test)

        test_time = time.perf_counter() - t0

        test_results = {
            "macro_f1": macro_f1(y_test, y_pred),
            "macro_recall": macro_recall(y_test, y_pred),
            "macro_sensitivity": macro_sensitivity(y_test, y_pred),
            "macro_roc_auc": macro_roc_auc(y_test, y_prob),
            "cross_entropy": categorical_cross_entropy(y_test, y_prob),
            "brier": multiclass_brier_score(y_test, y_prob),
            "ece": expected_calibration_error(y_test, y_prob),
        }

        print_test_results(test_results)

        # ---------------------------
        # Timing & memory report
        # ---------------------------
        print("\nPERFORMANCE METRICS")
        print(f"Dataset load time     : {dataset_load_time:.6f} s")
        print(f"Cross-validation time : {cv_time:.6f} s")
        print(f"Training time (fit)   : {fit_time:.6f} s")
        print(f"Testing time          : {test_time:.6f} s")
        print(f"Training RAM usage    : {fit_ram_mb:.4f} MB")

        # ---------------------------
        # Collect everything
        # ---------------------------
        record = {
            "dataset": ds,
            "n_neighbours": n_neighbors,
            "weights": weights,
            "distance_metric": metric,

            "dataset_load_time_s": dataset_load_time,
            "cv_time_s": cv_time,
            "fit_time_s": fit_time,
            "test_time_s": test_time,
            "fit_ram_mb": fit_ram_mb,
        }

        for name, (mean, std) in cv_results.items():
            record[f"cv_{name}_mean"] = mean
            record[f"cv_{name}_std"] = std

        for name, value in test_results.items():
            record[f"test_{name}"] = value

        all_results.append(record)

print("\nFinished evaluation.")
print(f"Total configurations evaluated: {len(all_results)}")

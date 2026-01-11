import time
import pandas as pd

from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


from utils_memory import measure_fit_ram_mb

from knn.data_loader import load_dataset
from knn.utils import cross_validate_knn, evaluate_on_dataset

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "tests" / "data"

# -------------------------------------------------
# Dataset selection
# -------------------------------------------------
DATASETS = [
    {"source": "builtin", "name": "iris"},
    {"source": "builtin", "name": "wine"},
    {"source": "builtin", "name": "digits"},
    {"source": "builtin", "name": "breast_cancer"},
    {
        "source": "csv",
        "filepath": DATA_DIR / "agaricus-lepiota.data",
        "target_column": 0,
        "header": None,
    },
    {
        "source": "csv",
        "filepath": DATA_DIR / "zoo.csv",
        "target_column": "class_type",
        "header": 0,
        "encode_features": True,
        "drop_columns": ["animal_name"],
    },
]


# -------------------------------------------------
# Exhaustive grid
# -------------------------------------------------
PARAM_GRID = {
    "n_neighbours": list(range(3, 22, 2)),
    "weights": ["uniform", "distance"],
    "distance_metric": ["euclidean", "manhattan"],
}


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    rows = []

    for ds in DATASETS:
        # ---------------------------------------------
        # Load dataset
        # ---------------------------------------------
        t0 = time.time()
        X, y = load_dataset(**ds)
        dataset_load_time = time.time() - t0

        # ---------------------------------------------
        # Train / test split
        # ---------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # ---------------------------------------------
        # Scaling (MANDATORY for KNN)
        # ---------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---------------------------------------------
        # Exhaustive evaluation
        # ---------------------------------------------
        for n_neighbors, weights, metric in product(
            PARAM_GRID["n_neighbours"],
            PARAM_GRID["weights"],
            PARAM_GRID["distance_metric"],
        ):
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
            )

            # -----------------------------------------
            # Cross-validation
            # -----------------------------------------
            t0 = time.time()
            cv_results = cross_validate_knn(
                model=model,
                X=X_train,
                y=y_train,
                cv=5,
            )
            cv_time = time.time() - t0

            # -----------------------------------------
            # Fit + RAM measurement
            # -----------------------------------------
            t0 = time.time()
            fit_ram_mb = measure_fit_ram_mb(
                model,
                X_train,
                y_train,
            )
            fit_time = time.time() - t0

            # -----------------------------------------
            # Test evaluation
            # -----------------------------------------
            t0 = time.time()
            test_results = evaluate_on_dataset(
                model=model,
                X=X_test,
                y=y_test,
            )
            test_time = time.time() - t0

            # -----------------------------------------
            # Collect row
            # -----------------------------------------

            if "name" in ds:
                dataset_name = ds["name"]
            else:
                dataset_name = Path(ds["filepath"]).name

            row = {
                "implementation": "sklearn",
                "dataset": dataset_name,
                "n_neighbours": n_neighbors,
                "weights": weights,
                "distance_metric": metric,

                "dataset_load_time_sec": dataset_load_time,
                "cv_time_sec": cv_time,
                "fit_time_sec": fit_time,
                "test_time_sec": test_time,
                "fit_ram_mb": fit_ram_mb,
            }

            # CV metrics (mean ± std kept)
            for metric_name, (mean, std) in cv_results.items():
                row[f"cv_{metric_name}_mean"] = mean
                row[f"cv_{metric_name}_std"] = std

            # Test metrics
            for metric_name, value in test_results.items():
                row[f"test_{metric_name}"] = value

            rows.append(row)

    # -------------------------------------------------
    # Export
    # -------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv("sklearn_knn_exhaustive_results.csv", index=False)


if __name__ == "__main__":
    main()

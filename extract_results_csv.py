import time
import pandas as pd

from pathlib import Path
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
    # {"source": "builtin", "name": "wine"},
    # {"source": "builtin", "name": "digits"},
    # {"source": "csv", "filepath": "data/mydata.csv", "target_column": "label"},
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
# Storage
# -------------------------------------------------
rows = []

for ds in DATASETS:
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
    # Exhaustive grid loop
    # -------------------------------
    for n_neighbors, weights, metric in product(
        PARAM_GRID["n_neighbours"],
        PARAM_GRID["weights"],
        PARAM_GRID["distance_metric"],
    ):
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

        # ---------------------------
        # Test metrics
        # ---------------------------
        test_metrics = {
            "test_macro_f1": macro_f1(y_test, y_pred),
            "test_macro_recall": macro_recall(y_test, y_pred),
            "test_macro_sensitivity": macro_sensitivity(y_test, y_pred),
            "test_macro_roc_auc": macro_roc_auc(y_test, y_prob),
            "test_cross_entropy": categorical_cross_entropy(y_test, y_prob),
            "test_brier": multiclass_brier_score(y_test, y_prob),
            "test_ece": expected_calibration_error(y_test, y_prob),
        }

        # ---------------------------
        # Single output row
        # ---------------------------
        row = {
            "dataset": ds.get("name", ds.get("filepath")),
            "n_neighbours": n_neighbors,
            "weights": weights,
            "distance_metric": metric,

            "dataset_load_time_s": dataset_load_time,
            "cv_time_s": cv_time,
            "fit_time_s": fit_time,
            "test_time_s": test_time,
            "fit_ram_mb": fit_ram_mb,
        }

        # CV means only (NO std)
        for metric_name, (mean, _) in cv_results.items():
            row[f"cv_{metric_name}"] = mean

        row.update(test_metrics)
        rows.append(row)

# -------------------------------------------------
# Export to CSV
# -------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv("knn_exhaustive_results.csv", index=False)

print(f"Exported {len(df)} rows to knn_exhaustive_results.csv")

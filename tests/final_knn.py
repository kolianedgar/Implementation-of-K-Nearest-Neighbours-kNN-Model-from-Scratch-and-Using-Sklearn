import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path

from knn.classifier import knn_classifier
from knn.utils import (
    grid_search_knn,
    cross_validate_knn,
    evaluate_on_dataset,
)
from knn.data_loader import load_dataset
from knn.reporting import *

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "tests" / "data"

# -------------------------------------------------
# Dataset selection
# -------------------------------------------------
DATASETS = [
    {
        "source": "csv",
        "filepath": DATA_DIR / "zoo.data",
        "target_column": -1,
        "header": None,
        "encode_features": True,
        "drop_columns": [0]
    },
]

for ds in DATASETS:
    print("\n" + "#" * 70)
    print("DATASET:", ds)
    print("#" * 70)

    X, y = load_dataset(**ds)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    best_params = grid_search_knn(
        X_train,
        y_train,
        param_grid={
            "n_neighbours": list(range(3, 22, 2)),
            "weights": ["uniform", "distance"],
            "distance_metric": ["euclidean", "manhattan"],
        },
        cv=5
    )

    print("\nBest parameters:", best_params)

    def model_factory():
        return knn_classifier(**best_params)

    cv_results = cross_validate_knn(X_train, y_train, model_factory, cv=5)
    print_cv_results(cv_results, "CROSS-VALIDATION RESULTS")

    final_model = knn_classifier(**best_params)
    final_model.fit(X_train, y_train)

    test_results = evaluate_on_dataset(final_model, X_test, y_test)
    print_test_results(test_results)

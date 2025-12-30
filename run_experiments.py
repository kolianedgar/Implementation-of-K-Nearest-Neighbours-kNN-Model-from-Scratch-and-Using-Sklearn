import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataset
from utils_own_knn import (
    grid_search_knn,
    cross_validate_knn,
    print_cv_results,
    evaluate_on_dataset,
    print_test_results,
)
from knn_classifier import *

# -------------------------------------------------
# Dataset selection
# -------------------------------------------------
DATASETS = [
    {"source": "builtin", "name": "iris"},
    {"source": "builtin", "name": "wine"},
    {"source": "builtin", "name": "digits"},
    # {"source": "csv", "filepath": "data/mydata.csv", "target_column": "label"},
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

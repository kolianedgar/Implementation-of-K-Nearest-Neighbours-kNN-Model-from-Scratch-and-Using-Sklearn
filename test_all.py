import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from knn.data_loader import (
    load_dataset,
)

from knn.utils import (
    grid_search_knn,
    cross_validate_knn,
    evaluate_on_dataset,
)

from knn.reporting import (
    print_cv_results,
    print_test_results,
)

# -------------------------------------------------
# Dataset selection
# -------------------------------------------------
DATASETS = [
    {"source": "builtin", "name": "iris"},
    # {"source": "builtin", "name": "wine"},
    # {"source": "builtin", "name": "digits"},
    # {"source": "csv", "filepath": "tests/data/mydata.csv", "target_column": "label"},
]

for ds in DATASETS:
    print("\n" + "#" * 80)
    print("DATASET CONFIGURATION")
    print(ds)
    print("#" * 80)

    # -------------------------------------------------
    # Load dataset (dataset-agnostic)
    # -------------------------------------------------
    X, y = load_dataset(**ds)

    # -------------------------------------------------
    # Train / test split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------------------------
    # Scaling (MANDATORY for KNN)
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------------------------
    # Hyperparameter search (TRAINING SET ONLY)
    # -------------------------------------------------
    best_params = grid_search_knn(
        X_train,
        y_train,
        param_grid={
            "n_neighbours": list(range(3, 22, 2)),
            "weights": ["uniform", "distance"],
            "distance_metric": ["euclidean", "manhattan"],
        },
        cv=5,
    )

    print("\nBest parameters:")
    print("Best parameters:", best_params)

    # -------------------------------------------------
    # Cross-validation with best model (TRAINING SET)
    # -------------------------------------------------
    best_model = KNeighborsClassifier(
        n_neighbors=best_params["n_neighbours"],
        weights=best_params["weights"],
        metric=best_params["distance_metric"],
    )
    
    cv_results = cross_validate_knn(
        model=best_model,
        X=X_train,
        y=y_train,
        cv=5,
    )

    print_cv_results(cv_results, title="CROSS-VALIDATION RESULTS")

    # -------------------------------------------------
    # Final test set evaluation
    # -------------------------------------------------
    best_model.fit(X_train, y_train)

    test_results = evaluate_on_dataset(
        model=best_model,
        X=X_test,
        y=y_test,
    )

    print_test_results(test_results)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from knn.data_loader import (
    load_dataset,
)

from knn.classifier import (
    knn_classifier,
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
    # Load dataset
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
    # Hyperparameter tuning (TRAINING SET ONLY)
    # -------------------------------------------------
    best_params = grid_search_knn(
        X=X_train,
        y=y_train,
        param_grid={
            "n_neighbours": list(range(3, 22, 2)),
            "weights": ["uniform", "distance"],
            "distance_metric": ["euclidean", "manhattan"],
        },
        cv=5,
    )

    print("\nBest parameters (custom KNN):")
    print(best_params)

    # -------------------------------------------------
    # Model factory using best hyperparameters
    # -------------------------------------------------
    def model_factory():
        return knn_classifier(
            n_neighbours=best_params["n_neighbours"],
            weights=best_params["weights"],
            distance_metric=best_params["distance_metric"],
        )

    # -------------------------------------------------
    # Cross-validation with best model (TRAINING SET)
    # -------------------------------------------------
    cv_results = cross_validate_knn(
        X=X_train,
        y=y_train,
        model_factory=model_factory,
        cv=5,
    )

    print_cv_results(
        cv_results,
        title="CROSS-VALIDATION RESULTS (CUSTOM KNN)",
    )

    # -------------------------------------------------
    # Final test set evaluation
    # -------------------------------------------------
    final_model = model_factory()
    final_model.fit(X_train, y_train)

    test_results = evaluate_on_dataset(
        model=final_model,
        X=X_test,
        y=y_test,
    )

    print_test_results(test_results)

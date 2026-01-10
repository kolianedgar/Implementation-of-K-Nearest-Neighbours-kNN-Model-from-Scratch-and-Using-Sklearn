import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

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
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "tests" / "data"

# -------------------------------------------------
# Dataset selection
# -------------------------------------------------
DATASETS = [
    {"source": "builtin", "name": "iris"},
    # {"source": "builtin", "name": "wine"},
    # {"source": "builtin", "name": "digits"},
    # {"source": "builtin", "name": "breast_cancer"},
    # {
    #     "source": "csv",
    #     "filepath": DATA_DIR / "agaricus-lepiota.data",
    #     "target_column": 0,
    #     "header": None,
    # },
    # {
    #     "source": "csv",
    #     "filepath": DATA_DIR / "zoo.csv",
    #     "target_column": "class_type",
    #     "header": 0,
    #     "encode_features": True,
    #     "drop_columns": ["animal_name"],
    # },
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
    Time_load = time.time()
    X, y = load_dataset(**ds)

    print(f"\nLoading the dataset took {time.time() - Time_load} seconds.")
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
    Time_grid_search = time.time()

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

    print(f"\nSearching the best combination of hyperparameters took {time.time() - Time_grid_search} seconds.")

    # -------------------------------------------------
    # Cross-validation with best model (TRAINING SET)
    # -------------------------------------------------

    best_model = KNeighborsClassifier(
        n_neighbors=best_params["n_neighbours"],
        weights=best_params["weights"],
        metric=best_params["distance_metric"],
    )
    
    Time_cv = time.time()

    cv_results = cross_validate_knn(
        model=best_model,
        X=X_train,
        y=y_train,
        cv=5,
    )

    print_cv_results(cv_results, title="CROSS-VALIDATION RESULTS")

    print(f"\nCross-validation took {time.time() - Time_cv} seconds.")
    # -------------------------------------------------
    # Final test set evaluation
    # -------------------------------------------------

    Time_test = time.time()

    best_model.fit(X_train, y_train)

    test_results = evaluate_on_dataset(
        model=best_model,
        X=X_test,
        y=y_test,
    )

    print_test_results(test_results)
    print(f"\nTesting the model took {time.time() - Time_test} seconds.")
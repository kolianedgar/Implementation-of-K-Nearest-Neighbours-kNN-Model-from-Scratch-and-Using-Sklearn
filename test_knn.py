from knn_classifier import *
from utils_own_knn import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load example data
X, y = load_iris(return_X_y=True)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Hyperparameter grid
# -------------------------------------------------
param_grid = {
    "n_neighbours": list(range(3, 22, 2)),
    "weights": ["uniform", "distance"],
    "distance_metric": ["euclidean", "manhattan"],
}

# -------------------------------------------------
# 1. Hyperparameter tuning
# -------------------------------------------------
print("\nStarting hyperparameter tuning...")
best_params = grid_search_knn(
    X_train,
    y_train,
    param_grid,
    cv=5
)

print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS")
print("=" * 60)
for k, v in best_params.items():
    print(f"{k:<20}: {v}")

# -------------------------------------------------
# 2. Cross-validated performance with best model
# -------------------------------------------------
def best_model_factory():
    return knn_classifier(**best_params)

cv_results = cross_validate_knn(
    X_train,
    y_train,
    best_model_factory,
    cv=5
)

print_cv_results(
    cv_results,
    title="CROSS-VALIDATION RESULTS (TRAINING SET)"
)

# -------------------------------------------------
# 3. Train final model on full training set
# -------------------------------------------------
final_model = knn_classifier(**best_params)
final_model.fit(X_train, y_train)

# -------------------------------------------------
# 4. Final evaluation on test set
# -------------------------------------------------
test_results = evaluate_on_dataset(
    final_model,
    X_test,
    y_test
)

print_test_results(test_results)
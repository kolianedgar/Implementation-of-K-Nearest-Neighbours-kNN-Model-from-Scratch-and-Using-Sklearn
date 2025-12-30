import os

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from knn_package_utils import (
    load_tabular_dataset,
    build_scoring_dict,
    summarize_cv_results
)

# ------------------------------------------------
# Load ANY dataset
# ------------------------------------------------
X, y = load_tabular_dataset(
    path=os.getcwd() + "\Iris.csv",     # <-- your dataset
    target_column="Species",      # <-- your label column
    drop_columns=None            # optional
)

# ------------------------------------------------
# Pipeline
# ------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# ------------------------------------------------
# Grid search
# ------------------------------------------------
param_grid = {
    "knn__n_neighbors": list(range(3, 22, 2)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["minkowski"],
    "knn__p": [1, 2]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid.fit(X, y)
best_model = grid.best_estimator_

print("Best parameters:")
print(grid.best_params_)
print()

# ------------------------------------------------
# Cross-validated evaluation
# ------------------------------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    estimator=best_model,
    X=X,
    y=y,
    scoring=build_scoring_dict(),
    cv=cv,
    n_jobs=-1
)

# ------------------------------------------------
# Results
# ------------------------------------------------
summary = summarize_cv_results(cv_results)

print("Cross-validated metrics (mean ± std):\n")
for metric, (mean, std) in summary.items():
    print(f"{metric}: {mean:.4f} ± {std:.4f}")
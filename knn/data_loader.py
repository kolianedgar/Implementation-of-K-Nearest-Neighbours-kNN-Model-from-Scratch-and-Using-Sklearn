import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)

# -------------------------------------------------
# Built-in datasets
# -------------------------------------------------
def load_builtin_dataset(name):
    name = name.lower()

    if name == "iris":
        X, y = load_iris(return_X_y=True)

    elif name == "wine":
        X, y = load_wine(return_X_y=True)

    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)

    elif name == "digits":
        X, y = load_digits(return_X_y=True)

    else:
        raise ValueError(f"Unknown built-in dataset: {name}")

    return np.asarray(X, dtype=float), np.asarray(y)


# -------------------------------------------------
# CSV / DATA datasets
# -------------------------------------------------
def load_csv_dataset(
    filepath,
    target_column,
    header="infer",
    drop_columns=None,
    encode_features=False,
):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(filepath, header=header)

    # ----------------------------
    # Resolve target column
    # ----------------------------
    if isinstance(target_column, int):
        if target_column < 0:
            target_column = df.shape[1] + target_column
        target_name = df.columns[target_column]
    else:
        target_name = target_column

    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found")

    y = df[target_name]
    X = df.drop(columns=[target_name])

    # ----------------------------
    # Drop non-feature columns
    # ----------------------------
    if drop_columns is not None:
        X = X.drop(columns=drop_columns)

    # ----------------------------
    # Encode features if needed
    # ----------------------------
    if encode_features:
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = LabelEncoder().fit_transform(X[col])

    # ----------------------------
    # Encode target
    # ----------------------------
    if not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.to_numpy()

    return X.to_numpy(dtype=float), np.asarray(y)

# -------------------------------------------------
# Unified loader
# -------------------------------------------------
def load_dataset(
    source,
    *,
    name=None,
    filepath=None,
    target_column=None,
    **kwargs,
):
    """
    Unified dataset loader.

    Parameters
    ----------
    source : {"builtin", "csv"}
    """

    if source == "builtin":
        if name is None:
            raise ValueError("Built-in dataset requires 'name'")
        return load_builtin_dataset(name)

    elif source == "csv":
        if filepath is None or target_column is None:
            raise ValueError("CSV dataset requires filepath and target_column")
        return load_csv_dataset(filepath, target_column, **kwargs)

    else:
        raise ValueError(f"Unknown dataset source: {source}")
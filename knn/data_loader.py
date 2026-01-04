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
    *,
    sep=",",
    header="infer",
    drop_columns=None,
    encode_target=True,
    encode_features=True,
):
    df = pd.read_csv(filepath, sep=sep, header=header)

    # Target
    y = df.iloc[:, target_column] if isinstance(target_column, int) else df[target_column]

    # Features
    X = df.drop(df.columns[target_column], axis=1)

    if drop_columns is not None:
        X = X.drop(columns=drop_columns)

    # Encode target
    if encode_target and not np.issubdtype(y.dtype, np.number):
        y = LabelEncoder().fit_transform(y)
    else:
        y = y.values

    # Encode features
    if encode_features:
        X = pd.get_dummies(X)
    else:
        X = X.values

    return X.values.astype(float), y

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
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)

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

    return np.asarray(X), np.asarray(y)

def load_csv_dataset(
    filepath,
    target_column,
    drop_columns=None,
    encoding=True,
):
    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if drop_columns is not None:
        X = X.drop(columns=drop_columns)

    # Convert to numpy
    X = X.values

    # Encode target if needed
    if encoding and not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values

    return X, y

def load_dataset(
    source,
    *,
    name=None,
    filepath=None,
    target_column=None,
    **kwargs
):
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
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_digits,
    load_breast_cancer,
)

def load_csv_dataset(
    filepath,
    target_column,
    header="infer",
    drop_columns=None,
    encode_features=True,
):
    """
    Load a CSV dataset and convert categorical data to numeric.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    target_column : int or str
        Target column index or name
    header : int or None
        Row index for header (None if no header)
    drop_columns : list[int | str], optional
        Columns to drop before splitting X/y
    encode_features : bool
        Whether to label-encode categorical features

    Returns
    -------
    X : np.ndarray
        Numeric feature matrix
    y : np.ndarray
        Encoded target labels
    """

    df = pd.read_csv(filepath, header=header)

    # ----------------------------
    # Drop unwanted columns
    # ----------------------------
    if drop_columns is not None:
        df = df.drop(columns=drop_columns)

    # ----------------------------
    # Extract target
    # ----------------------------
    if isinstance(target_column, int):
        y = df.iloc[:, target_column]
        X = df.drop(df.columns[target_column], axis=1)
    else:
        y = df[target_column]
        X = df.drop(columns=[target_column])

    # ----------------------------
    # Encode target labels
    # ----------------------------
    y = LabelEncoder().fit_transform(y)

    # ----------------------------
    # Encode categorical features
    # ----------------------------
    if encode_features:
        for col in X.columns:
            if X[col].dtype == object or isinstance(X[col], pd.CategoricalDtype):
                X[col] = LabelEncoder().fit_transform(X[col])

    # ----------------------------
    # Convert to numpy
    # ----------------------------
    X = X.to_numpy(dtype=float)
    y = y.astype(int)

    return X, y


def load_dataset(source, **kwargs):
    """
        Load a dataset from a specified source.

        Provides a unified interface for loading datasets either from
        built-in sources or from external CSV files.

        Parameters
        ----------
        source : {'builtin', 'csv'}
            Dataset source. If 'builtin', a scikit-learn built-in dataset
            is loaded. If 'csv', the dataset is loaded from a CSV file.

        **kwargs
            Additional keyword arguments controlling dataset loading.

            For ``source='builtin'``:
                name : {'iris', 'wine', 'digits', 'breast_cancer'}
                    Name of the built-in dataset to load.

            For ``source='csv'``:
                Passed directly to ``load_csv_dataset``.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.

        y : ndarray of shape (n_samples,)
            Target labels.

        Raises
        ------
        ValueError
            If an unknown dataset source or dataset name is specified.
    """

    if source == "csv":
        return load_csv_dataset(**kwargs)

    if source == "builtin":
        
        loaders = {
            "iris": load_iris,
            "wine": load_wine,
            "digits": load_digits,
            "breast_cancer": load_breast_cancer,
        }

        data = loaders[kwargs["name"]]()
        return data.data, data.target

    raise ValueError(f"Unknown dataset source: {source}")
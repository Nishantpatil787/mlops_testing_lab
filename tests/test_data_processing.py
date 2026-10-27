# tests/test_data_processing.py - unit tests of data processing functions
import numpy as np
import pandas as pd
import pytest
from data_processing import (load_data, fill_missing_values,
                             remove_duplicates, split_data, scale_features)
 
def test_load_data_shape():
    X, y = load_data()
    assert X.shape == (178, 13)
    assert len(y) == 178
 
def test_fill_missing_removes_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    assert fill_missing_values(df).isna().sum().sum() == 0
 
def test_fill_missing_uses_mean():
    df = pd.DataFrame({"a": [2.0, np.nan, 4.0]})
    assert fill_missing_values(df)["a"][1] == 3.0
 
def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 6]})
    assert len(remove_duplicates(df)) == 2
 
@pytest.mark.parametrize("test_size, n_test", [(0.2, 36), (0.3, 54)])
def test_split_data_sizes(test_size, n_test):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    assert len(X_test) == n_test
    assert len(X_train) + len(X_test) == 178
 
def test_split_data_is_repeatable():
    X, y = load_data()
    first = split_data(X, y)[1]
    second = split_data(X, y)[1]
    assert first.index.equals(second.index)
 
def test_scale_features_mean_and_std():
    X, y = load_data()
    X_train, X_test, _, _ = split_data(X, y)
    X_train_s, _ = scale_features(X_train, X_test)
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_train_s.std(axis=0), 1, atol=1e-6)

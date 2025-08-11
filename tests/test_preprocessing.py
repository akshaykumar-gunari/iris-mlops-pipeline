import pytest
import pandas as pd
from src.data_preprocessing import preprocess_and_save

def test_preprocessing_output_shape():
    """Check preprocessing returns expected columns."""
    raw_data = pd.DataFrame({
        "sepal_length": [5.1, 4.9],
        "sepal_width": [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width": [0.2, 0.2],
        "species": ["setosa", "setosa"]
    })
    X, y = preprocess_data(raw_data)
    assert X.shape[0] == 2
    assert y.shape[0] == 2

def test_preprocessing_no_missing_values():
    """Ensure preprocessing removes NaNs."""
    raw_data = pd.DataFrame({
        "sepal_length": [5.1, None],
        "sepal_width": [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width": [0.2, 0.2],
        "species": ["setosa", "setosa"]
    })
    X, y = preprocess_data(raw_data)
    assert not X.isnull().any().any()


import pickle
import numpy as np
import os
import pytest

@pytest.fixture(scope="module")
def loaded_model():
    """Load the trained model from model.pkl."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def test_model_file_exists():
    """Ensure model.pkl exists."""
    assert os.path.exists("model.pkl"), "model.pkl is missing!"

def test_model_prediction_shape(loaded_model):
    """Model should return predictions for one sample."""
    X_sample = np.random.rand(1, 4)
    prediction = loaded_model.predict(X_sample)
    assert prediction.shape == (1,)

def test_model_predicts_valid_class(loaded_model):
    """Model should return a valid Iris class (0, 1, or 2)."""
    X_sample = np.random.rand(1, 4)
    pred = loaded_model.predict(X_sample)[0]
    assert pred in [0, 1, 2]

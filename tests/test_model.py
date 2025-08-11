import pickle
import numpy as np
import os
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

@pytest.fixture(scope="module")
def loaded_model():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "model.pkl is missing!"

def test_model_prediction_shape(loaded_model):
    X_sample = np.random.rand(1, 4)
    prediction = loaded_model.predict(X_sample)
    assert prediction.shape == (1,)

def test_model_predicts_valid_class(loaded_model):
    pred = loaded_model.predict(np.random.rand(1, 4))[0]
    assert pred in [0, 1, 2]


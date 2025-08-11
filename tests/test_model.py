import mlflow.sklearn
import numpy as np
from pathlib import Path

def test_model_load_and_predict():
    model = mlflow.sklearn.load_model("../model.pkl")
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example Iris data
    prediction = model.predict(sample_input)

    assert prediction is not None
    assert len(prediction) == 1


import pytest
from app.api import app  # Flask app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_predict_endpoint(client):
    """Send a sample request to /predict and check response."""
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data
    assert json_data["prediction"] in [0, 1, 2]


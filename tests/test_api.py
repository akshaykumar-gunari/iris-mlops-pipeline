from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    json_data = response.json()

    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], list)


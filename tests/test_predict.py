# tests/test_predict.py
from fastapi.testclient import TestClient
from api.main import app

def test_predict_response():
    with TestClient(app) as client:  # important : déclenche startup/shutdown
        payload = {"text": "Hello world"}
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        # data = res.json()
        # assert "tags" in data
        # assert isinstance(data["tags"], list)
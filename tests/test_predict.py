from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_response():
    payload = {"text": "Hello world"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    # data = res.json()
    # assert "tags" in data
    # assert isinstance(data["tags"], list)
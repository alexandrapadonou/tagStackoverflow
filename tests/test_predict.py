from fastapi.testclient import TestClient
from api.main import app

def test_predict_schema():
    client = TestClient(app)
    r = client.post("/predict", json={"text": "How to parse JSON in Python?"})
    assert r.status_code == 200
    data = r.json()
    assert "tags" in data
    assert isinstance(data["tags"], list)
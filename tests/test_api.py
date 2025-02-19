from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API is running!"}

def test_predict():
    response = client.post("/predict/", json={"features": [1.2, 3.4, 5.6]})
    assert response.status_code == 200
    assert "fraud" in response.json()

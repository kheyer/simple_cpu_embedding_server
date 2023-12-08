from fastapi.testclient import TestClient

def test_model_api(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    response_dict = response.json()
    assert len(response_dict)>0
import pytest
from fastapi.testclient import TestClient
from rts.api.server import app
from rts.api.models import Feature
from .test_media import create_media
from rts.db.utils import reset_database

# Example feature data for testing
feature_data = {
    "feature_type": "test",
    "version": "0.0.1",
    "model_name": "TestModel",
    "model_params": {"param1": "value1", "param2": "value2"},
    "data": {"data1": "value1", "data2": "value2"},
    "embedding_size": 1024,
    # "embedding_1024": [1.0 for _ in range(1024)],
    # "embedding_1536": [1.0 for _ in range(1536)],
    # "embedding_2048": [1.0 for _ in range(2048)],
    "media_id": 1
}


@pytest.fixture
def db_setup():
    reset_database()


@pytest.fixture
def create_test_feature(create_media):
    response = create_media
    with TestClient(app) as client:
        response = client.post(
            "/feature/", json={**feature_data, "media_id": response.json()["media_id"]})
    return response


def test_create_feature(create_test_feature):
    response = create_test_feature
    assert response.status_code == 200
    assert response.json() == feature_data


def test_read_feature(create_test_feature):
    feature_id = create_test_feature["feature_id"]
    with TestClient(app) as client:
        response = client.get(f"/feature/{feature_id}")
    assert response.status_code == 200
    assert response.json() == feature_data


def test_read_features(create_test_feature):
    with TestClient(app) as client:
        response = client.get("/features/")
    assert response.status_code == 200
    assert feature_data in response.json()


def test_update_feature(create_test_feature):
    feature_id = create_test_feature["feature_id"]
    updated_data = {**feature_data, "version": "0.0.2"}
    with TestClient(app) as client:
        response = client.put(f"/feature/{feature_id}", json=updated_data)
    assert response.status_code == 200
    assert response.json() == updated_data


def test_delete_feature(create_test_feature):
    feature_id = create_test_feature["feature_id"]
    with TestClient(app) as client:
        response = client.delete(f"/feature/{feature_id}")
    assert response.status_code == 200
    assert response.json() == feature_data

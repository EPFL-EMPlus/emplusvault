import pytest
from fastapi.testclient import TestClient
from emv.api.server import app, mount_routers
from emv.api.api_settings import Settings, get_settings
from .conftest import reset_database, mock_authenticate
from .test_media_router import media_data
from emv.api.routers.auth_router import get_current_active_user
import random

client = TestClient(app)
settings = get_settings()
mount_routers(app, settings)

# Example feature data for testing
feature_data = {
    "feature_type": "test",
    "version": "0.0.1",
    "model_name": "TestModel",
    "model_params": {"param1": "value1", "param2": "value2"},
    "data": {"data1": "value1", "data2": "value2"},
    "embedding_size": 33,
    # "embedding_1024": [1.0 for _ in range(1024)],
    # "embedding_1536": [1.0 for _ in range(1536)],
    # "embedding_2048": [1.0 for _ in range(2048)],

    # This is the media_id of the test media, defined in test_media_router.py
    "media_id": "library-ID01",
}


@pytest.fixture
def db_setup():
    reset_database()


@pytest.fixture
def create_test_features():
    app.dependency_overrides[get_current_active_user] = mock_authenticate

    data = media_data.copy()

    for i in range(10):
        data["media_id"] = f"library-{i:02d}"
        data["media_path"] = f"/path/to/media-{i:02d}"
        data["hash"] = f"hash-{i:02d}"
        media = client.post("/media/", json=data)

        feature_data["media_id"] = f"library-{i:02d}"
        feature_data["embedding_33"] = [
            random.random() * 0.1 for _ in range(33)]
        response = client.post(
            "/feature/", json={**feature_data, "media_id": media.json()["media_id"]})
        # print(response.json())
        if response.status_code != 200:
            print("Error creating feature")
            print(response.status_code)
            print(response.json())

    # Create an additional feature that is far away from the others to test the similarity function
    data["media_id"] = f"library-{i+1:02d}"
    data["media_path"] = f"/path/to/media-{i+1:02d}"
    data["hash"] = f"hash-{i+1:02d}"
    media = client.post("/media/", json=data)

    feature_data["media_id"] = f"library-{i+1:02d}"
    feature_data["embedding_33"] = [
        1 - random.random() * 0.1 for _ in range(33)]
    response = client.post(
        "/feature/", json={**feature_data, "media_id": media.json()["media_id"]})
    if response.status_code != 200:
        print("Error creating feature")
        print(response.status_code)
        print(response.json())

    return response


def test_nearest_neighbors(create_test_features):
    response = client.get("/feature/similar/1/k/10")
    assert response.status_code == 200
    assert response.json()[-1]['media_id'] == 'library-10'

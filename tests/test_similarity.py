import pytest
from fastapi.testclient import TestClient
from emv.api.server import app, mount_routers
from emv.api.api_settings import Settings, get_settings
from .conftest import reset_database, mock_authenticate
from emv.api.routers.auth_router import get_current_active_user


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
    "embedding_size": 1024,
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
def create_test_features(create_media):
    app.dependency_overrides[get_current_active_user] = mock_authenticate
    response = create_media
    response = client.post(
        "/feature/", json={**feature_data, "media_id": create_media.json()["media_id"]})
    if response.status_code != 200:
        print("Error creating feature")
        print(response.status_code)
        print(response.json())
    return response


def nearest_neighbors(create_test_features):
    # Create some features where a few are close to each other
    ...

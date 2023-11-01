import pytest
from fastapi.testclient import TestClient
from emv.api.server import app, mount_routers
from emv.api.api_settings import Settings, get_settings
from emv.api.models import Feature
from .test_media_router import create_media, assert_media_response
from emv.api.routers.auth_router import get_current_active_user
from .conftest import reset_database, mock_authenticate
from json import JSONDecodeError
import json

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

client = TestClient(app)
settings = get_settings()
mount_routers(app, settings)


@pytest.fixture
def db_setup():
    reset_database()


@pytest.fixture
def create_test_feature(create_media):
    app.dependency_overrides[get_current_active_user] = mock_authenticate
    response = create_media
    response = client.post(
        "/feature/", json={**feature_data, "media_id": create_media.json()["media_id"]})
    return response


def test_create_feature(create_test_feature):
    response = create_test_feature
    assert response.status_code == 200
    assert_media_response(response.json(), feature_data)


def test_read_feature(create_test_feature):
    feature_id = create_test_feature.json()["feature_id"]
    response = client.get(f"/feature/{feature_id}")
    assert response.status_code == 200
    assert_media_response(response.json(), feature_data)


def test_read_features(create_test_feature):
    response = client.get("/features/")
    assert response.status_code == 200
    for row in response.json():
        assert_media_response(row, feature_data)

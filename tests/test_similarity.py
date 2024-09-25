import pytest
from fastapi.testclient import TestClient
from emv.api.server import app, mount_routers
from emv.api.api_settings import Settings, get_settings
from .conftest import reset_database, mock_authenticate
from .test_media_router import media_data
from .test_projection import create_projection_data, create_map_projection_feature_data
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


@pytest.fixture
def create_test_features_projection():
    reset_database()
    app.dependency_overrides[get_current_active_user] = mock_authenticate

    response = create_projection_data()
    data = media_data.copy()

    for i in range(10):
        data["media_id"] = f"library-{i:02d}"
        data["media_path"] = f"/path/to/media-{i:02d}"
        data["hash"] = f"hash-{i:02d}"
        media = client.post("/media/", json=data)

        feature_data["media_id"] = f"library-{i:02d}"
        feature_data["embedding_33"] = [
            1 - random.random() * 0.1 for _ in range(33)]
        response = client.post(
            "/feature/", json={**feature_data, "media_id": media.json()["media_id"]})

        create_map_projection_feature_data(
            response.json()["feature_id"], media.json()["media_id"])

    # Create an additional feature that is far away from the others to test the similarity function
    data["media_id"] = f"library-{i+1:02d}"
    data["media_path"] = f"/path/to/media-{i+1:02d}"
    data["hash"] = f"hash-{i+1:02d}"
    media = client.post("/media/", json=data)

    feature_data["media_id"] = f"library-{i+1:02d}"
    feature_data["embedding_33"] = [
        random.random() * 0.1 for _ in range(33)]
    response = client.post(
        "/feature/", json={**feature_data, "media_id": media.json()["media_id"]})

    create_map_projection_feature_data(
        response.json()["feature_id"], media.json()["media_id"])

    for i in range(10):
        # Create a few more feature of the same type, but don't add them to the projection
        data["media_id"] = f"library-{i+10:02d}"
        data["media_path"] = f"/path/to/media-{i+10:02d}"
        data["hash"] = f"hash-{i+10:02d}"
        media = client.post("/media/", json=data)

        feature_data["media_id"] = f"library-{i+10:02d}"
        feature_data["embedding_33"] = [
            random.random() * 0.1 for _ in range(33)]
        response = client.post(
            "/feature/", json={**feature_data, "media_id": media.json()["media_id"]})

    for i in range(100):
        # Create a few features with a different feature_type and without adding them to the projection to test if they don't appear
        data["media_id"] = f"library-{i+100:02d}"
        data["media_path"] = f"/path/to/media-{i+100:02d}"
        data["hash"] = f"hash-{i+100:02d}"
        media = client.post("/media/", json=data)

        feature_data["media_id"] = f"library-{i+100:02d}"
        feature_data["feature_type"] = "test2"
        feature_data["embedding_33"] = [
            random.random() * 0.1 for _ in range(33)]
        response = client.post(
            "/feature/", json={**feature_data, "media_id": media.json()["media_id"]})
        if response.status_code != 200:
            print("Error creating feature")
            print(response.status_code)
            print(response.json())

    return response


def test_nearest_neighbors_projection_by_keypoints(create_test_features_projection):

    # Example keypoints list. This is not a valid pose that could exist and is only for testing purposes
    keypoints = [
        [14.89, 14.09],
        [14.78, 14.19],
        [14.01, 14.21],
        [15.03, 14.23],
        [15.05, 14.25],
        [15.07, 14.27],
        [16.09, 14.29],
        [10, 10],  # hip1
        [20, 10],  # hip2
        [16.21, 14.31],
        [16.23, 14.33],
        [16.25, 14.35],
        [17.27, 14.37],
    ]

    # Make a POST request to the new endpoint
    response = client.post(
        "/feature/test/similar/projection/1/k/10",
        # Ensure this matches the expected format
        json={"keypoints": keypoints}
    )

    # Ensure the request was successful
    assert response.status_code == 200

    # Parse the response
    result = response.json()

    # Ensure that the response contains the correct number of neighbors
    assert len(result) == 10

    # Check that the furthest media_id is the last one, assuming the farthest point is 'library-10'
    assert result[0]['media_id'] == 'library-10'

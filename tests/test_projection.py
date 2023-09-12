import pytest
from fastapi.testclient import TestClient
from rts.api.server import app
from .conftest import reset_database, mock_authenticate
from rts.api.routers.auth_router import get_current_active_user
from rts.db.queries import create_library
from rts.api.models import LibraryCreate
import json


@pytest.fixture
def db_setup():
    reset_database()


@pytest.fixture
def create_projection(db_setup):
    app.dependency_overrides[get_current_active_user] = mock_authenticate
    with TestClient(app) as client:
        response = client.post("/projections/", json={
            "version": "1.0",
            "library_id": 1,
            "model_name": "Test Model",
            "model_params": {"param1": "value1"},
            "data": {"data1": "value1"},
            "dimension": 1,
            "atlas_folder_path": "/path/to/atlas",
            "atlas_width": 1,
            "tile_size": 1,
            "atlas_count": 1,
            "total_tiles": 1,
            "tiles_per_atlas": 1
        })
    return response.status_code, response.json()


def test_create_projection(create_projection):
    assert create_projection[0] == 200
    assert create_projection[1] == {"status": "Projection created"}


def test_read_projections(db_setup):
    with TestClient(app) as client:
        response = client.get("/projections/")
    assert response.status_code == 200
    assert response.json() == []


def test_read_projection(create_projection):
    # Assuming that a projection with id=1 exists in the database
    with TestClient(app) as client:
        response = client.get("/projections/1")
    assert response.status_code == 200
    # Assert that the response has the expected structure
    assert set(response.json().keys()) == {"projection_id", "version", "library_id", "created_at", "model_name", "model_params",
                                           "data", "dimension", "atlas_folder_path", "atlas_width", "tile_size", "atlas_count", "total_tiles", "tiles_per_atlas"}


# def test_update_projection(create_projection):
#     # Assuming that a projection with id=1 exists in the database
#     with TestClient(app) as client:
#         response = client.put("/projections/1", json={
#             "version": "1.1",
#             "library_id": 1,
#             "model_name": "Updated Model",
#             "model_params": {"param1": "value1"},
#             "data": {"data1": "value1"},
#             "dimension": 1,
#             "atlas_folder_path": "/path/to/atlas",
#             "atlas_width": 1,
#             "tile_size": 1,
#             "atlas_count": 1,
#             "total_tiles": 1,
#             "tiles_per_atlas": 1
#         })
#     assert response.status_code == 200
#     assert response.json() == {"status": "Projection updated"}


# def test_delete_projection(db_setup):
#     # Assuming that a projection with id=1 exists in the database
#     with TestClient(app) as client:
#         response = client.delete("/projections/1")
#     assert response.status_code == 200
#     assert response.json() == {"status": "Projection deleted"}

from fastapi.testclient import TestClient
from rts.api.server import app
from rts.db.utils import reset_database

client = TestClient(app)


def test_create_projection():
    reset_database()
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
    assert response.status_code == 200
    assert response.json() == {"status": "Projection created"}


def test_read_projections():
    reset_database()
    response = client.get("/projections/")
    assert response.status_code == 200
    assert response.json() == []


def test_read_projection():
    test_create_projection()
    # Assuming that a projection with id=1 exists in the database
    response = client.get("/projections/1")
    assert response.status_code == 200
    # Assert that the response has the expected structure
    assert set(response.json().keys()) == {"projection_id", "version", "library_id", "created_at", "model_name", "model_params",
                                           "data", "dimension", "atlas_folder_path", "atlas_width", "tile_size", "atlas_count", "total_tiles", "tiles_per_atlas"}


def test_update_projection():
    test_create_projection()
    # Assuming that a projection with id=1 exists in the database
    response = client.put("/projections/1", json={
        "version": "1.1",
        "library_id": 1,
        "model_name": "Updated Model",
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
    assert response.status_code == 200
    assert response.json() == {"status": "Projection updated"}


def test_delete_projection():
    reset_database()
    # Assuming that a projection with id=1 exists in the database
    response = client.delete("/projections/1")
    assert response.status_code == 200
    assert response.json() == {"status": "Projection deleted"}

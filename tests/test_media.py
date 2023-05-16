from fastapi.testclient import TestClient
import pytest
from rts.api.server import app
from rts.api.models import Media, LibraryCreate
from rts.db.utils import reset_database
from rts.db.queries import create_library
import json

@pytest.fixture
def db_setup():
    reset_database()

@pytest.fixture
def create_media(db_setup):
    create_library(LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    ))

# Example media data for testing
media_data = {
    "media_path": "/path/to/media",
    "original_path": "/original/path/to/media",
    "media_type": "video",
    "sub_type": "mp4",
    "size": 1024,
    "metadata": {"example": "metadata"},
    "library_id": 1,
    "hash": "123abc",
    "parent_id": 1,
    "start_ts": 0.0,
    "end_ts": 10.0,
    "start_frame": 0,
    "end_frame": 100,
    "frame_rate": 10.0
}


def test_create_media(create_media):
    with TestClient(app) as client:
        response = client.post("/media/", json=media_data)
    assert response.status_code == 200
    assert response.json() == media_data


def test_read_media(create_media):
    with TestClient(app) as client:
        response = client.get("/media/1")
    assert response.status_code == 200
    assert response.json() == media_data


def test_read_medias(create_media):
    with TestClient(app) as client:
        response = client.get("/media/")
    assert response.status_code == 200
    assert media_data in response.json()

def test_update_media(create_media):
    updated_data = {**media_data, "media_path": "/new/path/to/media"}
    with TestClient(app) as client:
        response = client.put("/media/1", json=updated_data)
    assert response.status_code == 200
    assert response.json() == updated_data


def test_delete_media(create_media):
    with TestClient(app) as client:
        response = client.delete("/media/1")
    assert response.status_code == 200
    assert response.json() == {"status": "Media deleted"}

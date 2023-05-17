from fastapi.testclient import TestClient
import pytest
from rts.api.server import app
from rts.api.models import Media, LibraryCreate
from rts.db.utils import reset_database
from rts.db.queries import create_library
import json
from json import JSONDecodeError


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


@pytest.fixture
def db_setup():
    reset_database()


@pytest.fixture
def create_media(db_setup: None):
    create_library(LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    ))
    with TestClient(app) as client:
        response = client.post("/media/", json=media_data)
    assert response.status_code == 200
    return response


def assert_media_response(response: dict, media_data: dict):
    for key in media_data.keys():
        try:
            assert media_data[key] == json.loads(response[key])
        except JSONDecodeError:
            assert media_data[key] == response[key]
        except TypeError:
            assert media_data[key] == response[key]


def test_create_media(create_media: None):
    response = create_media
    assert response.status_code == 200
    assert_media_response(response.json(), media_data)


def test_read_media(create_media: None):
    with TestClient(app) as client:
        response = client.get("/media/1")
    assert response.status_code == 200
    assert_media_response(response.json(), media_data)


def test_read_medias(create_media: None):
    with TestClient(app) as client:
        response = client.get("/media/")
    assert response.status_code == 200

    for row in response.json():
        assert_media_response(row, media_data)


def test_update_media(create_media: None):
    updated_data = {**media_data, "media_path": "/new/path/to/media"}
    with TestClient(app) as client:
        response = client.put("/media/1", json=updated_data)
    assert response.status_code == 200
    assert_media_response(response.json(), updated_data)


def test_delete_media(create_media: None):
    with TestClient(app) as client:
        response = client.delete("/media/1")
    assert response.status_code == 200
    assert response.json() == {"status": "Media deleted"}

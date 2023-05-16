from fastapi.testclient import TestClient
import pytest
from main import app  # Assuming main.py contains your FastAPI application
from models import Media  # Assuming models.py contains your Pydantic models
import json

client = TestClient(app)

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


@pytest.mark.asyncio
async def test_create_media():
    response = client.post("/media/", json=media_data)
    assert response.status_code == 200
    assert response.json() == media_data


@pytest.mark.asyncio
async def test_read_media():
    response = client.get("/media/1")
    assert response.status_code == 200
    assert response.json() == media_data


@pytest.mark.asyncio
async def test_read_medias():
    response = client.get("/media/")
    assert response.status_code == 200
    assert media_data in response.json()


@pytest.mark.asyncio
async def test_update_media():
    updated_data = {**media_data, "media_path": "/new/path/to/media"}
    response = client.put("/media/1", json=updated_data)
    assert response.status_code == 200
    assert response.json() == updated_data


@pytest.mark.asyncio
async def test_delete_media():
    response = client.delete("/media/1")
    assert response.status_code == 200
    assert response.json() == {"status": "Media deleted"}

from fastapi.testclient import TestClient
from rts.api.server import app, mount_routers
from rts.api.settings import Settings, get_settings
from rts.api.routers.auth_router import authenticate
from rts.api.models import LibraryCreate
from rts.db.utils import reset_database
from rts.db.queries import get_library_id_from_name
from rts.utils import get_logger
import json

LOG = get_logger()

client = TestClient(app)
settings = get_settings()
mount_routers(app, settings)


async def mock_authenticate(token: str = None):
    return True

app.dependency_overrides[authenticate] = mock_authenticate


def test_query_library():
    reset_database()
    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    )
    response = client.post("/libraries/", json=library.dict())

    response = client.get("/libraries/1")
    assert response.status_code == 200


def test_create_library():
    reset_database()
    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    )

    response = client.post("/libraries/", json=library.dict())
    assert response.status_code == 200
    assert response.json()["library_name"] == "test"
    assert response.json()["version"] == "0.0.1"
    assert response.json()["data"] == {"test": "test"}


def test_get_library_id_from_name():
    reset_database()
    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    )
    response = client.post("/libraries/", json=library.dict())
    assert response.status_code == 200
    request_library_id = response.json()["library_id"]

    library_id = get_library_id_from_name("test")
    assert library_id == request_library_id

    library_id = get_library_id_from_name("test2")
    assert library_id is None

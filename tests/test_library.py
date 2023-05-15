from fastapi.testclient import TestClient
from rts.api.server import app
from rts.api.models import LibraryCreate
from rts.utils import get_logger
import json

LOG = get_logger()


def test_query_library():

    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    )
    with TestClient(app) as client:
        response = client.post("/libraries/", json=library.dict())

    response = client.get("/libraries/1")
    assert response.status_code == 200

def test_create_library():

    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    )
    with TestClient(app) as client:
        response = client.post("/libraries/", json=library.dict())
    assert response.status_code == 200
    assert response.json()["library_name"] == "test"
    assert response.json()["version"] == "0.0.1"
    assert response.json()["data"] == {"test": "test"}

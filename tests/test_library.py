import pytest
from fastapi.testclient import TestClient
from rts.api.server import app
from rts.api.models import LibraryCreate
from rts.db.dao import DataAccessObject
from rts.settings import TEST_DATABASE_URL
from rts.db import create_database, database_exists, get_db
from rts.utils import get_logger
import json


LOG = get_logger()

app.testing = True
client = TestClient(app)

def test_query_library():

    response = client.get("/libraries/1")
    print(response.json())
    assert response.status_code == 200

def test_create_library():

    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        data=json.dumps({"test": "test"})
    )

    client.post("/libraries/", json=library.dict())

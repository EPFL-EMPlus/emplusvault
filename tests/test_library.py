from fastapi.testclient import TestClient
from rts.api.server import app, mount_routers
from rts.api.api_settings import Settings, get_settings
from rts.api.models import LibraryCreate
from rts.db.queries import get_library_id_from_name, create_library
from rts.api.routers.auth_router import get_current_active_user
from rts.utils import get_logger
from .conftest import reset_database, mock_authenticate
import json

LOG = get_logger()

client = TestClient(app)
settings = get_settings()
mount_routers(app, settings)


def test_query_library():
    app.dependency_overrides[get_current_active_user] = mock_authenticate
    reset_database()
    response = client.get("/libraries/1")
    assert response.status_code == 200

from fastapi.testclient import TestClient
import pytest
from emv.api.server import app, mount_routers
from emv.api.api_settings import Settings, get_settings
from emv.api.models import Media, LibraryCreate
from .conftest import reset_database, mock_authenticate, media_data
from emv.db.queries import create_library
from emv.api.routers.auth_router import get_current_active_user
import json
from json import JSONDecodeError
from dateutil.parser import parse


client = TestClient(app)
settings = get_settings()
mount_routers(app, settings)


@pytest.fixture
def db_setup():
    reset_database()


@pytest.fixture
def create_media(db_setup: None):
    app.dependency_overrides[get_current_active_user] = mock_authenticate
    response = client.post("/media/", json=media_data)
    assert response.status_code == 200
    return response


def assert_media_response(response: dict, media_data: dict):
    for key in media_data.keys():
        try:
            assert media_data[key] == json.loads(response[key])
        except (JSONDecodeError, KeyError, TypeError):
            # if both are strings and represent date-times, we compare parsed dates
            if (isinstance(media_data[key], str) and
                    isinstance(response.get(key, None), str)):
                try:
                    date1 = parse(media_data[key])
                    date2 = parse(response[key])
                    assert date1 == date2
                except ValueError:  # if parsing fails for any string, fall back to the simple assert
                    assert media_data[key] == response[key]
            else:
                assert media_data[key] == response[key]


def test_create_media(create_media: None):
    response = create_media
    assert response.status_code == 200
    assert_media_response(response.json(), media_data)

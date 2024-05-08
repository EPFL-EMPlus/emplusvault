from fastapi.testclient import TestClient
import pytest
from emv.api.server import app, mount_routers
from emv.api.api_settings import Settings, get_settings
from emv.db.queries import allow_user_to_access_library
from .conftest import media_data, reset_database

client = TestClient(app)
settings = get_settings()
mount_routers(app, settings)
data = {
    "grant_type": "password",
    "username": "testuser",
    "password": "secret",
}


def test_authentication():
    reset_database()
    response = client.get("/projections/")
    assert response.status_code == 400  # Token not provided because no authentication

    response = client.post(f"/gettoken", data=data)
    headers = {
        "Authorization": f"Bearer {response.json()['access_token']}"
    }
    response = client.get("/projections/", headers=headers)
    assert response.status_code == 200


# def test_rls():
#     reset_database()
#     response = client.post(f"/gettoken", data=data)
#     headers = {
#         "Authorization": f"Bearer {response.json()['access_token']}"
#     }
#     # import ipdb; ipdb.set_trace()
#     media_data2 = media_data.copy()
#     media_data2["library_id"] = 2

#     response = client.post("/media/", json=media_data2, headers=headers)
#     response = client.get("/media/library-ID01", headers=headers)
#     # import ipdb; ipdb.set_trace()
#     assert response.status_code == 401
#     assert response.json() == {"detail": "Not allowed"}

#     # explicitly allow user to access library
#     allow_user_to_access_library(1, 2)
#     response = client.post("/media/", json=media_data2, headers=headers)
#     assert response.status_code == 200

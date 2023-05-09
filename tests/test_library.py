import pytest
from fastapi.testclient import TestClient
from rts.api.server import app
from rts.api.models import LibraryCreate


client = TestClient(app)

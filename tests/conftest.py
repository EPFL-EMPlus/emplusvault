import pytest
from rts.db.dao import DataAccessObject
from rts.settings import TEST_DATABASE_URL
from rts.db import create_database, database_exists
from rts.utils import get_logger

LOG = get_logger()

@pytest.fixture(scope="session")
async def test_database():
    # Create test database
    await DataAccessObject().connect(TEST_DATABASE_URL)

    if not database_exists():
        create_database()

    yield DataAccessObject()

    await DataAccessObject().disconnect()

import pytest
from rts.api.dao import DataAccessObject
from rts.settings import TEST_DB_USER, TEST_DB_PASSWORD, TEST_DB_HOST, TEST_DB_NAME
from rts.db import create_database, database_exists

@pytest.fixture(scope="session")
async def test_database():
    # Create test database
    await DataAccessObject().connect(f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}/{TEST_DB_NAME}")

    if not database_exists():
        create_database()

    yield DataAccessObject()

    await DataAccessObject().disconnect()

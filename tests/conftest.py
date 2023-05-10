import pytest
from rts.db.dao import DataAccessObject
from rts.settings import TEST_DATABASE_URL
from rts.db import create_database, database_exists
from rts.utils import get_logger

LOG = get_logger()

# @pytest.fixture(scope="session")
# async def test_database():
#     # Create test database
#     await DataAccessObject().connect(TEST_DATABASE_URL)

#     if not database_exists():
#         create_database()

#     yield DataAccessObject()

#     await DataAccessObject().disconnect()

# async def pytest_sessionstart(session):
#     LOG.info("Creating test database")
#     await create_database("db/tables.sql")
#     LOG.info("Test database created")

# def pytest_sessionfinish(session, exitstatus):
#     LOG.info("Dropping test database")
#     # DataAccessObject().execute("DROP DATABASE IF EXISTS rts_test")
#     LOG.info("Test database dropped")

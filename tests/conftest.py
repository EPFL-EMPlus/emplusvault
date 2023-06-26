import pytest
from rts.utils import get_logger
from rts.db.utils import create_database
from rts.db.dao import DataAccessObject
from rts.api.routers.auth_router import authenticate
import time

LOG = get_logger()

TEST_DB_HOST = "localhost"
TEST_DB_PORT = 5435
TEST_DB_NAME = "testdb"
TEST_DB_USER = "postgres"
TEST_DB_PASSWORD = "testpassword"

TEST_DATABASE_URL = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"


def pytest_configure(config):
    LOG.info("Running pytest_configure")
    DataAccessObject().connect(TEST_DATABASE_URL)
    # Activate pgvector extension
    LOG.info("Activating pgvector extension")
    create_extension = "CREATE EXTENSION IF NOT EXISTS vector"
    DataAccessObject().execute_query(create_extension)

    create_extension = "CREATE EXTENSION IF NOT EXISTS postgis;"
    DataAccessObject().execute_query(create_extension)

    # Create the test database
    LOG.info(f"Creating test database {TEST_DATABASE_URL}")
    start = time.time()
    create_database("db/tables.sql")
    end = time.time()
    LOG.info(f"Test database created in {end - start} seconds")

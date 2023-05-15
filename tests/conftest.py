import pytest
from rts.settings import TEST_DATABASE_URL
from rts.utils import get_logger
from rts.db import create_database
from rts.db.dao import DataAccessObject
import time

LOG = get_logger()

def pytest_configure(config):
    LOG.info("Running pytest_configure")
    DataAccessObject().connect(TEST_DATABASE_URL)
    # Activate pgvector extension
    LOG.info("Activating pgvector extension")
    create_extension = "CREATE EXTENSION IF NOT EXISTS vector"
    DataAccessObject().execute_query(create_extension)

    # Create the test database
    LOG.info(f"Creating test database {TEST_DATABASE_URL}")
    start = time.time()
    create_database("db/tables.sql")
    end = time.time()
    LOG.info(f"Test database created in {end - start} seconds")

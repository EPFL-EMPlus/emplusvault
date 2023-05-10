import pytest
from rts.api.dao import DataAccessObject
from rts.settings import TEST_DATABASE_URL
from rts.db import create_database, database_exists
from rts.utils import get_logger

LOG = get_logger()

async def apply_sql_file(sql_file):
    # Read sql file and split it into individual statements
    with open(sql_file, "r") as f:
        statements = f.read().split(";")

    LOG.info(f"Applying {len(statements)} statements from {sql_file}")
    # Execute each statement
    for statement in statements:
        if statement.strip() != "":
            await DataAccessObject().execute(statement)

@pytest.fixture(scope="session")
async def test_database():
    # Create test database
    await DataAccessObject().connect(TEST_DATABASE_URL)

    if not database_exists():
        create_database()

    yield DataAccessObject()

    await DataAccessObject().disconnect()

import rts.db.fulltext
import rts.db.utils
from rts.db.dao import DataAccessObject
from rts.settings import DB_NAME, TEST_DATABASE_URL, DATABASE_URL
from rts.utils import get_logger

LOG = get_logger()

async def get_db():
    from rts.api.server import app
    print(f"Is in test mode: {app.testing}")
    db_url = TEST_DATABASE_URL if app.testing else DATABASE_URL
    await DataAccessObject().connect(db_url)
    await create_database("db/tables.sql")


def database_exists():
    return DataAccessObject().database_exists(DB_NAME)


async def create_database(sql_file):
    # Read sql file and split it into individual statements
    with open(sql_file, "r") as f:
        statements = f.read().split(";")

    LOG.info(f"Applying {len(statements)} statements from {sql_file}")
    # Execute each statement
    for statement in statements:
        print(statement)
        if statement.strip() != "":
            await DataAccessObject().execute_query(statement)
            print("-- query executed --")

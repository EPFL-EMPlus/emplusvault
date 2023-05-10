import rts.db.fulltext
import rts.db.utils
from rts.db.dao import DataAccessObject
from rts.settings import DB_NAME

def database_exists():
    return DataAccessObject().database_exists(DB_NAME)

async def create_database(sql_file):
    # Read sql file and split it into individual statements
    with open(sql_file, "r") as f:
        statements = f.read().split(";")

    LOG.info(f"Applying {len(statements)} statements from {sql_file}")
    # Execute each statement
    for statement in statements:
        if statement.strip() != "":
            await DataAccessObject().execute(statement)

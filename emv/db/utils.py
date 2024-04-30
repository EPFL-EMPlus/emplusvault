import os
import psycopg2
from typing import Optional, Dict
from emv.db.dao import DataAccessObject
from emv.settings import DB_NAME
from emv.utils import get_logger
from sqlalchemy.sql import text

LOG = get_logger()


def database_exists():
    return DataAccessObject().database_exists(DB_NAME)


def create_database(sql_file, output=False):
    # Read sql file and split it into individual statements
    with open(sql_file, "r") as f:
        statements = f.read().split(";")

    # Execute each statement
    for statement in statements:
        if output:
            LOG.info(statement)
        if statement.strip() != "":
            DataAccessObject().execute_query(text(statement))
            # print("-- query executed --")

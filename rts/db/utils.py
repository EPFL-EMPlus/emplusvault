import os
import psycopg2
from typing import Optional, Dict
from rts.db.dao import DataAccessObject
from rts.settings import DB_NAME
from rts.utils import get_logger

LOG = get_logger()


def database_exists():
    return DataAccessObject().database_exists(DB_NAME)


def create_database(sql_file, output=False):
    # Read sql file and split it into individual statements
    with open(sql_file, "r") as f:
        statements = f.read().split(";")

    # LOG.info(f"Applying {len(statements)} statements from {sql_file}")
    # Execute each statement
    for statement in statements:
        if output:
            print(statement)
        if statement.strip() != "":
            DataAccessObject().execute_query(statement)
            # print("-- query executed --")


def reset_database():
    # TODO: We need to apply the migrations after this command has been run
    create_database("db/tables.sql")

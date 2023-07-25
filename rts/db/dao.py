import sqlalchemy
from typing import Optional, Any, List, Dict, Union
from sqlalchemy.engine import Engine, Connection, ResultProxy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import text

from rts.utils import get_logger
from rts.settings import DATABASE_URL

LOG = get_logger()


# Create the engine
engine = sqlalchemy.create_engine(DATABASE_URL)


class DataAccessObject:
    def __init__(self, engine: Engine = engine) -> None:
        self._engine = engine

    def database_exists(self, db_name: str) -> bool:
        # Check if a database with the given name exists
        query = f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
        try:
            return self._engine.execute(query).scalar() is not None
        except SQLAlchemyError as e:
            LOG.error(f"An error occurred when executing query: {e}")
            raise e

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> ResultProxy:
        # Execute a SQL query on the connected database
        with self._engine.connect() as conn:
            trans = conn.begin()
            try:
                result = conn.execute(query, params)
                trans.commit()
                return result
            except SQLAlchemyError as e:
                trans.rollback()
                LOG.error(f"An error occurred when executing query: {e}")
                raise e

    def batch_insert(self, table: str, data: List[Dict[str, Any]]) -> None:
        # Insert a batch of data into the specified table
        with self._engine.connect() as conn:
            trans = conn.begin()
            try:
                for row in data:
                    conn.execute(text(f"INSERT INTO {table} VALUES (:values)"), values=row)
                trans.commit()
            except SQLAlchemyError as e:
                trans.rollback()
                LOG.error(f"An error occurred when inserting data: {e}")
                raise e

    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Execute a SQL query on the connected database and fetch all rows as a list of dictionaries
        with self._engine.connect() as conn:
            try:
                return conn.execute(query, params).mappings().fetchall()
            except SQLAlchemyError as e:
                LOG.error(f"An error occurred when fetching data: {e}")
                raise e

    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Execute a SQL query on the connected database and fetch the first row as a dictionary
        with self._engine.connect() as conn:
            try:
                return conn.execute(query, params).mappings().fetchone()
            except SQLAlchemyError as e:
                LOG.error(f"An error occurred when fetching data: {e}")
                raise e

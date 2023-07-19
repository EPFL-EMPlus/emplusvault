import sqlalchemy
from typing import Optional, Any, List, Dict, Union
from sqlalchemy.engine import Engine, Connection, ResultProxy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import text

from rts.utils import get_logger
from rts.settings import DATABASE_URL

LOG = get_logger()


class DataAccessObject:
    _engine = None  # SQLAlchemy engine object, used for connecting to the database
    _testing = False  # A flag indicating whether the DAO is in testing mode
    _test_setup_complete = False  # A flag indicating whether test setup is complete

    def __init__(self, db_url: Union[str, URL] = DATABASE_URL) -> None:
        self.connect(db_url)  # Automatically connect to the database

    def connect(self, db_url: Union[str, URL]) -> None:
        # Connect to the database using the given URL
        if not isinstance(db_url, (str, URL)):
            LOG.error("db_url must be a string or an instance of sqlalchemy.engine.url.URL")
            raise TypeError("db_url must be a string or an instance of sqlalchemy.engine.url.URL")

        try:
            self._engine = sqlalchemy.create_engine(db_url)
        except SQLAlchemyError as e:
            LOG.error(
                f"An error occurred when trying to connect to the database: {e}")
            raise e

    def disconnect(self) -> None:
        # Disconnect from the database by disposing the engine
        if self._engine:
            self._engine.dispose()

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
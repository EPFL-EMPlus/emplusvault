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
    _engine = None
    _user_id = None

    def __new__(cls, user_id=None, db_url=DATABASE_URL):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DataAccessObject, cls).__new__(cls)
            cls._engine = sqlalchemy.create_engine(db_url)
            cls._user_id = user_id

        return cls.instance

    def database_exists(self, db_name: str) -> bool:
        # Check if a database with the given name exists
        query = f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
        try:
            return self._engine.execute(query).scalar() is not None
        except SQLAlchemyError as e:
            LOG.error(f"An error occurred when executing query: {e}")
            raise e

    def set_user_id(self, user_id: int) -> None:
        self._user_id = user_id

    def set_rls_context(self, conn) -> None:
        conn.execute(
            text(f"SET LOCAL emplusvault.current_user_id TO {self._user_id};"))

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> ResultProxy:
        # Execute a SQL query on the connected database
        with self._engine.connect() as conn:
            trans = conn.begin()
            try:
                self.set_rls_context(conn)
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
                self.set_rls_context(conn)
                for row in data:
                    conn.execute(
                        text(f"INSERT INTO {table} VALUES (:values)"), values=row)
                trans.commit()
            except SQLAlchemyError as e:
                trans.rollback()
                LOG.error(f"An error occurred when inserting data: {e}")
                raise e

    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Execute a SQL query on the connected database and fetch all rows as a list of dictionaries
        with self._engine.connect() as conn:
            self.set_rls_context(conn)
            try:
                return conn.execute(query, params).mappings().fetchall()
            except SQLAlchemyError as e:
                LOG.error(f"An error occurred when fetching data: {e}")
                raise e

    def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Execute a SQL query on the connected database and fetch the first row as a dictionary
        with self._engine.connect() as conn:
            self.set_rls_context(conn)
            try:
                return conn.execute(query, params).mappings().fetchone()
            except SQLAlchemyError as e:
                LOG.error(f"An error occurred when fetching data: {e}")
                raise e

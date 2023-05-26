import sqlalchemy
from rts.utils import get_logger

LOG = get_logger()


class DataAccessObject:
    _instance = None
    _engine = None
    _testing = False
    _test_setup_complete = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataAccessObject, cls).__new__(cls)
        return cls._instance

    def connect(self, db_url):
        if self._engine:
            LOG.warning(
                "A connection to the database already exists, dispose first")
            return

        try:
            self._engine = sqlalchemy.create_engine(db_url)
        except sqlalchemy.exc.SQLAlchemyError as e:
            LOG.error(
                f"An error occurred when trying to connect to the database: {e}")
            raise e

    def disconnect(self):
        self._engine.dispose()
        self._engine = None

    def database_exists(self, db_name):
        query = f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
        return self._engine.execute(query).scalar() is not None

    def execute_query(self, query, params=None):
        with self._engine.connect() as conn:
            return conn.execute(query, params)

    def fetch_all(self, query, params=None):
        with self._engine.connect() as conn:
            return conn.execute(query, params).mappings().fetchall()

    def fetch_one(self, query, params=None):
        with self._engine.connect() as conn:
            return conn.execute(query, params).mappings().fetchone()

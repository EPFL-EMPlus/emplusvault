import sqlalchemy


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
        # from rts.api.server import app
        # from rts.settings import TEST_DATABASE_URL, DATABASE_URL
        # from rts.db import test_configure
        # DATABASE_URL = TEST_DATABASE_URL if app.testing else DATABASE_URL
        # self._testing = app.testing

        # if self._testing and not self._test_setup_complete:
        #     self._test_setup_complete = True
        #     test_configure()

        self._engine = sqlalchemy.create_engine(db_url)

    def database_exists(self, db_name):
        query = f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
        return self._engine.execute(query).scalar() is not None
    
    def execute_query(self, query, params=None):
        with self._engine.connect() as conn:
            conn.execute(query, params)

    def fetch_all(self, query, params=None):
        with self._engine.connect() as conn:
            return conn.execute(query, params).fetchall()

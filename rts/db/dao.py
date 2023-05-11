import asyncpg

class DataAccessObject:
    _instance = None
    _connection_pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataAccessObject, cls).__new__(cls)
            cls._connection_pool = None
        return cls._instance
    
    async def get_connection_pool(self):
        return self._connection_pool
    
    async def connect(self):
        from rts.api.server import app
        from rts.settings import TEST_DATABASE_URL, DATABASE_URL
        DATABASE_URL = TEST_DATABASE_URL if app.testing else DATABASE_URL
        
        if self._connection_pool is None:
            self._connection_pool = await asyncpg.create_pool(DATABASE_URL)

    async def fetch_one(self, query, values):
        if self._connection_pool is None:
            await self.connect()
        
        # values comes in the shape of (library_id, library_name)
        async with self._connection_pool.acquire() as connection:
            return await connection.fetchrow(query, *values)
        
    async def fetch_all(self, query, values):
        async with self._connection_pool.acquire() as connection:
            return await connection.fetch(query, *values)
        
    async def execute_query(self, query, values):
        async with self._connection_pool.acquire() as connection:
            return await connection.execute(query, *values)
    
    async def database_exists(self, db_name):
        async with self._connection_pool.acquire() as connection:
            return await connection.fetchval("SELECT EXISTS(SELECT datname FROM pg_catalog.pg_database WHERE datname = $1)", db_name)

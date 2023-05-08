from databases import Database
import os

class DataAccessObject:
    _instance = None
    _database = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self, database_url):
        if self._database is None:
            self._database = Database(database_url)
            await self._database.connect()

    async def disconnect(self):
        if self._database is not None:
            await self._database.disconnect()
            self._database = None

    async def execute_query(self, query, values=None):
        if self._database is None:
            raise ValueError("Database not connected")
        return await self._database.execute(query, values)

    async def fetch_one(self, query, values=None):
        if self._database is None:
            raise ValueError("Database not connected")
        return await self._database.fetch_one(query, values)

    async def fetch_all(self, query, values=None):
        if self._database is None:
            raise ValueError("Database not connected")
        return await self._database.fetch_all(query, values)

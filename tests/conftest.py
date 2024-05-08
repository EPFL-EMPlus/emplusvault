import pytest
from emv.utils import get_logger
from emv.db.utils import create_database
from emv.db.dao import DataAccessObject
import sqlalchemy
from sqlalchemy.sql import text
from alembic.config import Config
from emv.api.routers.auth_router import User
from emv.api.models import LibraryCreate
from emv.db.queries import allow_user_to_access_library, create_library
from alembic import command
import json
import time

LOG = get_logger()

TEST_DB_HOST = "localhost"
TEST_DB_PORT = 5435
TEST_DB_NAME = "testdb"
TEST_DB_USER = "testdb"
TEST_DB_PASSWORD = "testpassword"

TEST_DATABASE_URL = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"
# print("Testing...", TEST_DATABASE_URL)


media_data = {
    "media_id": "library-ID01",
    "media_path": "/path/to/media",
    "original_path": "/original/path/to/media",
    "created_at": "2021-01-01 00:00:00",
    "original_id": "123abc",
    "media_type": "video",
    "media_info": {"example": "media_info"},
    "sub_type": "mp4",
    "size": 1024,
    "metadata": {"example": "metadata"},
    "library_id": 1,
    "hash": "123abc",
    "parent_id": "library_ID00",
    "start_ts": 0.0,
    "end_ts": 10.0,
    "start_frame": 0,
    "end_frame": 100,
    "frame_rate": 10.0
}


async def mock_authenticate(token: str = None):
    return User(user_id=1, username="test", email="test@user.com", full_name="Test User", disabled=False)


def pytest_configure(config):
    LOG.info("Running pytest_configure")
    reset_database()


def reset_database():
    dao = DataAccessObject(user_id=1, db_url=TEST_DATABASE_URL)

    dao.execute_query(text("""
        DROP SCHEMA public CASCADE;
        CREATE SCHEMA public;
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'media') THEN
                DROP POLICY IF EXISTS user_media_access_policy ON media;
                ALTER TABLE media DISABLE ROW LEVEL SECURITY;

                DROP POLICY IF EXISTS user_feature_access_policy ON feature;
                ALTER TABLE feature DISABLE ROW LEVEL SECURITY;
            END IF;
        END$$;
    """))

    dao.execute_query(text("DROP table IF EXISTS alembic_version;"))
    dao.execute_query(text("DROP table IF EXISTS user_library_access;"))
    dao.execute_query(text("DROP table IF EXISTS entries_to_queue;"))
    dao.execute_query(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
    create_database("db/tables.sql")
    alembic_cfg = Config("alembic_test.ini")
    command.upgrade(alembic_cfg, "head")

    # create_new_user
    test_users = [{
        "username": "testuser",
        "full_name": "testuser",
        "email": "testuser@example.com",
        "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    },
    {
        "username": "testuser2",
        "full_name": "testuser2",
        "email": "testuser2@example.com",
        "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    }]

    query = text(
        "INSERT INTO users (username, full_name, email, password) VALUES (:username, :full_name, :email, :password) RETURNING user_id")
    result = dao.execute_query(query, test_users[0])
    result = dao.execute_query(query, test_users[1])
    user_id = result.fetchone()[0]

    # create new library for general access for the user
    library = LibraryCreate(
        library_name="test",
        version="0.0.1",
        prefix_path="/prefix/path/to/library",
        data=json.dumps({"test": "test"})
    )
    result = create_library(library)

    allow_user_to_access_library(user_id, result['library_id'])

    # Second library to test access control
    library = LibraryCreate(
        library_name="test2",
        version="0.0.1",
        prefix_path="/prefix/path/to/library",
        data=json.dumps({"test": "test"})
    )
    create_library(library)

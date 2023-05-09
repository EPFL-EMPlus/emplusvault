import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

TEST_DB_HOST="localhost"
TEST_DB_PORT=5432
TEST_DB_NAME="testdb"
TEST_DB_USER="testuser"
TEST_DB_PASSWORD="testpassword"

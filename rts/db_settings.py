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
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

DEV_DB_HOST = os.getenv("DEV_DB_HOST")
DEV_DB_PORT = os.getenv("DEV_DB_PORT")
DEV_DB_NAME = os.getenv("DEV_DB_NAME")
DEV_DB_USER = os.getenv("DEV_DB_USER")
DEV_DB_PASSWORD = os.getenv("DEV_DB_PASSWORD")
DEV_DATABASE_URL = f"postgresql://{DEV_DB_USER}:{DEV_DB_PASSWORD}@{DEV_DB_HOST}:{DEV_DB_PORT}/{DEV_DB_NAME}"

BUCKET_NAME = os.getenv("BUCKET_NAME")

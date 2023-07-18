import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BUCKET_NAME = os.getenv("BUCKET_NAME")
SUPERUSER_CLI_KEY = os.getenv("SUPERUSER_CLI_KEY")

RTS_LOCAL_DATA = os.getenv("RTS_LOCAL_DATA") or "/media/data/rts/"

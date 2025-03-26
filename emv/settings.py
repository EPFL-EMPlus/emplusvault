import os
from dotenv import load_dotenv
load_dotenv()


def get_secret(key):
    """
    Tries to get the secret from Docker secret file.
    If not available, falls back to environment variable.
    """
    try:
        with open(f'/run/secrets/{key}', 'r') as secret_file:
            return secret_file.read().strip()
    except IOError:
        # Secret not found as file, fallback to environment variable
        return os.getenv(key)


# SECRET_KEY FOR AUTH
# openssl rand -hex 32
SECRET_KEY = get_secret("SECRET_KEY")

S3_ACCESS_KEY = get_secret("S3_ACCESS_KEY")
S3_SECRET_KEY = get_secret("S3_SECRET_KEY")
S3_ENDPOINT = get_secret("S3_ENDPOINT")
S3_OUTSIDE_ENDPOINT = get_secret("S3_OUTSIDE_ENDPOINT")

DB_HOST = get_secret("DB_HOST")
DB_PORT = get_secret("DB_PORT")
DB_NAME = get_secret("DB_NAME")
DB_USER = get_secret("DB_USER")
DB_PASSWORD = get_secret("DB_PASSWORD")
LLM_DB_NAME = get_secret("LLM_DB_NAME")

DATABASE_URL = get_secret("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

LLM_DATABASE_URL = get_secret("LLM_DATABASE_URL")
if not LLM_DATABASE_URL:
    LLM_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{LLM_DB_NAME}"

BUCKET_NAME = get_secret("BUCKET_NAME")
SUPERUSER_CLI_KEY = get_secret("SUPERUSER_CLI_KEY")

HF_TOKEN = get_secret("HF_TOKEN")

RTS_ROOT_FOLDER = get_secret("RTS_ROOT_FOLDER") or "/media/data/rts/"
IOC_ROOT_FOLDER = get_secret("IOC_ROOT_FOLDER") or "/media/data/ioc/"
MJF_ROOT_FOLDER = get_secret("MJF_ROOT_FOLDER") or "/media/data/mjf/"
VIDEO_ROOT_FOLDER = get_secret("VIDEO_ROOT_FOLDER") or "/media/data/videos/"

RABBITMQ_SERVER = get_secret("RABBITMQ_SERVER")

LLM_ENDPOINT = get_secret("LLM_ENDPOINT")

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

DB_HOST = get_secret("DB_HOST")
DB_PORT = get_secret("DB_PORT")
DB_NAME = get_secret("DB_NAME")
DB_USER = get_secret("DB_USER")
DB_PASSWORD = get_secret("DB_PASSWORD")
DATABASE_URL = get_secret("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BUCKET_NAME = get_secret("BUCKET_NAME")
SUPERUSER_CLI_KEY = get_secret("SUPERUSER_CLI_KEY")

HF_TOKEN = get_secret("HF_TOKEN")

RTS_ROOT_FOLDER = get_secret("RTS_ROOT_FOLDER") or "/media/data/rts/"
IOC_ROOT_FOLDER = get_secret("IOC_ROOT_FOLDER") or "/media/data/ioc/"
MJF_ROOT_FOLDER = get_secret("MJF_ROOT_FOLDER") or "/media/data/mjf/"

API_BASE_URL = get_secret("API_BASE_URL")
API_MAX_CALLS = 100
API_USERNAME = get_secret("API_USERNAME")
API_PASSWORD = get_secret("API_PASSWORD")

DRIVE_PATH = "/media/data/"
IOC_DRIVE_PATH = DRIVE_PATH + "ioc/"
RTS_DRIVE_PATH = DRIVE_PATH + "rts/"

RABBITMQ_SERVER = get_secret("RABBITMQ_SERVER")

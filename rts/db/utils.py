import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def create_conn() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    return conn

def execute_read_query(query) -> list:
    conn = create_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
    return result

def execute_write_query(query) -> None:
    conn = create_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()

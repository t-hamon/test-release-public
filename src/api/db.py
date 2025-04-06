import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")  # Read the DATABASE_URL from environment variables
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    return conn

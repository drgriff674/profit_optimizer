import psycopg2
import os
from psycopg2.extras import RealDictCursor

# ✅ Railway PostgreSQL connection string
DB_URL = "postgresql://postgres:qzniBQaYcEdGRMKMqJessjlVGSLseaam@switchback.proxy.rlwy.net:14105/railway"

def get_db_connection(cursor_factory=None):
    return psycopg2.connect(
        DB_URL,
        cursor_factory=cursor_factory
    )

# ✅ Initialize database and users table if not exists
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    conn.commit()
    cursor.close()
    conn.close()

# ✅ Load all users as a dict
def load_users():
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password, role FROM users")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {row["username"]: {"password": row["password"], "role": row["role"]} for row in rows}

# ✅ Save a new user
def save_user(username, password, role):
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING",
        (username, password, role)
    )
    conn.commit()
    cursor.close()
    conn.close()
    debug_print_users()

# ✅ Debug: print all users directly from DB
def debug_print_users():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print("📌 Current users in DB:", rows)

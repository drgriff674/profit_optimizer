import os
import psycopg2
from psycopg2.extras import RealDictCursor

# ✅ Use your Supabase connection string
DATABASE_URL = "postgresql://postgres:Profit123!project@db.djzmdhodxvdckbqyhrqj.supabase.co:5432/postgres"

# ✅ Connect to the database
def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# ✅ Initialize users table if not exists
def init_db():
    conn = get_connection()
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

# ✅ Load all users
def load_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, password, role FROM users")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {row["username"]: {"password": row["password"], "role": row["role"]} for row in rows}

# ✅ Save a new user
def save_user(username, password, role):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                   (username, password, role))
    conn.commit()
    cursor.close()
    conn.close()

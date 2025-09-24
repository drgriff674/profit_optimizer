import sqlite3
import os

# ✅ Always use the project root folder (where app.py is located)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
DB_PATH = os.path.join(PROJECT_ROOT, "users.db")

print(f"📂 Using database at: {DB_PATH}")

# ✅ Initialize database and users table if not exists
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

# ✅ Load all users as a dict
def load_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    return {username: {"password": password, "role": role} for username, password, role in rows}

# ✅ Save a new user
def save_user(username, password, role):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        (username, password, role)
    )
    conn.commit()
    conn.close()
    debug_print_users()  # show saved users each time

# ✅ Debug: print all users directly from DB
def debug_print_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    print("📌 Current users in DB:", rows)

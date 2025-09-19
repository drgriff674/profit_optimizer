import sqlite3
import os
from werkzeug.security import generate_password_hash  # 👈 ADD THIS LINE

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# 🔽 REPLACE EVERYTHING in init_db() WITH THIS
def init_db():
    print(f"[INIT] Using database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ✅ Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)

    # ✅ Ensure admin user is always present
    hashed_admin_pass = generate_password_hash("admin123")
    cursor.execute("""
    INSERT OR IGNORE INTO users (username, password, role)
    VALUES (?, ?, ?)
    """, ("admin", hashed_admin_pass, "admin"))

    conn.commit()
    conn.close()
# 🔼 CHANGE STOPS HERE

# ✅ Load all users as a dict
def load_users():
    print(f"[LOAD] Reading users from:{DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    return {username: {"password": password, "role": role} for username, password, role in rows}

# ✅ Save a new user
def save_user(username, password, role):
    print(f"[SAVE] Saving user '{username}' to: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                   (username, password, role))
    conn.commit()
    conn.close()
    print(f"[SAVE] User '{username}' saved successfully!")

import psycopg2
import os
from psycopg2.extras import RealDictCursor

# ✅ Railway PostgreSQL connection string

def get_db_connection(cursor_factory=None):
    db_url = os.environ.get("DATABASE_URL")

    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    
    return psycopg2.connect(
        db_url,
        cursor_factory=cursor_factory
    )

# ✅ Initialize database and tables if not exists
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # --- USERS TABLE ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)

    # --- EXPENSES TABLE (NEW) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        amount NUMERIC NOT NULL,
        category TEXT,
        description TEXT,
        expense_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- INVENTORY MOVEMENTS TABLE ---
    cursor.execute("""
    DROP TABLE IF EXISTS inventory_movements;
    CREATE TABLE inventory_movements (
        id SERIAL PRIMARY KEY,
        business_id INTEGER NOT NULL,
        item_id INTEGER NOT NULL,
        movement_type TEXT NOT NULL,   -- sale, usage, restock
        quantity_change NUMERIC NOT NULL,
        note TEXT,
        created_by TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE,
        FOREIGN KEY (item_id) REFERENCES inventory_items(id) ON DELETE CASCADE
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS revenue_entries (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        category TEXT NOT NULL,
        amount NUMERIC NOT NULL,
        revenue_date DATE NOT NULL,
        locked BOOLEAN DEFAULT FALSE
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS businesses (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        business_name TEXT NOT NULL,
        paybill TEXT NOT NULL,
        account_number TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (paybill, account_number),
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory_items (
        id SERIAL PRIMARY KEY,
        business_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory_snapshots (
        id SERIAL PRIMARY KEY,
        business_id INTEGER NOT NULL,
        snapshot_date DATE NOT NULL,
        snapshot_type TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory_snapshot_items (
        id SERIAL PRIMARY KEY,
        snapshot_id INTEGER NOT NULL,
        item_id INTEGER NOT NULL,
        quantity NUMERIC NOT NULL,
        FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshots(id) ON DELETE CASCADE,
        FOREIGN KEY (item_id) REFERENCES inventory_items(id) ON DELETE CASCADE,
        UNIQUE (snapshot_id, item_id)
    )
    """)

    cursor.execute("""
    ALTER TABLE inventory_movements
    ADD COLUMN IF NOT EXISTS source TEXT;
    """)

    # --- MPESA TRANSACTIONS TABLE ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mpesa_transactions (
        id SERIAL PRIMARY KEY,
        transaction_id TEXT UNIQUE NOT NULL,
        amount NUMERIC(10,2),
        sender TEXT,
        receiver TEXT,
        transaction_type TEXT,
        account_reference TEXT,
        description TEXT,
        status TEXT DEFAULT 'pending',
        raw_payload JSON,
        origin_ip TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
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
    return {
        row["username"]: {
            "password": row["password"],
            "role": row["role"]
        }
        for row in rows
    }

# ✅ Save a new user
def save_user(username, password, role):
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO users (username, password, role)
        VALUES (%s, %s, %s)
        ON CONFLICT (username) DO NOTHING
        """,
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

# =====================================================
# ✅ EXPENSE HELPERS (NEW – USED BY MANUAL ENTRY)
# =====================================================

def save_expense(username, amount, category, description, expense_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO expenses (username, amount, category, description, expense_date)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (username, amount, category, description, expense_date)
    )
    conn.commit()
    cursor.close()
    conn.close()

def load_expenses(username):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            amount,
            category,
            description,
            expense_date,
            created_at
        FROM expenses
        WHERE username = %s
        ORDER BY expense_date DESC
        """,
        (username,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# =====================================================
# ✅ REVENUE ENTRY HELPERS (MANUAL DAILY SPLITS)
# =====================================================

def save_revenue_entry(username, category, amount, revenue_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO revenue_entries (username, category, amount, revenue_date)
        VALUES (%s, %s, %s, %s)
        """,
        (username, category, amount, revenue_date)
    )
    conn.commit()
    cursor.close()
    conn.close()



def load_expense_categories(username):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT category
        FROM expenses
        WHERE username = %s
          AND category IS NOT NULL
          AND category <> ''
        ORDER BY category ASC
        """,
        (username,)
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # rows is like [('Food',), ('Transport',)]
    return [row[0] for row in rows]

def lock_revenue_day(username, revenue_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE revenue_entries
        SET locked = TRUE
        WHERE username = %s AND revenue_date = %s
        """,
        (username, revenue_date)
    )
    conn.commit()
    cursor.close()
    conn.close()

def load_revenue_entries_for_day(username, revenue_date):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, category, amount, locked
        FROM revenue_entries
        WHERE username = %s AND revenue_date = %s
        ORDER BY id ASC
    """, (username, revenue_date))
    rows = cursor.fetchall()
    conn.close()
    return rows

def save_business(username, business_name, paybill, account_number):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO businesses (username, business_name, paybill, account_number)
        VALUES (%s, %s, %s, %s)
    """, (username, business_name, paybill, account_number))
    conn.commit()
    cursor.close()
    conn.close()

def load_user_business(username):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT business_name, paybill, account_number
        FROM businesses
        WHERE username = %s
        LIMIT 1
    """, (username,))
    business = cursor.fetchone()
    cursor.close()
    conn.close()
    return business

# =====================================================
# ✅ INVENTORY HELPERS
# =====================================================

def create_inventory_item(business_id, name, category, unit):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO inventory_items (business_id, name, category, unit)
        VALUES (%s, %s, %s, %s)
        """,
        (business_id, name, category, unit)
    )
    conn.commit()
    cursor.close()
    conn.close()


def create_inventory_snapshot(business_id, snapshot_date, snapshot_type, created_by):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO inventory_snapshots
        (business_id, snapshot_date, snapshot_type, created_by)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (business_id, snapshot_date, snapshot_type, created_by)
    )
    snapshot_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return snapshot_id

def save_snapshot_item(snapshot_id, item_id, quantity):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO inventory_snapshot_items
        (snapshot_id, item_id, quantity)
        VALUES (%s, %s, %s)
        """,
        (snapshot_id, item_id, quantity)
    )
    conn.commit()
    cursor.close()
    conn.close()


def load_latest_inventory_snapshot(business_id):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM inventory_snapshots
        WHERE business_id = %s
        ORDER BY snapshot_date DESC, created_at DESC
        LIMIT 1
        """,
        (business_id,)
    )

    snapshot = cursor.fetchone()
    cursor.close()
    conn.close()
    return snapshot

def load_snapshot_items(snapshot_id):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            ii.name,
            ii.category,
            ii.unit,
            isi.quantity
        FROM inventory_snapshot_items isi
        JOIN inventory_items ii ON ii.id = isi.item_id
        WHERE isi.snapshot_id = %s
        """,
        (snapshot_id,)
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

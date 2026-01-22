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

    cursor.execute("""
    ALTER TABLE mpesa_transactions
    ADD COLUMN IF NOT EXISTS seen BOOLEAN DEFAULT FALSE
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS revenue_anomalies (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        revenue_date DATE NOT NULL,
        anomaly_type TEXT NOT NULL,
        severity TEXT NOT NULL, -- info, warning, critical
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT FALSE
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS revenue_ai_summaries (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    revenue_date DATE NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (username, revenue_date)
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
    conn = get_db_connection()
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
    

# ✅ Debug: print all users directly from DB
def debug_print_users():
    conn = conn = get_db_connection()
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

def lock_manual_entries_for_the_day(username, revenue_date):
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

def get_existing_revenue_days(username):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get business identifiers
    cur.execute("""
        SELECT paybill, account_number
        FROM businesses
        WHERE username = %s
        LIMIT 1
    """, (username,))
    biz = cur.fetchone()

    if not biz:
        cur.close()
        conn.close()
        return []

    paybill, account_number = biz

    # Pull days from BOTH sources:
    # 1) MPesa transactions
    # 2) Manual revenue entries
    cur.execute("""
        SELECT DISTINCT day FROM (
            SELECT DATE(created_at) AS day
            FROM mpesa_transactions
            WHERE (
                (%s IS NOT NULL AND account_reference = %s)
                OR
                (%s IS NULL AND receiver = %s)
            )

            UNION

            SELECT revenue_date AS day
            FROM revenue_entries
            WHERE username = %s
        ) d
        ORDER BY day DESC
    """, (
        account_number,
        account_number,
        account_number,
        paybill,
        username
    ))

    days = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()
    return days
def ensure_revenue_day_exists(username, revenue_date):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO revenue_days (username, revenue_date)
        VALUES (%s, %s)
        ON CONFLICT (username, revenue_date) DO NOTHING
    """, (username, revenue_date))

    conn.commit()
    cursor.close()
    conn.close()

def get_ai_summary_for_day(username, revenue_date):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT summary
        FROM revenue_ai_summaries
        WHERE username = %s AND revenue_date = %s
    """, (username, revenue_date))

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    return row[0] if row else None

def save_ai_summary_for_day(username, revenue_date, summary):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO revenue_ai_summaries (username, revenue_date, summary)
        VALUES (%s, %s, %s)
        ON CONFLICT (username, revenue_date)
        DO UPDATE SET summary = EXCLUDED.summary
    """, (username, revenue_date, summary))

    conn.commit()
    cursor.close()
    conn.close()

def detect_revenue_anomalies(username, revenue_date):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    # --- totals ---
    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0) AS manual_total
        FROM revenue_entries
        WHERE username = %s AND revenue_date = %s
    """, (username, revenue_date))
    manual_total = float(cursor.fetchone()["manual_total"])

    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0) AS mpesa_total
        FROM mpesa_transactions
        WHERE status = 'confirmed'
          AND DATE(created_at) = %s
    """, (revenue_date,))
    mpesa_total = float(cursor.fetchone()["mpesa_total"])

    total = manual_total + mpesa_total

    anomalies = []

    # --- Rule A ---
    if manual_total > 0:
        diff_ratio = abs(mpesa_total - manual_total) / manual_total
        if diff_ratio > 0.2:
            anomalies.append((
                "MPESA_MISMATCH",
                "warning",
                f"MPesa differs from manual by {int(diff_ratio*100)}%"
            ))

    # --- Rule C ---
    if total == 0:
        anomalies.append((
            "ZERO_REVENUE",
            "critical",
            "Revenue day locked with zero total"
        ))

    # --- Rule B ---
    cursor.execute("""
        SELECT AVG(day_total) AS avg_total
        FROM (
            SELECT revenue_date, SUM(amount) AS day_total
            FROM revenue_entries
            WHERE username = %s
              AND revenue_date < %s
            GROUP BY revenue_date
            ORDER BY revenue_date DESC
            LIMIT 7
        ) t
    """, (username, revenue_date))

    avg_row = cursor.fetchone()
    if avg_row and avg_row["avg_total"]:
        avg_total = float(avg_row["avg_total"])
        if avg_total > 0 and total < avg_total * 0.7:
            anomalies.append((
                "SUDDEN_DROP",
                "warning",
                "Revenue dropped more than 30% vs recent average"
            ))

    # --- Save anomalies ---
    for a in anomalies:
        cursor.execute("""
            INSERT INTO revenue_anomalies
            (username, revenue_date, anomaly_type, severity, message)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, revenue_date, *a))

    conn.commit()
    cursor.close()
    conn.close()

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

def get_dashboard_revenue_intelligence(username):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Last 7 days revenue (locked only)
    cursor.execute("""
        SELECT
            revenue_date,
            SUM(amount) AS manual_total
        FROM revenue_entries
        WHERE username = %s
          AND locked = TRUE
          AND revenue_date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY revenue_date
        ORDER BY revenue_date DESC
    """, (username,))
    manual_days = cursor.fetchall()

    # MPesa totals for last 7 days
    cursor.execute("""
        SELECT
            DATE(created_at) AS revenue_date,
            SUM(amount) AS mpesa_total
        FROM mpesa_transactions
        WHERE status = 'confirmed'
          AND DATE(created_at) >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY DATE(created_at)
    """)
    mpesa_days = cursor.fetchall()

    # Anomaly count
    cursor.execute("""
        SELECT COUNT(DISTINCT revenue_date) AS anomaly_days
        FROM revenue_anomalies
        WHERE username = %s
          AND revenue_date >= CURRENT_DATE - INTERVAL '7 days'
    """, (username,))
    anomaly_count = cursor.fetchone()["anomaly_days"]

    cursor.close()
    conn.close()

    return {
        "manual_days": manual_days,
        "mpesa_days": mpesa_days,
        "anomaly_days": anomaly_count,
    }



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

def lock_business_day(username, business_date):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Calculate final total (manual + mpesa)
    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM revenue_entries
        WHERE username = %s AND revenue_date = %s
    """, (username, business_date))
    manual_total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM mpesa_transactions
        WHERE status = 'confirmed'
          AND DATE(created_at) = %s
    """, (business_date,))
    mpesa_total = cursor.fetchone()[0]

    final_total = manual_total + mpesa_total

    # Lock the day
    cursor.execute("""
        UPDATE daily_revenue_status
        SET is_locked = TRUE,
            total_revenue = %s
        WHERE username = %s AND business_date = %s
    """, (final_total, username, business_date))

    # Lock manual entries too (secondary)
    cursor.execute("""
        UPDATE revenue_entries
        SET locked = TRUE
        WHERE username = %s AND revenue_date = %s
    """, (username, business_date))

    conn.commit()
    cursor.close()
    conn.close()


def load_revenue_days(username):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT
            revenue_date,
            SUM(amount) AS total_amount,
            BOOL_AND(locked) AS locked
        FROM revenue_entries
        WHERE username = %s
        GROUP BY revenue_date
        ORDER BY revenue_date DESC
    """, (username,))

    days = cursor.fetchall()

    cursor.close()
    conn.close()
    return days


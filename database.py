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

    # --- CASH REVENUE ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cash_revenue (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
        revenue_date DATE NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_cash_revenue_user_date
    ON cash_revenue(username, revenue_date);
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
    CREATE TABLE IF NOT EXISTS inventory_movements (
        id SERIAL PRIMARY KEY,
        business_id INTEGER NOT NULL,
        item_id INTEGER NOT NULL,
        movement_type TEXT NOT NULL,
        quantity_change NUMERIC NOT NULL,
        note TEXT,
        created_by TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source TEXT,

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
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS revenue_days (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        revenue_date DATE NOT NULL,
        locked BOOLEAN DEFAULT FALSE,
        total_amount NUMERIC(12,2) DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (username, revenue_date)
    );
    """)

    conn.commit()
    cursor.close()
    conn.close()


#revenue cash functions
def add_cash_revenue(username, amount, revenue_date, description=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO cash_revenue (username, amount, revenue_date, description)
        VALUES (%s, %s, %s, %s)
    """, (username, amount, revenue_date, description))
    conn.commit()
    cur.close()
    conn.close()

def get_cash_revenue_for_day(username, revenue_date):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT amount, description
        FROM cash_revenue
        WHERE username = %s
          AND revenue_date = %s
        ORDER BY created_at ASC
    """, (username, revenue_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_cash_revenue_total_for_day(username, revenue_date):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM cash_revenue
        WHERE username = %s
          AND revenue_date = %s
    """, (username, revenue_date))
    total = cur.fetchone()[0]
    cur.close()
    conn.close()
    return float(total)

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

def get_expenses_for_day(username, date):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount, category, description
        FROM expenses
        WHERE username = %s
          AND expense_date = %s
    """, (username, date))

    rows = cursor.fetchall()
    total = sum(float(r["amount"]) for r in rows)

    cursor.close()
    conn.close()

    return {
        "entries": rows,
        "total": total
    }

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
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT revenue_date AS day
        FROM revenue_days
        WHERE username = %s
        ORDER BY revenue_date DESC
    """, (username,))

    days = [row["day"] for row in cursor.fetchall()]

    cursor.close()
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
        SELECT total_amount
        FROM revenue_days
        WHERE username = %s AND revenue_date = %s
    """, (username, revenue_date))

    row = cursor.fetchone()
    total = float(row["total_amount"]) if row else 0

    anomalies = []

    # --- Rule A: MPesa dominance (FIXED TIMEZONE) ---
    if total > 0:
        cursor.execute("""
            SELECT COALESCE(SUM(m.amount), 0) AS mpesa_total
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                   (b.account_number IS NOT NULL AND m.account_reference = b.account_number)
                   OR
                   (b.account_number IS NULL AND m.receiver = b.paybill)
                 )
            WHERE b.username = %s
              AND m.status = 'confirmed'
              AND DATE(m.created_at AT TIME ZONE 'Africa/Nairobi') = %s
        """, (username, revenue_date))

        mpesa_total = float(cursor.fetchone()["mpesa_total"])

        ratio = mpesa_total / total if total > 0 else 0

        if ratio > 0.8:
            anomalies.append((
                "MPESA_DOMINANCE",
                "info",
                "MPesa accounts for more than 80% of total revenue"
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

    # --- Manual revenue (locked only, last 7 days) ---
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

    # --- MPesa totals (timezone-safe, last 7 days) ---
    cursor.execute("""
        SELECT
            DATE(created_at AT TIME ZONE 'Africa/Nairobi') AS revenue_date,
            SUM(amount) AS mpesa_total
        FROM mpesa_transactions
        WHERE status = 'confirmed'
          AND DATE(created_at AT TIME ZONE 'Africa/Nairobi') >=
              CURRENT_DATE - INTERVAL '7 days'
        GROUP BY revenue_date
        ORDER BY revenue_date DESC
    """)
    mpesa_days = cursor.fetchall()

    # --- Anomaly count (last 7 days, timezone-safe) ---
    cursor.execute("""
        SELECT COUNT(DISTINCT revenue_date) AS anomaly_days
        FROM revenue_anomalies
        WHERE username = %s
          AND revenue_date >= CURRENT_DATE - INTERVAL '7 days'
    """, (username,))
    anomaly_count = cursor.fetchone()["anomaly_days"]

    # --- Forecast readiness & confidence ---
    locked_manual_days_count = len(manual_days)

    if locked_manual_days_count >= 7:
        forecast_ready = True
        confidence_label = "🟢 Forecast ready — high confidence"
    elif locked_manual_days_count >= 5:
        forecast_ready = False
        confidence_label = "🟡 Almost there — a few more locked days needed"
    elif locked_manual_days_count >= 3:
        forecast_ready = False
        confidence_label = "🟠 Learning in progress — limited data"
    else:
        forecast_ready = False
        confidence_label = "🔴 Forecast locked — system still learning"

    cursor.close()
    conn.close()

    return {
        "manual_days": manual_days,
        "mpesa_days": mpesa_days,
        "anomaly_days": anomaly_count,
        "forecast_ready":forecast_ready,
        "confidence_label":confidence_label,
    }

def get_dashboard_intelligence_snapshot(username, days=7):
    """
    High-level intelligence summary for the dashboard.
    Reads ONLY locked revenue days.
    Used for AI reasoning & forecast readiness.
    """

    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    # Locked revenue days in window
    cursor.execute("""
        SELECT
            COUNT(*) AS locked_days,
            COALESCE(SUM(total_amount), 0) AS total_revenue
        FROM revenue_days
        WHERE username = %s
          AND locked = TRUE
          AND revenue_date >= CURRENT_DATE - INTERVAL '%s days'
    """, (username, days))

    row = cursor.fetchone()
    locked_days = row["locked_days"] or 0
    total_revenue = float(row["total_revenue"] or 0)

    # Anomaly days in window
    cursor.execute("""
        SELECT COUNT(DISTINCT revenue_date) AS anomaly_days
        FROM revenue_anomalies
        WHERE username = %s
          AND revenue_date >= CURRENT_DATE - INTERVAL '%s days'
    """, (username, days))

    anomaly_days = cursor.fetchone()["anomaly_days"] or 0

    cursor.close()
    conn.close()

    avg_daily_revenue = (
        total_revenue / locked_days if locked_days > 0 else 0
    )

    return {
        "window_days": days,
        "locked_days": locked_days,
        "anomaly_days": anomaly_days,
        "avg_daily_revenue": avg_daily_revenue,
        "ready_for_forecast": locked_days >= 30
    }

def get_locked_revenue_for_forecast(username, min_days=7):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT
            revenue_date AS ds,
            total_amount AS y
        FROM revenue_days
        WHERE username = %s
          AND locked = TRUE
        ORDER BY revenue_date ASC
    """, (username,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Prophet needs enough history
    if len(rows) < min_days:
        return {
            "ready": False,
            "days": len(rows),
            "data": []
        }

    return {
        "ready": True,
        "days": len(rows),
        "data": rows
    }

def get_live_financial_performance(username):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT
            created_at AT TIME ZONE 'Africa/Nairobi' AS timestamp,
            amount
        FROM mpesa_transactions
        WHERE status = 'confirmed'
        AND username = %s
        ORDER BY timestamp ASC
    """,(username,))
    revenue = cursor.fetchall()

    cursor.execute("""
        SELECT
            expense_date AS date,
            SUM(amount) AS expenses
        FROM expenses
        WHERE username = %s
        GROUP BY expense_date
        ORDER BY expense_date
    """, (username,))
    expenses = cursor.fetchall()

    cursor.close()
    conn.close()

    return {
        "revenue": revenue,
        "expenses": expenses
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


def load_revenue_days(username):
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            revenue_date,
            total_amount,
            locked
        FROM revenue_days
        WHERE username = %s
        ORDER BY revenue_date DESC
    """, (username,))

    days = cursor.fetchall()
    cursor.close()
    conn.close()
    return days

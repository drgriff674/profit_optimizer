import psycopg2
import os
from openai import OpenAI
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

DATABASE_URL = os.environ.get("DATABASE_URL")
connection_pool = pool.ThreadedConnectionPool(
    4, #min connections
    20, #max connections
    DATABASE_URL
    )

#  Railway PostgreSQL connection string

def get_db_connection():
    conn = connection_pool.getconn()

    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
    except:
        conn = psycopg2.connect(
            DATABASE_URL,
            sslmode="require"
        )
    return conn
        
    

def run_db_operation(operation, commit=False):
    
    conn = get_db_connection()
    cur = None

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # ✅ SAFE SESSION HANDLING
        try:
            from flask import session
            username = session.get("username")
        except:
            username = None

        # ✅ ONLY SET RLS IF USER EXISTS
        if username:
            cur.execute(f"SET app.current_username = '{username}'")

        # --- run actual query ---
        result = operation(cur)

        if commit:
            conn.commit()

        return result

    except Exception as e:
        conn.rollback()
        print("DB ERROR:", e)
        raise e

    finally:
        if cur:
            cur.close()
        connection_pool.putconn(conn)

#  Initialize database and tables if not exists
def init_db():
    def operation(cur):
    

        # --- USERS TABLE ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
        """)

        # --- CASH REVENUE ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cash_revenue (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
            revenue_date DATE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_cash_revenue_user_date
        ON cash_revenue(username, revenue_date);
        """)
    
        cur.execute("""
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
        cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            amount NUMERIC NOT NULL,
            category TEXT,
            description TEXT,
            expense_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
        """)

    

        cur.execute("""
        CREATE TABLE IF NOT EXISTS revenue_entries (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            category TEXT NOT NULL,
            amount NUMERIC NOT NULL,
            revenue_date DATE NOT NULL,
            locked BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
        """)

    

        cur.execute("""
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
        cur.execute("""
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

        cur.execute("""
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

        cur.execute("""
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

        cur.execute("""
        ALTER TABLE inventory_movements
        ADD COLUMN IF NOT EXISTS source TEXT;
        """)

        # --- MPESA TRANSACTIONS TABLE ---
        cur.execute("""
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

        cur.execute("""
        ALTER TABLE mpesa_transactions
        ADD COLUMN IF NOT EXISTS local_date DATE;
        """)

        cur.execute("""
        ALTER TABLE mpesa_transactions
        ADD COLUMN IF NOT EXISTS seen BOOLEAN DEFAULT FALSE
        """)


        cur.execute("""
        CREATE TABLE IF NOT EXISTS revenue_anomalies (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            revenue_date DATE NOT NULL,
            anomaly_type TEXT NOT NULL,
            severity TEXT NOT NULL, -- info, warning, critical
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS revenue_ai_summaries (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            revenue_date DATE NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (username, revenue_date),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS revenue_days (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            revenue_date DATE NOT NULL,
            locked BOOLEAN DEFAULT FALSE,
            total_amount NUMERIC(12,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (username, revenue_date),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );
        """)
        #dashboard snapshot table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_snapshot (
            username TEXT PRIMARY KEY,
            total_revenue NUMERIC(12,2) DEFAULT 0,
            total_expenses NUMERIC(12,2) DEFAULT 0,
            total_profit NUMERIC(12,2) DEFAULT 0,
            largest_expense TEXT DEFAULT 'N/A',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        ALTER TABLE dashboard_snapshot
        ADD COLUMN IF NOT EXISTS profit_growth NUMERIC(6,2);
        """)

        cur.execute("""
        UPDATE dashboard_snapshot
        SET profit_growth = 0
        WHERE profit_growth IS NULL;
        """)

        # dashboard intelligence table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_intelligence (
            username TEXT PRIMARY KEY,
            locked_days INTEGER DEFAULT 0,
            anomaly_days INTEGER DEFAULT 0,
            mpesa_days INTEGER DEFAULT 0,
            last_insight_date DATE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cur.execute("""
        ALTER TABLE dashboard_intelligence
        ADD COLUMN IF NOT EXISTS last_insight_date DATE;
        """)

        # weekly ai_reports table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS weekly_ai_reports(
            id SERIAL PRIMARY KEY,
            username TEXT,
            locked_days INTEGER,
            report TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            username TEXT,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # --- PRODUCTS (WHAT CAN BE SOLD) ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            business_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            price NUMERIC(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
        )
        """)

        # --- SALES ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id SERIAL PRIMARY KEY,
            sale_id TEXT UNIQUE NOT NULL,
            business_id INTEGER NOT NULL,
            total_amount NUMERIC(10,2) NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
        )
        """)

        # --- SALE ITEMS ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sale_items (
            id SERIAL PRIMARY KEY,
            sale_id TEXT NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price NUMERIC(10,2) NOT NULL,
            FOREIGN KEY (sale_id) REFERENCES sales(sale_id) ON DELETE CASCADE,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        )
        """)

        # --- SALES SYSTEM INDEXES ---

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_products_business
        ON products(business_id);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sales_business
        ON sales(business_id);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sales_sale_id
        ON sales(sale_id);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sales_created_at
        ON sales(created_at);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sales_status
        ON sales(status);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sales_business_date_status
        ON sales(business_id, created_at, status);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sale_items_sale_id
        ON sale_items(sale_id);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sale_items_product_id
        ON sale_items(product_id);
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_revenue_days_username 
        ON revenue_days(username);

        CREATE INDEX IF NOT EXISTS idx_revenue_days_username_locked 
        ON revenue_days(username, locked);

        CREATE INDEX IF NOT EXISTS idx_expenses_username 
        ON expenses(username);

        CREATE INDEX IF NOT EXISTS idx_revenue_entries_username 
        ON revenue_entries(username);

        CREATE INDEX IF NOT EXISTS idx_revenue_anomalies_username 
        ON revenue_anomalies(username);

        CREATE INDEX IF NOT EXISTS idx_businesses_username 
        ON businesses(username);

        CREATE INDEX IF NOT EXISTS idx_audit_username 
        ON audit_logs(username);

        CREATE INDEX IF NOT EXISTS idx_audit_action 
        ON audit_logs(action);

        CREATE INDEX IF NOT EXISTS idx_audit_created_at 
        ON audit_logs(created_at);

        CREATE INDEX IF NOT EXISTS idx_mpesa_status 
        ON mpesa_transactions(status);

        CREATE INDEX IF NOT EXISTS idx_mpesa_account_reference 
        ON mpesa_transactions(account_reference);

        CREATE INDEX IF NOT EXISTS idx_mpesa_receiver 
        ON mpesa_transactions(receiver);

        CREATE INDEX IF NOT EXISTS idx_mpesa_created_at 
        ON mpesa_transactions(created_at);
        """)
    
        cur.execute("""
        ALTER TABLE mpesa_transactions
        ADD COLUMN IF NOT EXISTS local_date DATE;
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_mpesa_local_date
        ON mpesa_transactions(local_date);
        """)

        # --- SUBSCRIPTIONS TABLE ---
        cur.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,

            plan TEXT DEFAULT 'monthly',
            status TEXT DEFAULT 'trial',

            trial_start TIMESTAMP,
            trial_end TIMESTAMP,

            subscription_start TIMESTAMP,
            subscription_end TIMESTAMP,

            last_payment_reference TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # --- MPESA HARD SECURITY ---
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'chk_mpesa_status'
            ) THEN
                ALTER TABLE mpesa_transactions
                ADD CONSTRAINT chk_mpesa_status
                CHECK (status IN ('pending','confirmed','failed'));
            END IF;
        END$$;
        """)

        cur.execute("""
        ALTER TABLE mpesa_transactions
        ALTER COLUMN amount SET NOT NULL;
        """)

        cur.execute("""
        ALTER TABLE mpesa_transactions
        ALTER COLUMN transaction_id SET NOT NULL;
        """)

        cur.execute("""
        ALTER TABLE mpesa_transactions
        ALTER COLUMN status SET NOT NULL;
        """)

        # --- PREVENT UPDATES (IMMUTABLE TRANSACTIONS) ---
        cur.execute("""
        CREATE OR REPLACE FUNCTION prevent_mpesa_update()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'MPesa transactions cannot be updated';
        END;
        $$ LANGUAGE plpgsql;
        """)

        cur.execute("""
        DROP TRIGGER IF EXISTS no_update_mpesa ON mpesa_transactions;
        """)

        cur.execute("""
        CREATE TRIGGER no_update_mpesa
        BEFORE UPDATE ON mpesa_transactions
        FOR EACH ROW
        EXECUTE FUNCTION prevent_mpesa_update();
        """)

    run_db_operation(operation, commit=True)


def process_payment(username, amount, local_date):
    def operation(cur):

        # 🔥 ensure day exists (same logic as MPesa)
        cur.execute("""
            INSERT INTO revenue_days (username, revenue_date)
            VALUES (%s, %s)
            ON CONFLICT (username, revenue_date) DO NOTHING
        """, (username, local_date))

        # 💰 update total
        cur.execute("""
            UPDATE revenue_days
            SET total_amount = total_amount + %s
            WHERE username = %s AND revenue_date = %s
        """, (amount, username, local_date))

    run_db_operation(operation, commit=True)

def create_sale(business_id, items):

    import uuid

    def operation(cur):

        sale_id = str(uuid.uuid4())
        total = 0

        # calculate total
        for item in items:
            cur.execute("SELECT price FROM products WHERE id=%s", (item["product_id"],))
            product = cur.fetchone()
            if not product:
                continue

            total += float(product["price"]) * int(item["quantity"])

        # create sale
        cur.execute("""
            INSERT INTO sales (sale_id, business_id, total_amount, status)
            VALUES (%s, %s, %s, 'pending')
        """, (sale_id, business_id, total))

        # insert items
        for item in items:
            cur.execute("SELECT price FROM products WHERE id=%s", (item["product_id"],))
            product = cur.fetchone()
            if not product:
                continue

            cur.execute("""
                INSERT INTO sale_items (sale_id, product_id, quantity, price)
                VALUES (%s, %s, %s, %s)
            """, (
                sale_id,
                item["product_id"],
                item["quantity"],
                product["price"]
            ))

        return {"sale_id": sale_id, "total": total}

    return run_db_operation(operation, commit=True)


def get_top_products_for_day(username, date):

    def operation(cur):

        cur.execute("""
            SELECT b.id
            FROM businesses b
            WHERE b.username = %s
            LIMIT 1
        """, (username,))

        biz = cur.fetchone()
        if not biz:
            return []

        business_id = biz["id"]

        cur.execute("""
            SELECT 
                p.name,
                SUM(si.quantity) as total_sold,
                SUM(si.quantity * si.price) as revenue
            FROM sales s
            JOIN sale_items si ON s.sale_id = si.sale_id
            JOIN products p ON p.id = si.product_id
            WHERE s.business_id = %s
              AND DATE(s.created_at) = %s
              AND s.status = 'completed'
            GROUP BY p.name
            ORDER BY total_sold DESC
            LIMIT 5
        """, (business_id, date))

        return cur.fetchall()

    return run_db_operation(operation)


#revenue cash functions
def add_cash_revenue(username, amount, revenue_date, description=None):

    def operation(cur):
        cur.execute("""
            INSERT INTO cash_revenue (username, amount, revenue_date, description)
            VALUES (%s, %s, %s, %s)
        """, (username, amount, revenue_date, description))

    run_db_operation(operation, commit=True)

def get_cash_revenue_for_day(username, revenue_date):

    def operation(cur):
        cur.execute("""
            SELECT amount, description
            FROM cash_revenue
            WHERE username = %s
              AND revenue_date = %s
            ORDER BY created_at ASC
        """, (username, revenue_date))
        return cur.fetchall()

    return run_db_operation(operation)

def get_cash_revenue_total_for_day(username, revenue_date):

    def operation(cur):
        cur.execute("""
            SELECT COALESCE(SUM(amount), 0) AS total
            FROM cash_revenue
            WHERE username = %s
              AND revenue_date = %s
        """, (username, revenue_date))
        row = cur.fetchone()
        return float(row["total"] or 0)

    return run_db_operation(operation)

#load users dict
def load_users():

    def operation(cur):
        cur.execute("SELECT username, password, role FROM users")
        rows = cur.fetchall()
        return {
            row["username"]: {
                "password": row["password"],
                "role": row["role"]
            }
            for row in rows
        }

    return run_db_operation(operation)

#save users
def save_user(username, password, role):

    def operation(cur):
        cur.execute(
            """
            INSERT INTO users (username, password, role)
            VALUES (%s, %s, %s)
            ON CONFLICT (username) DO NOTHING
            """,
            (username, password, role)
        )

    run_db_operation(operation, commit=True)
    

#  Debug: print all users directly from DB
def debug_print_users():

    def operation(cur):
        cur.execute("SELECT username, role FROM users")
        return cur.fetchall()

    rows = run_db_operation(operation)
    print("📌 Current users in DB:", rows)
    

#  EXPENSE HELPERS (NEW – USED BY MANUAL ENTRY)
def save_expense(username, amount, category, description, expense_date):

    def operation(cur):
        cur.execute("""
            INSERT INTO expenses (username, amount, category, description, expense_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, amount, category, description, expense_date))

    run_db_operation(operation, commit=True)

def get_expenses_for_day(username, date):

    def operation(cur):
        cur.execute("""
            SELECT amount, category, description
            FROM expenses
            WHERE username = %s
              AND expense_date = %s
        """, (username, date))

        rows = cur.fetchall()
        total = sum(float(r["amount"]) for r in rows)

        return {
            "entries": rows,
            "total": total
        }

    return run_db_operation(operation)

def load_expenses(username):

    def operation(cur):
        cur.execute("""
            SELECT
                amount,
                category,
                description,
                expense_date,
                created_at
            FROM expenses
            WHERE username = %s
            ORDER BY expense_date DESC
        """, (username,))
        return cur.fetchall()

    return run_db_operation(operation)

#  REVENUE ENTRY HELPERS (MANUAL DAILY SPLITS)
def save_revenue_entry(username, category, amount, revenue_date):

    def operation(cur):
        cur.execute("""
            INSERT INTO revenue_entries (username, category, amount, revenue_date)
            VALUES (%s, %s, %s, %s)
        """, (username, category, amount, revenue_date))

    run_db_operation(operation, commit=True)


def load_expense_categories(username):

    def operation(cur):
        cur.execute("""
            SELECT DISTINCT category
            FROM expenses
            WHERE username = %s
              AND category IS NOT NULL
              AND category <> ''
            ORDER BY category ASC
        """, (username,))

        rows = cur.fetchall()
        return [row["category"] for row in rows]

    return run_db_operation(operation)

def lock_manual_entries_for_the_day(username, revenue_date):

    def operation(cur):
        cur.execute("""
            UPDATE revenue_entries
            SET locked = TRUE
            WHERE username = %s AND revenue_date = %s
        """, (username, revenue_date))

    run_db_operation(operation, commit=True)

def load_revenue_entries_for_day(username, revenue_date):

    def operation(cur):
        cur.execute("""
            SELECT id, category, amount, locked
            FROM revenue_entries
            WHERE username = %s AND revenue_date = %s
            ORDER BY id ASC
        """, (username, revenue_date))
        return cur.fetchall()

    return run_db_operation(operation)

def save_business(username, business_name, paybill, account_number):

    def operation(cur):
        cur.execute("""
            INSERT INTO businesses (username, business_name, paybill, account_number)
            VALUES (%s, %s, %s, %s)
        """, (username, business_name, paybill, account_number))

    run_db_operation(operation, commit=True)
    
def load_user_business(username):

    def operation(cur):
        cur.execute("""
            SELECT business_name, paybill, account_number
            FROM businesses
            WHERE username = %s
            LIMIT 1
        """, (username,))
        return cur.fetchone()

    return run_db_operation(operation)

def get_existing_revenue_days(username):

    def operation(cur):

        cur.execute("""
            SELECT revenue_date AS day
            FROM revenue_days
            WHERE username = %s
            ORDER BY revenue_date DESC
        """, (username,))

        rows = cur.fetchall()

        return [row["day"] for row in rows]

    return run_db_operation(operation)

def ensure_revenue_day_exists(username, revenue_date):

    def operation(cur):
        cur.execute("""
            INSERT INTO revenue_days (username, revenue_date)
            VALUES (%s, %s)
            ON CONFLICT (username, revenue_date) DO NOTHING
        """, (username, revenue_date))

    run_db_operation(operation, commit=True)

def get_ai_summary_for_day(username, revenue_date):

    def operation(cur):

        cur.execute("""
            SELECT summary
            FROM revenue_ai_summaries
            WHERE username = %s AND revenue_date = %s
        """, (username, revenue_date))

        row = cur.fetchone()

        return row["summary"] if row else None

    return run_db_operation(operation)

def save_ai_summary_for_day(username, revenue_date, summary):

    def operation(cur):
        cur.execute("""
            INSERT INTO revenue_ai_summaries (username, revenue_date, summary)
            VALUES (%s, %s, %s)
            ON CONFLICT (username, revenue_date)
            DO UPDATE SET summary = EXCLUDED.summary
        """, (username, revenue_date, summary))

    run_db_operation(operation, commit=True)

def detect_revenue_anomalies(username, revenue_date):

    def operation(cur):

        # --- totals ---
        cur.execute("""
            SELECT total_amount
            FROM revenue_days
            WHERE username = %s AND revenue_date = %s
        """, (username, revenue_date))

        row = cur.fetchone()
        total = float(row["total_amount"]) if row else 0

        anomalies = []

        # --- Rule A: MPesa dominance ---
        if total > 0:
            cur.execute("""
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
                  AND m.local_date = %s
            """, (username, revenue_date))

            mpesa_total = float(cur.fetchone()["mpesa_total"])
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
        cur.execute("""
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

        avg_row = cur.fetchone()

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
            cur.execute("""
                INSERT INTO revenue_anomalies
                (username, revenue_date, anomaly_type, severity, message)
                VALUES (%s, %s, %s, %s, %s)
            """, (username, revenue_date, *a))

    run_db_operation(operation, commit=True)

# INVENTORY HELPERS
def create_inventory_item(business_id, name, category, unit):

    def operation(cur):
        cur.execute(
            """
            INSERT INTO inventory_items (business_id, name, category, unit)
            VALUES (%s, %s, %s, %s)
            """,
            (business_id, name, category, unit)
        )

    run_db_operation(operation, commit=True)
    
def get_dashboard_revenue_intelligence(username):

    def operation(cur):

        # REAL locked business days (SOURCE OF TRUTH)
        cur.execute("""
            SELECT revenue_date
            FROM revenue_days
            WHERE username = %s
              AND locked = TRUE
              AND revenue_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY revenue_date DESC
        """, (username,))
        manual_days = cur.fetchall()

        # MPesa revenue DAYS (not totals)
        cur.execute("""
            SELECT DISTINCT
                m.local_date AS revenue_date
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                    (b.account_number IS NOT NULL AND m.account_reference = b.account_number)
                    OR
                    (b.account_number IS NULL AND m.receiver = b.paybill)
                 )
            WHERE b.username = %s
              AND m.status = 'confirmed'
              AND m.local_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY revenue_date DESC
        """, (username,))
        mpesa_days = cur.fetchall()

        # anomaly count
        cur.execute("""
            SELECT COUNT(DISTINCT revenue_date) AS anomaly_days
            FROM revenue_anomalies
            WHERE username = %s
              AND revenue_date >= CURRENT_DATE - INTERVAL '7 days'
        """, (username,))
        anomaly_count = cur.fetchone()["anomaly_days"]

        return {
            "manual_days": manual_days,
            "mpesa_days": mpesa_days,
            "anomaly_days": anomaly_count,
        }

    return run_db_operation(operation)

def get_dashboard_intelligence_snapshot(username, days=7):

    def operation(cur):

        # TRUE locked days count
        cur.execute("""
            SELECT
                COUNT(*) AS locked_days,
                COALESCE(SUM(total_amount),0) AS total_revenue
            FROM revenue_days
            WHERE username = %s
              AND locked = TRUE
              AND revenue_date >= CURRENT_DATE - (%s * INTERVAL '1 day')
        """, (username, days))

        row = cur.fetchone()

        locked_days = int(row["locked_days"] or 0)
        total_revenue = float(row["total_revenue"] or 0)

        # anomaly count
        cur.execute("""
            SELECT COUNT(DISTINCT revenue_date) AS anomaly_days
            FROM revenue_anomalies
            WHERE username = %s
              AND revenue_date >= CURRENT_DATE - (%s * INTERVAL '1 day')
        """, (username, days))

        anomaly_days = int(cur.fetchone()["anomaly_days"] or 0)

        avg_daily_revenue = total_revenue / locked_days if locked_days else 0

        # Forecast maturity tiers
        if locked_days < 7:
            confidence = "Insufficient"
            horizon = 0
            ready = False

        elif locked_days < 14:
            confidence = "Low"
            horizon = 7
            ready = True

        elif locked_days < 30:
            confidence = "Medium"
            horizon = 30
            ready = True

        else:
            confidence = "High"
            horizon = 90
            ready = True

        return {
            "locked_days": locked_days,
            "anomaly_days": anomaly_days,
            "avg_daily_revenue": avg_daily_revenue,
            "confidence": confidence,
            "horizon": horizon,
            "ready": ready
        }

    return run_db_operation(operation)

def get_locked_revenue_for_forecast(username):

    def operation(cur):

        cur.execute("""
            SELECT
                revenue_date AS ds,
                total_amount AS y
            FROM revenue_days
            WHERE username = %s
              AND locked = TRUE
            ORDER BY revenue_date ASC
        """, (username,))

        rows = cur.fetchall()

        days = len(rows)

        # Forecast maturity tiers
        if days < 7:
            return {
                "ready": False,
                "days": days,
                "confidence": "Insufficient",
                "forecast_period": 0,
                "data": []
            }

        elif days < 14:
            return {
                "ready": True,
                "days": days,
                "confidence": "Low",
                "forecast_period": 7,
                "data": rows
            }

        elif days < 30:
            return {
                "ready": True,
                "days": days,
                "confidence": "Medium",
                "forecast_period": 30,
                "data": rows
            }

        else:
            return {
                "ready": True,
                "days": days,
                "confidence": "High",
                "forecast_period": 90,
                "data": rows
            }

    return run_db_operation(operation)
    
def get_live_financial_performance(username):

    def operation(cur):

        # ---------- MPESA REVENUE ----------
        cur.execute("""
            SELECT
                m.local_date::timestamp AS timestamp,
                SUM(m.amount) AS amount
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                    (b.account_number IS NOT NULL AND m.account_reference = b.account_number)
                    OR
                    (b.account_number IS NULL AND m.receiver = b.paybill)
                 )
            WHERE b.username = %s
              AND m.status = 'confirmed'
              AND m.created_at IS NOT NULL
              AND m.created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY m.local_date
        """, (username,))
        mpesa = cur.fetchall()

        # ---------- CASH REVENUE ----------
        cur.execute("""
            SELECT
                revenue_date::timestamp AS timestamp,
                SUM(amount) AS amount
            FROM cash_revenue
            WHERE username = %s
              AND revenue_date > CURRENT_DATE - INTERVAL '30 days'
            GROUP BY revenue_date
        """, (username,))
        cash = cur.fetchall()

        # ---------- COMBINE BOTH ----------
        combined = {}

        for r in mpesa:
            combined[r["timestamp"]] = combined.get(r["timestamp"], 0) + float(r["amount"])

        for r in cash:
            combined[r["timestamp"]] = combined.get(r["timestamp"], 0) + float(r["amount"])

        revenue = [
            {"timestamp": k, "amount": v}
            for k, v in sorted(combined.items())
        ]

        # ---------- EXPENSES ----------
        cur.execute("""
            SELECT
                expense_date::timestamp AS timestamp,
                SUM(amount) AS expenses
            FROM expenses
            WHERE username = %s
            GROUP BY expense_date
            ORDER BY expense_date ASC
        """, (username,))
        expenses = cur.fetchall()

        return {
            "revenue": revenue,
            "expenses": expenses
        }

    return run_db_operation(operation)

def create_inventory_snapshot(business_id, snapshot_date, snapshot_type, created_by):

    def operation(cur):
        cur.execute(
            """
            INSERT INTO inventory_snapshots
            (business_id, snapshot_date, snapshot_type, created_by)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (business_id, snapshot_date, snapshot_type, created_by)
        )

        return cur.fetchone()["id"]

    return run_db_operation(operation, commit=True)

def save_snapshot_item(snapshot_id, item_id, quantity):

    def operation(cur):
        cur.execute(
            """
            INSERT INTO inventory_snapshot_items
            (snapshot_id, item_id, quantity)
            VALUES (%s, %s, %s)
            """,
            (snapshot_id, item_id, quantity)
        )

    run_db_operation(operation, commit=True)


def load_latest_inventory_snapshot(business_id):

    def operation(cur):
        cur.execute(
            """
            SELECT *
            FROM inventory_snapshots
            WHERE business_id = %s
            ORDER BY snapshot_date DESC, created_at DESC
            LIMIT 1
            """,
            (business_id,)
        )

        return cur.fetchone()

    return run_db_operation(operation)

def load_snapshot_items(snapshot_id):

    def operation(cur):
        cur.execute(
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

        return cur.fetchall()

    return run_db_operation(operation)


def load_revenue_days(username):

    def operation(cur):
        cur.execute("""
            SELECT
                revenue_date,
                total_amount,
                locked
            FROM revenue_days
            WHERE username = %s
            ORDER BY revenue_date DESC
        """, (username,))

        return cur.fetchall()

    return run_db_operation(operation)

def update_dashboard_snapshot(username):

    if not username:
        print("snapshot skipped: username missing")
        return

    def operation(cur):

        # CASH TOTAL
        cur.execute("""
            SELECT COALESCE(SUM(amount),0) AS total
            FROM cash_revenue
            WHERE username=%s
        """,(username,))
        cash=float(cur.fetchone()["total"])

        # MPESA TOTAL
        cur.execute("""
            SELECT COALESCE(SUM(m.amount),0) AS total
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                    (b.account_number IS NOT NULL AND m.account_reference=b.account_number)
                    OR
                    (b.account_number IS NULL AND m.receiver=b.paybill)
                 )
            WHERE b.username=%s
              AND m.status='confirmed'
        """,(username,))
        mpesa=float(cur.fetchone()["total"])

        # SALES TOTAL (ONLY COMPLETED SALES)
        cur.execute("""
            SELECT COALESCE(SUM(s.total_amount),0) AS total
            FROM sales s
            JOIN businesses b ON s.business_id = b.id
            WHERE b.username = %s
              AND s.status = 'completed'
        """, (username,))

        sales_total = float(cur.fetchone()["total"])

        revenue=cash+mpesa+sales_total

        # EXPENSE TOTAL
        cur.execute("""
            SELECT COALESCE(SUM(amount),0) AS total
            FROM expenses
            WHERE username=%s
        """,(username,))
        expenses=float(cur.fetchone()["total"])

        profit=revenue-expenses

        # PROFIT GROWTH
        cur.execute("""
            SELECT total_amount
            FROM revenue_days
            WHERE username=%s
              AND locked=TRUE
            ORDER BY revenue_date DESC
            LIMIT 2
        """,(username,))
        rows = cur.fetchall()

        if len(rows) >= 2:
            current = float(rows[0]["total_amount"])
            previous = float(rows[1]["total_amount"])

            if previous > 0:
                growth = round(((current-previous)/previous)*100,2)
                growth = min(growth, 999999)
            else:
                growth = 0
        else:
            growth = 0
        

        # LARGEST EXPENSE
        cur.execute("""
            SELECT COALESCE(category,'Uncategorized') AS category
            FROM expenses
            WHERE username=%s
            GROUP BY category
            ORDER BY SUM(amount) DESC
            LIMIT 1
        """,(username,))
        row = cur.fetchone()
        largest_expense = row["category"] if row else "N/A"

        # UPSERT SNAPSHOT
        cur.execute("""
            INSERT INTO dashboard_snapshot
            (username,total_revenue,total_expenses,total_profit,largest_expense,profit_growth)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT(username)
            DO UPDATE SET
                total_revenue=EXCLUDED.total_revenue,
                total_expenses=EXCLUDED.total_expenses,
                total_profit=EXCLUDED.total_profit,
                largest_expense=EXCLUDED.largest_expense,
                profit_growth=EXCLUDED.profit_growth,
                updated_at=CURRENT_TIMESTAMP
        """,(username,revenue,expenses,profit,largest_expense,growth))

    run_db_operation(operation, commit=True)

def update_dashboard_intelligence(username):

    if not username:
        print("intelligence skipped: username missing")
        return

    def operation(cur):

        # locked days
        cur.execute("""
            SELECT COUNT(*) AS c
            FROM revenue_days
            WHERE username=%s AND locked=TRUE
        """,(username,))
        locked_days = cur.fetchone()["c"]

        # anomaly days
        cur.execute("""
            SELECT COUNT(DISTINCT revenue_date) AS c
            FROM revenue_anomalies
            WHERE username=%s
        """,(username,))
        anomaly_days = cur.fetchone()["c"]

        # mpesa days
        cur.execute("""
            SELECT COUNT(DISTINCT m.local_date) AS c
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                  (b.account_number IS NOT NULL AND m.account_reference=b.account_number)
                  OR
                  (b.account_number IS NULL AND m.receiver=b.paybill)
              )
            WHERE b.username=%s
              AND m.status='confirmed'
        """,(username,))
        mpesa_days = cur.fetchone()["c"]

        # upsert intelligence snapshot
        cur.execute("""
            INSERT INTO dashboard_intelligence
            (username,locked_days,anomaly_days,mpesa_days)
            VALUES(%s,%s,%s,%s)
            ON CONFLICT(username)
            DO UPDATE SET
                locked_days=EXCLUDED.locked_days,
                anomaly_days=EXCLUDED.anomaly_days,
                mpesa_days=EXCLUDED.mpesa_days,
                updated_at=CURRENT_TIMESTAMP
        """,(username,locked_days,anomaly_days,mpesa_days))

    run_db_operation(operation, commit=True)

def get_dashboard_snapshot(username):

    def operation(cur):

        cur.execute(
            "SELECT * FROM dashboard_snapshot WHERE username=%s",
            (username,)
        )

        row = cur.fetchone()

        return row or {
            "total_revenue": 0,
            "total_expenses": 0,
            "total_profit": 0
        }

    return run_db_operation(operation)

def get_dashboard_intelligence(username):

    def operation(cur):

        cur.execute(
            "SELECT * FROM dashboard_intelligence WHERE username=%s",
            (username,)
        )

        row = cur.fetchone()

        return row or {
            "locked_days": 0,
            "anomaly_days": 0,
            "mpesa_days": 0
        }

    return run_db_operation(operation)

def maybe_generate_dashboard_insight(username):

    def operation(cur):

        # get last insight date
        cur.execute("""
            SELECT last_insight_date
            FROM dashboard_intelligence
            WHERE username=%s
        """,(username,))

        row = cur.fetchone()
        last_date = row["last_insight_date"] if row else None

        # count locked days AFTER last insight
        if last_date:
            cur.execute("""
                SELECT COUNT(*) AS c
                FROM revenue_days
                WHERE username=%s
                AND locked=TRUE
                AND revenue_date > %s
            """,(username,last_date))
        else:
            cur.execute("""
                SELECT COUNT(*) AS c
                FROM revenue_days
                WHERE username=%s
                AND locked=TRUE
            """,(username,))

        new_locked = cur.fetchone()["c"]

        if new_locked >= 7:

            cur.execute("""
                UPDATE dashboard_intelligence
                SET last_insight_date = CURRENT_DATE
                WHERE username=%s
            """,(username,))

    run_db_operation(operation, commit=True)
    
def run_weekly_intelligence(username):

    def operation(cur):

        cur.execute("""
            SELECT revenue_date,total_amount
            FROM revenue_days
            WHERE username=%s AND locked=TRUE
            ORDER BY revenue_date DESC
        """,(username,))

        rows = cur.fetchall()

        # not enough data
        if len(rows) < 7:
            return

        # only run every 7 days
        if len(rows) % 7 != 0:
            return

        last7 = rows[:7]

        values = [float(r["total_amount"]) for r in last7]

        avg = sum(values) / 7
        sorted_values = sorted(values)
        mx = sorted_values[-2]
        mn = sorted_values[1]

        if mx - mn > avg * 1.2:
            summary = "Revenue unstable this week. Large daily fluctuations detected."
        elif avg > 0 and values[0] > values[-1]:
            summary = "Revenue trending upward. Positive business momentum."
        else:
            summary = "Revenue stable. Business operating normally."

        cur.execute("""
            INSERT INTO revenue_ai_summaries
            (username,revenue_date,summary)
            VALUES(%s,CURRENT_DATE,%s)
            ON CONFLICT DO NOTHING
        """,(username,summary))

    run_db_operation(operation, commit=True)
    
def generate_weekly_ai_report_if_ready(username):

    def operation(cur):

        cur.execute("""
            SELECT COUNT(*) AS locked_days
            FROM revenue_days
            WHERE username=%s AND locked=TRUE
        """,(username,))

        locked = cur.fetchone()["locked_days"]

        # only fire every 7 days
        if locked % 7 != 0:
            return

        # check if this report already exists
        cur.execute("""
            SELECT report
            FROM weekly_ai_reports
            WHERE username=%s AND locked_days=%s
        """,(username,locked))

        existing = cur.fetchone()

        if existing and existing["report"] != "AI Insight temporarily unavailable.":
            return

        # LOAD LAST 7 LOCKED DAYS DATA
        cur.execute("""
            SELECT revenue_date,total_amount
            FROM revenue_days
            WHERE username=%s AND locked=TRUE
            ORDER BY revenue_date DESC
            LIMIT 7
        """,(username,))

        rows = cur.fetchall()

        # build summary text for OpenAI
        summary_lines = []
        for i, r in enumerate(rows, start=1):
            amount = float(r["total_amount"] or 0)
            summary_lines.append(f"Day {i}: KSh {amount:,.0f}")
        summary = "Weekly revenue data:\n" + "\n".join(summary_lines)

        # ---- CALL OPENAI ----
        ai_text = call_openai(summary)

        # SAVE REPORT
        cur.execute("""
            INSERT INTO weekly_ai_reports(username,locked_days,report)
            VALUES(%s,%s,%s)
        """,(username,locked,ai_text))

    run_db_operation(operation, commit=True)
    
def call_openai(summary):

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a business intelligence assistant for small businesses.

Analyze weekly revenue data and produce short actionable insights.

Rules:
- Write EXACTLY 3 bullet-point insights.
- Each insight must be ONE sentence.
- Focus on unusual spikes, drops, or patterns.
- Suggest a simple action when relevant.
- Keep each insight under 20 words.

Format output exactly like this:

• Insight 1
• Insight 2
• Insight 3
"""
                },
                {
                    "role": "user",
                    "content": f"""
Weekly revenue data:

{summary}
"""
                }
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("OpenAI error:", e)
        return f"AI error: {str(e)}"



def detect_weekly_alerts(rows):

    values = [float(r["total_amount"]) for r in rows]

    avg = sum(values) / len(values)
    mx = max(values)
    mn = min(values)

    alerts = []

    if mx > avg * 2:
        alerts.append("⚠ Revenue spike detected this week.")

    if mn < avg * 0.5:
        alerts.append("📉 Revenue dropped significantly on one day.")

    if max(values) - min(values) > avg:
        alerts.append("⚠ Revenue volatility detected across the week.")

    return alerts
    
def get_latest_weekly_report(username):

    def operation(cur):

        cur.execute("""
            SELECT report, locked_days, created_at
            FROM weekly_ai_reports
            WHERE username=%s
            ORDER BY created_at DESC
            LIMIT 1
        """, (username,))

        return cur.fetchone()

    return run_db_operation(operation)

def get_user(username):

    def operation(cur):
        cur.execute(
            "SELECT username,email, password, role FROM users WHERE username=%s",
            (username,)
        )
        return cur.fetchone()

    return run_db_operation(operation)

def get_weekly_inventory_insights(username):

    def operation(cur):

        cur.execute("""
            SELECT id
            FROM businesses
            WHERE username = %s
            LIMIT 1
        """, (username,))

        biz = cur.fetchone()
        if not biz:
            return []

        business_id = biz["id"]

        # movements last 7 days
        cur.execute("""
            SELECT movement_type, COUNT(*) as count
            FROM inventory_movements
            WHERE business_id = %s
            AND created_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY movement_type
        """, (business_id,))

        rows = cur.fetchall()

        insights = []

        for r in rows:
            mtype = r["movement_type"]
            count = r["count"]

            if mtype == "adjustment":
                insights.append(
                    f"📦 {count} inventory adjustments were recorded this week."
                )

            elif mtype == "restock":
                insights.append(
                    f"📥 {count} inventory restocks occurred this week."
                )

            elif mtype == "sale":
                insights.append(
                    f"📤 Inventory decreased {count} times due to sales."
                )
            elif mtype == "usage":
                insights.append(
                    f"📤 Inventory decreased {count} times due to usage."
                )
                    

        return insights

    return run_db_operation(operation)

def create_user_with_business(username, email, password, role, business_name, paybill, account_number):

    def operation(cur):

        # check if user exists
        cur.execute(
            "SELECT 1 FROM users WHERE username=%s OR email=%s",
            (username, email)
        )

        if cur.fetchone():
            return "exists"

        # insert user
        cur.execute(
            """
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, %s)
            """,
            (username, email, password, role)
        )

        # insert business
        cur.execute(
            """
            INSERT INTO businesses (username, business_name, paybill, account_number)
            VALUES (%s, %s, %s, %s)
            """,
            (username, business_name, paybill, account_number)
        )


        from datetime import datetime, timedelta

        now = datetime.utcnow()
        trial_end = now + timedelta(days=7)

        cur.execute("""
            INSERT INTO subscriptions (
                username,
                status,
                trial_start,
                trial_end
            )
            VALUES (%s, %s, %s, %s)
        """, (
            username,
            "trial",
            now,
            trial_end
        ))

        return "created"

    return run_db_operation(operation, commit=True)

def get_subscription(username):

    def operation(cur):
        cur.execute("""
            SELECT *
            FROM subscriptions
            WHERE username = %s
            LIMIT 1
        """, (username,))
        return cur.fetchone()

    return run_db_operation(operation)

def get_business_info(username):

    def operation(cur):
        cur.execute("""
            SELECT business_name, paybill, account_number
            FROM businesses
            WHERE username = %s
            LIMIT 1
        """, (username,))

        return cur.fetchone()

    return run_db_operation(operation)

def get_user_by_email(email):

    def operation(cur):

        cur.execute(
            "SELECT*FROM users WHERE email=%s",
            (email,)
        )

        return cur.fetchone()

    return run_db_operation(operation)

def log_audit(username, action, details=None, ip_address=None):

    def operation(cur):
        cur.execute("""
            INSERT INTO audit_logs (username, action, details, ip_address)
            VALUES (%s, %s, %s, %s)
        """, (username, action, details, ip_address))

    run_db_operation(operation, commit=True)

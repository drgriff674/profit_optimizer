"""
Migration script: Fix 'amount' column type in mpesa_transactions table.

Purpose:
Ensures 'amount' column is numeric to support financial calculations.
This was originally used as a patch inside app.py and is now extracted for clarity.
"""

import os
import psycopg2

def fix_amount_column():
    """Convert 'amount' column from text/money to numeric if needed"""
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()
        cur.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'mpesa_transactions'
                      AND column_name = 'amount'
                      AND data_type IN ('text', 'money')
                ) THEN
                    ALTER TABLE mpesa_transactions
                    ALTER COLUMN amount TYPE numeric 
                    USING REPLACE(amount::text, '$', '')::numeric;
                END IF;
            END $$;
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Migration completed: 'amount' column converted to numeric.")
    except Exception as e:
        print(f"⚠️ Migration failed or skipped: {e}")

if __name__ == "__main__":
    fix_amount_column()

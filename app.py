print("app.py import started")
from flask import (
    Flask,
    render_template,
    make_response,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file,
    jsonify,
)
from flask import Response
import csv
from io import StringIO
from functools import wraps
from xhtml2pdf import pisa
import json
import os
from openai import OpenAI
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
from urllib.parse import quote
from werkzeug.utils import secure_filename
from datetime import datetime
from fpdf import FPDF
import io
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
import requests
from requests.auth import HTTPBasicAuth
from flask_mail import Mail, Message
from dotenv import load_dotenv
load_dotenv()
from prophet import Prophet
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2.extras import RealDictCursor
from database import(
    load_users,
    save_user,
    init_db,
    save_expense,
    load_expenses,
    get_db_connection,
    load_revenue_entries_for_day,
    load_expense_categories,
    save_revenue_entry,
    lock_manual_entries_for_the_day,
    load_revenue_days,
    get_ai_summary_for_day,
    save_ai_summary_for_day,
    detect_revenue_anomalies,
    get_dashboard_revenue_intelligence,
    get_existing_revenue_days,
    ensure_revenue_day_exists,
    get_expenses_for_day,
    add_cash_revenue,
    get_cash_revenue_for_day,
    get_cash_revenue_total_for_day,
)
import pytz
from flask_caching import Cache
from datetime import date
import re

# ============================
# Revenue helpers
# ============================

def generate_revenue_day_export_data(username, revenue_date):
    manual_entries = load_revenue_entries_for_day(username, revenue_date)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # 1️⃣ Get business identifiers
    cursor.execute("""
        SELECT paybill, account_number
        FROM businesses
        WHERE username = %s
        LIMIT 1
    """, (username,))
    biz = cursor.fetchone()

    if not biz:
        cursor.close()
        conn.close()
        manual_total = sum(float(e["amount"]) for e in manual_entries)

        expenses = get_expenses_for_day(username, revenue_date)
        expense_total = float(expenses["total"])

        net_total = manual_total - expense_total

        return {
            "date": revenue_date,
            "manual_entries": manual_entries,
            "mpesa_entries": [],
            "expense_entries": expenses["entries"],
            "manual_total": manual_total,
            "mpesa_total": 0,
            "expense_total": expense_total,
            "net_total": net_total,
        }
    paybill = biz["paybill"]
    account_number = biz["account_number"]

    # 2️⃣ Pull MPesa scoped to THIS business + THIS day
    cursor.execute("""
        SELECT amount, sender, transaction_id, created_at
        FROM mpesa_transactions
        WHERE DATE(created_at) = %s
          AND (
                (%s IS NOT NULL AND account_reference = %s)
                OR
                (%s IS NULL AND receiver = %s)
          )
        ORDER BY created_at ASC
    """, (
        revenue_date,
        account_number,
        account_number,
        account_number,
        paybill
    ))

    mpesa_entries = cursor.fetchall()

    cursor.close()
    conn.close()

    manual_total = sum(float(e["amount"]) for e in manual_entries)
    mpesa_total = sum(float(m["amount"]) for m in mpesa_entries)

    expenses = get_expenses_for_day(username, revenue_date)
    expense_total = float(expenses["total"])

    gross_total = manual_total + mpesa_total
    net_total = gross_total - expense_total

    return {
        "date": revenue_date,
        "manual_entries": manual_entries,
        "mpesa_entries": mpesa_entries,
        "expense_entries": expenses["entries"],
        "manual_total": manual_total,
        "mpesa_total": mpesa_total,
        "expense_total": expense_total,
        "net_total": net_total,
    }

def generate_revenue_ai_summary(date, manual_total, mpesa_total, manual_entries):
    if not AI_ENABLED or client is None:
        return "AI summary unavailable. OpenAI API key not configured."

    categories = {}
    for e in manual_entries:
        categories[e["category"]] = categories.get(e["category"], 0) + float(e["amount"])

    category_lines = "\n".join(
        f"- {k}: KSh {v}" for k, v in categories.items()
    )

    prompt = f"""
You are a financial analyst.

Analyze the following daily revenue:

Date: {date}
Manual revenue: KSh {manual_total}
MPesa revenue: KSh {mpesa_total}
Total revenue: KSh {manual_total + mpesa_total}

Breakdown by category:
{category_lines}

Tasks:
- Explain performance in simple business language
- Mention anomalies (spikes, drops, imbalance)
- Keep it under 120 words
- Be factual, not motivational
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def run_cron_jobs():
    print("⏰ Running background jobs")

    # 1. Sync Google Sheets
    sync_google_sheets()

    # 2. Generate AI insights
    generate_ai_insights()

    # 3. Run forecasts
    run_forecasts()

    # 4. Clear dashboard cache
    cache.delete("dashboard_data")

    print("✅ Cron jobs done, cache refreshed")

# ✅ Safe import for pdfkit (Render may not have wkhtmltopdf)
try:
    import pdfkit
    pdfkit_available = True
except ImportError:
    from xhtml2pdf import pisa
    pdfkit_available = False


# Railway PostgreSQL connection
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


# ✅ Flask app setup
app = Flask(__name__)
cache = Cache(app, config={
    "CACHE_TYPE": "SimpleCache",  # safe for localhost
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutes
})
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 📧 Mail configuration (uses environment variables)
mail_port = os.getenv("MAIL_PORT")

app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER") or "smtp.gmail.com"
app.config["MAIL_PORT"] = int(mail_port)if mail_port else 587
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS") == "True"
app.config["MAIL_USE_SSL"] = os.getenv("MAIL_USE_SSL") == "True"
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = ("OptiGain Reports",os.getenv("MAIL_USERNAME")or "noreply@optigain.local")

mail = Mail(app)

init_db()

# Load OpenAI only if API key exists
# Load environment variables
load_dotenv()

# Optional AI setup (works both local + Render)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    AI_ENABLED = True
    print("AI Enabled (API key detected)")
else:
    client = None
    AI_ENABLED = False
    print("⚠️ No OpenAI API key found — AI features disabled locally.")


def generate_ai_insights(kpis):
    """Generate smart business insights from KPI metrics."""
    insights = []
    try:
        total_profit = float(
            kpis.get("total_profit", "0").replace("$", "").replace(",", "")
        )
        avg_profit = float(
            kpis.get("avg_profit", "0").replace("$", "").replace(",", "")
        )
        profit_growth = float(
            kpis.get("profit_growth", "0").replace("%", "").replace(",", "")
        )

        if profit_growth > 15:
            insights.append(
                "🚀 Profit growth is impressive! Consider reinvesting profits into marketing or expansion."
            )
        elif profit_growth < 0:
            insights.append(
                "⚠️ Profit is declining. Review cost centers and adjust revenue strategies."
            )
        else:
            insights.append(
                "📊 Profit growth is steady. Maintain your current operational efficiency."
            )

        if avg_profit < total_profit * 0.05:
            insights.append(
                "💡 Low average profit per period — optimize product pricing or reduce overhead."
            )
        else:
            insights.append(
                "✅ Average profit margins look healthy. Keep optimizing your revenue streams."
            )

        insights.append(
            f"📅 Latest KPI snapshot — Growth: {profit_growth:.2f}%, Total Profit: ${total_profit:,.2f}"
        )
    except Exception as e:
        insights.append(f"Error generating insights: {str(e)}")

    return insights


uploaded_csvs = {}


@app.route("/")
def index():
    return redirect(url_for("login"))



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT username, password, role FROM users WHERE username = %s",
                (username,),
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user and check_password_hash(user["password"], password):
                session["username"] = username
                return redirect(url_for("dashboard"))

            flash("❌ Invalid username or password", "error")
            return redirect(url_for("login"))

        except Exception as e:
            flash(f"⚠️ Database error: {str(e)}", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        new_user = request.form["username"].strip()
        new_email = request.form["email"].strip()
        new_pass = request.form["password"].strip()

        business_name = request.form["business_name"].strip()
        paybill = request.form["paybill"].strip()
        account_number = request.form.get("account_number","").strip()

        try:
            conn = get_connection()
            cursor = conn.cursor()

            # ✅ Ensure users table exists (with email column)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL
                )
            """
            )

            # ✅ Check if username or email already exists
            cursor.execute(
                "SELECT * FROM users WHERE username = %s OR email = %s",
                (new_user, new_email),
            )
            existing_user = cursor.fetchone()
            if existing_user:
                flash(
                    "⚠️ Username or email already exists, please choose another.",
                    "error",
                )
                cursor.close()
                conn.close()
                return redirect(url_for("register"))

            # ✅ Validate password
            if len(new_pass) < 4:
                flash("⚠️ Password must be at least 4 characters long.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for("register"))

            # ✅ Validate paybill (5–7 digits only)
            if not re.fullmatch(r"\d{5,7}", paybill):
                flash("⚠️ Invalid paybill number. Must be 5–7 digits.", "error")
                cursor.close()
                conn.close()
                return redirect(url_for("register"))

            # ✅ Validate account number (optional, but digits only if provided)
            if account_number and not re.fullmatch(r"[A-Za-z0-9_-]{2,30}", account_number):
                flash("⚠️ Invalid account number.Use letters and numbers only (2-30 chars).",)
                cursor.close()
                conn.close()
                return redirect(url_for("register"))

            # ✅ Determine role
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()["count"]
            role = "admin" if user_count == 0 else "user"

            # ✅ Hash password and save new user
            hashed_password = generate_password_hash(new_pass)

            cursor.execute(
                """
                INSERT INTO users (username, email, password, role)
                VALUES (%s, %s, %s, %s)
                """,
                (new_user, new_email, hashed_password, role),
            )

            # ✅ Save business details
            cursor.execute(
                """
                INSERT INTO businesses (username, business_name, paybill, account_number)
                VALUES (%s, %s, %s, %s)
                """,
                (new_user, business_name, paybill, account_number or None),
            )

            conn.commit()
            cursor.close()
            conn.close() 

            # ✅ Create upload folder
            os.makedirs(
                os.path.join(app.config["UPLOAD_FOLDER"], new_user), exist_ok=True
            )

            # ✅ Save session info
            session["username"] = new_user
            session["email"] = new_email
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"❌ Error creating user: {str(e)}", "error")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@cache.cached(timeout=300, key_prefix=lambda:f"dashboard_data:{session.get('username')}")
def get_dashboard_data(username):
    data={}
    data["intelligence"] = get_dashboard_revenue_intelligence(username)
    """
    Read-heavy dashboard data only.
    NO Google Sheets sync.
    NO AI calls.
    NO Prophet.
    """
    data = {}

    # Example:
    if os.path.exists("financial_data.csv"):
        df = pd.read_csv("financial_data.csv")
        data["df"] = df
    else:
        data["df"] = pd.DataFrame()

    return data

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]


    # 🔹 Cached dashboard data
    cached_data = get_dashboard_data(username)
    notifications = cached_data.get("notifications", [])
    forecast_data = cached_data.get("forecast_data", [])
    last_synced = cached_data.get("last_synced")
    intelligence = cached_data.get("intelligence")

    answer = None

    # 🔹 User files (not cached)
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], username)
    os.makedirs(user_folder, exist_ok=True)
    files = sorted(os.listdir(user_folder))

    # ===============================
    # 📊 KPI LOGIC (DEFENSIVE)
    # ===============================
    kpis = {
        "total_profit": "KSh 0",
        "avg_profit": "KSh 0",
        "total_expenses":"KSh 0",
        "profit_growth": "0%",
        "largest_expense": "N/A",
    }

    conn = get_db_connection(cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # 🔹 Fetch user's business (paybill & account number)
    cur.execute("""
        SELECT paybill, account_number
        FROM businesses
        WHERE username = %s
        LIMIT 1
    """, (username,))
    biz = cur.fetchone()

    paybill = biz["paybill"] if biz else None
    account_number = biz["account_number"] if biz else None

    # 🔹 MPesa Revenue (USER-SCOPED)
    mpesa_total = 0

    if paybill:
        cur.execute("""
            SELECT COALESCE(SUM(amount), 0) AS mpesa_total
            FROM mpesa_transactions
            WHERE transaction_type = 'C2B Payment'
              AND (
                    (%s IS NOT NULL AND account_reference = %s)
                    OR
                    (%s IS NULL AND receiver = %s)
              )
        """, (
            account_number,
            account_number,
            account_number,
            paybill
        ))
        row = cur.fetchone()
        mpesa_total = float(row["mpesa_total"]) if row else 0

    # 🔹 Cash Revenue (NEW)
    cur.execute("""
        SELECT COALESCE(SUM(amount), 0) AS cash_total
        FROM cash_revenue
        WHERE username = %s
    """, (username,))
    row = cur.fetchone()
    cash_total=float(row["cash_total"]) if row and "cash_total" in row else 0

    live_total_revenue = mpesa_total + cash_total

    latest_payment = None

    if paybill:
        cur.execute("""
            SELECT id, amount, created_at
            FROM mpesa_transactions
            WHERE seen = FALSE
            AND (
                (%s IS NOT NULL AND account_reference = %s)
                OR
                (%s IS NULL AND receiver = %s)
            )
            ORDER BY created_at DESC
            LIMIT 1
        """, (
            account_number,
            account_number,
            account_number,
            paybill
        ))

        latest_payment = cur.fetchone()

        # Mark as seen immediately (so it doesn't repeat)
        if latest_payment:
            cur.execute(
                "UPDATE mpesa_transactions SET seen = TRUE WHERE id = %s",
                (latest_payment["id"],)
            )
            conn.commit()    

    # 🔹 Expenses — SAFE SANITIZATION
    cur.execute("""
        SELECT COALESCE(SUM(amount), 0) AS total_expenses
        FROM expenses
        WHERE username = %s
    """, (username,))
    total_expenses = float(cur.fetchone()["total_expenses"])
    # 🔹 Largest expense category — SAFE
    cur.execute("""
        SELECT
            COALESCE(category, 'Uncategorized') AS category,
            SUM(amount) AS total
        FROM expenses
        WHERE username = %s
        GROUP BY category
        ORDER BY total DESC
        LIMIT 1
    """, (username,))
    largest = cur.fetchone()
    cur.close()
    conn.close()

    profit = (mpesa_total + cash_total)-total_expenses
    
    
    kpis["total_profit"] = f"KSh {profit:,.0f}"
    kpis["avg_profit"] = f"KSh {profit:,.0f}"
    kpis["total_expenses"] = f"KSh{total_expenses:,.0f}"
    kpis["largest_expense"] = largest["category"] if largest else "N/A"

    # ===============================
    # 🤖 AI INSIGHTS
    # ===============================
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}],
                )
                answer = response.choices[0].message.content.strip()
                notifications.append(f"💡 Smart Insight: {answer}")
            except Exception as e:
                notifications.append(f"AI error: {str(e)}")

    # ===============================
    # 📈 Forecast chart prep
    # ===============================
    forecast_chart = []
    for item in forecast_data:
        try:
            forecast_chart.append({
                "date": item["date"],
                "predicted_revenue": float(item["predicted_revenue"])
            })
        except Exception:
            continue
    

    return render_template(
        "dashboard.html",
        files=files,
        notifications=notifications,
        answer=answer,
        kpis=kpis,
        forecast_data=forecast_data,
        forecast_chart=json.dumps(forecast_chart),
        last_synced=last_synced,
        current_year=datetime.now().year,
        latest_payment=latest_payment,
        intelligence=intelligence,
        live_total_revenue=live_total_revenue
    )

@app.route("/revenue/day/<date>/delete", methods=["POST"])
@login_required
def delete_revenue_day(date):
    username = session["username"]

    conn = get_db_connection()
    cur = conn.cursor()

    # 🔒 BLOCK deletion if day is locked
    cur.execute("""
        SELECT locked
        FROM revenue_days
        WHERE username = %s
          AND revenue_date = %s
    """, (username, date))
    row = cur.fetchone()

    if row and row[0]:
        cur.close()
        conn.close()
        flash("This revenue day is locked and cannot be deleted.", "error")
        return redirect(url_for("revenue_overview"))

    # ✅ Delete manual revenue entries only
    cur.execute("""
        DELETE FROM revenue_entries
        WHERE username = %s
          AND revenue_date = %s
    """, (username, date))

    # ✅ Delete the revenue day container
    cur.execute("""
        DELETE FROM revenue_days
        WHERE username = %s
          AND revenue_date = %s
    """, (username, date))

    conn.commit()
    cur.close()
    conn.close()

    # Refresh dashboard numbers
    cache.delete_memoized(get_dashboard_data, username)

    flash("Revenue day deleted.", "success")
    return redirect(url_for("revenue_overview"))


@app.route("/revenue/day/<date>/ai-summary", methods=["POST"])
@login_required
def generate_ai_summary_for_day_route(date):
    if not AI_ENABLED:
        flash("AI summary is unavailable...")
        return redirect(...)

    username = session["username"]

    data = generate_revenue_day_export_data(username, date)

    summary = generate_revenue_ai_summary(
        date,
        data["manual_total"],
        data["mpesa_total"],
        data["manual_entries"]
    )

    save_ai_summary_for_day(username, date, summary)

    # 🔒 LOCK DAY + SNAPSHOT TOTAL

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE revenue_days
        SET locked = TRUE,
            total_amount = %s
        WHERE username = %s
          AND revenue_date = %s
    """, (
        data["manual_total"] + data["mpesa_total"],
        username,
        date
    ))

    conn.commit()
    cur.close()
    conn.close()
    

    flash("AI summary generated.")
    return redirect(url_for("revenue_day_detail", date=date))


@app.route("/revenue/day/<date>/export/csv")
@login_required
def export_revenue_day_csv(date):
    username = session["username"]

    data = generate_revenue_day_export_data(username, date)

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Revenue Report"])
    writer.writerow(["Date", data["date"]])
    writer.writerow([])
    
    # Totals
    writer.writerow(["Totals"])
    writer.writerow(["MPesa (Gross)", data["mpesa_total"]])
    writer.writerow(["Manual Split", data["manual_total"]])
    writer.writerow(["Expenses", -data["expense_total"]])
    writer.writerow(["Net Revenue", data["net_total"]])
    writer.writerow([])

    # Manual entries
    writer.writerow(["Manual Entries"])
    writer.writerow(["Category", "Amount"])
    for e in data["manual_entries"]:
        writer.writerow([e["category"], e["amount"]])

    writer.writerow([])

    # Expenses
    writer.writerow(["Expenses"])
    writer.writerow(["Category", "Amount"])
    for e in data["expense_entries"]:
        writer.writerow([e["category"], -e["amount"]])

    writer.writerow([])

    # MPesa entries
    writer.writerow(["MPesa Transactions"])
    writer.writerow(["Sender", "Amount", "Transaction ID", "Time"])
    for m in data["mpesa_entries"]:
        writer.writerow([
            m["sender"],
            m["amount"],
            m["transaction_id"],
            m["created_at"]
        ])

    csv_data = output.getvalue()
    output.close()

    response = Response(
        csv_data,
        mimetype="text/csv"
    )
    response.headers["Content-Disposition"] = (
        f"attachment; filename=revenue_{date}.csv"
    )

    return response

@app.route("/revenue/day/<date>/export/pdf")
@login_required
def export_revenue_day_pdf(date):
    username = session["username"]

    manual_entries = load_revenue_entries_for_day(username, date)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # 1️⃣ Get business identifiers
    cursor.execute("""
        SELECT paybill, account_number
        FROM businesses
        WHERE username = %s
        LIMIT 1
    """, (username,))
    biz = cursor.fetchone()

    mpesa_entries = []

    if biz:
        paybill = biz["paybill"]
        account_number = biz["account_number"]

        cursor.execute("""
            SELECT amount, sender, created_at
            FROM mpesa_transactions
            WHERE
                (
                    (%s IS NOT NULL AND account_reference = %s)
                    OR
                    (%s IS NULL AND receiver = %s)
                )
              AND DATE(created_at) = %s
            ORDER BY created_at ASC
        """, (
            account_number,
            account_number,
            account_number,
            paybill,
            date
        ))

        mpesa_entries = cursor.fetchall()
    cursor.close()
    conn.close()

    manual_total = sum(float(e["amount"]) for e in manual_entries)
    mpesa_total = sum(float(e["amount"]) for e in mpesa_entries)
    grand_total = manual_total + mpesa_total

    # Expenses
    expenses = get_expenses_for_day(username, date)
    expense_total = float(expenses["total"])
    expense_entries = expenses["entries"]

    # Net revenue
    net_total = grand_total - expense_total

    html = render_template(
        "revenue_day_pdf.html",
        date=date,
        manual_entries=manual_entries,
        mpesa_entries=mpesa_entries,
        expense_entries=expense_entries,
        manual_total=manual_total,
        mpesa_total=mpesa_total,
        expense_total=expense_total,
        net_total=net_total
    )

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)

    response = make_response(pdf_buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=revenue_{date}.pdf"

    return response

@app.route("/revenue/day/<date>")
@login_required
def revenue_day_detail(date):
    username = session["username"]

    manual_entries = load_revenue_entries_for_day(username, date)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # 🔹 get business identifiers
    cursor.execute("""
        SELECT paybill, account_number
        FROM businesses
        WHERE username = %s
        LIMIT 1
    """, (username,))
    biz = cursor.fetchone()

    mpesa_entries = []

    if biz:
        paybill = biz["paybill"]
        account_number = biz["account_number"]

        cursor.execute("""
            SELECT amount, sender, transaction_id, created_at
            FROM mpesa_transactions
            WHERE (
                (%s IS NOT NULL AND account_reference = %s)
                OR
                (%s IS NULL AND receiver = %s)
            )
            AND created_at >= %s::date
            AND created_at < (%s::date + INTERVAL '1 day')
            ORDER BY created_at ASC
        """, (
            account_number,
            account_number,
            account_number,
            paybill,
            date,
            date
        ))
        mpesa_entries = cursor.fetchall()

    # 💵 CASH revenue for the day (NOT manual splits)
    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0) AS cash_total
        FROM cash_revenue
        WHERE username = %s
          AND revenue_date = %s
    """, (username, date))

    row = cursor.fetchone()
    cash_total = float(row["cash_total"]) if row else 0.0

    # 🔹 anomalies
    cursor.execute("""
        SELECT anomaly_type, severity, message
        FROM revenue_anomalies
        WHERE username = %s
          AND revenue_date = %s
    """, (username, date))
    anomalies = cursor.fetchall()

    # 🔒 CHECK LOCK STATUS (MUST BE BEFORE CLOSE)
    cursor.execute("""
        SELECT locked
        FROM revenue_days
        WHERE username = %s
          AND revenue_date = %s
        LIMIT 1
    """, (username, date))
    row = cursor.fetchone()
    is_locked = row["locked"] if row else False

    # ✅ NOW CLOSE
    cursor.close()
    conn.close()

    manual_total = sum(float(e["amount"]) for e in manual_entries)
    mpesa_total = sum(float(e["amount"]) for e in mpesa_entries)
    gross_total = mpesa_total + cash_total

    ai_summary = get_ai_summary_for_day(username, date)

    expenses = get_expenses_for_day(username, date)
    expense_total = expenses["total"]
    expense_entries = expenses["entries"]

    net_revenue = gross_total - expense_total

    return render_template(
        "revenue_day_detail.html",
        date=date,
        manual_entries=manual_entries,
        mpesa_entries=mpesa_entries,
        manual_total=manual_total,
        mpesa_total=mpesa_total,
        is_locked=is_locked,
        anomalies=anomalies,
        ai_summary=ai_summary,
        expense_total=expense_total,
        expense_entries=expense_entries,
        net_revenue=net_revenue,
        cash_total=cash_total,
        gross_total=gross_total
    )


@app.route("/revenue/overview")
@login_required
def revenue_overview():
    username = session["username"]

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            revenue_date,
            locked,
            total_amount AS total_revenue
        FROM revenue_days
        WHERE username = %s
        ORDER BY revenue_date DESC
    """, (username,))

    days = cur.fetchall()

    cur.close()
    conn.close()

    return render_template(
        "revenue_overview.html",
        days=days
    )

@app.route("/revenue/lock", methods=["POST"])
@login_required
def lock_revenue_day_route():
    username = session["username"]
    revenue_date = request.form["revenue_date"]

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1️⃣ cash revenue total
    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM cash_revenue
        WHERE username = %s
          AND revenue_date = %s
    """, (username, revenue_date))
    cash_total = float(cursor.fetchone()[0])

    # 2️⃣ MPesa total
    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM mpesa_transactions
        WHERE status = 'confirmed'
          AND DATE(created_at) = %s
    """, (revenue_date,))
    mpesa_total = float(cursor.fetchone()[0])

    gross_total = mpesa_total + cash_total

    # 3️⃣ Expenses total
    expenses = get_expenses_for_day(username, revenue_date)
    expense_total = float(expenses["total"])

    # 4️⃣ NET revenue (this is the truth)
    net_total = gross_total - expense_total

    # 5️⃣ Ensure revenue_day exists
    cursor.execute("""
        INSERT INTO revenue_days (username, revenue_date)
        VALUES (%s, %s)
        ON CONFLICT (username, revenue_date) DO NOTHING
    """, (username, revenue_date))

    # 6️⃣ Lock day + store NET revenue
    cursor.execute("""
        UPDATE revenue_days
        SET locked = TRUE,
            total_amount = %s
        WHERE username = %s
          AND revenue_date = %s
    """, (net_total, username, revenue_date))

    conn.commit()
    cursor.close()
    conn.close()

    # 7️⃣ Lock manual entries (secondary safety)
    lock_manual_entries_for_the_day(username, revenue_date)

    # 8️⃣ Detect anomalies AFTER final numbers are known
    detect_revenue_anomalies(username, revenue_date)

    flash("Revenue day locked successfully.")
    return redirect(url_for("revenue_overview"))

@app.route("/revenue/cash", methods=["GET", "POST"])
@login_required
def cash_revenue_entry():
    username = session["username"]

    if request.method == "POST":
        amount = float(request.form["amount"])
        date = request.form["date"]
        description = request.form.get("description")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cash_revenue (username, amount, description, revenue_date)
            VALUES (%s, %s, %s, %s)
        """, (username, amount, description, date))
        conn.commit()
        cur.close()
        conn.close()

        flash("✅ Cash revenue added", "success")
        return redirect(url_for("dashboard"))

    return render_template("cash_revenue_entry.html")

@app.route("/api/latest-payment")
def latest_payment():
    if "username" not in session:
        return jsonify({"payment": None})

    username = session["username"]

    conn = get_db_connection(cursor_factory=RealDictCursor)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, amount, created_at
        FROM mpesa_transactions
        WHERE seen = FALSE
        ORDER BY created_at DESC
        LIMIT 1
    """)
    payment = cur.fetchone()

    if payment:
        # Mark as seen immediately
        cur.execute(
            "UPDATE mpesa_transactions SET seen = TRUE WHERE id = %s",
            (payment["id"],)
        )
        conn.commit()

    cur.close()
    conn.close()

    return jsonify({"payment": payment})

@app.route("/api/financial_data")
def financial_data():
    import psycopg2, os
    from flask import jsonify, session

    if "username" not in session:
        return jsonify({})

    username = session["username"]

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        # Revenue per day
        cur.execute("""
            SELECT
                DATE(created_at) AS date,
                COALESCE(SUM(amount), 0) AS revenue
            FROM mpesa_transactions
            WHERE username = %s
            GROUP BY DATE(created_at)
        """, (username,))
        revenue_rows = cur.fetchall()

        # Expenses per day
        cur.execute("""
            SELECT
                DATE(expense_date) AS date,
                COALESCE(SUM(amount), 0) AS expenses
            FROM expenses
            WHERE username = %s
            GROUP BY DATE(expense_date)
        """, (username,))
        expense_rows = cur.fetchall()

        cur.close()
        conn.close()

        # Convert to dicts for easy merge
        revenue_map = {r[0]: float(r[1]) for r in revenue_rows}
        expense_map = {e[0]: float(e[1]) for e in expense_rows}

        # Merge all dates
        all_dates = sorted(set(revenue_map.keys()) | set(expense_map.keys()))

        dates = [d.strftime("%b %d") for d in all_dates]
        revenue = [revenue_map.get(d, 0) for d in all_dates]
        expenses = [expense_map.get(d, 0) for d in all_dates]
        profit = [r - e for r, e in zip(revenue, expenses)]

        return jsonify({
            "dates": dates,
            "revenue": revenue,
            "expenses": expenses,
            "profit": profit
        })

    except Exception as e:
        print("⚠️ Error generating financial data:", e)
        return jsonify({})


@app.route("/api/transactions_summary")
def transactions_summary():
    import psycopg2, os
    from flask import jsonify, session

    if "username" not in session:
        return jsonify({})

    username = session["username"]

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                COALESCE(SUM(amount), 0) AS total_revenue,
                COALESCE(AVG(amount), 0) AS avg_transaction,
                COUNT(*) AS txn_count
            FROM mpesa_transactions
            WHERE username = %s
        """, (username,))

        total_revenue, avg_transaction, txn_count = cur.fetchone()

        cur.close()
        conn.close()

        return jsonify({
            "total_revenue": round(float(total_revenue), 2),
            "avg_transaction": round(float(avg_transaction), 2),
            "txn_count": txn_count,
            "profit_growth": "N/A"  # real growth comes later
        })

    except Exception as e:
        print("⚠️ Error generating transaction summary:", e)
        return jsonify({})
    
# ✅ GET ACCESS TOKEN
@app.route("/get_token")
def get_token():
    import os, requests
    from flask import jsonify

    consumer_key = os.getenv("MPESA_CONSUMER_KEY")
    consumer_secret = os.getenv("MPESA_CONSUMER_SECRET")

    if not consumer_key or not consumer_secret:
        return jsonify({"error": "Missing M-Pesa credentials in environment variables."}), 400

    try:
        auth_url = "https://api.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials"
        response = requests.get(auth_url, auth=(consumer_key, consumer_secret), timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": f"Failed to get token: {str(e)}"}), 500

# ✅ REGISTER CALLBACK URL
@app.route("/register_url", methods=["POST"])
def register_url():
    import os, requests
    from flask import jsonify

    shortcode = os.getenv("MPESA_SHORTCODE")

    consumer_key = os.getenv("MPESA_CONSUMER_KEY")
    consumer_secret = os.getenv("MPESA_CONSUMER_SECRET")

    # PRODUCTION OAuth URL
    token_url = "https://api.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials"

    try:
        token_response = requests.get(
            token_url,
            auth=(consumer_key, consumer_secret),
            timeout=10
        )
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]
    except Exception as e:
        return jsonify({"error": "Token generation failed", "details": str(e)}), 400

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "ShortCode": shortcode,
        "ResponseType": "Completed",
        "ConfirmationURL": "https://profitoptimizer-production.up.railway.app/payment/confirm",
        "ValidationURL": "https://profitoptimizer-production.up.railway.app/payment/validate",
    }

    try:
        res = requests.post(
            "https://api.safaricom.co.ke/mpesa/c2b/v2/registerurl",
            headers=headers,
            json=payload,
            timeout=15
        )
        return jsonify(res.json()), res.status_code
    except Exception as e:
        return jsonify({"error": "Register URL failed", "details": str(e)}), 500

    

@app.route("/payment/validate", methods=["POST"])
def payment_validate():
    data = request.get_json(silent=True)
    print("📥 VALIDATION Callback:", data)

    return jsonify({
        "ResultCode": 0,
        "ResultDesc": "Accepted"
    })

@app.route("/payment/confirm", methods=["POST"])
def payment_confirm():
    import psycopg2, os, json
    from datetime import datetime
    from flask import request, jsonify

    data = request.get_json(silent=True)
    if not data:
        print("Empty or invalid JSON received")
        return jsonify({
            "ResultCode": 0,
            "ResultDesc": "No JSON body"
            })

    try:
        # ============================================================
        # 1️⃣ CASE A — C2B SIMULATOR (V1 CALLBACK)
        # ============================================================
        if "TransID" in data and "TransAmount" in data:
            print("✔ Detected: C2B Simulator")

            transaction_id = data.get("TransID")
            amount = float(data.get("TransAmount", 0))
            sender_name = data.get("FirstName", "Unknown")
            sender_phone = data.get("MSISDN", "")
            description = data.get("TransactionType", "C2B Paybill")
            account_ref = data.get("BillRefNumber", "")
            shortcode = data.get("BusinessShortCode")

        # ============================================================
        # 2️⃣ CASE B — DARAJA V2 CALLBACK (STK-STYLE)
        # ============================================================
        elif "Body" in data and "stkCallback" in data["Body"]:
            print("✔ Detected: V2 STK Callback")

            stk = data["Body"]["stkCallback"]
            transaction_id = stk.get("CheckoutRequestID", "")
            description = stk.get("ResultDesc", "")
            amount = 0.0
            sender_phone = ""
            sender_name = "Unknown"
            account_ref = "V2 Callback"

            items = stk.get("CallbackMetadata", {}).get("Item", [])

            for item in items:
                if item.get("Name") == "Amount":
                    amount = float(item.get("Value", 0))
                if item.get("Name") == "MpesaReceiptNumber":
                    transaction_id = item.get("Value", transaction_id)
                if item.get("Name") == "PhoneNumber":
                    sender_phone = str(item.get("Value", ""))
        else:
            print("❌ Unknown callback format")
            return jsonify({"ResultCode": 1, "ResultDesc": "Invalid callback format"})

        if not transaction_id or amount <=0:
            print("Invalid transaction data, skipping insert")
            return jsonify({"ResultCode": 0, "ResultDesc": "Ignored"})

        # ============================================================
        # SAVE TO DATABASE
        # ============================================================
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        # 🔍 Find business linked to this payment
        cur.execute("""
            SELECT id, username
            FROM businesses
            WHERE paybill = %s
            AND account_number = %s
            LIMIT 1
        """, (shortcode, account_ref))

        biz = cur.fetchone()

        if not biz:
            business_id = None
            username = None
        else:
            business_id = biz[0]
            username = biz[1]

        cur.execute("""
            INSERT INTO mpesa_transactions
            (
                transaction_id,
                amount,
                sender,
                receiver,
                transaction_type,
                account_reference,
                description,
                raw_payload,
                origin_ip,
                created_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW());
        """, (
            transaction_id,
            amount,
            sender_phone or sender_name,
            "OptiGain",
            "C2B Payment",
            account_ref,
            description,
            json.dumps(data),
            request.remote_addr
        ))
        conn.commit()

        payment_date = datetime.utcnow().date()
        ensure_revenue_day_exists(username,payment_date)
        
        cur.close()
        conn.close()
        if username:
            today = datetime.utcnow().date()
            ensure_revenue_day_exists(username, today)

            cache.delete_memoized(get_dashboard_data, username)
        

        print("✅ PAYMENT SAVED:", amount)

        return jsonify({"ResultCode": 0, "ResultDesc": "Success"})

    except Exception as e:
        print("❌ ERROR in confirm callback:", e)
        return jsonify({"ResultCode": 1, "ResultDesc": "Internal Error"})

    
# ✅ Query M-Pesa Account Balance (safe naming)
@app.route("/api/account_balance")
def account_balance():
    return jsonify({
        "status": "disabled",
        "message": "Account balance will be enabled after Safaricom initiator approval"
    })


# ✅ Payment Timeout Callback
@app.route("/payment/timeout", methods=["POST"])
def payment_timeout():
    data = request.get_json(silent=True)
    print("⏱️ Payment Timeout:", data or "No payload")
    return jsonify({"ResultCode": 1, "ResultDesc": "Request timed out"})

# ✅ Balance Result Callback
@app.route("/payment/balance_result", methods=["POST"])
def payment_balance_result():
    data = request.get_json(silent=True)
    print("💰 Account Balance Result:", data or "No payload")
    return jsonify({"ResultCode": 0, "ResultDesc": "Balance result received"})


# ✅ AI Insight Engine — analyzes latest financial data and generates insights
@app.route("/api/ai_insights")
def ai_insights():
    try:
        if "username" not in session:
            return jsonify([])

        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])

        if not os.path.exists(user_folder):
            return jsonify([])

        files = [f for f in os.listdir(user_folder) if f.endswith(".csv")]
        if not files:
            return jsonify([])

        latest_file = sorted(files)[-1]
        df = pd.read_csv(os.path.join(user_folder, latest_file))
        df.columns = df.columns.str.lower().str.strip()

        insights = []

        # Require revenue & expenses
        if "revenue" not in df.columns or "expenses" not in df.columns:
            return jsonify(["Uploaded file is missing revenue or expenses columns."])

        df["profit"] = df["revenue"] - df["expenses"]

        # ---- Recent trend (last 2 rows) ----
        if len(df) >= 2:
            if df["profit"].iloc[-1] > df["profit"].iloc[-2]:
                insights.append("Profit increased in the latest period ✅")
            elif df["profit"].iloc[-1] < df["profit"].iloc[-2]:
                insights.append("Profit decreased recently ⚠️ Check expenses or pricing.")
            else:
                insights.append("Profit remained stable recently.")

        # ---- Average profit margin (safe) ----
        valid = df[df["revenue"] > 0]
        if not valid.empty:
            avg_margin = (valid["profit"] / valid["revenue"]).mean() * 100
            insights.append(f"Average profit margin: {avg_margin:.1f}%")
        else:
            insights.append("Average profit margin unavailable (no revenue data).")

        # ---- Best period (optional month column) ----
        if "month" in df.columns:
            best_month = df.loc[df["profit"].idxmax(), "month"]
            insights.append(f"Highest profit recorded in {best_month}.")

        return jsonify(insights)

    except Exception as e:
        print("AI insight error:", e)
        return jsonify([])

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        files = request.files.getlist("files")

        # ✅ Validate uploads
        if not files or not all(allowed_file(f.filename) for f in files):
            return "Only CSV files are allowed.", 400

        if len(files) > 3:
            return "Please upload up to 3 files only.", 400

        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
        os.makedirs(user_folder, exist_ok=True)

        username = session.get("username")
        if username not in uploaded_csvs:
            uploaded_csvs[username] = []

        saved_paths = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(user_folder, filename)
            file.save(filepath)
            uploaded_csvs[username].append(filename)
            saved_paths.append(filepath)
            session["active_csv"] = filename

        cache.delete(f"dashboard_data_{session['username']}")    

        # ✅ If only one file uploaded → just show summary, no comparison
        if len(saved_paths) == 1:
            df = pd.read_csv(saved_paths[0])
            df.columns = df.columns.str.lower().str.strip()

            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
            df["expenses"] = pd.to_numeric(df["expenses"], errors="coerce")

            summary = {
                "total_revenue": df["revenue"].sum(),
                "total_expenses": df["expenses"].sum(),
                "profit": df["revenue"].sum() - df["expenses"].sum(),
            }

            return render_template(
                "results.html",
                revenue=summary["total_revenue"],
                expenses=summary["total_expenses"],
                profit=summary["profit"],
                margin=(
                    round(summary["profit"] / summary["total_revenue"] * 100, 2)
                    if summary["total_revenue"]
                    else 0
                ),
                advice="📊 Uploaded single file — showing summary only.",
            )

        # ✅ If two or more files → compare the latest two
        df1 = pd.read_csv(saved_paths[-2])
        df2 = pd.read_csv(saved_paths[-1])

        # Standardize and clean
        for df in (df1, df2):
            df.columns = df.columns.str.lower().str.strip()
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
            df["expenses"] = pd.to_numeric(df["expenses"], errors="coerce")

        comparison = pd.DataFrame(
            {
                "Month": df1["month"],
                "Revenue Difference": df2["revenue"] - df1["revenue"],
                "Expense Difference": df2["expenses"] - df1["expenses"],
            }
        )

        session["comparison_data"] = comparison.to_json()

        # ✅ Smart notifications
        notifications = []
        revenue_diff = comparison["Revenue Difference"].sum()
        expense_diff = comparison["Expense Difference"].sum()

        if revenue_diff > 0:
            notifications.append("📈 Your revenue increased overall!")
        elif revenue_diff < 0:
            notifications.append("📉 Revenue dropped — check details.")

        if expense_diff > 0:
            notifications.append("⚠️ Expenses went up — monitor costs.")
        elif expense_diff < 0:
            notifications.append("✅ Great job! Expenses decreased.")

        session["notifications"] = notifications

        table_html = comparison.to_html(index=False)

        return render_template(
            "comparison.html", table_html=table_html, notifications=notifications
        )

    return render_template("upload.html")


@app.route("/view/<filename>", methods=["GET", "POST"])
def view_file(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    from urllib.parse import quote

    # Search text (if user submits search)
    search_query = request.form.get("search") if request.method == "POST" else ""

    # Locate file in user's folder
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found.", 404

    # ✅ Read SINGLE CSV (faster than merging many)
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return "Failed to load data.", 400

    # Normalise columns
    df.columns = df.columns.str.lower().str.strip()

    # Ensure columns exist
    if "revenue" not in df.columns or "expenses" not in df.columns:
        return "CSV must contain 'revenue' and 'expenses' columns.", 400

    # Convert numeric safely
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["expenses"] = pd.to_numeric(df["expenses"], errors="coerce").fillna(0)

    # Optional: simple search filter
    if search_query:
        mask = df.astype(str).apply(lambda col: col.str.contains(search_query, case=False, na=False))
        df = df[mask.any(axis=1)]

    # ✅ Categorisation (same logic as before)
    def categorize(description):
        desc = str(description).lower()
        if any(word in desc for word in ["facebook", "ad", "campaign", "seo"]):
            return "Marketing"
        elif any(word in desc for word in ["sales", "client", "deal"]):
            return "Sales"
        elif any(word in desc for word in ["research", "development", "prototype"]):
            return "R&D"
        elif any(word in desc for word in ["office", "admin", "maintenance"]):
            return "Operations"
        else:
            return "Other"

    if "description" in df.columns:
        df["category"] = df["description"].apply(categorize)

    # ✅ Core metrics
    total_revenue = float(df["revenue"].sum())
    total_expenses = float(df["expenses"].sum())
    profit = total_revenue - total_expenses
    profit_margin = round((profit / total_revenue) * 100, 2) if total_revenue > 0 else 0.0
    avg_revenue = round(float(df["revenue"].mean()), 2)
    avg_expenses = round(float(df["expenses"].mean()), 2)

    # Try to use "month" column if present, otherwise fake index
    x_axis = df["month"] if "month" in df.columns else df.index.astype(str)

    # ✅ Bar chart: Revenue vs Expenses
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="Revenue", x=x_axis, y=df["revenue"]))
    bar_fig.add_trace(go.Bar(name="Expenses", x=x_axis, y=df["expenses"]))
    bar_fig.update_layout(
        barmode="group",
        title="Monthly Revenue vs Expenses",
        xaxis_title="Period",
        yaxis_title="Amount (KSh)",
    )
    # Only first chart includes plotly.js to keep payload smaller
    bar_chart = pyo.plot(bar_fig, output_type="div", include_plotlyjs=True)

    # ✅ Pie Chart: Profit vs Expenses
    pie_fig = go.Figure(
        data=[go.Pie(labels=["Profit", "Expenses"], values=[max(profit, 0), total_expenses])]
    )
    pie_fig.update_layout(title="Profit vs Expenses")
    pie_chart = pyo.plot(pie_fig, output_type="div", include_plotlyjs=False)

    # ✅ Line Chart: revenue & expenses over time
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=x_axis, y=df["revenue"], mode="lines+markers", name="Revenue"))
    line_fig.add_trace(go.Scatter(x=x_axis, y=df["expenses"], mode="lines+markers", name="Expenses"))
    line_fig.update_layout(
        title="Financial Trends",
        xaxis_title="Period",
        yaxis_title="Amount",
    )
    line_chart = pyo.plot(line_fig, output_type="div", include_plotlyjs=False)

    # ✅ Category breakdown (if available)
    if "category" in df.columns:
        category_totals = df.groupby("category")["expenses"].sum()
        cat_fig = go.Figure(
            data=[go.Pie(labels=category_totals.index, values=category_totals.values)]
        )
        cat_fig.update_layout(title="Expense Breakdown by Category")
        category_pie_chart = pyo.plot(cat_fig, output_type="div", include_plotlyjs=False)
    else:
        category_pie_chart = "<div class='alert alert-warning'>No 'Category' data available.</div>"

    # ✅ Insights
    insights = []

    if profit_margin < 10:
        insights.append("⚠️ Low profit margin — consider revising pricing or cutting costs.")
    elif profit_margin < 25:
        insights.append("🟡 Moderate margin — optimise operations and marketing ROI.")
    else:
        insights.append("🟢 Strong profit margin — keep scaling what works.")

    if "month" in df.columns:
        rev_trend = df["revenue"].diff().fillna(0)
        if (rev_trend > 0).all():
            insights.append("📈 Revenue has increased in every period in this dataset.")
        elif (rev_trend < 0).all():
            insights.append("📉 Revenue has decreased in every period — investigate causes.")
        else:
            insights.append("↕️ Revenue is fluctuating — monitor key months and drivers.")

    if total_expenses > 0 and total_revenue > 0:
        cost_ratio = total_expenses / total_revenue * 100
        insights.append(f"💸 Total expenses are {cost_ratio:.1f}% of revenue.")

    final_advice = "<br>".join(insights)
    escaped_advice = quote(final_advice)

    # ✅ Safer / lighter table HTML (limit rows if very large)
    df_for_table = df.copy()
    max_rows = 300
    if len(df_for_table) > max_rows:
        df_for_table = df_for_table.head(max_rows)

    table_html = df_for_table.to_html(
        classes="table table-hover table-dark table-bordered",
        index=False,
        border=0,
        justify="center",
    ).replace(
        'style="',
        'style="color:white !important; background-color:#121212 !important; '
    )

    return render_template(
        "results.html",
        revenue=total_revenue,
        expenses=total_expenses,
        profit=profit,
        margin=profit_margin,
        avg_revenue=avg_revenue,
        avg_expenses=avg_expenses,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        line_chart=line_chart,
        category_pie_chart=category_pie_chart,  # 🔴 matches your template variable
        advice=final_advice,
        trend_insights=insights,
        escaped_advice=escaped_advice,
        filename=filename,
        table_html=table_html,
        search_query=search_query,
    )
@app.route("/download_report", methods=["POST"])
def download_report():
    data = request.form

    html = render_template(
        "report.html",
        username=session.get("user", "User"),
        metrics={
            "revenue": data.get("revenue", "N/A"),
            "expenses": data.get("expenses", "N/A"),
            "profit": data.get("profit", "N/A"),
            "margin": data.get("margin", "N/A"),
        },
        advice=data.get("advice", "No advice provided."),
    )

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.BytesIO(html.encode("utf-8")), dest=pdf_buffer)
    pdf_buffer.seek(0)

    filename = f"Financial_Report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    return send_file(pdf_buffer, download_name=filename, as_attachment=True)


@app.route("/preview/<filename>")
def preview_file(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    try:
        df = pd.read_csv(filepath)
        table_html = df.to_html(
            classes="table table-bordered table-striped", index=False
        )
    except Exception as e:
        return f"Error reading file: {e}", 500

    return render_template("preview.html", table_html=table_html, filename=filename)


@app.route("/download_excel/<filename>")
def download_excel(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    df = pd.read_csv(filepath)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Financial Data")
    output.seek(0)

    return send_file(
        output,
        download_name="financial_data.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/download_csv/<filename>")
def download_csv(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    return send_file(filepath, as_attachment=True)


@app.route("/download_cleaned/<filename>")
def download_cleaned(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    try:
        df = pd.read_csv(filepath)

        # Clean the data
        df_cleaned = df.dropna()
        for col in ["Revenue", "Expenses"]:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

        df_cleaned = df_cleaned.dropna()

        # Save to buffer
        cleaned_path = os.path.join(user_folder, f"cleaned_{filename}")
        df_cleaned.to_csv(cleaned_path, index=False)

        return send_file(cleaned_path, as_attachment=True)

    except Exception as e:
        return str(e), 500


@app.route("/send_summary", methods=["POST"])
def send_summary():
    if "username" not in session:
        return redirect(url_for("login"))

    recipient_email = request.form.get("email")

    # get values from form
    revenue = request.form.get("revenue", "0")
    expenses = request.form.get("expenses", "0")
    profit = request.form.get("profit", "0")
    margin = request.form.get("margin", "0")
    advice = request.form.get("advice", "No advice available.")

    # Render HTML for PDF
    html = render_template(
        "summary_report.html",
        revenue=revenue,
        expenses=expenses,
        profit=profit,
        margin=margin,
        advice=advice
    )

    # Convert to PDF
    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer)
    pdf_buffer.seek(0)

    # Build Email
    msg = Message(
        subject="📊 OptiGain Summary Report",
        recipients=[recipient_email],
        sender=("OptiGain Reports", os.getenv("MAIL_USERNAME")),
        body="Your financial summary report is attached.",
    )

    # HTML body (Improves deliverability)
    msg.html = f"""
    <p>Hello,</p>
    <p>Your financial summary report from <strong>OptiGain</strong> is ready.</p>

    <h4>📌 Quick Overview</h4>
    <ul>
        <li><strong>Revenue:</strong> {revenue}</li>
        <li><strong>Expenses:</strong> {expenses}</li>
        <li><strong>Profit:</strong> {profit}</li>
        <li><strong>Margin:</strong> {margin}%</li>
    </ul>

    <p><em>Your full detailed summary is attached as a PDF.</em></p>
    <p>Best regards,<br>OptiGain Team</p>
    """

    # Attach PDF
    msg.attach("OptiGain_Summary.pdf", "application/pdf", pdf_buffer.read())

    # Send email
    try:
        mail.send(msg)
        flash("✅ Summary report sent successfully!", "success")
    except Exception as e:
        print("Email Error:", e)
        flash("❌ Failed to send summary.", "danger")

    return redirect(url_for("dashboard"))
@app.route("/download_advice", methods=["POST"])
def download_advice():
    advice_text = request.form.get("advice", "")
    advice_file = io.BytesIO()
    advice_file.write(advice_text.encode("utf-8"))
    advice_file.seek(0)
    return send_file(advice_file, download_name="ai_advice.txt", as_attachment=True)


@app.route("/download_raw_pdf/<filename>", methods=["GET"])
def download_raw_pdf(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    df = pd.read_csv(filepath)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Raw Financial Data", ln=True, align="C")
    pdf.ln(10)

    # Add table headers
    col_width = pdf.w / (len(df.columns) + 1)
    for col in df.columns:
        pdf.cell(col_width, 10, txt=str(col), border=1)
    pdf.ln()

    # Add table rows
    for i in range(len(df)):
        for col in df.columns:
            value = str(df.iloc[i][col])
            pdf.cell(col_width, 10, txt=value, border=1)
        pdf.ln()

    # Output to BytesIO
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_stream = io.BytesIO(pdf_bytes)

    # Send file to browser as download
    return send_file(
        pdf_stream,
        download_name="raw_financial_data.pdf",
        as_attachment=True,
        mimetype="application/pdf",
    )


@app.route("/download_summary_txt", methods=["POST"])
def download_summary_txt():
    data = request.form

    summary_text = (
        f"Financial Summary\n\n"
        f"Total Revenue: ${data['revenue']}\n"
        f"Total Expenses: ${data['expenses']}\n"
        f"Profit: ${data['profit']}\n"
        f"Profit Margin: {data['margin']}%\n\n"
        f"AI Advice:\n{data['advice']}"
    )

    txt_bytes = summary_text.encode("utf-8")
    txt_stream = io.BytesIO(txt_bytes)
    txt_stream.seek(0)

    return send_file(
        txt_stream,
        as_attachment=True,
        download_name="financial_summary.txt",
        mimetype="text/plain",
    )


@app.route("/compare/<filenames>")
def compare_files(filenames):
    if "username" not in session:
        return redirect(url_for("login"))

    files = filenames.split(",")
    summaries = []
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])

    for file in files:
        path = os.path.join(user_folder, file)
        try:
            df = pd.read_csv(path)
            total_revenue = df["Revenue"].sum()
            total_expenses = df["Expenses"].sum()
            profit = total_revenue - total_expenses
            margin = (
                round((profit / total_revenue) * 100, 2) if total_revenue > 0 else 0
            )

            summaries.append(
                {
                    "filename": file,
                    "revenue": total_revenue,
                    "expenses": total_expenses,
                    "profit": profit,
                    "margin": margin,
                }
            )
        except:
            summaries.append({"filename": file, "error": "Could not process file"})

    return render_template("compare.html", summaries=summaries)


@app.route("/download_comparison_csv")
def download_comparison_csv():
    if "comparison_data" not in session:
        return "No comparison data to download", 400

    df = pd.read_json(session["comparison_data"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="comparison.csv",
    )


@app.route("/download_comparison_pdf")
def download_comparison_pdf():
    if "comparison_data" not in session:
        return "No comparison data to download", 400

    df = pd.read_json(session["comparison_data"])

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Comparison Report", ln=True, align="C")
    pdf.ln(10)

    # Table header
    for col in df.columns:
        pdf.cell(60, 10, txt=col, border=1)
    pdf.ln()

    # Table rows
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(60, 10, txt=str(item), border=1)
        pdf.ln()

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(pdf_output, download_name="comparison.pdf", as_attachment=True)


@app.route("/download_bar_chart")
def download_bar_chart():
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Dummy data (you can replace this with real data)
    months = ["Jan", "Feb", "Mar", "Apr"]
    revenues = [10000, 15000, 20000, 18000]
    expenses = [7000, 9000, 11000, 10500]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(months))

    plt.bar(index, revenues, bar_width, label="Revenue", color="green")
    plt.bar(
        [i + bar_width for i in index],
        expenses,
        bar_width,
        label="Expenses",
        color="red",
    )
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.title("Revenue vs Expenses")
    plt.xticks([i + bar_width / 2 for i in index], months)
    plt.legend()

    from io import BytesIO
    import os
    from flask import send_file

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return send_file(
        buffer, mimetype="image/png", as_attachment=True, download_name="bar_chart.png"
    )


@app.route("/download_line_chart")
def download_line_chart():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    from flask import send_file

    # Dummy data (replace with your actual trend data)
    months = ["Jan", "Feb", "Mar", "Apr"]
    revenues = [10000, 15000, 20000, 18000]

    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(months, revenues, marker="o", linestyle="-", color="blue", label="Revenue")
    plt.title("Revenue Trend Over Time")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.legend()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return send_file(
        buffer, mimetype="image/png", as_attachment=True, download_name="line_chart.png"
    )


@app.route("/download_pie_chart")
def download_pie_chart():
    import matplotlib.pyplot as plt
    from io import BytesIO
    from flask import send_file

    # Dummy data (replace with your actual category data)
    labels = ["Marketing", "Sales", "Operations", "R&D"]
    expenses = [4000, 3000, 2500, 1500]

    plt.figure(figsize=(6, 6))
    plt.pie(expenses, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Expense Distribution by Category")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return send_file(
        buffer, mimetype="image/png", as_attachment=True, download_name="pie_chart.png"
    )


@app.route("/download_full_report", methods=["GET"])
def download_full_report():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session.get("username", "User")

    # 📊 Step 1: Locate the latest uploaded CSV
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], username)
    if not os.path.exists(user_folder):
        return "No uploaded files found.", 404

    files = [f for f in os.listdir(user_folder) if f.endswith(".csv")]
    if not files:
        return "No CSV files found.", 404

    # Use the most recent file
    latest_file = max([os.path.join(user_folder, f) for f in files], key=os.path.getctime)
    df = pd.read_csv(latest_file)

    # 📈 Step 2: Basic calculations
    revenue = df["Revenue"].sum() if "Revenue" in df.columns else 0
    expenses = df["Expenses"].sum() if "Expenses" in df.columns else 0
    profit = revenue - expenses
    margin = round((profit / revenue * 100), 2) if revenue else 0
    avg_revenue = df["Revenue"].mean() if "Revenue" in df.columns else 0
    avg_expenses = df["Expenses"].mean() if "Expenses" in df.columns else 0

    metrics = {
        "revenue": f"{revenue:,.2f}",
        "expenses": f"{expenses:,.2f}",
        "profit": f"{profit:,.2f}",
        "margin": margin,
        "avg_revenue": f"{avg_revenue:,.2f}",
        "avg_expenses": f"{avg_expenses:,.2f}"
    }

    # 🧠 Step 3: Example AI advice (or load your saved one)
    advice = (
        f"Your current profit margin is {margin}%. "
        f"Consider reducing operational costs to improve profitability further."
    )

    # 🖼️ Step 4: Generate charts safely (Render-compatible)
    chart_html = ""
    if "Month" in df.columns and "Revenue" in df.columns and "Expenses" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Month"], y=df["Revenue"], name="Revenue", marker_color="green"))
        fig.add_trace(go.Bar(x=df["Month"], y=df["Expenses"], name="Expenses", marker_color="red"))
        fig.update_layout(
            title="Monthly Revenue vs Expenses",
            barmode="group",
            xaxis_title="Month",
            yaxis_title="Amount (KSh)",
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="#ffffff"
        )

        try:
            # Try saving as image (only works locally)
            chart_path = f"static/reports/revenue_expenses_{username}.png"
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            pio.write_image(fig, chart_path)
            chart_html = f'<img src="/{chart_path}" style="width:100%; border-radius:10px; margin-top:10px;">'
        except Exception as e:
            print(f"[Render-safe mode] Kaleido not available, using HTML chart instead: {e}")
            chart_html = pyo.plot(fig, include_plotlyjs="cdn", output_type="div")

    # 🧾 Step 5: Render the HTML report
    html = render_template(
        "report.html",
        username=username,
        metrics=metrics,
        advice=advice,
        chart_html=chart_html,  # ✅ use this instead of charts list
        current_date=datetime.now().strftime("%B %d, %Y"),
        current_year=datetime.now().year
    )

    # 🪄 Step 6: Convert to PDF using pdfkit or fallback to xhtml2pdf
    if pdfkit_available:
        pdf = pdfkit.from_string(html, False, options={
            "page-size": "A4",
            "encoding": "UTF-8",
            "margin-top": "10mm",
            "margin-bottom": "10mm",
            "margin-left": "10mm",
            "margin-right": "10mm"
        })
    else:
        # Render-safe fallback (Render doesn’t support wkhtmltopdf)
        pdf_buffer = io.BytesIO()
        pisa.CreatePDF(io.BytesIO(html.encode("utf-8")), dest=pdf_buffer)
        pdf = pdf_buffer.getvalue()

    # 📤 Step 7: Return as downloadable PDF
    filename = f"OptiGain_Full_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response
@app.route("/send_report", methods=["POST"])
def send_report():
    if "username" not in session:
        return redirect(url_for("login"))

    # receiver email
    recipient_email = request.form.get("email")
    if not recipient_email:
        flash("❌ No email provided.", "danger")
        return redirect(url_for("dashboard"))

    # financial data from form
    revenue = request.form.get("revenue", "0")
    expenses = request.form.get("expenses", "0")
    profit = request.form.get("profit", "0")
    margin = request.form.get("margin", "0")
    advice = request.form.get("advice", "No advice available.")

    # Build ABSOLUTE logo path
    logo_path = os.path.join(app.root_path, "static", "logo.png").replace("\\", "/")

    # Render HTML template → PDF
    html = render_template(
        "email_report.html",
        logo_path=logo_path,
        revenue=revenue,
        expenses=expenses,
        profit=profit,
        margin=margin,
        advice=advice
    )

    # Generate PDF
    pdf_buffer = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer)
    pdf_buffer.seek(0)

    if result.err:
        print("PDF ERROR:", result.err)
        flash("❌ Failed to generate PDF report.", "danger")
        return redirect(url_for("dashboard"))

    # Build email
    msg = Message(
        subject="📊 OptiGain Financial Report",
        recipients=[recipient_email],
        sender=("OptiGain Reports", os.getenv("MAIL_USERNAME")),
        body="Your full OptiGain report is attached."
    )

    # Add HTML body (professional + improves inbox deliverability)
    msg.html = f"""
    <p>Hello,</p>

    <p>Your detailed <strong>OptiGain Financial Report</strong> is attached as a PDF.</p>

    <h4>📌 Quick Snapshot</h4>
    <ul>
        <li><strong>Total Revenue:</strong> {revenue}</li>
        <li><strong>Total Expenses:</strong> {expenses}</li>
        <li><strong>Total Profit:</strong> {profit}</li>
        <li><strong>Profit Margin:</strong> {margin}%</li>
    </ul>

    <h4>🧠 AI Insight</h4>
    <p>{advice}</p>

    <p>Best regards,<br><strong>OptiGain Team</strong></p>
    """

    # Attach PDF
    msg.attach("OptiGain_Full_Report.pdf", "application/pdf", pdf_buffer.read())

    try:
        mail.send(msg)
        flash(f"📧 Report sent to {recipient_email}", "success")
    except Exception as e:
        print("MAIL ERROR:", e)
        flash("❌ Failed to send report.", "danger")

    return redirect(url_for("dashboard"))
@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], username)

    if os.path.exists(user_folder):
        uploaded_files = sorted(os.listdir(user_folder))
    else:
        uploaded_files = []

    return render_template("profile.html", uploaded_files=uploaded_files)


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], username)
    file_path = os.path.join(user_folder, filename)

    # Remove from disk
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from tracking list
    if username in uploaded_csvs:
        if filename in uploaded_csvs[username]:
            uploaded_csvs[username].remove(filename)

    flash(f"{filename} deleted successfully.")
    return redirect(url_for("profile"))


@app.route("/rename_file", methods=["POST"])
def rename_file():
    if "username" not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for("login"))

    old_filename = request.form["old_filename"]
    new_filename = request.form["new_filename"]

    username = session.get("username")
    user_folder = os.path.join("uploads", username)

    old_path = os.path.join(user_folder, old_filename)
    new_path = os.path.join(user_folder, new_filename)

    # Rename the file if it exists
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

        # Optional: update the uploaded_csvs dictionary if used
        if username in uploaded_csvs:
            uploaded_csvs[username] = [
                new_filename if f == old_filename else f
                for f in uploaded_csvs[username]
            ]

        flash(f"Renamed {old_filename} to {new_filename}", "success")
    else:
        flash("File not found.", "danger")

    return redirect(url_for("profile"))


@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    if "username" not in session:
        return redirect(url_for("login"))

    answer = None

    if request.method == "POST":
        question = request.form.get("question")
        username = session["username"]
        files = uploaded_csvs.get(username, [])
        if not files:
            answer = "No uploaded CSV files found."
        else:
            # We'll use the latest uploaded file for simplicity
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], username, files[-1])
            df = pd.read_csv(file_path)

            # Simple rule-based responses
            if "highest revenue" in question.lower():
                row = df.loc[df["Revenue"].idxmax()]
                answer = (
                    f"The highest revenue was in {row['Month']} with ${row['Revenue']}"
                )
            elif "average expense" in question.lower():
                avg_exp = df["Expenses"].mean()
                answer = f"The average expense is ${avg_exp:.2f}"
            elif "lowest revenue" in question.lower():
                row = df.loc[df["Revenue"].idxmin()]
                answer = (
                    f"The lowest revenue was in {row['Month']} with ${row['Revenue']}"
                )
            else:
                answer = "Sorry, I didn't understand the question."

    return render_template("advisor.html", answer=answer)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        # When someone just visits the Ask page
        return render_template("ask.html")

    # For POST (form submission or API)
    question = request.form.get("question") or (
        request.json.get("question") if request.is_json else None
    )

    if not question:
        if request.is_json:
            return jsonify({"error": "No question provided"}), 400
        return render_template("ask.html", answer="Please enter a question.")

    if not AI_ENABLED:
        msg = "AI functionality is temporarily disabled."
        return jsonify({"error": msg}), (
            503 if request.is_json else render_template("ask.html", answer=msg)
        )

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = "Missing OpenAI API key."
            return jsonify({"error": msg}), (
                500 if request.is_json else render_template("ask.html", answer=msg)
            )

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart business assistant that gives concise, actionable insights.",
                },
                {"role": "user", "content": question},
            ],
            temperature=0.7,
        )

        answer = (
            response.choices[0].message.content.strip()
            if response.choices
            else "No response received."
        )

        if request.is_json:
            return jsonify({"answer": answer})
        return render_template("ask.html", answer=answer)

    except Exception as e:
        msg = f"Error generating answer: {str(e)}"
        if request.is_json:
            return jsonify({"error": msg}), 500
        return render_template("ask.html", answer=msg)


@app.route("/admin")
def admin():
    # ✅ Must be logged in
    if "username" not in session:
        flash("You must be logged in to access the admin panel.", "error")
        return redirect(url_for("login"))

    current_user = session["username"]

    # ✅ Restrict admin panel to specific usernames
    allowed_admins = ["griffin", "diana", "rose"]
    if current_user not in allowed_admins:
        flash("Unauthorized: You don’t have admin access.", "error")
        return redirect(url_for("dashboard"))

    # ✅ Load users from the PostgreSQL database
    from database import load_users

    users = load_users()

    total_users = len(users)

    # ✅ Render admin page
    return render_template(
        "admin.html", user=current_user, users=users, total_users=total_users
    )


@app.route("/use-demo-data")
def use_demo_data():
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    os.makedirs(user_folder, exist_ok=True)

    demo_file = os.path.join(user_folder, "demo_data.csv")

    # Copy sample file for this user
    sample_path = os.path.join("sample_data.csv")
    if os.path.exists(sample_path):
        import shutil

        shutil.copy(sample_path, demo_file)
        session["uploaded_file"] = "demo_data.csv"
        flash("Demo data loaded successfully!", "success")
    else:
        flash("Sample data file missing. Please contact admin.", "error")

    return redirect(url_for("dashboard"))


@app.route("/profit_calculator", methods=["GET", "POST"])
def profit_calculator():
    profit = None
    if request.method == "POST":
        try:
            revenue = float(request.form["revenue"])
            expenses = float(request.form["expenses"])
            profit = revenue - expenses
        except ValueError:
            flash("⚠️ Please enter valid numbers.", "error")

    return render_template("profit_calculator.html", profit=profit)


@app.route("/trend_forecaster", methods=["GET"])
def trend_forecaster():
    if "username" not in session:
        return redirect(url_for("login"))

    forecast_plot = None

    try:
        conn = get_db_connection(cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        # 1️⃣ Get revenue from MPesa (grouped monthly)
        cursor.execute("""
            SELECT
                DATE_TRUNC('month', created_at) AS month,
                SUM(amount) AS revenue
            FROM mpesa_transactions
            WHERE result_code = 0
            GROUP BY month
            ORDER BY month
        """)
        revenue_rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not revenue_rows:
            flash("Not enough revenue data to generate forecast.", "warning")
            return render_template("trend_forecaster.html", forecast_plot=None)

        revenue_df = pd.DataFrame(revenue_rows)
        revenue_df["month"] = pd.to_datetime(revenue_df["month"])
        revenue_df["revenue"] = pd.to_numeric(revenue_df["revenue"], errors="coerce")

        # 2️⃣ Load manual expenses (CSV for now)
        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
        manual_path = os.path.join(user_folder, "manual_entries.csv")

        if os.path.exists(manual_path):
            expenses_df = pd.read_csv(manual_path)
            expenses_df["Month"] = pd.to_datetime(expenses_df["Month"])
            expenses_df["Expenses"] = pd.to_numeric(expenses_df["Expenses"], errors="coerce")

            expenses_df = (
                expenses_df
                .groupby(pd.Grouper(key="Month", freq="M"))["Expenses"]
                .sum()
                .reset_index()
                .rename(columns={"Month": "month"})
            )
        else:
            expenses_df = pd.DataFrame(columns=["month", "Expenses"])

        # 3️⃣ Merge revenue and expenses
        merged = pd.merge(
            revenue_df,
            expenses_df,
            on="month",
            how="left"
        )

        merged["Expenses"] = merged["Expenses"].fillna(0)
        merged["net_revenue"] = merged["revenue"] - merged["Expenses"]

        # 4️⃣ Prepare Prophet data
        prophet_df = merged.rename(
            columns={"month": "ds", "net_revenue": "y"}
        )[["ds", "y"]]

        if len(prophet_df) < 2:
            flash("Not enough historical data for forecasting.", "warning")
            return render_template("trend_forecaster.html", forecast_plot=None)

        # 5️⃣ Train Prophet
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=6, freq="M")
        forecast = model.predict(future)

        # 6️⃣ Save plot
        fig = model.plot(forecast)
        img_path = os.path.join("static", "forecast.png")
        fig.savefig(img_path)
        forecast_plot = img_path

    except Exception as e:
        print("Trend forecast error:", e)
        flash("Error generating forecast.", "danger")

    return render_template("trend_forecaster.html", forecast_plot=forecast_plot)
# 🗑 Delete a user
@app.route("/delete_user/<username>", methods=["POST"])
def delete_user(username):
    if "username" not in session:
        flash("Login required.", "error")
        return redirect(url_for("login"))

    current_user = session["username"]
    if current_user not in ["griffin", "diana", "rose"]:
        flash("Unauthorized access.", "error")
        return redirect(url_for("dashboard"))

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    cursor.close()
    conn.close()

    flash(f"User '{username}' deleted successfully.", "success")
    return redirect(url_for("admin"))


# ⭐ Promote a user (make them admin)
@app.route("/promote_user/<username>", methods=["POST"])
def promote_user(username):
    if "username" not in session:
        flash("Login required.", "error")
        return redirect(url_for("login"))

    current_user = session["username"]
    if current_user not in ["griffin", "diana", "rose"]:
        flash("Unauthorized access.", "error")
        return redirect(url_for("dashboard"))

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = 'admin' WHERE username = %s", (username,))
    conn.commit()
    cursor.close()
    conn.close()

    flash(f"User '{username}' promoted to admin.", "success")
    return redirect(url_for("admin"))


# 🔁 Reset user password (default: '1234')
@app.route("/reset_password/<username>", methods=["POST"])
def reset_password(username):
    if "username" not in session:
        flash("Login required.", "error")
        return redirect(url_for("login"))

    current_user = session["username"]
    if current_user not in ["griffin", "diana", "rose"]:
        flash("Unauthorized access.", "error")
        return redirect(url_for("dashboard"))

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET password = %s WHERE username = %s", ("1234", username)
    )
    conn.commit()
    cursor.close()
    conn.close()

    flash(f"Password for '{username}' reset to '1234'.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/toggle_ai", methods=["POST"])
def toggle_ai():
    # Only admin can access
    if "username" not in session or session["username"] != "griffin":
        return "Unauthorized", 403

    # Flip the current value
    global AI_ENABLED
    AI_ENABLED = not AI_ENABLED

    status = "enabled" if AI_ENABLED else "disabled"
    return f"AI insights are now {status}."


@app.route("/admin/send_reports", methods=["POST"])
def send_reports():
    if "username" not in session or session["username"] != "griffin":
        flash("Access denied.", "danger")
        return redirect(url_for("login"))

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT username, email FROM users")
    users = cur.fetchall()
    cur.close()
    conn.close()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sent_count = 0
    for user in users:
        username = user["username"]
        email = user["email"]

        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], username)
        if not os.path.exists(user_folder):
            continue

        files = [f for f in os.listdir(user_folder) if f.endswith(".csv")]
        if not files:
            continue

        latest_file = sorted(files)[-1]
        file_path = os.path.join(user_folder, latest_file)

        try:
            df = pd.read_csv(file_path)
            preview = df.head().to_string()

            prompt = f"""
            Generate a professional business summary for user {username} based on the following financial data:
            {preview}

            Include:
            - 3 key insights
            - 1 improvement recommendation
            - A short motivational line
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            ai_text = response.choices[0].message.content.strip()

            # Generate PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(
                200, 10, txt=f"Weekly Business Report - {username}", ln=True, align="C"
            )
            pdf.ln(10)
            pdf.multi_cell(0, 10, txt=ai_text)
            pdf_bytes = pdf.output(dest="S").encode("latin1")

            # Send via email
            msg = Message(subject="📊 Your Weekly OptiGain Report", recipients=[email])
            msg.body = (
                "Attached is your latest AI-generated business performance report."
            )
            msg.attach(f"{username}_report.pdf", "application/pdf", pdf_bytes)

            mail.send(msg)
            sent_count += 1

        except Exception as e:
            print(f"Error sending report to {username}: {e}")

    flash(f"✅ Reports sent successfully to {sent_count} users.", "success")
    return redirect(url_for("admin"))


@app.route("/edit_entry/<int:index>", methods=["GET", "POST"])
def edit_entry(index):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    manual_path = os.path.join(user_folder, "manual_entries.csv")

    if not os.path.exists(manual_path):
        flash("No data found to edit.", "error")
        return redirect(url_for("manual_entry"))

    df = pd.read_csv(manual_path)

    if index >= len(df):
        flash("Invalid entry selected.", "error")
        return redirect(url_for("manual_entry"))

    if request.method == "POST":
        df.at[index, "Month"] = request.form.get("month")
        df.at[index, "Revenue"] = float(request.form.get("revenue"))
        df.at[index, "Expenses"] = float(request.form.get("expenses"))
        df.at[index, "Description"] = request.form.get("description")
        df.to_csv(manual_path, index=False)

        flash("✅ Entry updated successfully!", "success")
        return redirect(url_for("manual_entry"))

    entry = df.iloc[index].to_dict()
    return render_template("edit_entry.html", entry=entry, index=index)

@app.route("/delete_entry/<int:index>")
def delete_entry(index):
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    manual_path = os.path.join(user_folder, "manual_entries.csv")

    if not os.path.exists(manual_path):
        flash("No data found.", "error")
        return redirect(url_for("manual_entry"))

    df = pd.read_csv(manual_path)

    if index < len(df):
        df = df.drop(index).reset_index(drop=True)
        df.to_csv(manual_path, index=False)
        flash("🗑️ Entry deleted successfully!", "success")
    else:
        flash("Invalid entry selected.", "error")

    return redirect(url_for("manual_entry"))


@app.route("/expense_entry", methods=["GET", "POST"])
def expense_entry():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    if request.method == "POST":
        try:
            save_expense(
                username=username,
                amount=float(request.form["amount"]),
                category=request.form.get("category"),
                description=request.form.get("description"),
                expense_date=request.form["date"]
            )
            flash("✅ Expense added successfully!", "success")
            return redirect(url_for("expense_entry"))
        except Exception as e:
            print(e)
            flash("❌ Failed to save expense.", "error")

    # 🔹 NEW: load user-defined categories
    categories = load_expense_categories(username)

    return render_template(
        "expense_entry.html",
        categories=categories
    )

@app.route("/delete_expense")
def delete_expense():
    if "username" not in session:
        return redirect(url_for("login"))

    expenses = load_expenses(session["username"])
    return render_template("delete_expense.html", entries=expenses)

@app.route("/revenue_entry", methods=["GET", "POST"])
def revenue_entry():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    # ✅ Determine selected date
    selected_date = request.form.get("date") or request.args.get("date")
    if not selected_date:
        selected_date = date.today().isoformat()

    if request.method == "POST":
        save_revenue_entry(
            username=username,
            category=request.form["category"],
            amount=float(request.form["amount"]),
            revenue_date=selected_date
        )
        flash("✅ Revenue entry added", "success")

        # 🔑 Redirect WITH date preserved
        return redirect(url_for("revenue_entry", date=selected_date))

    # ✅ Load entries for that date
    entries = load_revenue_entries_for_day(session["username"], selected_date)

    return render_template(
        "revenue_entry.html",
        entries=entries,
        selected_date=selected_date
    )


@app.route("/inventory_setup", methods=["GET", "POST"])
def inventory_setup():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    # 🔹 GET business_id FIRST (REQUIRED)
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cur = conn.cursor()

    cur.execute("""
        SELECT id FROM businesses
        WHERE username = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (username,))

    business = cur.fetchone()

    if not business:
        flash("❌ No business found. Please create a business first.", "error")
        return redirect(url_for("dashboard"))

    business_id = business["id"]

    cur.close()
    conn.close()

    # 🔹 HANDLE FORM SUBMIT
    if request.method == "POST":
        try:
            name = request.form["item_name"].strip().title()
            category = request.form["category"].strip().title()
            unit = request.form["unit"].strip().lower()
            quantity = float(request.form["starting_quantity"])
            snapshot_date = request.form["snapshot_date"]

            if not name or not unit:
                flash("⚠️ Item name and unit are required.", "error")
                return redirect(url_for("inventory_setup"))

            conn = get_db_connection()
            cur = conn.cursor()

            # 1️⃣ Insert item
            cur.execute("""
                INSERT INTO inventory_items (business_id, name, category, unit)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (business_id, name, category, unit))

            item_id = cur.fetchone()[0]

            # 2️⃣ Create snapshot
            cur.execute("""
                INSERT INTO inventory_snapshots
                (business_id, snapshot_date, snapshot_type, created_by)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (business_id, snapshot_date, "initial", username))

            snapshot_id = cur.fetchone()[0]

            # 3️⃣ Snapshot quantity
            cur.execute("""
                INSERT INTO inventory_snapshot_items
                (snapshot_id, item_id, quantity)
                VALUES (%s, %s, %s)
            """, (snapshot_id, item_id, quantity))

            conn.commit()
            cur.close()
            conn.close()

            return redirect(url_for("inventory_setup"))

        except Exception as e:
            print("Inventory error:", e)
            flash("❌ Failed to save inventory item.", "error")

    # 🔹 LOAD INVENTORY FOR DISPLAY
    conn = get_db_connection(cursor_factory=RealDictCursor)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            i.name,
            i.category,
            i.unit,
            s.snapshot_date,
            si.quantity
        FROM inventory_items i
        JOIN inventory_snapshot_items si ON si.item_id = i.id
        JOIN inventory_snapshots s ON s.id = si.snapshot_id
        WHERE i.business_id = %s
        ORDER BY s.snapshot_date DESC
    """, (business_id,))

    items = cur.fetchall()
    cur.close()
    conn.close()

    return render_template("inventory_setup.html", items=items)


@app.route("/inventory/adjust", methods=["GET", "POST"])
def inventory_adjust():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    conn = get_db_connection(cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # Get business_id for user
    cur.execute(
        "SELECT id FROM businesses WHERE username = %s",
        (username,)
    )
    business = cur.fetchone()
    if not business:
        flash("No business found.", "error")
        return redirect(url_for("dashboard"))

    business_id = business["id"]

    if request.method == "POST":
        item_id = request.form["item_id"]
        movement_type = request.form["movement_type"]
        quantity = float(request.form["quantity"])
        note = request.form.get("note")

        # Normalize quantity direction
        if movement_type in ("sale", "usage"):
            quantity = -abs(quantity)
        else:
            quantity = abs(quantity)

        cur.execute("""
            INSERT INTO inventory_movements
            (business_id, item_id, quantity_change, movement_type, source, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            business_id,
            item_id,
            quantity,
            movement_type,
            note,
            username
        ))

        conn.commit()
        flash("Inventory updated successfully.", "success")

    # Load items for dropdown
    cur.execute("""
        SELECT id, name
        FROM inventory_items
        WHERE business_id = %s
        ORDER BY name
    """, (business_id,))
    items = cur.fetchall()

    conn.close()

    return render_template(
        "inventory_adjust.html",
        items=items,
        success=True
    )

@app.route("/charts/revenue-forecast")
def revenue_forecast():
    if "username" not in session:
        return redirect(url_for("login"))

    # Example data
    forecast_dates = ["Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    forecast_values = [120000, 135000, 150000, 160000, 175000, 190000]

    return render_template(
        "charts/revenue_forecast.html",
        forecast_dates=forecast_dates,
        forecast_values=forecast_values
    )

@app.route("/charts/live-performance")
def live_performance():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT revenue_date, amount
        FROM revenue_entries
        WHERE username = %s
        ORDER BY revenue_date ASC
    """, (session["username"],))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Safe defaults 
    dates = [str(r[0]) for r in rows] if rows else []
    values = [float(r[1]) for r in rows] if rows else []

    return render_template(
        "charts/live_performance.html",
        dates=dates,
        values=values
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print("app is about to start listening on port",port)
    app.run(host="0.0.0.0", port=port)

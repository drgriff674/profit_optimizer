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
import threading
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
    run_db_operation,
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
    get_dashboard_intelligence_snapshot,
    get_live_financial_performance,
    get_locked_revenue_for_forecast,
    get_dashboard_snapshot,
    update_dashboard_snapshot,
    get_dashboard_intelligence,
    update_dashboard_intelligence,
    run_weekly_intelligence,
    maybe_generate_dashboard_insight,
    generate_weekly_ai_report_if_ready,
    get_latest_weekly_report,
    call_openai,
    get_user,
    create_user_with_business,
    get_weekly_inventory_insights,
    get_business_info,
)
import pytz
from flask_caching import Cache
from datetime import date
import re
import random
from email.mime.text import MIMEText

def send_otp_email(receiver_email, otp):

    msg = MIMEText(f"Your OptiGain verification code is: {otp}")
    msg["Subject"] = "OptiGain Email Verification"
    msg["From"] = "yourgmail@gmail.com"
    msg["To"] = receiver_email

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login("yourgmail@gmail.com", "YOUR_APP_PASSWORD")

    server.sendmail(msg["From"], receiver_email, msg.as_string())
    server.quit()

    print("OTP email sent successfully")

# ============================
# Revenue helpers
# ============================

def generate_revenue_day_export_data(username, revenue_date): 

    manual_entries = load_revenue_entries_for_day(username, revenue_date)

    def operation(cur):

        # --- business identifiers ---
        cur.execute("""
            SELECT paybill, account_number
            FROM businesses
            WHERE username = %s
            LIMIT 1
        """, (username,))
        biz = cur.fetchone()

        # --- locked day check ---
        cur.execute("""
            SELECT locked, total_amount
            FROM revenue_days
            WHERE username=%s AND revenue_date=%s
            LIMIT 1
        """,(username,revenue_date))

        day = cur.fetchone()

        is_locked = day["locked"] if day else False
        locked_total = float(day["total_amount"]) if day else 0

        # --- cash ---
        cur.execute("""
            SELECT COALESCE(SUM(amount),0) AS cash_total
            FROM cash_revenue
            WHERE username=%s AND revenue_date=%s
        """,(username,revenue_date))

        row=cur.fetchone()
        cash_total=float(row["cash_total"]) if row else 0.0

        mpesa_total = 0
        mpesa_entries = []

        # --- mpesa if business exists ---
        if biz:

            paybill=biz["paybill"]
            account_number=biz["account_number"]

            cur.execute("""
                SELECT COALESCE(SUM(amount),0) AS mpesa_total
                FROM mpesa_transactions
                WHERE status='confirmed'
                  AND (
                        (%s IS NOT NULL AND account_reference=%s)
                        OR
                        (%s IS NULL AND receiver=%s)
                      )
                  AND local_date=%s
            """,(
                account_number,
                account_number,
                account_number,
                paybill,
                revenue_date
            ))

            row=cur.fetchone()
            mpesa_total=float(row["mpesa_total"]) if row else 0.0

            cur.execute("""
                SELECT amount, created_at
                FROM mpesa_transactions
                WHERE status='confirmed'
                  AND (
                        (%s IS NOT NULL AND account_reference=%s)
                        OR
                        (%s IS NULL AND receiver=%s)
                      )
                  AND local_date=%s
                ORDER BY created_at ASC
            """,(
                account_number,
                account_number,
                account_number,
                paybill,
                revenue_date
            ))

            mpesa_entries = cur.fetchall()

        return {
            "date": revenue_date,
            "biz": biz,
            "is_locked": is_locked,
            "locked_total": locked_total,
            "cash_total": cash_total,
            "mpesa_total": mpesa_total,
            "mpesa_entries": mpesa_entries,
            "manual_entries": manual_entries,
            "expense_total": 0
        }

    db = run_db_operation(operation)
    return db

    
def generate_revenue_ai_summary(username, date):

    data = generate_revenue_day_export_data(username, date)

    if not data:
        return "No revenue available for this day."
        

    cash_total = data["cash_total"]
    mpesa_total = data["mpesa_total"]
    expense_total = data["expense_total"]
    manual_entries = data["manual_entries"]

    if not AI_ENABLED or client is None:
        return "AI summary unavailable. OpenAI API key not configured."

    categories={}
    for e in manual_entries:
        categories[e["category"]] = categories.get(e["category"],0)+float(e["amount"])

    category_lines="\n".join(
        f"- {k}: KSh {v:,.0f}" for k,v in categories.items()
    ) or "No manual category splits recorded."

    gross_total=cash_total+mpesa_total
    net_total=gross_total-expense_total

    prompt=f"""
You are a financial analyst.

Date: {date}

Cash revenue: KSh {cash_total:,.0f}
MPesa revenue: KSh {mpesa_total:,.0f}
Gross revenue: KSh {gross_total:,.0f}
Expenses: KSh {expense_total:,.0f}
Net revenue: KSh {net_total:,.0f}

Revenue category breakdown:
{category_lines}

Explain performance under 90 words.
Be factual.
"""

    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
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




# ✅ Flask app setup
app = Flask(__name__)

print ("Initializing database on startup...")
init_db()

cache = Cache(app, config={
    "CACHE_TYPE": "SimpleCache",  # safe for localhost
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutes
})
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key")

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
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

mail = Mail(app)

def initialize_database():
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


def generate_ai_insights(snapshot):
    insights = []

    if snapshot["locked_days"] < snapshot["window_days"]:
        insights.append(
            "📊 Data is still being finalized. Lock all revenue days for stronger insights."
        )

    if snapshot["anomaly_days"] > 0:
        insights.append(
            f"⚠️ {snapshot['anomaly_days']} day(s) show revenue inconsistencies."
        )

    if snapshot["ready_for_forecast"]:
        insights.append(
            "📈 Enough historical data collected. Forecasting confidence is improving."
        )
    else:
        insights.append(
            "⏳ Forecast model is still learning. More locked days needed."
        )

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

            user = get_user(username)

            if user and check_password_hash(user["password"], password):


                session["username"] = username
                print("LOGIN SUCCESS:",username)
                print("SESSION SET:",session.get("username"))
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
        confirm_pass = request.form["confirm_password"].strip()

        if new_pass != confirm_pass:
            flash("❌ Passwords do not match.", "error")
            return redirect(url_for("register"))

        business_name = request.form["business_name"].strip()
        paybill = request.form["paybill"].strip()
        account_number = request.form.get("account_number","").strip()

        try:
            # determine role
            users = load_users()
            role = "admin" if len(users) == 0 else "user"

            # hash password
            hashed_password = generate_password_hash(new_pass)

            result = create_user_with_business(
                new_user,
                new_email,
                hashed_password,
                role,
                business_name,
                paybill,
                account_number or None
            )

            if result == "exists":
                flash("⚠️ Username or email already exists.", "error")
                return redirect(url_for("register"))
        
            # ✅ Create upload folder
            os.makedirs(
                os.path.join(app.config["UPLOAD_FOLDER"], new_user), exist_ok=True
            )

            # Generate OTP
            otp = random.randint(100000, 999999)

            session["pending_user"] = new_user
            session["pending_email"] = new_email
            session["otp"] = otp

            # Send OTP email
            send_otp_email(new_email, otp)

            flash("📧 Verification code sent to your email.", "success")

            return redirect(url_for("verify_email"))

        except Exception as e:
            flash(f"❌ Error creating user: {str(e)}", "error")
            return redirect(url_for("register"))

    return render_template("register.html")

@app.route("/verify-email", methods=["GET", "POST"])
def verify_email():

    if request.method == "POST":

        code = request.form["otp"]

        if str(code) == str(session.get("otp")):

            session["username"] = session.get("pending_user")
            session["email"] = session.get("pending_email")
            session["verified"] = True

            flash("✅ Email verified successfully.", "success")

            return redirect(url_for("dashboard"))

        flash("❌ Invalid verification code.", "error")

    return render_template("verify_email.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@cache.memoize(timeout=3600)
def get_business_info(username):

    def operation(cur):
        cur.execute("""
            SELECT paybill, account_number
            FROM businesses
            WHERE username = %s
            LIMIT 1
        """, (username,))
        return cur.fetchone()

    return run_db_operation(operation)

@cache.memoize(timeout=120)
def get_dashboard_data(username):
    data = {}

    # 1️Historical revenue data (for charts)
    data["revenue_history"] = get_dashboard_revenue_intelligence(username)

    # 2️Intelligence snapshot (AI + dashboard counters)
    data["intelligence"] = get_dashboard_intelligence_snapshot(username)

    # 3️Forecast readiness (LOCKED days only)
    data["forecast_status"] = get_locked_revenue_for_forecast(username)
        

    # Optional legacy / placeholder
    if os.path.exists("financial_data.csv"):
        df = pd.read_csv("financial_data.csv")
        data["df"] = df
    else:
        data["df"] = pd.DataFrame()

    return data

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    print("SESSION USER ON DASHBOARD:",session.get("username"))
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    update_dashboard_snapshot(username)
    update_dashboard_intelligence(username)
    
    run_weekly_intelligence(username)
    generate_weekly_ai_report_if_ready(username)

    latest_payment = None


    #  Cached dashboard data
    cached_data = {}
    notifications = []
    forecast_data = []
    last_synced = None
    import time

    start_total = time.time()

    start = time.time()
    snapshot = get_dashboard_snapshot(username)
    print("Snapshot took:", time.time() - start)

    start = time.time()
    intelligence = get_dashboard_intelligence(username)
    print("Intelligence took:", time.time() - start)

    start = time.time()
    forecast_status = get_locked_revenue_for_forecast(username)
    print("Forecast took:", time.time() - start)

    start = time.time()
    latest_report = get_latest_weekly_report(username)
    print("Weekly report took:", time.time() - start)

    start = time.time()
    inventory_insights = get_weekly_inventory_insights(username)
    print ("Inventory_insights took:",time.time() - start)

    print("TOTAL dashboard time:", time.time() - start_total)

    answer = None

    #  User files (not cached)
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], username)
    os.makedirs(user_folder, exist_ok=True)
    files = sorted(os.listdir(user_folder))

    
    # kpi logic
    
    kpis = {
        "total_profit": "KSh 0",
        "avg_profit": "KSh 0",
        "total_expenses":"KSh 0",
        "profit_growth": "0%",
        "largest_expense": "N/A",
    }

    
    
    if not snapshot:
        snapshot = {}

    total_revenue = snapshot.get("total_revenue", 0)
    total_expenses = snapshot.get("total_expenses", 0)
    profit = snapshot.get("total_profit", 0)

    kpis["total_profit"] = f"KSh {profit:,.0f}"
    kpis["avg_profit"] = f"KSh {profit:,.0f}"
    kpis["total_expenses"] = f"KSh {total_expenses:,.0f}"
    kpis["largest_expense"] = snapshot.get("largest_expense","N/A")

    growth = snapshot.get("profit_growth",0)
    kpis["profit_growth"] = f"{growth:.2f}%"

    live_total_revenue = total_revenue

    
    
    # AI Insights
    
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if AI_ENABLED and question:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}],
                )
                answer = response.choices[0].message.content.strip()
                notifications.append(f"💡 Smart Insight: {answer}")
            except Exception as e:
                notifications.append(f"AI error: {str(e)}")

    
    # forecast chart
    
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
        forecast_status=forecast_status,
        live_total_revenue=live_total_revenue,
        weekly_report=latest_report,
        inventory_insights=inventory_insights
    )

@app.route("/api/dashboard-snapshot")
@login_required
def api_dash():

    username=session["username"]
    snap=get_dashboard_snapshot(username)

    return jsonify(snap)

@app.route("/revenue/day/<date>/delete", methods=["POST"])
@login_required
def delete_revenue_day(date):

    username = session["username"]

    def operation(cur):

        # check if day is locked
        cur.execute("""
            SELECT locked
            FROM revenue_days
            WHERE username = %s
              AND revenue_date = %s
        """, (username, date))

        row = cur.fetchone()

        if row and row["locked"]:
            return "locked"

        # delete manual entries
        cur.execute("""
            DELETE FROM revenue_entries
            WHERE username = %s
              AND revenue_date = %s
        """, (username, date))

        # delete container day
        cur.execute("""
            DELETE FROM revenue_days
            WHERE username = %s
              AND revenue_date = %s
        """, (username, date))

        return "deleted"

    result = run_db_operation(operation, commit=True)

    if result == "locked":
        flash("This revenue day is locked and cannot be deleted.", "error")
        return redirect(url_for("revenue_overview"))

    # refresh dashboard cache
    cache.delete_memoized(get_dashboard_data, username)

    flash("Revenue day deleted.", "success")
    return redirect(url_for("revenue_overview"))

@app.route("/revenue/day/<date>/ai-summary", methods=["POST"])
@login_required
def generate_ai_summary_for_day_route(date):

    if not AI_ENABLED:
        flash("AI summary unavailable")
        return redirect(url_for("revenue_day_detail", date=date))

    username = session["username"]

    data = generate_revenue_day_export_data(username, date)

    if not data:
        flash("No revenue data available")
        return redirect(url_for("revenue_day_detail", date=date))

    summary = generate_revenue_ai_summary(
        date=date,
        cash_total=data["cash_total"],
        mpesa_total=data["mpesa_total"],
        expense_total=data["expense_total"],
        manual_entries=data["manual_entries"]
    )

    save_ai_summary_for_day(username, date, summary)

    locked_total = data["cash_total"] + data["mpesa_total"]

    def operation(cur):
        cur.execute("""
            UPDATE revenue_days
            SET locked = TRUE,
                total_amount = %s
            WHERE username = %s
              AND revenue_date = %s
        """, (locked_total, username, date))

    run_db_operation(operation, commit=True)

    flash("AI summary generated.")
    return redirect(url_for("revenue_day_detail", date=date))

@app.route("/revenue/day/<date>/export/csv")
@login_required
def export_revenue_day_csv(date):

    username = session["username"]
    data = generate_revenue_day_export_data(username, date)

    if not data:
        abort(404)

    cash_total = data.get("cash_total", 0)
    mpesa_total = data.get("mpesa_total", 0)
    expense_total = data.get("expense_total", 0)

    gross_total = cash_total + mpesa_total
    net_total = gross_total - expense_total

    manual_entries = data.get("manual_entries", [])
    mpesa_entries = data.get("mpesa_entries", [])
    expense_entries = data.get("expense_entries", [])

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(["Revenue Report"])
    writer.writerow(["Date", data["date"]])
    writer.writerow([])

    writer.writerow(["Totals"])
    writer.writerow(["Cash", data["cash_total"]])
    writer.writerow(["MPesa", data["mpesa_total"]])
    writer.writerow(["Gross Revenue", gross_total])
    writer.writerow(["Expenses",-expense_total])
    writer.writerow(["Net Revenue", net_total])
    writer.writerow([])

    writer.writerow(["Manual Split Entries"])
    writer.writerow(["Category", "Amount"])
    for e in manual_entries:
        writer.writerow([e["category"], e["amount"]])

    writer.writerow([])
    writer.writerow(["Expenses"])
    writer.writerow(["Category", "Amount"])
    for e in expense_entries:
        writer.writerow([e["category"], -e["amount"]])

    writer.writerow([])
    writer.writerow(["MPesa Transactions"])
    writer.writerow(["Amount", "Time"])
    for m in mpesa_entries:
        writer.writerow([m["amount"], m["created_at"]])

    csv_data = output.getvalue()
    output.close()

    response = Response(csv_data, mimetype="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=revenue_{date}.csv"

    return response

@app.route("/revenue/day/<date>/export/pdf")
@login_required
def export_revenue_day_pdf(date):

    username = session["username"]
    data = generate_revenue_day_export_data(username, date)

    if not data:
        abort(404)

    # --- SAFE VALUES ---
    manual_entries = data.get("manual_entries", [])
    mpesa_entries = data.get("mpesa_entries", [])
    expense_entries = data.get("expense_entries", [])

    cash_total = float(data.get("cash_total", 0))
    mpesa_total = float(data.get("mpesa_total", 0))
    expense_total = float(data.get("expense_total", 0))

    gross_total = cash_total + mpesa_total
    net_total = gross_total - expense_total

    html = render_template(
        "revenue_day_pdf.html",
        date=data.get("date"),
        manual_entries=manual_entries,
        mpesa_entries=mpesa_entries,
        expense_entries=expense_entries,
        cash_total=cash_total,
        mpesa_total=mpesa_total,
        gross_total=gross_total,
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
    expenses = get_expenses_for_day(username, date)
    ai_summary = get_ai_summary_for_day(username, date)

    def operation(cur):

        # business identifiers
        cur.execute("""
            SELECT paybill, account_number
            FROM businesses
            WHERE username = %s
            LIMIT 1
        """, (username,))
        biz = cur.fetchone()

        # cash revenue
        cur.execute("""
            SELECT COALESCE(SUM(amount),0) AS cash_total
            FROM cash_revenue
            WHERE username=%s AND revenue_date=%s
        """,(username,date))

        row = cur.fetchone()
        cash_total = float(row["cash_total"]) if row else 0.0

        # anomalies
        cur.execute("""
            SELECT anomaly_type, severity, message
            FROM revenue_anomalies
            WHERE username=%s AND revenue_date=%s
        """,(username,date))
        anomalies = cur.fetchall()

        # lock status
        cur.execute("""
            SELECT locked, total_amount
            FROM revenue_days
            WHERE username=%s AND revenue_date=%s
            LIMIT 1
        """,(username,date))

        row = cur.fetchone()

        if row:
            is_locked = row.get("locked", False)
            locked_total = float(row.get("total_amount") or 0)
        else:
            is_locked = False
            locked_total = 0

        mpesa_total = 0.0

        if not is_locked and biz:

            paybill = biz["paybill"]
            account_number = biz["account_number"]

            cur.execute("""
                SELECT COALESCE(SUM(amount),0) AS mpesa_total
                FROM mpesa_transactions
                WHERE status='confirmed'
                  AND local_date=%s
                  AND (
                        account_reference=%s
                        OR receiver=%s
                      )
            """,(date,account_number,paybill))

            row = cur.fetchone()
            mpesa_total = float(row["mpesa_total"]) if row else 0.0

        return {
            "biz": biz,
            "cash_total": cash_total,
            "anomalies": anomalies,
            "is_locked": is_locked,
            "locked_total": locked_total,
            "mpesa_total": mpesa_total
        }

    db = run_db_operation(operation)

    cash_total = db["cash_total"]
    anomalies = db["anomalies"]
    is_locked = db["is_locked"]
    locked_total = db["locked_total"]
    mpesa_total = db["mpesa_total"]

    manual_total = sum(float(e["amount"]) for e in manual_entries)

    expense_total = expenses["total"]
    expense_entries = expenses["entries"]

    if is_locked:

        gross_total = locked_total
        net_revenue = locked_total
        mpesa_total = max(0, locked_total - cash_total + expense_total)

    else:

        gross_total = mpesa_total + cash_total
        net_revenue = gross_total - expense_total

        if abs(manual_total - net_revenue) > 0.01:
            anomalies.append({
                "anomaly_type": "manual_split_mismatch",
                "severity": "warning",
                "message": (
                    f"Manual split total (KSh {manual_total:.2f}) "
                    f"does not match net revenue (KSh {net_revenue:.2f})."
                )
            })

    return render_template(
        "revenue_day_detail.html",
        date=date,
        manual_total=manual_total,
        manual_entries=manual_entries,
        mpesa_total=mpesa_total,
        cash_total=cash_total,
        gross_total=gross_total,
        net_revenue=net_revenue,
        is_locked=is_locked,
        anomalies=anomalies,
        ai_summary=ai_summary,
        expense_total=expense_total,
        expense_entries=expense_entries
    )

@app.route("/revenue/overview")
@login_required
def revenue_overview():

    username = session["username"]

    def operation(cur):
        cur.execute("""
            SELECT
                revenue_date,
                locked,
                total_amount AS total_revenue
            FROM revenue_days
            WHERE username = %s
            ORDER BY revenue_date DESC
        """, (username,))

        return cur.fetchall()

    days = run_db_operation(operation)

    return render_template(
        "revenue_overview.html",
        days=days
    )


def run_post_lock_tasks(username, revenue_date):
    try:
        lock_manual_entries_for_the_day(username, revenue_date)
        detect_revenue_anomalies(username, revenue_date)
        update_dashboard_snapshot(username)
        update_dashboard_intelligence(username)
        run_weekly_intelligence(username)
        maybe_generate_dashboard_insight(username)
        generate_weekly_ai_report_if_ready(username)
    except Exception as e:
        print("Background task error:", e)

@app.route("/revenue/lock", methods=["POST"])
@login_required
def lock_revenue_day_route():

    username = session["username"]
    revenue_date = request.form["revenue_date"]

    expenses = get_expenses_for_day(username, revenue_date)
    expense_total = float(expenses["total"])

    def operation(cur):

        # stop if already locked
        cur.execute("""
            SELECT locked
            FROM revenue_days
            WHERE username=%s AND revenue_date=%s
        """,(username,revenue_date))

        row = cur.fetchone()

        if row and row["locked"]:
            return {"already_locked": True}

        # cash total
        cur.execute("""
            SELECT COALESCE(SUM(amount),0) AS cash_total
            FROM cash_revenue
            WHERE username=%s AND revenue_date=%s
        """,(username,revenue_date))

        cash_total = float(cur.fetchone()["cash_total"])

        # mpesa total
        cur.execute("""
            SELECT COALESCE(SUM(m.amount),0) AS mpesa_total
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                    (b.account_number IS NOT NULL AND m.account_reference=b.account_number)
                    OR
                    (b.account_number IS NULL AND m.receiver=b.paybill)
                 )
            WHERE b.username=%s
              AND m.status='confirmed'
              AND m.local_date=%s
        """,(username,revenue_date))

        mpesa_total = float(cur.fetchone()["mpesa_total"])

        gross_total = cash_total + mpesa_total
        net_total = gross_total - expense_total

        # ensure day exists
        cur.execute("""
            INSERT INTO revenue_days (username,revenue_date)
            VALUES(%s,%s)
            ON CONFLICT(username,revenue_date) DO NOTHING
        """,(username,revenue_date))

        # lock day
        cur.execute("""
            UPDATE revenue_days
            SET locked=TRUE,
                total_amount=%s
            WHERE username=%s
              AND revenue_date=%s
        """,(net_total,username,revenue_date))

        return {"already_locked": False}

    result = run_db_operation(operation, commit=True)

    if result["already_locked"]:
        flash("This day is already locked and cannot be changed.", "warning")
        return redirect(url_for("revenue_overview"))

    # run background intelligence pipeline
    threading.Thread(
        target=run_post_lock_tasks,
        args=(username, revenue_date)
    ).start()

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

        def operation(cur):
            cur.execute("""
                INSERT INTO cash_revenue (username, amount, description, revenue_date)
                VALUES (%s, %s, %s, %s)
            """, (username, amount, description, date))

        run_db_operation(operation, commit=True)

        update_dashboard_snapshot(username)
        update_dashboard_intelligence(username)

        cache.delete_memoized(get_dashboard_data, username)

        flash("✅ Cash revenue added", "success")
        return redirect(url_for("dashboard"))

    return render_template("cash_revenue_entry.html")

@app.route("/revenue/entry/<int:entry_id>/edit", methods=["GET", "POST"])
@login_required
def edit_revenue_entry(entry_id):

    username = session["username"]

    # -------- fetch entry --------
    def fetch_entry(cur):
        cur.execute("""
            SELECT *
            FROM revenue_entries
            WHERE id = %s AND username = %s
            LIMIT 1
        """, (entry_id, username))
        return cur.fetchone()

    entry = run_db_operation(fetch_entry)

    if not entry:
        abort(404)

    # -------- update entry --------
    if request.method == "POST":

        new_category = request.form["category"]
        new_amount = float(request.form["amount"])

        def update_entry(cur):
            cur.execute("""
                UPDATE revenue_entries
                SET category = %s,
                    amount = %s
                WHERE id = %s AND username = %s
            """, (new_category, new_amount, entry_id, username))

        run_db_operation(update_entry, commit=True)

        flash("Entry updated. You can continue editing", "success")

        return redirect(url_for(
            "revenue_entry",
            date=entry["revenue_date"]
        ))

    return render_template(
        "edit_revenue_entry.html",
        entry=entry
    )

@app.route("/charts/revenue-forecast")
@login_required
def revenue_forecast():

    username = session["username"]

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
        return cur.fetchall()

    rows = run_db_operation(operation)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        return render_template(
            "charts/revenue_forecast.html",
            not_ready=True,
            days=0
        )

    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)

    # Remove zero or invalid revenue days
    df = df[df["y"] > 0]
    df = df.sort_values("ds")

    locked_days = len(df)

    if locked_days < 7:
        return render_template(
            "charts/revenue_forecast.html",
            not_ready=True,
            days=locked_days
        )

    # Forecast maturity tiers
    if locked_days < 14:
        forecast_period = 7
        confidence_level = "Low"
    elif locked_days < 30:
        forecast_period = 30
        confidence_level = "Medium"
    else:
        forecast_period = 90
        confidence_level = "High"

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_mode="additive"
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

    dates = forecast["ds"].dt.strftime("%Y-%m-%d").tolist()
    predictions = forecast["yhat"].tolist()
    upper = forecast["yhat_upper"].tolist()
    lower = forecast["yhat_lower"].tolist()

    actual_dates = df["ds"].dt.strftime("%Y-%m-%d").tolist()
    actual_values = df["y"].tolist()

    return render_template(
        "charts/revenue_forecast.html",
        dates=dates,
        predictions=predictions,
        upper=upper,
        lower=lower,
        actual_dates=actual_dates,
        actual_values=actual_values,
        confidence_level=confidence_level,
        forecast_period=forecast_period,
        not_ready=False
    )

@app.route("/charts/live-performance")
@login_required
def live_performance():

    username = session["username"]

    def operation(cur):

        # CASH per day
        cur.execute("""
            SELECT revenue_date AS day, SUM(amount) AS cash
            FROM cash_revenue
            WHERE username=%s
            GROUP BY revenue_date
        """,(username,))
        cash_rows = cur.fetchall()

        # MPESA per day
        cur.execute("""
            SELECT
                m.local_date AS day,
                SUM(m.amount) AS mpesa
            FROM mpesa_transactions m
            JOIN businesses b
              ON (
                    (b.account_number IS NOT NULL AND m.account_reference=b.account_number)
                    OR
                    (b.account_number IS NULL AND m.receiver=b.paybill)
                 )
            WHERE b.username=%s
              AND m.status='confirmed'
            GROUP BY day
        """,(username,))
        mpesa_rows = cur.fetchall()

        # EXPENSES per day
        cur.execute("""
            SELECT expense_date AS day, SUM(amount) AS expenses
            FROM expenses
            WHERE username=%s
            GROUP BY expense_date
        """,(username,))
        expense_rows = cur.fetchall()

        return cash_rows, mpesa_rows, expense_rows

    cash_rows, mpesa_rows, expense_rows = run_db_operation(operation)

    # maps
    cash_map = {str(r["day"]): float(r["cash"]) for r in cash_rows}
    mpesa_map = {str(r["day"]): float(r["mpesa"]) for r in mpesa_rows}
    expense_map = {str(r["day"]): float(r["expenses"]) for r in expense_rows}

    dates = sorted(set(cash_map) | set(mpesa_map) | set(expense_map))

    revenue_values = []
    expense_values = []
    profit_values = []

    for d in dates:
        gross = cash_map.get(d, 0) + mpesa_map.get(d, 0)
        exp = expense_map.get(d, 0)

        revenue_values.append(gross)
        expense_values.append(exp)
        profit_values.append(gross - exp)

    return render_template(
        "charts/live_performance.html",
        dates=dates,
        revenue_values=revenue_values,
        expense_values=expense_values,
        profit_values=profit_values
    )

@app.route("/profile")
def profile():

    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]

    business = get_business_info(username)
    intelligence = get_dashboard_intelligence(username)
    snapshot = get_dashboard_snapshot(username)

    return render_template(
        "profile.html",
        business=business,
        intelligence=intelligence,
        snapshot=snapshot
    )

@app.route("/api/latest-payment")
def latest_payment():

    if "username" not in session:
        return jsonify({"payment": None})

    username = session["username"]

    def operation(cur):

        SQL = """
        SELECT m.id, m.amount, m.created_at
        FROM mpesa_transactions m
        JOIN businesses b
        ON (
            (b.account_number IS NOT NULL AND m.account_reference = b.account_number)
            OR
            (b.account_number IS NULL AND m.receiver = b.paybill)
        )
        WHERE b.username = %s
        AND m.seen = FALSE
        AND m.status = 'confirmed'
        ORDER BY m.created_at DESC
        LIMIT 1
        """

        cur.execute(SQL, (username,))
        payment = cur.fetchone()

        if payment:
            cur.execute("""
                UPDATE mpesa_transactions m
                SET seen = TRUE
                FROM businesses b
                WHERE m.id = %s
                AND (
                        (b.account_number IS NOT NULL AND m.account_reference = b.account_number)
                        OR
                        (b.account_number IS NULL AND m.receiver = b.paybill)
                    )
                AND b.username = %s
            """, (payment["id"], username))

        return payment

    payment = run_db_operation(operation, commit=True)

    return jsonify({"payment": payment})


@app.route("/api/financial_data")
def financial_data():

    if "username" not in session:
        return jsonify({})

    username = session["username"]

    try:

        def operation(cur):

            # Revenue per day
            cur.execute("""
                SELECT
                    m.local_date AS date,
                    COALESCE(SUM(m.amount),0) AS revenue
                FROM mpesa_transactions m
                JOIN businesses b
                  ON (
                        (b.account_number IS NOT NULL AND m.account_reference = b.account_number)
                        OR
                        (b.account_number IS NULL AND m.receiver = b.paybill)
                     )
                WHERE b.username = %s
                  AND m.status = 'confirmed'
                GROUP BY m.local_date
            """, (username,))
            revenue_rows = cur.fetchall()

            # Expenses per day
            cur.execute("""
                SELECT
                    DATE(expense_date) AS date,
                    COALESCE(SUM(amount),0) AS expenses
                FROM expenses
                WHERE username = %s
                GROUP BY DATE(expense_date)
            """, (username,))
            expense_rows = cur.fetchall()

            return revenue_rows, expense_rows

        revenue_rows, expense_rows = run_db_operation(operation)

        # Convert to maps
        revenue_map = {r["date"]: float(r["revenue"]) for r in revenue_rows}
        expense_map = {e["date"]: float(e["expenses"]) for e in expense_rows}

        all_dates = sorted(set(revenue_map) | set(expense_map))

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

    if "username" not in session:
        return jsonify({})

    username = session["username"]

    try:

        def operation(cur):
            cur.execute("""
                SELECT 
                    COALESCE(SUM(m.amount),0) AS total_revenue,
                    COALESCE(AVG(m.amount),0) AS avg_transaction,
                    COUNT(*) AS txn_count
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

            return cur.fetchone()

        row = run_db_operation(operation)

        total_revenue = float(row["total_revenue"] or 0)
        avg_transaction = float(row["avg_transaction"] or 0)
        txn_count = int(row["txn_count"] or 0)

        return jsonify({
            "total_revenue": round(total_revenue, 2),
            "avg_transaction": round(avg_transaction, 2),
            "txn_count": txn_count,
            "profit_growth": "N/A"
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
            shortcode = None

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

        if not transaction_id:
            print("Missing transaction_id, skipping insert")
            return jsonify({"ResultCode": 0, "ResultDesc": "Ignored"})
        
        #DEBUG ......
        print("Callback amount received",amount)

        # ============================================================
        # SAVE TO DATABASE
        # ============================================================
        def operation(cur):

            # find linked business
            cur.execute("""
                SELECT id, username
                FROM businesses
                WHERE paybill = %s
                LIMIT 1
                """, (shortcode,))
            biz = cur.fetchone()

            if not biz:
                business_id = None
                username_local = None
            else:
                business_id = biz["id"]
                username_local = biz["username"]

            print("BUSINESS FOUND:",username_local)

            import pytz
            from datetime import datetime

            nairobi = pytz.timezone("Africa/Nairobi")
            local_date = datetime.now(nairobi).date()

            # insert transaction (safe against duplicates)
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
                    status,
                    created_at,
                    local_date
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'confirmed',NOW(),%s)
                ON CONFLICT (transaction_id) DO NOTHING
            """, (
                transaction_id,
                amount,
                sender_phone or sender_name,
                shortcode,
                "C2B Payment",
                account_ref,
                description,
                json.dumps(data),
                request.remote_addr,
                local_date
            ))

            return username_local, local_date

        username_local, local_date = run_db_operation(operation, commit=True)
        
        if username_local:
            ensure_revenue_day_exists(username_local, local_date)
            update_dashboard_snapshot(username_local)
            update_dashboard_intelligence(username_local)
            cache.delete_memoized(get_dashboard_data, username_local)
        

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
            DATE_TRUNC('month', revenue_date) AS month,
            SUM(total_amount) AS revenue
        FROM revenue_days
        WHERE username = %s
        AND locked = TRUE
        GROUP BY DATE_TRUNC('month', revenue_date)
        ORDER BY month
        """, (session["username"],))
        
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

            update_dashboard_snapshot(username)
            update_dashboard_intelligence(username)
            cache.delete_memoized(get_dashboard_data, username)
            
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

        update_dashboard_intelligence(username)
        cache.delete_memoized(get_dashboard_data, username)
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

    # 🔹 GET business_id FIRST
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

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

            cur.execute("""
                INSERT INTO inventory_items (business_id, name, category, unit)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (business_id, name, category, unit))

            item_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO inventory_snapshots
                (business_id, snapshot_date, snapshot_type, created_by)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (business_id, snapshot_date, "initial", username))

            snapshot_id = cur.fetchone()[0]

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
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
    SELECT
        i.name,
        i.category,
        i.unit,
        s.snapshot_date,
        si.quantity + COALESCE(SUM(m.quantity_change),0) AS quantity
    FROM inventory_items i
    JOIN inventory_snapshot_items si ON si.item_id = i.id
    JOIN inventory_snapshots s ON s.id = si.snapshot_id
    LEFT JOIN inventory_movements m
        ON m.item_id = i.id
        AND m.business_id = i.business_id
        AND m.created_at > s.created_at
    WHERE i.business_id = %s
    GROUP BY
        i.name,
        i.category,
        i.unit,
        s.snapshot_date,
        si.quantity,
        s.created_at
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

    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get business_id
        cur.execute(
            "SELECT id FROM businesses WHERE username=%s",
            (username,)
        )
        business = cur.fetchone()

        if not business:
            flash("No business found.", "error")
            return redirect(url_for("dashboard"))

        business_id = business["id"]

        if request.method == "POST":

            item_id = int(request.form["item_id"])
            movement_type = request.form["movement_type"]
            quantity = float(request.form["quantity"])
            note = request.form.get("note")

            # normalize direction
            if movement_type in ("sale", "usage"):
                quantity = -abs(quantity)
            else:
                quantity = abs(quantity)

            cur.execute("""
                INSERT INTO inventory_movements
                (business_id, item_id, quantity_change, movement_type, source, created_by)
                VALUES (%s,%s,%s,%s,%s,%s)
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

        # load items
        cur.execute("""
            SELECT id, name
            FROM inventory_items
            WHERE business_id=%s
            ORDER BY name
        """, (business_id,))

        items = cur.fetchall()

        return render_template(
            "inventory_adjust.html",
            items=items,
            success=True
        )

    except Exception as e:
        print("Inventory adjust error:", e)
        flash("Inventory update failed.", "error")
        return redirect(url_for("dashboard"))

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":


    port = int(os.environ.get("PORT", 5000))
    print("app is about to start listening on port", port)

    app.run(host="0.0.0.0", port=port)

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
from prophet import Prophet
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2.extras import RealDictCursor
from database import load_users, save_user, init_db
import pytz

# ✅ Safe import for pdfkit (Render may not have wkhtmltopdf)
try:
    import pdfkit
    pdfkit_available = True
except ImportError:
    from xhtml2pdf import pisa
    pdfkit_available = False


# Railway PostgreSQL connection
DATABASE_URL = "postgresql://postgres:qzniBQaYcEdGRMKMqJessjlVGSLseaam@switchback.proxy.rlwy.net:14105/railway"


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


# ✅ Flask app setup
app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 📧 Mail configuration (uses environment variables)
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT"))
app.config["MAIL_USE_TLS"] = os.getenv("MAIL_USE_TLS") == "True"
app.config["MAIL_USE_SSL"] = os.getenv("MAIL_USE_SSL") == "True"
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = ("OptiGain Reports",os.getenv("MAIL_USERNAME"))

mail = Mail(app)

init_db()

# ✅ Disable AI if not needed
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_KEY"):
    print("WARNING: No OpenAI API key found - AI features will be limited.")
DISABLE_AI = False

AI_ENABLED = True


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
                flash("✅ Login successful!", "success")
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

            # ✅ Determine role
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()["count"]
            role = "admin" if user_count == 0 else "user"

            # ✅ Hash password and save new user
            hashed_password = generate_password_hash(new_pass)
            cursor.execute(
                "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
                (new_user, new_email, hashed_password, role),
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
            flash("✅ Registration successful! You are now logged in.", "success")
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"❌ Error creating user: {str(e)}", "error")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    # 🔄 Auto-refresh Google Sheets data before loading dashboard
    try:
        from sheets_helper import read_data
        import csv

        sheet_name = request.args.get("sheet","sheet1")
        data = read_data(sheet_name)
        csv_file = "financial_data.csv"
        fieldnames = ["Date", "Expenses", "Profit", "Revenue"]
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print("✅ Google Sheets auto-sync complete.")

        # 🔮 Generate AI insights for synced data
        try:
            df = pd.read_csv("financial_data.csv")
            preview = df.head().to_string()
            prompt = f"""
            You are a business data analyst AI for OptiGain.
            Review this synced financial data and provide:
            - 2 short trend insights
            - 1 actionable recommendation
            - 1 one-sentence performance summary
            Data sample:
            {preview}
            """
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are OptiGain's smart financial assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            notifications.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"⚠️ Failed to generate AI insights after sync: {e}")
        from datetime import datetime

        nairobi_tz = pytz.timezone("Africa/Nairobi")
        last_synced = datetime.now(nairobi_tz).strftime("%b %d, %y . %I:%M %p")
    except Exception as e:
        print(f"⚠️ Google Sheets sync failed: {e}")

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    os.makedirs(user_folder, exist_ok=True)
    files = os.listdir(user_folder)

    latest_file = None
    notifications = []
    answer = None
    forecast_data = []

    # ✅ Handle uploaded files
    if files:
        if "uploaded_file" in session and session["uploaded_file"] in files:
            latest_file = session["uploaded_file"]
        else:
            latest_file = sorted(files)[-1]

        file_path = os.path.join(user_folder, latest_file)

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            df = pd.read_csv(file_path)

            # Merge with manual entries
            manual_path = os.path.join(user_folder, "manual_entries.csv")
            if os.path.exists(manual_path):
                manual_df = pd.read_csv(manual_path)
                df = pd.concat([df, manual_df], ignore_index=True)

            preview = df.head().to_string()
            prompt = f"""
            You are a business data analyst AI for OptiGain. Review the latest uploaded company data below.
            Generate:
            1. 2 short trend insights (e.g. revenue uptrend, expense spike)
            2. 1 actionable recommendation (e.g. optimize marketing, reduce overhead)
            3. 1 performance summary (1 sentence, professional tone)

            Data sample:
            {preview}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are OptiGain's smart business assistant. Give data-driven, clear financial insights.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )

            insights = response.choices[0].message.content.strip()
            notifications.append(insights)
        except Exception as e:
            notifications.append(f"Error generating insights: {str(e)}")

    # ✅ Handle question form submission
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            notifications.append("Please enter a question before submitting.")
        else:
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}],
                )
                answer = response.choices[0].message.content.strip()
                notifications.append(f"💡 Smart Insight: {answer}")
            except Exception as e:
                notifications.append(f"Error generating insights: {str(e)}")

    # ✅ KPI Calculation
    kpis = {}
    try:
        df = pd.read_csv("financial_data.csv")
        print("📊 Loaded data from Google Sheets")
    except Exception:
        if latest_file:
            filepath = os.path.join(user_folder, latest_file)
            df = pd.read_csv(filepath)
            print("📂 Fallback to uploaded CSV")
        else:
            df = pd.DataFrame()

    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
        if "revenue" in df.columns and "expenses" in df.columns:
            df["profit"] = df["revenue"] - df["expenses"]

            total_profit = df["profit"].sum()
            avg_profit = df["profit"].mean()
            profit_growth = (
                ((df["profit"].iloc[-1] - df["profit"].iloc[0]) / df["profit"].iloc[0])
                * 100
                if df["profit"].iloc[0] != 0
                else 0
            )

            largest_expense = "Unknown"
            if "description" in df.columns:
                largest_expense = df.groupby("description")["expenses"].sum().idxmax()

            kpis = {
                "total_profit": f"${total_profit:,.2f}",
                "avg_profit": f"${avg_profit:,.2f}",
                "profit_growth": f"{profit_growth:.2f}%",
                "largest_expense": largest_expense,
            }
        else:
            kpis = {
                "total_profit": "N/A",
                "avg_profit": "N/A",
                "profit_growth": "N/A",
                "largest_expense": "N/A",
            }
    else:
        kpis = {
            "total_profit": "N/A",
            "avg_profit": "N/A",
            "profit_growth": "N/A",
            "largest_expense": "N/A",
        }

    # 🔮 Forecasting with Prophet
    try:
        # Always use the latest synced Google Sheets data
        df = pd.read_csv("financial_data.csv")

        # Merge with manual entries if they exist
        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
        manual_path = os.path.join(user_folder, "manual_entries.csv")
        if os.path.exists(manual_path):
            manual_df = pd.read_csv(manual_path)
            df = pd.concat([df, manual_df], ignore_index=True)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        if "date" in df.columns and "revenue" in df.columns:
            df["ds"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["ds"])
            df["y"] = df["revenue"]

            model = Prophet()
            model.fit(df[["ds", "y"]])
            future = model.make_future_dataframe(periods=6, freq="M")
            forecast = model.predict(future)
            future_forecast = forecast.tail(6)

            forecast_data = [
                {
                    "date": row["ds"].strftime("%b %Y"),
                    "predicted_revenue": f"${row['yhat']:,.2f}",
                }
                for _, row in future_forecast.iterrows()
            ]
    except Exception as e:
        forecast_data = [{"date": "Error", "predicted_revenue": str(e)}]

    # 📊 Prepare forecast chart data
    forecast_chart = []
    for item in forecast_data:
        try:
            if "predicted_revenue" in item:
                value = item["predicted_revenue"]
                if isinstance(value, str):
                    value = value.replace("$", "").replace(",", "")
                forecast_chart.append(
                    {"date": item["date"], "predicted_revenue": float(value)}
                )
        except Exception:
            continue

    # 📊 Live Performance (Revenue vs Expenses)
    performance_chart = []
    try:
        df = pd.read_csv("financial_data.csv")
        df.columns = df.columns.str.lower().str.strip()

        if (
            "date" in df.columns
            and "revenue" in df.columns
            and "expenses" in df.columns
        ):
            # Prepare data (limit to 12 most recent months)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date").tail(12)

            performance_chart = [
                {
                    "month": d.strftime("%b %Y"),
                    "revenue": float(r),
                    "expenses": float(e),
                }
                for d, r, e in zip(df["date"], df["revenue"], df["expenses"])
            ]
    except Exception as e:
        print(f"⚠️ Performance chart load failed: {e}")

    # ✅ Render dashboard
    return render_template(
        "dashboard.html",
        files=files,
        latest_file=latest_file,
        notifications=notifications,
        answer=answer,
        kpis=kpis,
        forecast_data=forecast_data,
        forecast_chart=json.dumps(forecast_chart),
        last_synced=last_synced,
        current_year=datetime.now().year
    )


@app.route("/api/financial_data")
def financial_data():
    import psycopg2, os
    from flask import jsonify

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        # Group totals per day (or month) for your chart
        cur.execute("""
            SELECT
                DATE(created_at) AS date,
                SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS revenue,
                SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) AS expenses
            FROM mpesa_transactions
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
            LIMIT 12;
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Convert DB rows to lists
        months = [r[0].strftime("%b %d") for r in rows]
        revenue = [float(r[1]) for r in rows]
        expenses = [float(r[2]) for r in rows]
        profit = [r1 - r2 for r1, r2 in zip(revenue, expenses)]

        return jsonify({
            "month": months,
            "revenue": revenue,
            "expenses": expenses,
            "profit": profit
        })

    except Exception as e:
        print(f"⚠️ Error generating live financial data: {e}")
        return jsonify({"error": str(e)})



@app.route("/api/transactions_summary")
def transactions_summary():
    import psycopg2, os
    from flask import jsonify

    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COALESCE(SUM(amount),0) AS total_profit,
                COALESCE(AVG(amount),0) AS avg_profit,
                COUNT(*) AS txn_count
            FROM mpesa_transactions;
        """)
        total_profit, avg_profit, txn_count = cur.fetchone()
        cur.close()
        conn.close()

        return jsonify({
            "total_profit": round(total_profit, 2),
            "avg_profit": round(avg_profit, 2),
            "profit_growth": f"{(avg_profit/total_profit*100 if total_profit else 0):.1f}%",
            "largest_expense": 0.0  # placeholder until expenses are tracked
        })
    except Exception as e:
        print(f"⚠️ Error generating summary: {e}")
        return jsonify({"error": str(e)})
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
        auth_url = "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials"
        response = requests.get(auth_url, auth=(consumer_key, consumer_secret))
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": f"Failed to get token: {str(e)}"}), 500

# ✅ REGISTER CALLBACK URL
@app.route("/register_url", methods=["GET"])
def register_url():
    import os, requests, json
    from flask import jsonify

    consumer_key = os.getenv("MPESA_CONSUMER_KEY")
    consumer_secret = os.getenv("MPESA_CONSUMER_SECRET")

    # Step 1: Get access token
    try:
        token_response = requests.get(
            "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials",
            auth=(consumer_key, consumer_secret),
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
    except Exception as e:
        return jsonify({"error": f"Failed to get token: {str(e)}"}), 400

    # Step 2: Register callback URLs (avoid using “mpesa” in the URL)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "ShortCode": "600983",  # Sandbox Shortcode (will replace with live shortcode later)
        "ResponseType": "Completed",
        "ConfirmationURL": "https://profitoptimizer-production.up.railway.app/payment/confirm",
        "ValidationURL": "https://profitoptimizer-production.up.railway.app/payment/validate",
    }

    try:
        res = requests.post(
            "https://sandbox.safaricom.co.ke/mpesa/c2b/v1/registerurl",
            headers=headers,
            json=payload,
            timeout=15,
        )
        return jsonify(res.json()), res.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to register URL: {str(e)}"}), 500

# ================================
# C2B VALIDATION (Sandbox)
# ================================
@app.route("/payment/validate", methods=["POST"])
def payment_validate():
    data = request.get_json()
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

    data = request.get_json()
    print("📥 PAYMENT CALLBACK RECEIVED:", json.dumps(data, indent=2))

    try:
        # ============================================================
        # 1️⃣ CASE A — C2B SIMULATOR (V1 CALLBACK)
        # ============================================================
        if "TransID" in data:
            print("✔ Detected: C2B Simulator (V1)")

            transaction_id = data.get("TransID", "")
            amount = float(data.get("TransAmount", 0))
            sender_name = data.get("FirstName", "Unknown")
            sender_phone = data.get("MSISDN", "")
            description = "C2B Payment (V1)"
            account_ref = data.get("BillRefNumber", "")

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

        # ============================================================
        # SAVE TO DATABASE
        # ============================================================
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO mpesa_transactions
            (transaction_id, amount, sender, receiver, transaction_type,
             account_reference, description, timestamp, raw_payload, origin_ip, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
        """, (
            transaction_id,
            amount,
            sender_phone or sender_name,
            "OptiGain",
            "C2B Payment",
            account_ref,
            description,
            datetime.utcnow(),
            json.dumps(data),
            request.remote_addr
        ))

        conn.commit()
        cur.close()
        conn.close()

        print("✅ PAYMENT SAVED:", amount)

        return jsonify({"ResultCode": 0, "ResultDesc": "Success"})

    except Exception as e:
        print("❌ ERROR in confirm callback:", e)
        return jsonify({"ResultCode": 1, "ResultDesc": "Internal Error"})
# ✅ Query M-Pesa Account Balance (safe naming)
@app.route("/api/account_balance")
def account_balance():
    try:
        import os, requests
        from requests.auth import HTTPBasicAuth
        from flask import jsonify

        consumer_key = os.getenv("MPESA_CONSUMER_KEY")
        consumer_secret = os.getenv("MPESA_CONSUMER_SECRET")
        shortcode = os.getenv("MPESA_SHORTCODE")
        passkey = os.getenv("MPESA_PASSKEY")

        # 1️⃣ Get Access Token
        token_url = "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials"
        token_response = requests.get(
            token_url, auth=HTTPBasicAuth(consumer_key, consumer_secret)
        )
        access_token = token_response.json().get("access_token")

        if not access_token:
            return jsonify({"error": "Failed to get access token"}), 500

        # 2️⃣ Query Account Balance
        balance_url = "https://sandbox.safaricom.co.ke/mpesa/accountbalance/v1/query"
        headers = {"Authorization": f"Bearer {access_token}"}

        payload = {
            "Initiator": "testapi",
            "SecurityCredential": "Safaricom123!",
            "CommandID": "AccountBalance",
            "PartyA": shortcode,
            "IdentifierType": "4",
            "Remarks": "Checking account balance",
            "QueueTimeOutURL": "https://profit-optimizer.onrender.com/payment/timeout",
            "ResultURL": "https://profit-optimizer.onrender.com/payment/balance_result",
        }

        response = requests.post(balance_url, json=payload, headers=headers)

        try:
            data = response.json()
        except ValueError:
            print("⚠️ Invalid JSON from M-Pesa:", response.text)
            return jsonify({"error": "Invalid or empty JSON response from M-Pesa", "raw": response.text}), 500

        print("✅ M-Pesa raw response:", data)
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Payment Timeout Callback
@app.route("/payment/timeout", methods=["POST"])
def payment_timeout():
    data = request.get_json()
    print("⏱️ Payment Timeout:", data)
    return jsonify({"ResultCode": 1, "ResultDesc": "Request timed out"})

# ✅ Balance Result Callback
@app.route("/payment/balance_result", methods=["POST"])
def payment_balance_result():
    data = request.get_json()
    print("💰 Account Balance Result:", data)
    return jsonify({"ResultCode": 0, "ResultDesc": "Balance result received"})
# ✅ AI Insight Engine — analyzes latest financial data and generates insights
@app.route("/api/ai_insights")
def ai_insights():
    try:
        if "username" not in session:
            return jsonify([])

        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
        files = [f for f in os.listdir(user_folder) if f.endswith(".csv")]
        if not files:
            return jsonify([])

        latest_file = sorted(files)[-1]
        df = pd.read_csv(os.path.join(user_folder, latest_file))
        df.columns = df.columns.str.lower().str.strip()

        insights = []

        # --- Trend analysis ---
        if all(col in df.columns for col in ["revenue", "expenses"]):
            df["profit"] = df["revenue"] - df["expenses"]
            recent = df.tail(3)

            if recent["profit"].iloc[-1] > recent["profit"].iloc[-2]:
                insights.append("Profit increased in the latest period ✅")
            elif recent["profit"].iloc[-1] < recent["profit"].iloc[-2]:
                insights.append(
                    "Profit decreased recently ⚠️ Check expenses or pricing."
                )
            else:
                insights.append("Profit remained stable recently.")

            avg_margin = (df["profit"] / df["revenue"]).mean() * 100
            insights.append(f"Average profit margin: {avg_margin:.1f}%")

            if avg_margin < 15:
                insights.append(
                    "Profit margin is below 15%. Consider reducing costs or revising prices."
                )
            elif avg_margin > 30:
                insights.append(
                    "Strong profit margin (>30%). Business is performing well! 💪"
                )

            # Highest & lowest month
            max_profit_month = df.loc[df["profit"].idxmax(), "month"]
            insights.append(f"Highest profit recorded in {max_profit_month}.")

        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)})


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
    uploaded_files = uploaded_csvs.get(username, [])

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


@app.route("/trend_forecaster", methods=["GET", "POST"])
def trend_forecaster():
    forecast_plot = None

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)

            # Ensure correct column names
            df.columns = [c.lower() for c in df.columns]
            if "date" not in df or "revenue" not in df:
                flash("⚠️ CSV must have 'date' and 'revenue' columns.", "error")
                return redirect(url_for("trend_forecaster"))

            # Prepare data
            df["date"] = pd.to_datetime(df["date"])
            prophet_df = df.rename(columns={"date": "ds", "revenue": "y"})

            # Fit Prophet model
            model = Prophet()
            model.fit(prophet_df)

            # Forecast next 30 days
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            # Plot
            fig = model.plot(forecast)
            img_path = os.path.join("static", "forecast.png")
            fig.savefig(img_path)
            forecast_plot = img_path

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


@app.route("/manual_entry", methods=["GET", "POST"])
def manual_entry():
    if "username" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    os.makedirs(user_folder, exist_ok=True)
    manual_path = os.path.join(user_folder, "manual_entries.csv")

    # ✅ Load existing data
    if os.path.exists(manual_path):
        df = pd.read_csv(manual_path)
    else:
        df = pd.DataFrame(columns=["Month", "Revenue", "Expenses", "Description"])

    # ✅ Handle form submission
    if request.method == "POST":
        month = request.form.get("month")
        revenue = request.form.get("revenue")
        expenses = request.form.get("expenses")
        description = request.form.get("description")

        if not month or not revenue or not expenses:
            flash("Please fill in all required fields.", "error")
        else:
            new_entry = {
                "Month": month,
                "Revenue": float(revenue),
                "Expenses": float(expenses),
                "Description": description,
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(manual_path, index=False)
            flash("✅ Data added successfully!", "success")
            return redirect(url_for("manual_entry"))

    return render_template("manual_entry.html", entries=df.to_dict(orient="records"))


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
        df = df.drop(index)
        df.to_csv(manual_path, index=False)
        flash("🗑️ Entry deleted successfully!", "success")
    else:
        flash("Invalid entry selected.", "error")

    return redirect(url_for("manual_entry"))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

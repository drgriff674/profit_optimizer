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
from sheets_helper import read_data, add_row
import pytz


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


# ✅ Mail configuration
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "griffinnnnn77@gmail.com"
app.config["MAIL_PASSWORD"] = "abcdefghijklmnop"  # Replace with Gmail App Password
app.config["MAIL_DEFAULT_SENDER"] = "griffinnnnn77@gmail.com"

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


@app.route("/sync_sheets")
def sync_sheets():
    from sheets_helper import read_data
    import csv

    data = read_data()

    # Save data into your local CSV so dashboard can use it
    csv_file = "financial_data.csv"
    fieldnames = ["Date", "Expenses", "Profit", "Revenue"]

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    return {"status": "success", "rows_synced": len(data)}


# Auto-sync Google Sheets data on startup (Render-compatible)
with app.app_context():
    try:
        from sheets_helper import read_data
        import csv

        data = read_data()
        csv_file = "financial_data.csv"
        fieldnames = ["Date", "Expenses", "Profit", "Revenue"]
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print("✅ Google Sheets auto-sync complete.")
    except Exception as e:
        print(f"⚠️ Google Sheets sync failed: {e}")


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
    )


@app.route("/api/financial_data")
def financial_data():
    try:
        import psycopg2, os
        from flask import jsonify
        import pandas as pd

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        query = """
            SELECT 
                DATE_TRUNC('month', created_at) AS month,
                SUM(amount) AS revenue
            FROM mpesa_transactions
            GROUP BY 1
            ORDER BY 1;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return jsonify({"month": [], "revenue": [], "expenses": [], "profit": []})

        # For now, assume expenses = 0 (or calculate if you store them later)
        df["expenses"] = 0
        df["profit"] = df["revenue"] - df["expenses"]

        df["month"] = df["month"].dt.strftime("%b %Y")

        return jsonify({
            "month": df["month"].tolist(),
            "revenue": df["revenue"].tolist(),
            "expenses": df["expenses"].tolist(),
            "profit": df["profit"].tolist(),
        })

    except Exception as e:
        print("⚠️ Error generating live financial data:", e)
        return jsonify({"error": str(e)})


@app.route("/api/transactions_summary")
def transactions_summary():
    try:
        import psycopg2, os
        from flask import jsonify

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(SUM(amount),0) FROM mpesa_transactions;")
        total = float(cur.fetchone()[0])

        cur.execute("SELECT COALESCE(AVG(amount),0) FROM mpesa_transactions;")
        avg = float(cur.fetchone()[0])

        cur.execute("""
            SELECT MAX(amount) FROM mpesa_transactions WHERE transaction_type='Expense';
        """)
        largest_expense = float(cur.fetchone()[0] or 0)

        cur.close()
        conn.close()

        return jsonify({
            "total_profit": total,
            "average_profit": avg,
            "largest_expense": largest_expense,
            "profit_growth": 0.0  # we’ll compute real growth later
        })

    except Exception as e:
        print("⚠️ Error loading summary:", e)
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

# ✅ PAYMENT CONFIRMATION CALLBACK
@app.route("/payment/confirm", methods=["POST"])
def payment_confirm():
    from flask import request, jsonify
    import psycopg2, json, os
    from datetime import datetime

    data = request.get_json()
    print("📥 Confirmation received:", data)

    try:
        result = data.get("Result", {})
        result_desc = result.get("ResultDesc", "")
        transaction_id = result.get("ConversationID", "")
        # Try to extract amount from multiple possible places
        amount = 0.0
        try:
            # Case 1: Normal sandbox structure (direct TransAmount)
            if "TransAmount" in data:
                amount = float(data.get("TransAmount", 0))

            # Case 2: Callback wrapped under "Result"
            elif "Result" in data and isinstance(data["Result"], dict):
                params = data["Result"].get("ResultParameters", {}).get("ResultParameter", [])
                for p in params:
                    if p.get("Key") == "TransAmount":
                        amount = float(p.get("Value", 0))
                        break
        except Exception:
            amount = 0.0
        # Extract sender details if available
        sender_name = "Unknown"
        phone_number = ""

        # Some simulator responses include "FirstName", "MiddleName", "LastName"
        if "FirstName" in data:
            names = [data.get("FirstName", ""), data.get("MiddleName", ""), data.get("LastName", "")]
            sender_name = " ".join([n for n in names if n]).strip()
            phone_number = data.get("MSISDN", "")

        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO mpesa_transactions
            (transaction_id, amount, sender, receiver, transaction_type, account_reference, description, timestamp, raw_payload, origin_ip, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
            """,
            (
                transaction_id,
                amount,
                sender_name,
                "OptiGain",
                "C2B Payment",
                "Confirmation",
                result_desc,
                datetime.utcnow(),
                json.dumps(data),
                request.remote_addr,
            ),
        )

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Transaction saved successfully.")

        return jsonify({"ResultCode": 0, "ResultDesc": "Callback received successfully"})
    except Exception as e:
        print("⚠️ Error saving callback:", e)
        return jsonify({"ResultCode": 1, "ResultDesc": str(e)})

# ✅ PAYMENT VALIDATION CALLBACK
@app.route("/payment/validate", methods=["POST"])
def payment_validate():
    data = request.get_json()
    print("📥 Validation request:", data)
    return jsonify({"ResultCode": 0, "ResultDesc": "Validation received successfully"})


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

    search_query = request.form.get("search") if request.method == "POST" else ""
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["username"])
    filepaths = []

    # ✅ Support up to 3 uploaded files per user
    uploaded_files = uploaded_csvs.get(session["username"], [])
    for f in uploaded_files:
        path = os.path.join(user_folder, f)
        if os.path.exists(path):
            filepaths.append(path)

    # If the file being viewed isn't already in the list, include it
    filepath = os.path.join(user_folder, filename)
    if filepath not in filepaths:
        filepaths.insert(0, filepath)

    if not filepaths:
        return "No uploaded files found", 404

    # ✅ Read all CSVs (1–3) safely
    dataframes = []
    for path in filepaths[:3]:
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
            df.columns = df.columns.str.lower().str.strip()
            if "revenue" in df.columns and "expenses" in df.columns:
                df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
                df["expenses"] = pd.to_numeric(df["expenses"], errors="coerce")
                dataframes.append(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    if not dataframes:
        return "No valid financial data found in uploaded files.", 400

    # ✅ Categorization (kept from your version)
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

    for df in dataframes:
        if "description" in df.columns:
            df["category"] = df["description"].apply(categorize)

    # ✅ Combine or compare datasets dynamically
    if len(dataframes) == 1:
        df = dataframes[0]
    else:
        base = dataframes[0]
        for i, compare_df in enumerate(dataframes[1:], start=2):
            base = base.merge(
                compare_df, on="month", suffixes=("", f"_file{i}"), how="outer"
            )
        df = base

    # ✅ Core metrics
    total_revenue = sum(df[col].sum() for col in df.columns if "revenue" in col)
    total_expenses = sum(df[col].sum() for col in df.columns if "expenses" in col)
    profit = total_revenue - total_expenses
    profit_margin = round((profit / total_revenue) * 100, 2) if total_revenue > 0 else 0
    avg_revenue = round(df[[c for c in df.columns if "revenue" in c]].mean().mean(), 2)
    avg_expenses = round(
        df[[c for c in df.columns if "expenses" in c]].mean().mean(), 2
    )

    # ✅ Charts
    bar_fig = go.Figure()
    for col in df.columns:
        if "revenue" in col:
            bar_fig.add_trace(go.Bar(name=f"Revenue {col}", x=df["month"], y=df[col]))
        elif "expenses" in col:
            bar_fig.add_trace(go.Bar(name=f"Expenses {col}", x=df["month"], y=df[col]))
    bar_fig.update_layout(
        barmode="group", title="Monthly Revenue vs Expenses (All Files)"
    )
    bar_chart = pyo.plot(bar_fig, output_type="div", include_plotlyjs=True)

    # Pie Chart
    pie_fig = go.Figure(
        data=[go.Pie(labels=["Profit", "Expenses"], values=[profit, total_expenses])]
    )
    pie_chart = pyo.plot(pie_fig, output_type="div", include_plotlyjs=True)

    # Line Chart
    line_fig = go.Figure()
    for col in df.columns:
        if "revenue" in col or "expenses" in col:
            line_fig.add_trace(
                go.Scatter(x=df["month"], y=df[col], mode="lines+markers", name=col)
            )
    line_fig.update_layout(
        title="📉 Financial Trends", xaxis_title="Month", yaxis_title="Amount"
    )
    line_chart = pyo.plot(line_fig, output_type="div", include_plotlyjs=True)

    # ✅ Advanced insights
    insights = []
    if len(dataframes) > 1:
        for i in range(1, len(dataframes)):
            rev1, rev2 = (
                dataframes[i - 1]["revenue"].sum(),
                dataframes[i]["revenue"].sum(),
            )
            diff = rev2 - rev1
            pct = (diff / rev1 * 100) if rev1 else 0
            insights.append(
                f"💼 Between File {i} and File {i+1}, revenue changed by {pct:.2f}% ({'↑' if pct > 0 else '↓'} {abs(diff):,.2f})."
            )

    if profit_margin < 10:
        insights.append("⚠️ Low profit margin — consider pricing or cost adjustments.")
    elif profit_margin < 25:
        insights.append(
            "🟡 Average margin. Try optimizing operations or marketing ROI."
        )
    else:
        insights.append("🟢 Healthy margin — maintain efficiency and growth.")

    # Trend stability
    if "month" in df.columns and "revenue" in df.columns:
        trend = df["revenue"].diff().fillna(0)
        if (trend > 0).all():
            insights.append("📈 Steady revenue growth each period.")
        elif (trend < 0).all():
            insights.append("📉 Consistent decline — investigate causes.")
        else:
            insights.append("↕️ Mixed revenue trend — review month-to-month changes.")

    # ✅ Category chart (optional)
    if "category" in df.columns:
        category_totals = df.groupby("category")["expenses"].sum()
        cat_fig = go.Figure(
            data=[go.Pie(labels=category_totals.index, values=category_totals.values)]
        )
        cat_fig.update_layout(title="🧾 Expense Breakdown by Category")
        category_chart = pyo.plot(cat_fig, output_type="div", include_plotlyjs=False)
    else:
        category_chart = (
            "<div class='alert alert-warning'>No 'Category' data available.</div>"
        )

    # ✅ Safe HTML for results
    table_html = df.to_html(classes="table table-bordered table-striped", index=False)
    final_advice = "<br>".join(insights)
    escaped_advice = quote(final_advice)

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
        category_chart=category_chart,
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

    email = request.form.get("email")
    revenue = request.form.get("revenue")
    expenses = request.form.get("expenses")
    profit = request.form.get("profit")
    margin = request.form.get("margin")
    advice = request.form.get("advice")

    # Simulate email sending
    print(f"📨 Sending summary to: {email}")
    print("Summary content:")
    print(f"Revenue: {revenue}")
    print(f"Expenses: {expenses}")
    print(f"Profit: {profit}")
    print(f"Margin: {margin}%")
    print("Advice:", advice)

    flash("✅ Summary sent to your email (simulated).", "success")
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


@app.route("/download_full_report")
def download_full_report():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    from io import BytesIO
    from flask import send_file
    from fpdf import FPDF
    import os

    # Example data (replace with actual session or DB data as needed)
    months = ["Jan", "Feb", "Mar", "Apr"]
    revenues = [10000, 12000, 14000, 13000]
    expenses = [8000, 8500, 9000, 9500]
    advice = (
        "Consider reducing operations cost in March and investing more in marketing."
    )

    # Create plots and save to buffers
    def create_chart():
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        sns.lineplot(x=months, y=revenues, label="Revenue", ax=ax)
        sns.lineplot(x=months, y=expenses, label="Expenses", ax=ax)
        ax.set_title("Financial Trend")
        ax.legend()
        buf = BytesIO()
        fig.tight_layout()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)
        return buf

    chart_buf = create_chart()

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Financial Summary Report", ln=True, align="C")
    pdf.ln(10)

    # Summary stats
    avg_rev = sum(revenues) / len(revenues)
    avg_exp = sum(expenses) / len(expenses)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Average Revenue: ${avg_rev:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Average Expenses: ${avg_exp:.2f}", ln=True)
    pdf.ln(10)

    # Save chart image temporarily
    img_path = os.path.join("static", "temp_chart.png")
    with open(img_path, "wb") as f:
        f.write(chart_buf.read())

    # Add chart to PDF
    pdf.image(img_path, x=10, y=None, w=180)
    pdf.ln(10)

    # Add advice
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"AI Financial Advice:\n{advice}")

    # Save PDF to buffer
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    pdf_buffer = BytesIO(pdf_bytes)
    pdf_buffer.seek(0)

    # Clean up
    os.remove(img_path)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="full_dashboard_report.pdf",
        mimetype="application/pdf",
    )


@app.route("/send_report", methods=["POST"])
def send_report():
    # Get the recipient email (from form or session)
    recipient_email = request.form.get("email") or session.get("email")
    if not recipient_email:
        flash("❌ No recipient email provided.", "danger")
        return redirect(url_for("dashboard"))

    try:
        # Generate sample PDF report in memory
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Financial Report Summary", ln=True, align="C")
        pdf.cell(200, 10, txt="Generated via OptiGain", ln=True, align="C")

        # Convert to BytesIO for attachment
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        pdf_buffer = io.BytesIO(pdf_bytes)

        # Create email
        msg = Message(
            subject="📊 Your Financial Report - OptiGain",
            recipients=[recipient_email],
            body="Hello,\n\nPlease find attached your latest financial report.\n\nBest,\nThe OptiGain Team",
        )

        # Attach PDF
        msg.attach("Financial_Report.pdf", "application/pdf", pdf_buffer.read())

        # Send email
        mail.send(msg)

        flash(f"✅ Report sent successfully to {recipient_email}.", "success")

    except Exception as e:
        print("Error sending email:", e)
        flash(f"⚠️ Failed to send report: {str(e)}", "danger")

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

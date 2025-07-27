from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import os
from openai import OpenAI
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
from urllib.parse import quote
from werkzeug.utils import secure_filename
from fpdf import FPDF
import json
import io
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from flask_mail import Mail, Message
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'your_secret_key'

uploaded_csvs = {}

# Mail Configuration for griffinnnnn77@gmail.com
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'griffinnnnn77@gmail.com'
app.config['MAIL_PASSWORD'] = 'abcdefghijklmnop'  # Replace with Gmail App Password
app.config['MAIL_DEFAULT_SENDER'] = 'griffinnnnn77@gmail.com'

mail = Mail(app)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}



USER_FILE = 'users.json'

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f)
        
def allowed_file(filename):
    return','in filename and filename.rsplit(',',1)[1].lower()=='csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_users()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            session['user_id'] = username
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    users = load_users()
    if request.method == 'POST':
        new_user = request.form['username']
        new_pass = request.form['password']
        if new_user in users:
            flash("Username already exists", "error")
            return redirect(url_for('register'))
        users[new_user] = new_pass
        save_users(users)
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], new_user), exist_ok=True)
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'],session['user'])
    os.makedirs(user_folder, exist_ok=True)
    files = os.listdir(user_folder)

    latest_file = None
    notifications = []

    # ✅ FIX 1: Try using the most recently uploaded file from session
    if 'uploaded_file' in session and session['uploaded_file'] in files:
        latest_file = session['uploaded_file']
    elif files:
        # ✅ FIX 2: If no session file, fall back to last file in folder
        latest_file = sorted(files)[-1]

    if latest_file:
        filepath = os.path.join(user_folder, latest_file)
        try:
            df = pd.read_csv(filepath)

            # ✅ NO CHANGE: Sample analysis logic
            if 'Revenue' in df.columns and 'Expenses' in df.columns:
                recent_revenue = df['Revenue'].tail(3).mean()
                recent_expenses = df['Expenses'].tail(3).mean()
                older_expenses = df['Expenses'].head(3).mean()

                if recent_expenses > older_expenses * 1.3:
                    notifications.append("Alert: Your expenses increased by over 30% in recent months.")

                if df['Revenue'].iloc[-1] > df['Revenue'].iloc[0]:
                    notifications.append("Good job: Revenue is steadily increasing.")
        except Exception as e:
            notifications.append("Could not read latest financial data.")

    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question}]
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI Error:{e}")
                answer = f"Error:{str(e)}"

    # ✅ NO CHANGE: Pass list of files and notifications to dashboard template
    return render_template('dashboard.html', files=files, notifications=notifications, answer=answer)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        files = request.files.getlist('files')

        if len(files) < 2:
            return "Please upload two files to compare.", 400

        # Save files to the user's folder
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
        os.makedirs(user_folder, exist_ok=True)

        username = session.get('username')
        if username:
            if username not in uploaded_csvs:
                uploaded_csvs[username] = []
            for file in files:
                filename = file.filename
                uploaded_csvs[username].append(filename)

        file1 = files[0]
        file2 = files[1]

        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        filepath1 = os.path.join(user_folder, filename1)
        filepath2 = os.path.join(user_folder, filename2)

        file1.save(filepath1)
        file2.save(filepath2)

        # Save one of the files to session so dashboard can use it
        session['uploaded_file'] = filename1

        # Load into pandas for comparison
        df1 = pd.read_csv(filepath1)
        df2 = pd.read_csv(filepath2)

        comparison = pd.DataFrame({
            'Month': df1['Month'],
            'Revenue Difference': df2['Revenue'] - df1['Revenue'],
            'Expense Difference': df2['Expenses'] - df1['Expenses']
        })
        session['comparison_data'] = comparison.to_json()

        table_html = comparison.to_html(index=False)

        return render_template('comparison.html', table_html=table_html)

    return render_template('upload.html')

@app.route('/view/<filename>',methods=['GET','POST'])
def view_file(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    search_query = request.form.get('search') if request.method == 'POST' else''

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, on_bad_lines='skip')
    elif filepath.endswith('.xls') or filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        return "Unsupported file format"

    # Example keyword-based categorization
    def categorize(description):
        desc = str(description).lower()
        if any(word in desc for word in ['facebook', 'ad', 'campaign', 'seo']):
            return 'Marketing'
        elif any(word in desc for word in ['sales', 'client', 'deal']):
            return 'Sales'
        elif any(word in desc for word in ['research', 'development', 'prototype']):
            return 'R&D'
        elif any(word in desc for word in ['office', 'admin', 'maintenance']):
            return 'Operations'
        else:
            return 'Other'

# Only run if 'Description' column exists
    if 'Description' in df.columns:
        df['Category'] = df['Description'].apply(categorize)

    if 'username' in session:
        username = session['username']
        if username not in uploaded_csvs:
            uploaded_csvs[username] = []
        if filename not in uploaded_csvs[username]:
            uploaded_csvs[username].append(filename)

    # Handle optional category filter
    category_filter = request.args.get('category')
    if category_filter:
        df = df[df['Category'] == category_filter]

    # Get unique categories for dropdown (only if 'Category' column exists)
    categories = df['Category'].unique().tolist() if 'Category' in df.columns else []

    total_revenue = df['Revenue'].sum()
    total_expenses = df['Expenses'].sum()
    profit = total_revenue - total_expenses
    profit_margin = round((profit / total_revenue) * 100, 2) if total_revenue > 0 else 0

    # Calculate averages
    avg_revenue = round(df['Revenue'].mean(), 2)
    avg_expenses = round(df['Expenses'].mean(), 2)

    bar_fig = go.Figure(data=[
        go.Bar(name='Revenue', x=df['Month'], y=df['Revenue']),
        go.Bar(name='Expenses', x=df['Month'], y=df['Expenses'])
    ])
    bar_fig.update_layout(barmode='group', title='Monthly Revenue vs Expenses')
    bar_chart = pyo.plot(bar_fig, output_type='div', include_plotlyjs=True)

    pie_fig = go.Figure(data=[
        go.Pie(labels=['Profit', 'Expenses'], values=[profit, total_expenses])
    ])
    pie_chart = pyo.plot(pie_fig, output_type='div', include_plotlyjs=True)

    if 'Expense Category' in df.columns:
        category_totals = df.groupby('Expense Category')['Expenses'].sum()
        category_fig = go.Figure(data=[
            go.Pie(labels=category_totals.index, values=category_totals.values)
        ])
        category_fig.update_layout(title="🧾 Expense Breakdown by Category")
        category_chart = pyo.plot(category_fig, output_type='div', include_plotlyjs=False)
    else:
        category_chart = "<div class='alert alert-warning'>No 'Expense Category' data available.</div>"

    # Line Chart for Trends
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=df['Month'], y=df['Revenue'], mode='lines+markers', name='Revenue'))
    line_fig.add_trace(go.Scatter(x=df['Month'], y=df['Expenses'], mode='lines+markers', name='Expenses'))
    line_fig.update_layout(title='📉 Trend Over Time', xaxis_title='Month', yaxis_title='Amount')
    line_chart = pyo.plot(line_fig, output_type='div', include_plotlyjs=True)

    
    advice = []

    if profit_margin < 15:
        advice.append("💡 Profit margin is below 15%. Consider cutting costs or boosting revenue.")
    elif profit_margin < 25:
        advice.append("🟡 Profit margin is okay, but there's room for improvement.")
    else:
        advice.append("🟢 Excellent profit margin! Keep doing what works.")

    max_expense = df['Expenses'].max()
    max_month = df.loc[df['Expenses'].idxmax(), 'Month']
    if max_expense > (df['Expenses'].mean() * 1.5):
        advice.append(f"⚠️ Expenses were unusually high in {max_month}. Investigate large spending.")

    if df['Revenue'].iloc[-1] < df['Revenue'].iloc[0]:
        advice.append("📉 Revenue is trending down. Try boosting sales or marketing.")
    else:
        advice.append("📈 Revenue is trending upward. Well done!")

    revenue_diff = df['Revenue'].diff().fillna(0)
    if revenue_diff.abs().max() > df['Revenue'].mean() * 0.5:
        advice.append("🔍 Significant spike/drop in revenue detected — check for causes.")

    expenses_diff = df['Expenses'].diff().fillna(0)
    if expenses_diff.abs().max() > df['Expenses'].mean() * 0.5:
        advice.append("🔍 Significant change in expenses — investigate cost fluctuations.")

    final_advice = "<br>".join(advice)
    escaped_advice = quote(final_advice)
    table_html = df.to_html(classes='table table-bordered table-stripped',index=False)

    # Trend analysis
    trend_insights = []

    if df['Revenue'].is_monotonic_increasing:
        trend_insights.append("📈 Revenue has been consistently increasing.")
    elif df['Revenue'].is_monotonic_decreasing:
        trend_insights.append("📉 Revenue has been consistently decreasing.")
    else:
        trend_insights.append("↕️ Revenue shows fluctuations over time.")

    if df['Expenses'].max() > df['Expenses'].mean() * 1.5:
        spike_month = df.loc[df['Expenses'].idxmax(), 'Month']
        trend_insights.append(f"⚠️ Expense spike detected in {spike_month}.")
    else:
        trend_insights.append("✅ Expenses remain stable without major spikes.")

    return render_template(
        'results.html',
        revenue=total_revenue,
        expenses=total_expenses,
        profit=profit,
        margin=profit_margin,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        category_chart=category_chart,
        line_chart=line_chart,
        advice=final_advice,
        trend_insights=trend_insights,
        escaped_advice=escaped_advice,
        filename=filename,
        table_html=table_html,
        search_query=search_query,
        selected_category=category_filter
    
    )
def remove_emojis(text):
    return''.join(c for c in text if ord(c) < 128)

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.form
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Financial Report", ln=True, align='C')
        pdf.ln(10)

        pdf.cell(200, 10, txt=f"Total Revenue: {data['revenue']}", ln=True)
        pdf.cell(200, 10, txt=f"Total Expenses: {data['expenses']}", ln=True)
        pdf.cell(200, 10, txt=f"Profit: {data['profit']}", ln=True)
        pdf.cell(200, 10, txt=f"Profit Margin: {data['margin']}%", ln=True)
        pdf.ln(5)
        
        clean_advice=remove_emojis(data['advice'])
        
        pdf.multi_cell(0,10,txt="AI Advice:\n" + clean_advice)

        pdf_data=pdf.output(dest='S').encode('latin1')
        pdf_output=io.BytesIO(pdf_data)

        return send_file(pdf_output, download_name='financial_report.pdf', as_attachment=True)
    except Exception as e:
        return str(e), 500
    
@app.route('/preview/<filename>')
def preview_file(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    try:
        df = pd.read_csv(filepath)
        table_html = df.to_html(classes='table table-bordered table-striped', index=False)
    except Exception as e:
        return f"Error reading file: {e}", 500

    return render_template('preview.html', table_html=table_html, filename=filename)

@app.route('/download_excel/<filename>')
def download_excel(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    df = pd.read_csv(filepath)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Financial Data')
    output.seek(0)

    return send_file(output,
                     download_name='financial_data.xlsx',
                     as_attachment=True,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/download_csv/<filename>')
def download_csv(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    return send_file(filepath, as_attachment=True)

@app.route('/download_cleaned/<filename>')
def download_cleaned(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    try:
        df = pd.read_csv(filepath)

        # Clean the data
        df_cleaned = df.dropna()
        for col in ['Revenue', 'Expenses']:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        df_cleaned = df_cleaned.dropna()

        # Save to buffer
        cleaned_path = os.path.join(user_folder, f"cleaned_{filename}")
        df_cleaned.to_csv(cleaned_path, index=False)

        return send_file(cleaned_path, as_attachment=True)

    except Exception as e:
        return str(e), 500
    
@app.route('/send_summary', methods=['POST'])
def send_summary():
    if 'user' not in session:
        return redirect(url_for('login'))

    email = request.form.get('email')
    revenue = request.form.get('revenue')
    expenses = request.form.get('expenses')
    profit = request.form.get('profit')
    margin = request.form.get('margin')
    advice = request.form.get('advice')

    # Simulate email sending
    print(f"📨 Sending summary to: {email}")
    print("Summary content:")
    print(f"Revenue: {revenue}")
    print(f"Expenses: {expenses}")
    print(f"Profit: {profit}")
    print(f"Margin: {margin}%")
    print("Advice:", advice)

    flash("✅ Summary sent to your email (simulated).", "success")
    return redirect(url_for('dashboard'))

@app.route('/download_advice', methods=['POST'])
def download_advice():
    advice_text = request.form.get('advice', '')
    advice_file = io.BytesIO()
    advice_file.write(advice_text.encode('utf-8'))
    advice_file.seek(0)
    return send_file(advice_file, download_name='ai_advice.txt', as_attachment=True)

@app.route('/download_raw_pdf/<filename>', methods=['GET'])
def download_raw_pdf(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    df = pd.read_csv(filepath)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Raw Financial Data", ln=True, align='C')
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
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_stream = io.BytesIO(pdf_bytes)
    

    # Send file to browser as download
    return send_file(
        pdf_stream,
        download_name='raw_financial_data.pdf',
        as_attachment=True,
        mimetype='application/pdf'
    )

@app.route('/download_summary_txt', methods=['POST'])
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

    txt_bytes = summary_text.encode('utf-8')
    txt_stream = io.BytesIO(txt_bytes)
    txt_stream.seek(0)

    return send_file(
        txt_stream,
        as_attachment=True,
        download_name='financial_summary.txt',
        mimetype='text/plain'
    )

@app.route('/compare/<filenames>')
def compare_files(filenames):
    if 'user' not in session:
        return redirect(url_for('login'))

    files = filenames.split(',')
    summaries = []
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])

    for file in files:
        path = os.path.join(user_folder, file)
        try:
            df = pd.read_csv(path)
            total_revenue = df['Revenue'].sum()
            total_expenses = df['Expenses'].sum()
            profit = total_revenue - total_expenses
            margin = round((profit / total_revenue) * 100, 2) if total_revenue > 0 else 0

            summaries.append({
                'filename': file,
                'revenue': total_revenue,
                'expenses': total_expenses,
                'profit': profit,
                'margin': margin
            })
        except:
            summaries.append({'filename': file, 'error': 'Could not process file'})

    return render_template('compare.html', summaries=summaries)

@app.route('/download_comparison_csv')
def download_comparison_csv():
    if 'comparison_data' not in session:
        return "No comparison data to download", 400

    df = pd.read_json(session['comparison_data'])
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name='comparison.csv'
    )

@app.route('/download_comparison_pdf')
def download_comparison_pdf():
    if 'comparison_data' not in session:
        return "No comparison data to download", 400

    df = pd.read_json(session['comparison_data'])

    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Comparison Report", ln=True, align='C')
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
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(pdf_output, download_name='comparison.pdf', as_attachment=True)

@app.route('/download_bar_chart')
def download_bar_chart():
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Dummy data (you can replace this with real data)
    months = ['Jan', 'Feb', 'Mar', 'Apr']
    revenues = [10000, 15000, 20000, 18000]
    expenses = [7000, 9000, 11000, 10500]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(months))

    plt.bar(index, revenues, bar_width, label='Revenue', color='green')
    plt.bar([i + bar_width for i in index], expenses, bar_width, label='Expenses', color='red')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.title('Revenue vs Expenses')
    plt.xticks([i + bar_width / 2 for i in index], months)
    plt.legend()

    from io import BytesIO
    import os
    from flask import send_file

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='bar_chart.png')

@app.route('/download_line_chart')
def download_line_chart():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    from flask import send_file

    # Dummy data (replace with your actual trend data)
    months = ['Jan', 'Feb', 'Mar', 'Apr']
    revenues = [10000, 15000, 20000, 18000]

    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(months, revenues, marker='o', linestyle='-', color='blue', label='Revenue')
    plt.title('Revenue Trend Over Time')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.legend()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='line_chart.png')

@app.route('/download_pie_chart')
def download_pie_chart():
    import matplotlib.pyplot as plt
    from io import BytesIO
    from flask import send_file

    # Dummy data (replace with your actual category data)
    labels = ['Marketing', 'Sales', 'Operations', 'R&D']
    expenses = [4000, 3000, 2500, 1500]

    plt.figure(figsize=(6, 6))
    plt.pie(expenses, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Expense Distribution by Category')

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='pie_chart.png')

@app.route('/download_full_report')
def download_full_report():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    from io import BytesIO
    from flask import send_file
    from fpdf import FPDF
    import os

    # Example data (replace with actual session or DB data as needed)
    months = ['Jan', 'Feb', 'Mar', 'Apr']
    revenues = [10000, 12000, 14000, 13000]
    expenses = [8000, 8500, 9000, 9500]
    advice = "Consider reducing operations cost in March and investing more in marketing."

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
    pdf.cell(200, 10, txt="Financial Summary Report", ln=True, align='C')
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
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_bytes)
    pdf_buffer.seek(0)

    # Clean up
    os.remove(img_path)

    return send_file(pdf_buffer, as_attachment=True, download_name="full_dashboard_report.pdf", mimetype='application/pdf')

@app.route('/send_report', methods=['POST'])
def send_report():
    email = request.form.get('email')
    if not email:
        flash("No email provided.", "danger")
        return redirect(url_for('dashboard'))
    try:
        flash(f"report sent to {email} successfully.","success")
    except Exception as e:
        print("Error:",e)
        flash("An error occured while sending the email,", "danger")
    return redirect(url_for('dashboard'))

    # Example PDF content
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Financial Report Summary", ln=True, align='C')
    pdf.cell(200, 10, txt="Generated via Profit Optimizer", ln=True, align='C')

    # Save PDF to memory (not disk)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf.buffer = io.BytesIO(pdf_bytes)
    pdf_buffer.seek(0)  # Go to start of BytesIO stream

    # Compose email
    msg = EmailMessage()
    msg['Subject'] = 'Your Financial Report'
    msg['From'] = 'griffinnnnn77@gmail.com'
    msg['To'] = recipient_email
    msg.set_content('Please find attached your financial report.')

    # Attach the PDF (with filename)
    msg.add_attachment(pdf_output.read(), maintype='application', subtype='pdf', filename='report.pdf')

    # Send the email using Gmail’s SMTP server
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('griffinnnnn77@gmail.com', 'abcdefghijklmnop')
        smtp.send_message(msg)

    return 'Report sent successfully!'

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    uploaded_files = uploaded_csvs.get(username, [])

    return render_template('profile.html', uploaded_files=uploaded_files)

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    username = session['user']
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
    file_path = os.path.join(user_folder, filename)

    # Remove from disk
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from tracking list
    if username in uploaded_csvs:
        if filename in uploaded_csvs[username]:
            uploaded_csvs[username].remove(filename)

    flash(f"{filename} deleted successfully.")
    return redirect(url_for('profile'))
@app.route('/rename_file', methods=['POST'])
def rename_file():
    if 'user_id' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))

    old_filename = request.form['old_filename']
    new_filename = request.form['new_filename']

    username = session.get('username')
    user_folder = os.path.join('uploads', username)

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

    return redirect(url_for('profile'))

@app.route('/advisor', methods=['GET', 'POST'])
def advisor():
    if 'username' not in session:
        return redirect(url_for('login'))

    answer = None

    if request.method == 'POST':
        question = request.form.get('question')
        username = session['username']
        files = uploaded_csvs.get(username, [])
        if not files:
            answer = "No uploaded CSV files found."
        else:
            # We'll use the latest uploaded file for simplicity
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], username, files[-1])
            df = pd.read_csv(file_path)

            # Simple rule-based responses
            if 'highest revenue' in question.lower():
                row = df.loc[df['Revenue'].idxmax()]
                answer = f"The highest revenue was in {row['Month']} with ${row['Revenue']}"
            elif 'average expense' in question.lower():
                avg_exp = df['Expenses'].mean()
                answer = f"The average expense is ${avg_exp:.2f}"
            elif 'lowest revenue' in question.lower():
                row = df.loc[df['Revenue'].idxmin()]
                answer = f"The lowest revenue was in {row['Month']} with ${row['Revenue']}"
            else:
                answer = "Sorry, I didn't understand the question."

    return render_template('advisor.html', answer=answer)

@app.route('/add_entry/<filename>', methods=['GET', 'POST'])
def add_entry(filename):
    if 'user' not in session:
        return redirect(url_for('login'))

    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user'])
    filepath = os.path.join(user_folder, filename)

    if not os.path.exists(filepath):
        return "CSV file not found.", 404

    if request.method == 'POST':
        # Get form values
        month = request.form['month']
        revenue = request.form['revenue']
        expenses = request.form['expenses']
        description = request.form['description']  # optional

        # Append new row to CSV
        with open(filepath, 'a') as f:
            line = f"{month},{revenue},{expenses},{description}\n"
            f.write(line)

        return redirect(url_for('view_file', filename=filename))

    return render_template('add_entry.html', filename=filename)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    answer = None

    if request.method == "POST":
        question = request.form.get("question")

        if question:
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": question}
                    ]
                )

                answer = response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI Error: {e}")
                answer = f"Error: {str(e)}"

    return render_template("ask.html", answer=answer)

@app.route("/admin")
def admin():
    if "user" not in session or session["user"] != ["griff","teresia","zachary","mutuma"]
        return "Unauthorized", 403
    with open("users.json","r")as f:
        users = json.load(f)
    
    total_users = len(users)

    # Later we’ll add real stats here
    return render_template("admin.html", user=session["user"],total_users=len(users),user_list=list(users.keys()))

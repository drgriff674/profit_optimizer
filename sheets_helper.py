import gspread
from google.oauth2.service_account import Credentials

# Path to your downloaded JSON key file
SERVICE_ACCOUNT_FILE = "profit-optimizer-sheets.json"  # <-- update filename if different

# Define the scopes (permissions)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Authenticate
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# Open your Google Sheet by name (rename this to your actual sheet name)
SHEET_NAME = "ProfitOptimizerData"
sheet = client.open(SHEET_NAME).sheet1

def read_data():
    """Read all rows from the sheet"""
    return sheet.get_all_records()

def add_row(data):
    """Append a new row"""
    sheet.append_row(data)

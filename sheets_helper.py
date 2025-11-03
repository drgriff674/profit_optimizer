import os
import json
import gspread
from google.oauth2.service_account import Credentials

# ✅ Load Google credentials safely from environment or local file
try:
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

    if credentials_json:
        # From environment (Render / Railway)
        credentials_info = json.loads(credentials_json)
    else:
        # From local file (for local testing)
        with open("instance/google-credentials.json", "r") as f:
            credentials_info = json.load(f)

except Exception as e:
    raise RuntimeError(f"❌ Failed to load Google credentials: {e}")

# ✅ Scopes for Google Sheets and Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Authenticate and authorize
creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
client = gspread.authorize(creds)

# ✅ Your Google Sheet ID
SHEET_ID = "1vL-wLddVruVCIyjgB35Ej66UsbdnoIZkrlIe4_eK4xs"

def get_sheet(sheet_name="Sheet1"):
    """Open a specific sheet by name, fallback to Sheet1."""
    spreadsheet = client.open_by_key(SHEET_ID)
    try:
        return spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.sheet1

def read_data(sheet_name="Sheet1"):
    """Read all rows from a specific sheet"""
    sheet = get_sheet(sheet_name)
    return sheet.get_all_records()

def add_row(data, sheet_name="Sheet1"):
    """Append a new row to a specific sheet"""
    sheet = get_sheet(sheet_name)
    sheet.append_row(data)

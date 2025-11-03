import os
import json
import gspread
from google.oauth2.service_account import Credentials

# Load credentials from environment variable
credentials_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

# Define the correct scopes (for Sheets + Drive access)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Authenticate with Google
creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
client = gspread.authorize(creds)

# ✅ Your Google Sheet ID
SHEET_ID = "1vL-wLddVruVCIyjgB35Ej66UsbdnoIZkrlIe4_eK4xs"

def get_sheet(sheet_name="Sheet1"):
    """
    Open a specific sheet by name.
    Defaults to 'Sheet1' if no name is provided.
    """
    spreadsheet = client.open_by_key(SHEET_ID)
    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # Fallback to the first sheet if the given one doesn't exist
        sheet = spreadsheet.sheet1
    return sheet

def read_data(sheet_name="Sheet1"):
    """Read all rows from a specific sheet"""
    sheet = get_sheet(sheet_name)
    return sheet.get_all_records()

def add_row(data, sheet_name="Sheet1"):
    """Append a new row to a specific sheet"""
    sheet = get_sheet(sheet_name)
    sheet.append_row(data)

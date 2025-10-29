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

# ✅ Use your spreadsheet ID (replace this with your actual ID!)
SHEET_ID = "1vL-wLddVruVCIyjgB35Ej66UsbdnoIZkrlIe4_eK4xs/edit?gid=0#gid=0"

# Open the sheet by key (ID)
sheet = client.open_by_key(SHEET_ID).sheet1

def read_data():
    """Read all rows from the sheet"""
    return sheet.get_all_records()

def add_row(data):
    """Append a new row"""
    sheet.append_row(data)

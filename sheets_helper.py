import os
import json
import gspread
from google.oauth2.service_account import Credentials

# Load the Google credentials JSON from environment variable
# (You’ll add this in Render dashboard as GOOGLE_APPLICATION_CREDENTIALS_JSON)
credentials_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

# Define Google Sheets API scopes (permissions)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Create credentials object
creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)

# Authorize the gspread client
client = gspread.authorize(creds)

# Open your Google Sheet by name — update this to match your sheet’s name
SHEET_NAME = "ProfitOptimizerData"
sheet = client.open(SHEET_NAME).sheet1

def read_data():
    """Read all rows from the sheet"""
    return sheet.get_all_records()

def add_row(data):
    """Append a new row"""
    sheet.append_row(data)

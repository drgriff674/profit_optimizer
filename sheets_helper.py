import gspread
import os, json
from google.oauth2.service_account import Credentials

credentials_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
client = gspread.authorize(creds)

SHEET_NAME = "1vL-wLddVruVCIyjgB35Ej66UsbdnoIZkrlIe4_eK4xs/edit?gid=0#gid=0"  # your Google Sheet name
sheet = client.open_by_key(SHEET_ID).sheet1
def read_data():
    """Read all rows from the sheet"""
    return sheet.get_all_records()

def add_row(data):
    """Append a new row"""
    sheet.append_row(data)

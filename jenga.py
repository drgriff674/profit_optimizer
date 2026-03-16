import requests
import os

JENGA_API_KEY = os.environ.get("JENGA_API_KEY")
JENGA_API_SECRET = os.environ.get("JENGA_API_SECRET")
JENGA_MERCHANT_CODE = os.environ.get("JENGA_MERCHANT_CODE")

def get_access_token():

    url = "https://v3.jengahq.io/identity/v3/token"

    headers = {
        "Content-Type": "application/json",
        "consumer-key": JENGA_API_KEY,
        "consumer-secret": JENGA_API_SECRET
    }

    response = requests.get(url, headers=headers)

    return response.text

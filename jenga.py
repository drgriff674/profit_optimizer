import requests
import os

JENGA_API_KEY = os.environ.get("JENGA_API_KEY")
JENGA_API_SECRET = os.environ.get("JENGA_API_SECRET")
JENGA_MERCHANT_CODE = os.environ.get("JENGA_MERCHANT_CODE")

def get_access_token():

    url = "https://sandbox.jengahq.io/identity/v3/token"

    payload = {
        "merchantCode": JENGA_MERCHANT_CODE,
        "consumerKey": JENGA_API_KEY,
        "consumerSecret": JENGA_API_SECRET
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()

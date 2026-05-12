import resend
import os

resend.api_key = os.getenv("RESEND_API_KEY")

def send_email(to_email, subject, html_content):

    params = {
        "from": "OptiGain Security <security@optigainapp.com>",
        "to": to_email,
        "subject": subject,
        "html": html_content,
    }

    email = resend.Emails.send(params)

    return email

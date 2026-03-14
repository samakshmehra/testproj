import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE_NUMBER")
to_number = os.getenv("ALERT_PHONE_NUMBER")

# Replace with your ngrok URL
NGROK_URL = "https://your-ngrok-url.ngrok-free.app"

client = Client(account_sid, auth_token)

call = client.calls.create(
    url=f"{NGROK_URL}/voice",
    to=to_number,
    from_=from_number
)

print(f"Grievance call initiated: {call.sid}")
print(f"Calling {to_number}...")

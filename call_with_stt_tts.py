import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE_NUMBER")
to_number = os.getenv("ALERT_PHONE_NUMBER")

client = Client(account_sid, auth_token)

# Make call with custom TTS message
call = client.calls.create(
    twiml='<Response><Say>Hello! This is an automated call. Please respond.</Say><Record transcribe="true" maxLength="30"/></Response>',
    to=to_number,
    from_=from_number
)

print(f"Call initiated: {call.sid}")

import os
from twilio.rest import Client
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class TwilioAlert:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.to_number = os.getenv("ALERT_PHONE_NUMBER")
        
        if not all([self.account_sid, self.auth_token, self.from_number, self.to_number]):
            logger.warning("Twilio credentials not configured - alerts disabled")
            self.client = None
            return
        
        self.client = Client(self.account_sid, self.auth_token)
    
    def make_accident_call(self, description: str = "An accident has been detected"):
        """Make a phone call to alert about an accident"""
        if not self.client:
            logger.warning("Twilio not configured - skipping call")
            return None
        
        try:
            call = self.client.calls.create(
                twiml=f'<Response><Say>{description}. Please check the surveillance system immediately.</Say></Response>',
                to=self.to_number,
                from_=self.from_number
            )
            logger.info(f"Accident call made: {call.sid}")
            return call.sid
        except Exception as e:
            logger.error(f"Failed to make call: {e}")
            return None
    
    def send_sms(self, message: str):
        """Send SMS alert"""
        if not self.client:
            logger.warning("Twilio not configured - skipping SMS")
            return None
        
        try:
            sms = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            logger.info(f"SMS sent: {sms.sid}")
            return sms.sid
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return None

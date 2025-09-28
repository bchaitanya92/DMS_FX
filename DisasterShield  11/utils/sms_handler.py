import os
import logging
from twilio.rest import Client
from datetime import datetime

logger = logging.getLogger(__name__)

class SMSHandler:
    def __init__(self):
        """Initialize SMS handler with Twilio credentials"""
        self.account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        self.auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        self.from_number = os.environ.get('TWILIO_PHONE_NUMBER')
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio credentials not found in environment variables")
            self.client = None
        else:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("SMS handler initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SMS handler: {str(e)}")
                self.client = None

    def send_alert(self, to_number, message, alert_id=None):
        """Send an alert SMS to the specified number"""
        if self.client is None:
            logger.warning("SMS handler not initialized - skipping alert")
            return None

        try:
            # Add alert ID to message if provided
            if alert_id:
                message = f"{message}\nAlert ID: {alert_id}"

            # Send message using Twilio
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            logger.info(f"Alert SMS sent successfully to {to_number}")
            return message.sid
        except Exception as e:
            logger.error(f"Failed to send alert SMS to {to_number}: {str(e)}")
            return None

    def send_confirmation(self, to_number, alert_id):
        """Send a confirmation message to acknowledge receipt of alert"""
        if self.client is None:
            logger.warning("SMS handler not initialized - skipping confirmation")
            return None

        try:
            message = (
                f"Thank you for confirming receipt of alert {alert_id}.\n"
                f"Stay safe and follow all emergency guidelines."
            )
            
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            logger.info(f"Confirmation SMS sent successfully to {to_number}")
            return message.sid
        except Exception as e:
            logger.error(f"Failed to send confirmation SMS to {to_number}: {str(e)}")
            return None 
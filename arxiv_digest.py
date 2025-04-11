import os

# OpenAI API configuration
OPENAI_CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "model": "gpt-4o-mini",
    "use_ai_summaries": True,
}

# Email configuration
EMAIL_CONFIG = {
    "send_email": True,
    "sender_email": os.environ.get("EMAIL_ADDRESS"),
    "sender_password": os.environ.get("EMAIL_PASSWORD"),
    "recipient_email": os.environ.get("RECIPIENT_EMAIL"),
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 465,
}

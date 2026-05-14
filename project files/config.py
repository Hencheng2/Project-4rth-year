# config.py

import os

# --- Flask Secret Key ---
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', '09da35833ef9cb699888f08d66a0cfb827fb10e53f6c1549')

# --- Database Configuration ---
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///parish.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False


# --- Admin User Credentials ---
ADMIN_USERNAME = "Henry"
ADMIN_PASSWORD_RAW = "Dec@2003"
ADMIN_EMAIL = 'hochieng86@gmail.com'

# --- Mail Configuration ---
MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', 'false').lower() in ['true', 'on', '1']
MAIL_USERNAME = os.environ.get('MAIL_USERNAME', 'hochieng86@gmail.com')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', 'ozwuasguuuotojgs')
MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'hochieng86@gmail.com')


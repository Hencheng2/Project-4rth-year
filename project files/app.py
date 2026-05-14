import os
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import hashlib
import base64
from functools import wraps
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import io
import csv
import warnings
import schedule
import logging
from contextlib import contextmanager

# Suppress TensorFlow warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import joblib

# Statsmodels for comparison models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# For PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

import requests
import re as _re

# Import configuration
import config

def _make_model_key(category, region):
    """Sanitize category/region into a filesystem-safe model key."""
    raw = f"{category}_{region}".lower()
    # Replace any character that isn't alphanumeric or underscore with underscore
    key = _re.sub(r'[^a-z0-9]+', '_', raw)
    # Collapse multiple underscores and strip leading/trailing ones
    key = _re.sub(r'_+', '_', key).strip('_')
    return key

# Configure logging - ACTIVITY LOGS ONLY (no debug)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress debug logs from other libraries
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('reportlab').setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Session configuration
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['DATABASE'] = 'sales_forecast.db'
app.config['MODEL_PATH'] = 'models/'
app.config['SCALER_PATH'] = 'scalers/'
app.config['REPORT_PATH'] = 'reports/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['MODEL_PATH'], exist_ok=True)
os.makedirs(app.config['SCALER_PATH'], exist_ok=True)
os.makedirs(app.config['REPORT_PATH'], exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# Email configuration from config.py
EMAIL_CONFIG = {
    'server': config.MAIL_SERVER,
    'port': config.MAIL_PORT,
    'use_tls': config.MAIL_USE_TLS,
    'use_ssl': config.MAIL_USE_SSL,
    'username': config.MAIL_USERNAME,
    'password': config.MAIL_PASSWORD,
    'sender': config.MAIL_DEFAULT_SENDER
}

# Pollinations AI Configuration
POLLINATIONS_API_URL = "https://text.pollinations.ai/"

# ============================================================================
# DATABASE CONNECTION WITH LOCK HANDLING
# ============================================================================

@contextmanager
def get_db_connection():
    """Context manager for database connections with WAL mode and retry logic"""
    conn = None
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(app.config['DATABASE'], timeout=30.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-20000")
            conn.execute("PRAGMA temp_store=MEMORY")
            yield conn
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            if conn:
                conn.rollback()
                conn.close()
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            raise
        finally:
            if conn:
                conn.close()

def get_db():
    """Legacy function for compatibility - use get_db_connection() for new code"""
    return get_db_connection().__enter__()

# ============================================================================
# DATABASE SCHEMA (Embedded)
# ============================================================================

DB_SCHEMA = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(200) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sales table
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    region VARCHAR(100) NOT NULL,
    units_sold INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_sales DECIMAL(12,2) NOT NULL,
    promotion_flag BOOLEAN DEFAULT 0,
    holiday_flag BOOLEAN DEFAULT 0,
    discount_percent DECIMAL(5,2) DEFAULT 0.0,
    stock_level INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date DATE NOT NULL,
    forecast_date DATE NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    region VARCHAR(100) NOT NULL,
    predicted_sales DECIMAL(12,2) NOT NULL,
    predicted_units INTEGER NOT NULL,
    confidence_interval_lower DECIMAL(12,2),
    confidence_interval_upper DECIMAL(12,2),
    model_type VARCHAR(50) NOT NULL,
    is_primary INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(200) NOT NULL,
    mae DECIMAL(10,4),
    rmse DECIMAL(10,4),
    mape DECIMAL(10,4),
    r2_score DECIMAL(10,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trends table
CREATE TABLE IF NOT EXISTS trends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trend_type VARCHAR(50) NOT NULL,
    trend_value DECIMAL(10,4),
    trend_direction VARCHAR(20),
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scheduled reports table
CREATE TABLE IF NOT EXISTS scheduled_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(200) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    frequency VARCHAR(20) NOT NULL,
    day VARCHAR(20),
    time TIME NOT NULL,
    recipients TEXT NOT NULL,
    format VARCHAR(10) NOT NULL,
    active BOOLEAN DEFAULT 1,
    last_run TIMESTAMP,
    next_run TIMESTAMP,
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Anomalies table
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_date DATE NOT NULL,
    anomaly_date DATE NOT NULL,
    product_category VARCHAR(100),
    region VARCHAR(100),
    sales_value DECIMAL(12,2),
    units_value INTEGER,
    detection_method VARCHAR(50),
    explanation TEXT,
    reviewed BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLM Explanations cache table
CREATE TABLE IF NOT EXISTS llm_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_type VARCHAR(50) NOT NULL,
    context_id VARCHAR(200),
    explanation TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User activity log
CREATE TABLE IF NOT EXISTS user_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action VARCHAR(100) NOT NULL,
    details TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- What-if scenarios table
CREATE TABLE IF NOT EXISTS what_if_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    base_category VARCHAR(100),
    base_region VARCHAR(100),
    parameters TEXT NOT NULL,
    results TEXT,
    created_by INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_sales_category ON sales(product_category);
CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(forecast_date);
CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions(product_category);
CREATE INDEX IF NOT EXISTS idx_sales_promotion ON sales(promotion_flag);
CREATE INDEX IF NOT EXISTS idx_sales_holiday ON sales(holiday_flag);
CREATE INDEX IF NOT EXISTS idx_anomalies_date ON anomalies(anomaly_date);
CREATE INDEX IF NOT EXISTS idx_scheduled_next ON scheduled_reports(next_run);
CREATE INDEX IF NOT EXISTS idx_model_metrics_date ON model_metrics(training_date);
CREATE INDEX IF NOT EXISTS idx_trends_type ON trends(trend_type);
CREATE INDEX IF NOT EXISTS idx_llm_context ON llm_explanations(context_type, context_id);
"""

def init_db():
    """Initialize database with all tables and default admin user"""
    try:
        logger.info("📁 Initializing database...")
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Execute schema
            cursor.executescript(DB_SCHEMA)
            
            # Check if admin user exists
            admin = cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',)).fetchone()
            if not admin:
                password_hash = generate_password_hash('admin123')
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, role) 
                    VALUES (?, ?, ?, ?)
                ''', ('admin', 'admin@example.com', password_hash, 'admin'))
                logger.info("👤 Default admin user created: admin / admin123")

        # Migration: add new columns inside a fresh connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Check existing columns
            existing = {row[1] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()}

            if 'forecast_session_id' not in existing:
                cursor.execute("ALTER TABLE predictions ADD COLUMN forecast_session_id VARCHAR(40)")
                logger.info("✅ Migration: added forecast_session_id column")

            if 'forecast_days' not in existing:
                cursor.execute("ALTER TABLE predictions ADD COLUMN forecast_days INTEGER DEFAULT 0")
                logger.info("✅ Migration: added forecast_days column")

            if 'is_primary' not in existing:
                cursor.execute("ALTER TABLE predictions ADD COLUMN is_primary INTEGER DEFAULT 1")
                logger.info("✅ Migration: added is_primary column")

            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(forecast_session_id)")
            except Exception:
                pass

        logger.info("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization error: {str(e)}")
        return False

# ============================================================================
# LLM INTEGRATION (Pollinations AI)
# ============================================================================

class LLMExplainer:
    def __init__(self):
        self.api_url = POLLINATIONS_API_URL
    
    def generate_explanation(self, prompt, max_length=500):
        """Generate text explanation using Pollinations AI"""
        try:
            logger.info("🤖 Generating AI explanation...")
            full_prompt = f"{prompt}\n\nProvide a concise, professional explanation suitable for business decision-makers."
            
            response = requests.get(
                f"{self.api_url}{full_prompt}",
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ AI explanation generated successfully")
                return response.text
            else:
                logger.warning(f"⚠️ AI explanation failed with status: {response.status_code}")
                return f"Unable to generate explanation. Status: {response.status_code}"
        except Exception as e:
            logger.warning(f"⚠️ AI explanation error: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def explain_forecast(self, category, region, forecast_data, historical_data):
        """Generate natural language explanation for forecast"""
        logger.info(f"📊 Generating forecast explanation for {category} / {region}")
        prompt = f"""
        As a retail sales analyst, explain the sales forecast for {category} in the {region} region.
        
        Historical Sales Summary:
        - Average daily sales: ${historical_data['avg_sales']:.2f}
        - Sales trend: {historical_data['trend']}
        - Seasonal pattern: {historical_data['seasonality']}
        
        Forecast Summary (next {len(forecast_data['predictions'])} days):
        - Total predicted sales: ${sum(forecast_data['predictions']):.2f}
        - Expected growth: {forecast_data['growth_rate']:.1f}%
        - Peak predicted day: ${max(forecast_data['predictions']):.2f}
        - Confidence level: {forecast_data['avg_confidence']:.1f}%
        
        Provide a clear explanation of what these numbers mean for business decisions, 
        highlighting any important patterns, risks, or opportunities.
        """
        return self.generate_explanation(prompt)
    
    def explain_trends(self, trends_data, analysis_period):
        """Generate natural language explanation for trends analysis"""
        logger.info(f"📈 Generating trends explanation for {analysis_period}")
        prompt = f"""
        As a retail business analyst, explain the key trends identified in our sales data for the {analysis_period}.
        
        Key Findings:
        - Overall growth rate: {trends_data['growth_rate']:.1f}%
        - Most volatile category: {trends_data['most_volatile']}
        - Best performing day: {trends_data['best_day']}
        - Seasonal strength: {trends_data['seasonal_strength']:.1f}%
        - Peak sales periods: {trends_data['peak_periods']}
        
        Provide actionable insights for inventory planning, staffing, and marketing strategies.
        """
        return self.generate_explanation(prompt)
    
    def explain_model_performance(self, model_metrics, comparison_metrics=None):
        """Generate explanation of model performance"""
        logger.info("📉 Generating model performance explanation")
        prompt = f"""
        As a data science consultant, explain the performance of our sales forecasting model.
        
        Current Model Metrics:
        - MAE: {model_metrics['mae']:.2f}
        - RMSE: {model_metrics['rmse']:.2f}
        - MAPE: {model_metrics['mape']:.1f}%
        - R² Score: {model_metrics['r2']:.3f}
        """
        
        if comparison_metrics:
            prompt += f"""
            
            Comparison with Classical Models:
            - ARIMA MAE: {comparison_metrics['arima']['mae']:.2f}
            - Exponential Smoothing MAE: {comparison_metrics['exp_smoothing']['mae']:.2f}
            - LSTM improvement: {comparison_metrics['improvement']:.1f}%
            """
        
        prompt += """
        Explain what these metrics mean for forecast reliability and business decisions.
        """
        
        return self.generate_explanation(prompt)
    
    def generate_recommendations(self, analysis_data):
        """Generate business recommendations based on analysis"""
        logger.info("💡 Generating business recommendations")
        prompt = f"""
        Based on our sales analysis, provide 5 specific, actionable recommendations for:
        
        Sales Performance:
        - Total sales: ${analysis_data['total_sales']:,.2f}
        - Growth rate: {analysis_data['growth_rate']:.1f}%
        
        Inventory Optimization:
        - Best selling categories: {analysis_data['top_categories']}
        - Seasonal patterns: {analysis_data['seasonal_patterns']}
        
        Provide numbered recommendations with expected impact and implementation timeline.
        """
        return self.generate_explanation(prompt)

# ============================================================================
# EMAIL SERVICE
# ============================================================================

class EmailService:
    def __init__(self, config):
        self.config = config
    
    def send_email(self, to_emails, subject, body, attachments=None):
        """Send email with optional attachments"""
        try:
            logger.info(f"📧 Sending email to {to_emails if isinstance(to_emails, str) else ', '.join(to_emails)} - Subject: {subject}")
            msg = MIMEMultipart()
            msg['From'] = self.config['sender']
            msg['To'] = ', '.join(to_emails) if isinstance(to_emails, list) else to_emails
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            if attachments:
                for attachment in attachments:
                    part = MIMEApplication(attachment['content'], Name=attachment['filename'])
                    part['Content-Disposition'] = f'attachment; filename="{attachment["filename"]}"'
                    msg.attach(part)
            
            if self.config['use_ssl']:
                server = smtplib.SMTP_SSL(self.config['server'], self.config['port'])
            else:
                server = smtplib.SMTP(self.config['server'], self.config['port'])
                if self.config['use_tls']:
                    server.starttls()
            
            server.login(self.config['username'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("✅ Email sent successfully")
            return True, "Email sent successfully"
        except Exception as e:
            logger.error(f"❌ Email sending failed: {str(e)}")
            return False, str(e)
        
    def send_report_with_attachment(self, to_emails, subject, body_html, filepath, filename):
        """Send email with file attachment"""
        try:
            logger.info(f"📎 Sending report with attachment to {to_emails if isinstance(to_emails, str) else ', '.join(to_emails)}")
            msg = MIMEMultipart()
            msg['From'] = self.config['sender']
            msg['To'] = ', '.join(to_emails) if isinstance(to_emails, list) else to_emails
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body_html, 'html'))
            
            # Attach file
            with open(filepath, 'rb') as f:
                attachment = MIMEApplication(f.read(), Name=filename)
                attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
                msg.attach(attachment)
            
            if self.config['use_ssl']:
                server = smtplib.SMTP_SSL(self.config['server'], self.config['port'])
            else:
                server = smtplib.SMTP(self.config['server'], self.config['port'])
                if self.config['use_tls']:
                    server.starttls()
            
            server.login(self.config['username'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("✅ Report email sent successfully")
            return True, "Email sent successfully"
        except Exception as e:
            logger.error(f"❌ Report email failed: {str(e)}")
            return False, str(e)
    
    def send_report(self, to_emails, report_name, report_data, report_format='pdf'):
        """Send report via email"""
        subject = f"Sales Forecast Report: {report_name}"
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Sales Forecasting System</h1>
                <p>{report_name} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            <div class="content">
                <p>Dear User,</p>
                <p>Your requested report "{report_name}" is attached to this email.</p>
                
                <h3>Report Summary:</h3>
                <div class="metric">
                    <strong>Report Type:</strong> {report_data.get('type', 'N/A')}<br>
                    <strong>Date Range:</strong> {report_data.get('date_range', 'N/A')}<br>
                    <strong>Generated By:</strong> {report_data.get('generated_by', 'System')}
                </div>
                
                <p>Please find the complete report attached.</p>
                
                <p>Best regards,<br>Sales Forecasting AI System</p>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to_emails, subject, body)

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.report_path = app.config['REPORT_PATH']
    
    def generate_sales_report(self, data, date_range, format='pdf'):
        """Generate comprehensive sales report"""
        logger.info(f"📄 Generating {format.upper()} report for period {date_range['start']} to {date_range['end']}")
        if format == 'pdf':
            return self._generate_pdf_sales_report(data, date_range)
        elif format == 'excel':
            return self._generate_excel_sales_report(data, date_range)
        elif format == 'csv':
            return self._generate_csv_sales_report(data, date_range)
        else:
            return None, "Unsupported format"
    
    def _generate_pdf_sales_report(self, data, date_range):
        """Generate PDF sales report"""
        filename = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.report_path, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        elements = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        elements.append(Paragraph("Sales Performance Report", title_style))
        
        date_style = ParagraphStyle(
            'DateStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        elements.append(Paragraph(f"Period: {date_range['start']} to {date_range['end']}", date_style))
        elements.append(Spacer(1, 20))
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Sales', f"${data['summary']['total_sales']:,.2f}"],
            ['Total Units', f"{data['summary']['total_units']:,}"],
            ['Average Daily Sales', f"${data['summary']['avg_daily']:,.2f}"],
            ['Growth Rate', f"{data['summary']['growth_rate']:.1f}%"],
            ['Best Category', data['summary']['best_category']],
            ['Best Region', data['summary']['best_region']]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(Paragraph("Key Performance Metrics", self.styles['Heading2']))
        elements.append(Spacer(1, 10))
        elements.append(metrics_table)
        elements.append(Spacer(1, 30))
        
        if 'categories' in data:
            elements.append(Paragraph("Category Performance", self.styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            cat_data = [['Category', 'Sales', 'Units', 'Share']]
            for cat in data['categories'][:10]:
                cat_data.append([
                    cat['name'],
                    f"${cat['sales']:,.2f}",
                    f"{cat['units']:,}",
                    f"{cat['share']:.1f}%"
                ])
            
            cat_table = Table(cat_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch])
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(cat_table)
        
        doc.build(elements)
        
        logger.info(f"✅ PDF report generated: {filename}")
        return filepath, filename
    
    def _generate_excel_sales_report(self, data, date_range):
        """Generate Excel report using pandas"""
        filename = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(self.report_path, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary_df = pd.DataFrame([{
                'Metric': 'Total Sales',
                'Value': data['summary']['total_sales']
            }, {
                'Metric': 'Total Units',
                'Value': data['summary']['total_units']
            }, {
                'Metric': 'Average Daily Sales',
                'Value': data['summary']['avg_daily']
            }, {
                'Metric': 'Growth Rate (%)',
                'Value': data['summary']['growth_rate']
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            if 'daily' in data:
                daily_df = pd.DataFrame(data['daily'])
                daily_df.to_excel(writer, sheet_name='Daily Sales', index=False)
            
            if 'categories' in data:
                cat_df = pd.DataFrame(data['categories'])
                cat_df.to_excel(writer, sheet_name='Categories', index=False)
            
            if 'regions' in data:
                region_df = pd.DataFrame(data['regions'])
                region_df.to_excel(writer, sheet_name='Regions', index=False)
        
        logger.info(f"✅ Excel report generated: {filename}")
        return filepath, filename
    
    def _generate_csv_sales_report(self, data, date_range):
        """Generate CSV report"""
        filename = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.report_path, filename)
        
        if 'daily' in data:
            df = pd.DataFrame(data['daily'])
            df.to_csv(filepath, index=False)
        
        logger.info(f"✅ CSV report generated: {filename}")
        return filepath, filename

# ============================================================================
# DATA PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        
    def create_features(self, df):
        """Create comprehensive time-based and lag features"""
        logger.info("🔧 Creating time-based and lag features for dataset")
        df = df.copy()
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Cyclical encoding
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df['total_sales'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df['total_sales'].rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df['total_sales'].rolling(window=window).std()
        
        # Difference features
        df['sales_diff_1'] = df['total_sales'].diff(1)
        df['sales_diff_7'] = df['total_sales'].diff(7)
        df['sales_pct_change_1'] = df['total_sales'].pct_change(1) * 100
        
        # Promotional impact
        df['promo_impact'] = df['promotion_flag'] * df['discount_percent']
        
        # Fill NaN values
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        logger.info(f"✅ Created {len(df.columns)} features from original data")
        return df
    
    def prepare_sequences(self, data, sequence_length=60, target_length=7, step=1):
        """Prepare sequences for LSTM model"""
        X, y = [], []
        
        for i in range(0, len(data) - sequence_length - target_length + 1, step):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length:i + sequence_length + target_length, 0])
        
        logger.info(f"📊 Prepared {len(X)} sequences (seq_len={sequence_length}, target_len={target_length})")
        return np.array(X), np.array(y)
    
    def scale_data(self, data, scaler_type='minmax'):
        """Scale data using specified scaler"""
        if scaler_type == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()
        
        scaled_data = scaler.fit_transform(data)
        logger.info(f"📏 Scaled data using {scaler_type} scaler: {data.shape[1]} features")
        return scaled_data, scaler
    
    def detect_anomalies(self, df, column='total_sales', method='iqr'):
        """Detect anomalies in sales data"""
        anomalies = []
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
            anomalies = df.loc[anomaly_indices, ['date', column]].to_dict('records')
            logger.info(f"🔍 IQR anomaly detection: found {len(anomalies)} anomalies")
        
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            z_scores = (df[column] - mean) / std
            anomaly_indices = df[abs(z_scores) > 3].index
            anomalies = df.loc[anomaly_indices, ['date', column]].to_dict('records')
            logger.info(f"🔍 Z-score anomaly detection: found {len(anomalies)} anomalies")
        
        return anomalies

# ============================================================================
# LSTM FORECASTER
# ============================================================================

class LSTMForecaster:
    # Class-level caches for loaded models (shared across all instances)
    _model_cache = {}
    _scaler_cache = {}
    _features_cache = {}
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.llm_explainer = LLMExplainer()
    
    def _get_cached_model(self, model_key):
        """Get model from cache or load it"""
        if model_key not in self._model_cache:
            model_path = f"{app.config['MODEL_PATH']}{model_key}.h5"
            scaler_path = f"{app.config['SCALER_PATH']}{model_key}_scaler.pkl"
            features_path = f"{app.config['SCALER_PATH']}{model_key}_features.pkl"
            
            if os.path.exists(model_path):
                logger.info(f"📦 Loading cached model: {model_key}")
                self._model_cache[model_key] = load_model(model_path)
                self._scaler_cache[model_key] = joblib.load(scaler_path)
                self._features_cache[model_key] = joblib.load(features_path)
            else:
                return None, None, None
        return (self._model_cache[model_key], 
                self._scaler_cache[model_key], 
                self._features_cache[model_key])
        
    def build_model(self, input_shape, output_length=7):
        """Build advanced LSTM model"""
        logger.info(f"🏗️ Building LSTM model with input shape {input_shape}, output length {output_length}")
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Bidirectional(LSTM(128, return_sequences=True, 
                             kernel_regularizer=l2(0.0001))),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(64, return_sequences=True,
                             kernel_regularizer=l2(0.0001))),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
            Dropout(0.2),
            
            Dense(output_length)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(optimizer=optimizer, 
                     loss='huber',
                     metrics=['mae', 'mse'])
        
        logger.info("✅ LSTM model built and compiled")
        return model
    
    def train_model(self, category, region, sequence_length=None, forecast_days=7, epochs=100):
        """Train LSTM model for specific category and region"""
        logger.info(f"🚀 Starting model training for {category} / {region}")
        start_time = time.time()
        
        try:
            with get_db_connection() as conn:
                query = '''
                    SELECT date, total_sales, units_sold, promotion_flag, 
                           holiday_flag, discount_percent, stock_level
                    FROM sales 
                    WHERE product_category = ? AND region = ?
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params=(category, region))
            
            logger.info(f"📊 Retrieved {len(df)} records for training")
            
            if len(df) < 10:
                logger.warning(f"⚠️ Insufficient data for training: {len(df)} records (need at least 10)")
                return None, f"Insufficient data for training. Found {len(df)} records, need at least 10 records."
            elif len(df) < 50:
                logger.warning(f"⚠️ Limited data for training: {len(df)} records (50+ recommended)")
            
            # Adapt sequence length to available data
            if sequence_length is None:
                if len(df) >= 90: sequence_length = 60
                elif len(df) >= 60: sequence_length = 45
                elif len(df) >= 40: sequence_length = 30
                elif len(df) >= 20: sequence_length = 15
                else: sequence_length = max(5, len(df) // 2)
            
            # Adapt forecast_days to data size
            forecast_days = min(forecast_days, max(1, len(df) // 5))
            
            preprocessor = DataPreprocessor()
            df = preprocessor.create_features(df)
            
            features = [
                'total_sales', 'units_sold', 'day_of_week', 'month', 
                'is_weekend', 'is_month_start', 'is_month_end',
                'sin_day_of_week', 'cos_day_of_week',
                'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
                'sales_rolling_mean_7', 'sales_rolling_std_7',
                'sales_rolling_mean_30', 'sales_rolling_std_30',
                'sales_diff_1', 'sales_diff_7', 'sales_pct_change_1',
                'promotion_flag', 'holiday_flag', 'discount_percent',
                'promo_impact'
            ]
            
            features = [f for f in features if f in df.columns]
            data = df[features].values
            
            scaled_data, scaler = preprocessor.scale_data(data)
            
            X, y = preprocessor.prepare_sequences(scaled_data, sequence_length, forecast_days, step=1)
            
            if len(X) < 2:
                logger.error(f"❌ Insufficient sequences: {len(X)} sequences, need at least 2")
                return None, "Insufficient sequences for training. Need more data records."
            
            train_size = max(1, int(len(X) * 0.8))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            if len(X_val) == 0:
                X_val, y_val = X_train, y_train
            
            logger.info(f"📊 Training split: {len(X_train)} training, {len(X_val)} validation")
            
            model = self.build_model((sequence_length, len(features)), forecast_days)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001)
            ]
            
            logger.info(f"🏋️ Training model for up to {epochs} epochs...")
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=0,
                shuffle=False
            )
            
            y_pred = model.predict(X_val, verbose=0)
            
            # Inverse-transform predictions & actuals back to real dollar values
            # so that MAE/RMSE/MAPE are meaningful (not on 0-1 scaled data)
            def _inv(arr_2d):
                """Inverse-transform a 2-D array of target values (first feature column)."""
                n_feat = len(features)
                out = []
                for row in arr_2d:
                    pad = np.zeros((len(row), n_feat))
                    pad[:, 0] = row
                    out.append(scaler.inverse_transform(pad)[:, 0])
                return np.array(out)
            
            y_val_inv  = _inv(y_val)
            y_pred_inv = _inv(y_pred)
            
            mae  = mean_absolute_error(y_val_inv.flatten(), y_pred_inv.flatten())
            rmse = np.sqrt(mean_squared_error(y_val_inv.flatten(), y_pred_inv.flatten()))
            # Safe MAPE: avoid division by near-zero actual values
            actual_flat = y_val_inv.flatten()
            pred_flat   = y_pred_inv.flatten()
            safe_denom  = np.where(np.abs(actual_flat) < 1.0, 1.0, np.abs(actual_flat))
            mape = np.mean(np.abs(actual_flat - pred_flat) / safe_denom) * 100
            r2   = r2_score(actual_flat, pred_flat)
            
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time:.2f}s - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, MAPE: {mape:.1f}%, R²: {r2:.3f}")
            
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO model_metrics (model_name, mae, rmse, mape, r2_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (f"{category}_{region}", float(mae), float(rmse), float(mape), float(r2)))
            
            model_key = _make_model_key(category, region)
            model.save(f"{app.config['MODEL_PATH']}{model_key}.h5")
            joblib.dump(scaler, f"{app.config['SCALER_PATH']}{model_key}_scaler.pkl")
            joblib.dump(features, f"{app.config['SCALER_PATH']}{model_key}_features.pkl")
            
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            return model, "Model trained successfully"
            
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            return None, f"Training error: {str(e)}"
    
    def predict(self, category, region, days_ahead=30, force_use_available=False):
        """Generate predictions using trained model"""
        logger.info(f"🔮 Generating {days_ahead}-day forecast for {category} / {region}")
        start_time = time.time()
        
        try:
            with get_db_connection() as conn:
                query = '''
                    SELECT date, total_sales, units_sold, promotion_flag, 
                           holiday_flag, discount_percent, stock_level
                    FROM sales 
                    WHERE product_category = ? AND region = ?
                    ORDER BY date DESC 
                    LIMIT 200
                '''
                df = pd.read_sql_query(query, conn, params=(category, region))
            
            logger.info(f"📊 Retrieved {len(df)} historical records for forecasting")
            
            # Check if we have sufficient data
            min_records_required = 90
            if len(df) < min_records_required:
                if force_use_available and len(df) >= 10:
                    logger.warning(f"⚠️ Using limited data ({len(df)} records) - forecast accuracy may be reduced")
                    min_records_required = 10
                else:
                    if len(df) < 10:
                        logger.error(f"❌ Insufficient data: {len(df)} records (need at least 10)")
                        return None, f"Insufficient data for prediction. Found {len(df)} records, need at least 10 records for basic prediction."
                    elif not force_use_available:
                        logger.warning(f"⚠️ Insufficient data for optimal prediction: {len(df)} records (90+ recommended)")
                        return None, f"Insufficient data for optimal prediction ({len(df)} records). Check 'Use available records' to forecast with current data (reduced accuracy)."
            
            df = df.iloc[::-1]
            
            model_key = _make_model_key(category, region)
            
            # Try to get from cache first
            model, scaler, features = self._get_cached_model(model_key)
            
            if model is None:
                logger.info(f"📦 No cached model found, training new model for {category} / {region}")
                # Train new model if not exists
                model, message = self.train_model(category, region)
                if model is None:
                    logger.error(f"❌ Model training failed: {message}")
                    return None, message
                # Load from disk and cache
                model, scaler, features = self._get_cached_model(model_key)
                if model is None:
                    logger.error(f"❌ Failed to load trained model")
                    return None, "Failed to load trained model"
            
            preprocessor = DataPreprocessor()
            df = preprocessor.create_features(df)
            
            available_features = [f for f in features if f in df.columns]
            
            # Determine sequence length based on available data
            data_points = len(df)
            if data_points >= 90:
                lookback_days = 60
                recent_data_points = 90
            elif data_points >= 60:
                lookback_days = 45
                recent_data_points = 60
            elif data_points >= 40:
                lookback_days = 30
                recent_data_points = 40
            elif data_points >= 30:
                lookback_days = 25
                recent_data_points = 30
            elif data_points >= 20:
                lookback_days = 20
                recent_data_points = 20
            else:
                lookback_days = data_points - 1
                recent_data_points = data_points
            
            recent_data = df[available_features].values[-recent_data_points:]
            
            scaled_data = scaler.transform(recent_data)
            
            predictions = []
            lower_bounds = []
            upper_bounds = []
            
            if len(scaled_data) >= lookback_days:
                current_sequence = scaled_data[-lookback_days:].reshape(1, lookback_days, len(available_features))
            else:
                padding_needed = lookback_days - len(scaled_data)
                padded_data = np.vstack([np.zeros((padding_needed, len(available_features))), scaled_data])
                current_sequence = padded_data.reshape(1, lookback_days, len(available_features))
            
            n_mc_samples = 5  # Reduced from 20 for 75% speed improvement
            
            for i in range(days_ahead):
                mc_predictions = []
                for _ in range(n_mc_samples):
                    pred = model(current_sequence, training=True).numpy()
                    mc_predictions.append(pred[0, 0])
                
                mean_pred = np.mean(mc_predictions)
                std_pred = np.std(mc_predictions)
                
                new_point = np.zeros((1, len(available_features)))
                new_point[0, 0] = mean_pred
                if len(available_features) > 1:
                    new_point[0, 1:] = scaled_data[-1, 1:]
                
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                            new_point.reshape(1, 1, len(available_features)), 
                                            axis=1)
                
                dummy_array = np.zeros((1, len(available_features)))
                dummy_array[0, 0] = mean_pred
                pred_original = scaler.inverse_transform(dummy_array)[0, 0]
                
                lower_bound = pred_original - 1.96 * std_pred * (pred_original + 1)
                upper_bound = pred_original + 1.96 * std_pred * (pred_original + 1)
                
                predictions.append(max(0, pred_original))
                lower_bounds.append(max(0, lower_bound))
                upper_bounds.append(upper_bound)
            
            last_date = pd.to_datetime(df['date'].iloc[-1])
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
            
            avg_price = df['total_sales'].sum() / df['units_sold'].sum() if df['units_sold'].sum() > 0 else 50
            predicted_units = [int(p / avg_price) for p in predictions]
            
            # Batch insert all predictions at once (much faster)
            predictions_batch = []
            current_date = datetime.now().date().isoformat()
            forecast_session_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
            for i in range(len(prediction_dates)):
                predictions_batch.append((
                    current_date,
                    prediction_dates[i].date().isoformat(),
                    category,
                    region,
                    float(predictions[i]),
                    int(predicted_units[i]),
                    float(lower_bounds[i]),
                    float(upper_bounds[i]),
                    'LSTM',
                    forecast_session_id,
                    days_ahead,
                    1  # is_primary
                ))
            
            with get_db_connection() as conn:
                conn.executemany('''
                    INSERT INTO predictions 
                    (prediction_date, forecast_date, product_category, region, 
                     predicted_sales, predicted_units, confidence_interval_lower, 
                     confidence_interval_upper, model_type, forecast_session_id, forecast_days,
                     is_primary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', predictions_batch)
            
            trend_analysis = self.analyze_trends(category, region, predictions, prediction_dates)
            
            historical_data = {
                'avg_sales': float(df['total_sales'].mean()),
                'trend': 'increasing' if df['total_sales'].iloc[-30:].mean() > df['total_sales'].iloc[:30].mean() else 'decreasing',
                'seasonality': 'strong' if len(df) > 90 else 'moderate'
            }
            
            forecast_data = {
                'predictions': predictions,
                'growth_rate': ((predictions[-1] / predictions[0]) - 1) * 100 if len(predictions) > 0 and predictions[0] > 0 else 0,
                'avg_confidence': 95 - (np.mean([(u - l) / (p + 1) for p, l, u in zip(predictions, lower_bounds, upper_bounds)]) * 100)
            }
            
            prediction_time = time.time() - start_time
            total_predicted_sales = sum(predictions)
            logger.info(f"✅ Forecast generated in {prediction_time:.2f}s - Total predicted sales: ${total_predicted_sales:,.2f}, Growth: {forecast_data['growth_rate']:.1f}%")
            
            try:
                llm_explanation = self.llm_explainer.explain_forecast(category, region, forecast_data, historical_data)
            except:
                llm_explanation = "Unable to generate AI explanation at this time."
            
            return {
                'dates': [d.date().isoformat() for d in prediction_dates],
                'predictions': predictions,
                'predicted_units': predicted_units,
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds,
                'avg_price': float(avg_price),
                'trend_analysis': trend_analysis,
                'llm_explanation': llm_explanation
            }, "Success"
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return None, f"Prediction error: {str(e)}"
    
    def analyze_trends(self, category, region, predictions, dates):
        """Analyze trends from predictions"""
        logger.info(f"📈 Analyzing trends for {category} / {region}")
        
        if len(predictions) < 7:
            return {}
        
        dates_series = pd.to_datetime(dates)
        pred_series = pd.Series(predictions, index=dates_series)
        
        analysis = {}
        
        weekly_avg = pred_series.groupby(dates_series.dayofweek).mean()
        best_day = weekly_avg.idxmax()
        worst_day = weekly_avg.idxmin()
        analysis['best_day'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][best_day]
        analysis['worst_day'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][worst_day]
        
        analysis['weekly_growth'] = ((pred_series[-7:].mean() / pred_series[:7].mean()) - 1) * 100 if len(predictions) >= 14 and pred_series[:7].mean() > 0 else 0
        analysis['monthly_growth'] = ((pred_series[-30:].mean() / pred_series[:30].mean()) - 1) * 100 if len(predictions) >= 60 and pred_series[:30].mean() > 0 else analysis['weekly_growth'] * 4.33
        
        analysis['volatility'] = float(pred_series.pct_change().std() * 100)
        
        rolling_mean = pred_series.rolling(window=7, min_periods=1).mean()
        peaks = (pred_series > rolling_mean * 1.2).sum()
        analysis['peak_count'] = int(peaks)
        analysis['trough_count'] = int((pred_series < rolling_mean * 0.8).sum())
        
        logger.info(f"📊 Trend analysis: Best day: {analysis['best_day']}, Weekly growth: {analysis['weekly_growth']:.1f}%, Volatility: {analysis['volatility']:.1f}%")
        
        try:
            with get_db_connection() as conn:
                conn.execute('DELETE FROM trends WHERE trend_type LIKE ?', (f"{category}_{region}_%",))
                
                trends = [
                    ('weekly_growth', analysis['weekly_growth'], 
                     'up' if analysis['weekly_growth'] > 0 else 'down', dates[0], dates[-1]),
                    ('monthly_growth', analysis['monthly_growth'],
                     'up' if analysis['monthly_growth'] > 0 else 'down', dates[0], dates[-1]),
                    ('volatility', analysis['volatility'],
                     'high' if analysis['volatility'] > 20 else 'normal', dates[0], dates[-1]),
                    ('peak_count', analysis['peak_count'], 
                     'high' if analysis['peak_count'] > len(predictions)/7 else 'normal', dates[0], dates[-1])
                ]
                
                for trend_type, value, direction, start_date, end_date in trends:
                    conn.execute('''
                        INSERT INTO trends (trend_type, trend_value, trend_direction, start_date, end_date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (f"{category}_{region}_{trend_type}", float(value), direction, start_date, end_date))
        except:
            pass
        
        return analysis

# ============================================================================
# BASELINE MODELS
# ============================================================================

class BaselineModels:
    @staticmethod
    def arima_forecast(data, days_ahead=30):
        """ARIMA model forecasting"""
        try:
            logger.info(f"📊 Running ARIMA forecast for {days_ahead} days ahead")
            result = adfuller(data)
            d = 0 if result[1] < 0.05 else 1
            
            best_aic = float('inf')
            best_model = None
            
            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_model = model_fit
                    except:
                        continue
            
            if best_model is None:
                model = ARIMA(data, order=(5, 1, 0))
                best_model = model.fit()
            
            forecast = best_model.forecast(steps=days_ahead)
            
            fitted = best_model.fittedvalues
            mae = mean_absolute_error(data[-len(fitted):], fitted)
            rmse = np.sqrt(mean_squared_error(data[-len(fitted):], fitted))
            mape = mean_absolute_percentage_error(data[-len(fitted):], fitted) * 100
            
            logger.info(f"✅ ARIMA forecast completed - MAE: ${mae:.2f}, MAPE: {mape:.1f}%")
            
            return {
                'forecast': forecast.tolist(),
                'metrics': {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
            }, "Success"
        except Exception as e:
            logger.error(f"❌ ARIMA Error: {str(e)}")
            return None, f"ARIMA Error: {str(e)}"
    
    @staticmethod
    def exponential_smoothing_forecast(data, days_ahead=30):
        """Exponential Smoothing forecasting"""
        try:
            logger.info(f"📊 Running Exponential Smoothing forecast for {days_ahead} days ahead")
            seasonal_periods = 7 if len(data) >= 14 else None
            
            if seasonal_periods:
                try:
                    model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=seasonal_periods)
                    model_fit = model.fit()
                except:
                    model = ExponentialSmoothing(data)
                    model_fit = model.fit()
            else:
                model = ExponentialSmoothing(data)
                model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=days_ahead)
            
            fitted = model_fit.fittedvalues
            mae = mean_absolute_error(data[-len(fitted):], fitted)
            rmse = np.sqrt(mean_squared_error(data[-len(fitted):], fitted))
            mape = mean_absolute_percentage_error(data[-len(fitted):], fitted) * 100
            
            logger.info(f"✅ Exponential Smoothing forecast completed - MAE: ${mae:.2f}, MAPE: {mape:.1f}%")
            
            return {
                'forecast': forecast.tolist(),
                'metrics': {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
            }, "Success"
        except Exception as e:
            logger.error(f"❌ Exponential Smoothing Error: {str(e)}")
            return None, f"Exponential Smoothing Error: {str(e)}"

# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.llm_explainer = LLMExplainer()
    
    def detect_anomalies(self, df, methods=['iqr', 'zscore', 'isolation_forest']):
        """Detect anomalies using multiple methods with enhanced business context"""
        logger.info(f"🔍 Running anomaly detection using methods: {', '.join(methods)}")
        anomalies = {}
        
        if 'iqr' in methods:
            anomalies['iqr'] = self.preprocessor.detect_anomalies(df, method='iqr')
        
        if 'zscore' in methods:
            anomalies['zscore'] = self.preprocessor.detect_anomalies(df, method='zscore')
        
        if 'isolation_forest' in methods:
            anomalies['isolation_forest'] = self._isolation_forest_detection(df)
        
        total = sum(len(v) for v in anomalies.values())
        logger.info(f"📊 Anomaly detection completed: Found {total} anomalies total")
        
        # Store anomalies with full context
        try:
            if any(anomalies.values()):
                with get_db_connection() as conn:
                    # Clear old anomalies for this filter combination to avoid duplicates
                    # (keep last 30 days of history)
                    cutoff_date = (datetime.now().date() - timedelta(days=30)).isoformat()
                    conn.execute('DELETE FROM anomalies WHERE detection_date < ?', (cutoff_date,))
                    
                    for method, method_anomalies in anomalies.items():
                        for anomaly in method_anomalies:
                            # Get category and region for this anomaly
                            anomaly_date = anomaly['date']
                            matching_rows = df[df['date'] == anomaly_date]
                            
                            category = 'Unknown'
                            region = 'Unknown'
                            units_value = 0
                            
                            if not matching_rows.empty:
                                category = matching_rows.iloc[0].get('product_category', 'Unknown')
                                region = matching_rows.iloc[0].get('region', 'Unknown')
                                units_value = int(matching_rows.iloc[0].get('units_sold', 0))
                            
                            sales_value = anomaly.get('total_sales', anomaly.get('sales_value', 0))
                            
                            # Calculate deviation magnitude
                            avg_sales = float(df['total_sales'].mean()) if 'total_sales' in df.columns else 1
                            deviation_pct = ((sales_value - avg_sales) / avg_sales) * 100 if avg_sales > 0 else 0
                            
                            # Generate explanation for this specific anomaly
                            explanation = self._generate_single_anomaly_explanation(
                                anomaly_date, sales_value, units_value, category, region, 
                                deviation_pct, method, avg_sales
                            )
                            
                            conn.execute('''
                                INSERT INTO anomalies 
                                (detection_date, anomaly_date, product_category, region, 
                                 sales_value, units_value, detection_method, explanation, reviewed)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                datetime.now().date().isoformat(),
                                anomaly_date,
                                category,
                                region,
                                float(sales_value),
                                int(units_value),
                                method,
                                explanation,
                                0
                            ))
        except Exception as e:
            logger.warning(f"⚠️ Failed to store anomalies in database: {str(e)}")
        
        return anomalies
    
    def _generate_single_anomaly_explanation(self, date, sales_value, units_value, category, region, deviation_pct, method, avg_sales):
        """Generate a specific explanation for a single anomaly"""
        direction = "spike" if deviation_pct > 0 else "drop"
        abs_pct = abs(deviation_pct)
        
        if abs_pct > 100:
            severity = "extreme"
            action = "Immediately investigate this transaction - possible data entry error or extraordinary event"
        elif abs_pct > 50:
            severity = "major"
            action = "Review this date's sales records and verify accuracy; check for promotions or stockouts"
        elif abs_pct > 25:
            severity = "moderate"
            action = "Monitor this pattern; consider if it aligns with holidays or marketing activities"
        else:
            severity = "minor"
            action = "No immediate action required but keep this pattern noted"
        
        explanation = f"{direction.upper()} of {abs_pct:.1f}% in {category}/{region} on {date}. "
        explanation += f"Sales: ${sales_value:,.2f} vs daily avg ${avg_sales:,.2f}. "
        explanation += f"Detected by {method.upper()}. {action}"
        
        return explanation
    
    def _isolation_forest_detection(self, df):
        """Detect anomalies using Isolation Forest"""
        try:
            if 'total_sales' not in df.columns or 'units_sold' not in df.columns:
                return []
            
            data = df[['total_sales', 'units_sold']].values
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data)
            
            anomaly_indices = df[outliers == -1].index
            anomalies = df.loc[anomaly_indices, ['date', 'total_sales', 'units_sold']].to_dict('records')
            
            logger.info(f"🌲 Isolation Forest detected {len(anomalies)} anomalies")
            return anomalies
        except Exception as e:
            logger.warning(f"⚠️ Isolation Forest error: {str(e)}")
            return []
    
    def get_anomaly_explanation(self, anomalies):
        """Generate explanation for detected anomalies"""
        if not anomalies or all(len(v) == 0 for v in anomalies.values()):
            return "No significant anomalies detected in the sales data."
        
        total_anomalies = sum(len(v) for v in anomalies.values())
        explanation = f"Detected {total_anomalies} potential anomalies across detection methods.\n\n"
        
        for method, method_anomalies in anomalies.items():
            if method_anomalies:
                explanation += f"Using {method.upper()} method: Found {len(method_anomalies)} anomalies\n"
                for anomaly in method_anomalies[:5]:
                    explanation += f"  - {anomaly['date']}: ${anomaly.get('total_sales', 0):,.2f}\n"
        
        return explanation
    
    def generate_business_explanation(self, anomalies, df, context=None):
        """Convert raw anomaly data into actionable business insights with specific data"""
        
        if not anomalies or all(len(v) == 0 for v in anomalies.values()):
            logger.info("📊 No anomalies detected in the data")
            return {
                'summary': "✅ No unusual sales patterns detected. Your sales data looks consistent.",
                'what_it_means': "All sales values fall within expected ranges based on historical patterns. Your inventory and staffing can follow normal schedules.",
                'actions': [
                    "Continue monitoring sales data weekly to catch emerging patterns early.",
                    "Run another anomaly scan after adding 30+ more sales records for better statistical power.",
                    "Export your data periodically to maintain a backup of normal patterns."
                ],
                'priority': "LOW"
            }
        
        # Prepare detailed analysis data with full context
        all_anomalies = []
        for method, method_anomalies in anomalies.items():
            for a in method_anomalies:
                # Get full row data for this anomaly
                anomaly_date = a['date']
                matching_rows = df[df['date'] == anomaly_date]
                
                category_val = 'Unknown'
                region_val = 'Unknown'
                units_val = 0
                if not matching_rows.empty:
                    category_val = matching_rows.iloc[0].get('product_category', 'Unknown')
                    region_val = matching_rows.iloc[0].get('region', 'Unknown')
                    units_val = int(matching_rows.iloc[0].get('units_sold', 0))
                
                all_anomalies.append({
                    'date': anomaly_date,
                    'value': a.get('total_sales', a.get('sales_value', 0)),
                    'units': units_val,
                    'category': category_val,
                    'region': region_val,
                    'method': method
                })
        
        # Calculate statistics
        all_anomalies.sort(key=lambda x: x['value'], reverse=True)
        top_anomaly = all_anomalies[0] if all_anomalies else None
        
        category = context.get('category', 'All') if context else 'All'
        region = context.get('region', 'All') if context else 'All'
        
        sales_values = [a['value'] for a in all_anomalies]
        avg_anomaly = sum(sales_values) / len(sales_values) if sales_values else 0
        total_count = len(all_anomalies)
        
        # Calculate average daily sales for context
        avg_daily_sales = float(df['total_sales'].mean()) if 'total_sales' in df.columns else 0
        
        # Determine recency
        thirty_days_ago = (datetime.now().date() - timedelta(days=30)).isoformat()
        recent_anomalies = [a for a in all_anomalies if a['date'] >= thirty_days_ago]
        recent_count = len(recent_anomalies)
        
        # Calculate priority based on actual data
        priority = "MEDIUM"
        if top_anomaly:
            deviation_ratio = top_anomaly['value'] / avg_daily_sales if avg_daily_sales > 0 else 1
            if deviation_ratio > 5 or top_anomaly['value'] > 10000:
                priority = "HIGH"
            elif deviation_ratio > 2.5 or top_anomaly['value'] > 5000:
                priority = "MEDIUM-HIGH"
            elif recent_count > 5:
                priority = "MEDIUM"
            else:
                priority = "LOW"
        
        logger.info(f"📊 Anomaly analysis: {total_count} anomalies, priority {priority}")
        
        # Build detailed prompt with actual anomaly data
        anomaly_details = []
        for a in all_anomalies[:8]:  # Top 8 anomalies
            direction = "↑ SPIKE" if a['value'] > avg_daily_sales else "↓ DROP"
            pct_diff = ((a['value'] - avg_daily_sales) / avg_daily_sales) * 100 if avg_daily_sales > 0 else 0
            anomaly_details.append(
                f"  • {a['date']}: {direction} of {abs(pct_diff):.1f}% | "
                f"Sales: ${a['value']:,.2f} (vs ${avg_daily_sales:,.2f} avg) | "
                f"Units: {a['units']} | {a['category']}/{a['region']}"
            )
        
        prompt = f"""You are a retail data analyst. Based on ACTUAL anomaly data below, provide SPECIFIC business insights.

CONTEXT:
- Filtered for: {category} / {region if region != 'All' else 'all regions'}
- Normal daily sales average: ${avg_daily_sales:,.2f}
- Total anomalies detected in this scan: {total_count} (recent: {recent_count} in last 30 days)

ACTUAL ANOMALIES DETECTED (with real sales values):
{chr(10).join(anomaly_details)}

PRIORITY LEVEL: {priority}

Please respond in EXACTLY this format with plain text (no markdown):

SUMMARY: (One sentence stating what the data shows - mention the actual dates and values)

WHAT THIS MEANS: (2-3 specific sentences explaining business impact for inventory, staffing, or promotions)

RECOMMENDED ACTIONS:
- Action 1 (specific to the anomalies shown above)
- Action 2 (specific to the anomalies shown above)
- Action 3 (general best practice)

PRIORITY: {priority}

CONCERN LEVEL: (LOW/MEDIUM/HIGH with specific reason based on the anomaly values)"""

        try:
            response = requests.get(f"{POLLINATIONS_API_URL}{prompt}", timeout=30)
            if response.status_code == 200:
                ai_response = response.text
                
                import re
                summary_match = re.search(r'SUMMARY:\s*(.+?)(?=\n\n|\nWHAT|\Z)', ai_response, re.DOTALL)
                meaning_match = re.search(r'WHAT THIS MEANS:\s*(.+?)(?=\n\n|\nRECOMMENDED|\Z)', ai_response, re.DOTALL)
                actions_match = re.search(r'RECOMMENDED ACTIONS:\s*(.+?)(?=\n\n|\nPRIORITY|\Z)', ai_response, re.DOTALL)
                concern_match = re.search(r'CONCERN LEVEL:\s*(.+?)(?=\n\n|\Z)', ai_response, re.DOTALL)
                
                actions = []
                if actions_match:
                    actions_text = actions_match.group(1)
                    actions = [a.strip('-•0123456789. ').strip() for a in actions_text.split('\n') if a.strip() and len(a.strip()) > 5]
                    actions = actions[:5]
                
                if not actions:
                    actions = [
                        f"Investigate the {top_anomaly['date'] if top_anomaly else 'largest'} anomaly - verify if this was a data entry error or real sales event.",
                        "Cross-reference anomaly dates with your marketing calendar to identify potential cause.",
                        "Run a data quality check on the affected dates to ensure accuracy."
                    ]
                
                concern_level = concern_match.group(1).strip() if concern_match else f"{priority} - Based on {total_count} anomalies detected"
                
                logger.info("✅ Generated specific AI-powered anomaly explanation with actual data")
                return {
                    'summary': summary_match.group(1).strip() if summary_match else f"Found {total_count} anomalies in your sales data from {all_anomalies[0]['date'] if all_anomalies else 'recent dates'}.",
                    'what_it_means': meaning_match.group(1).strip() if meaning_match else f"Sales on these dates deviated from the ${avg_daily_sales:,.2f} daily average by significant margins. This affects inventory planning and revenue forecasting accuracy.",
                    'actions': actions,
                    'priority': priority,
                    'concern_level': concern_level,
                    'anomaly_count': total_count,
                    'recent_count': recent_count,
                    'top_anomaly': {
                        'date': top_anomaly['date'] if top_anomaly else None,
                        'value': top_anomaly['value'] if top_anomaly else 0,
                        'vs_avg': f"{((top_anomaly['value'] - avg_daily_sales) / avg_daily_sales * 100):.1f}%" if top_anomaly and avg_daily_sales > 0 else "N/A"
                    } if top_anomaly else None
                }
        except Exception as e:
            logger.warning(f"⚠️ LLM explanation failed: {str(e)}")
        
        return self._generate_fallback_explanation(all_anomalies, total_count, recent_count, priority, category, region, avg_daily_sales)
    
    def _generate_fallback_explanation(self, all_anomalies, total_count, recent_count, priority, category, region, avg_daily_sales):
        """Generate detailed fallback explanation with actual anomaly data"""
        
        if not all_anomalies:
            return {
                'summary': "No anomalies detected in the current data set.",
                'what_it_means': "Your sales patterns are within normal statistical ranges.",
                'actions': ["Continue monitoring and add more data for better detection."],
                'priority': "LOW"
            }
        
        top_3 = all_anomalies[:3]
        top_anomaly = top_3[0] if top_3 else None
        
        # Build detailed summary with actual values
        summary_parts = []
        for a in top_3:
            pct_diff = ((a['value'] - avg_daily_sales) / avg_daily_sales) * 100 if avg_daily_sales > 0 else 0
            direction = "spike" if pct_diff > 0 else "drop"
            summary_parts.append(f"{a['date']}: {direction} of {abs(pct_diff):.1f}% (${a['value']:,.2f})")
        
        summary = f"Found {total_count} anomalies in {category}/{region}. " + "; ".join(summary_parts[:2])
        
        # Build what-it-means with specific impact
        what_it_means = f"Your normal daily average is ${avg_daily_sales:,.2f}. "
        if top_anomaly:
            pct = ((top_anomaly['value'] - avg_daily_sales) / avg_daily_sales) * 100 if avg_daily_sales > 0 else 0
            what_it_means += f"The largest deviation was on {top_anomaly['date']} with a {abs(pct):.1f}% {'increase' if pct > 0 else 'decrease'} to ${top_anomaly['value']:,.2f}. "
        
        what_it_means += "Such fluctuations can indicate data entry errors, successful promotions, stockouts, or external factors affecting sales."
        
        # Build actions based on anomaly patterns
        actions = []
        
        if recent_count > 0:
            actions.append(f"Check sales records for the last {recent_count} anomaly dates to verify data accuracy")
        
        if top_anomaly:
            actions.append(f"Investigate {top_anomaly['date']} - compare with any promotions, holidays, or events on that date")
        
        actions.append("Export anomaly data and cross-reference with your marketing/promotion calendar")
        actions.append("Review inventory levels around anomaly dates to identify potential stockout patterns")
        
        if total_count > 10:
            actions.append("Consider a data quality audit - multiple anomalies may indicate systematic recording issues")
        
        concern_level = f"{priority.upper()} - {total_count} anomalies detected"
        if recent_count > 5:
            concern_level += f", with {recent_count} occurring in the last 30 days"
        
        logger.info("📊 Using fallback anomaly explanation with actual data")
        
        return {
            'summary': summary,
            'what_it_means': what_it_means,
            'actions': actions[:4],
            'priority': priority,
            'concern_level': concern_level,
            'anomaly_count': total_count,
            'recent_count': recent_count,
            'top_anomaly': {
                'date': top_anomaly['date'] if top_anomaly else None,
                'value': top_anomaly['value'] if top_anomaly else 0,
                'vs_avg': f"{((top_anomaly['value'] - avg_daily_sales) / avg_daily_sales * 100):.1f}%" if top_anomaly and avg_daily_sales > 0 else "N/A"
            } if top_anomaly else None
        }
    
    def _generate_fallback_explanation(self, all_anomalies, total_count, recent_anomalies, priority, category, region):
        """Generate fallback explanation when LLM is unavailable"""
        top_3 = all_anomalies[:3]
        top_values = [f"${a['value']:,.2f}" for a in top_3]
        
        category_text = f"for {category}" if category != 'All' else "across all categories"
        region_text = f"in {region}" if region != 'All' else "across all regions"
        
        if total_count > 50:
            severity_text = "a large number of"
            recommendation = "Run a data quality check and verify recent data entry accuracy."
        elif total_count > 20:
            severity_text = "multiple"
            recommendation = "Review sales data from the affected dates and check for patterns."
        else:
            severity_text = "a few"
            recommendation = "Monitor these dates for recurring patterns and validate data accuracy."
        
        summary = f"Detected {severity_text} ({total_count}) unusual sales patterns {category_text} {region_text}."
        
        if recent_anomalies > 0:
            summary += f" {recent_anomalies} of these occurred in the last 30 days."
        
        what_it_means = f"The largest anomalies range from {top_values[0] if top_values else 'N/A'} to {top_values[-1] if len(top_values) > 1 else top_values[0] if top_values else 'N/A'}. "
        what_it_means += "These spikes or drops may indicate data entry errors, successful promotions, stockouts, or external market factors affecting sales."
        
        actions = [
            f"Review sales records for dates: {', '.join([a['date'] for a in top_3[:3]])}",
            "Verify if these dates had any special promotions or marketing campaigns",
            recommendation,
            "Export anomaly data for further analysis using the Export button"
        ]
        
        logger.info("📊 Using fallback anomaly explanation (LLM unavailable)")
        return {
            'summary': summary,
            'what_it_means': what_it_means,
            'actions': actions,
            'priority': priority
        }

# ============================================================================
# MODEL SCHEDULER
# ============================================================================

class ModelScheduler:
    def __init__(self):
        self.scheduled_jobs = {}
        self.running = True
    
    def start_scheduler(self):
        """Start the scheduler for automatic retraining"""
        def schedule_check():
            logger.info("🔄 Model retraining scheduler started")
            while self.running:
                try:
                    self.check_and_retrain()
                except Exception as e:
                    logger.error(f"❌ Error in scheduler: {str(e)}")
                time.sleep(3600)
        
        thread = threading.Thread(target=schedule_check, daemon=True)
        thread.start()
        logger.info("✅ Model scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("🛑 Model scheduler stopped")
    
    def check_and_retrain(self):
        """Check if models need retraining"""
        try:
            with get_db_connection() as conn:
                combinations = conn.execute('''
                    SELECT DISTINCT product_category, region, COUNT(*) as count
                    FROM sales 
                    GROUP BY product_category, region
                    HAVING count >= 100
                ''').fetchall()
            
            for combo in combinations:
                category = combo['product_category']
                region = combo['region']
                
                model_key = _make_model_key(category, region)
                model_path = f"{app.config['MODEL_PATH']}{model_key}.h5"
                
                if os.path.exists(model_path):
                    last_training = os.path.getmtime(model_path)
                    days_since_training = (time.time() - last_training) / (24 * 3600)
                    
                    if days_since_training > 7:
                        logger.info(f"🔄 Auto-retraining model for {category} / {region} (last trained {days_since_training:.1f} days ago)")
                        try:
                            forecaster = LSTMForecaster()
                            forecaster.train_model(category, region)
                            logger.info(f"✅ Auto-retraining completed for {category} / {region}")
                        except Exception as e:
                            logger.error(f"❌ Auto-retraining failed for {category} / {region}: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error in check_and_retrain: {str(e)}")

# ============================================================================
# EMAIL SCHEDULER
# ============================================================================

class EmailScheduler:
    def __init__(self, email_service):
        self.email_service = email_service
        self.running = True
        self.schedules = {}
    
    def load_schedules(self):
        """Load scheduled reports from database"""
        try:
            with get_db_connection() as conn:
                schedules = conn.execute('''
                    SELECT * FROM scheduled_reports 
                    WHERE active = 1 AND (next_run IS NULL OR next_run <= datetime('now'))
                    ORDER BY next_run
                ''').fetchall()
            logger.info(f"📋 Loaded {len(schedules)} pending scheduled reports")
            return schedules
        except Exception as e:
            logger.error(f"❌ Error loading schedules: {str(e)}")
            return []
    
    def generate_report_file(self, schedule, start_date, end_date):
        """Generate report file and return filepath and filename"""
        try:
            with get_db_connection() as conn:
                if schedule['report_type'] == 'sales':
                    df = pd.read_sql_query('''
                        SELECT date, product_category, region, units_sold, unit_price, total_sales
                        FROM sales 
                        WHERE date BETWEEN ? AND ?
                        ORDER BY date
                    ''', conn, params=(start_date.isoformat(), end_date.isoformat()))
                else:
                    df = pd.read_sql_query('''
                        SELECT * FROM predictions 
                        WHERE forecast_date BETWEEN ? AND ?
                        ORDER BY forecast_date
                    ''', conn, params=(start_date.isoformat(), end_date.isoformat()))
            
            if df.empty:
                logger.warning(f"⚠️ No data for report {schedule['name']} in period {start_date} to {end_date}")
                return None, None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_format = schedule.get('format', 'csv').lower()
            
            if report_format == 'csv':
                filename = f"report_{schedule['name'].replace(' ', '_')}_{timestamp}.csv"
                filepath = os.path.join(app.config['REPORT_PATH'], filename)
                df.to_csv(filepath, index=False)
                logger.info(f"📄 Generated CSV report: {filename}")
                return filepath, filename
            
            elif report_format == 'excel':
                filename = f"report_{schedule['name'].replace(' ', '_')}_{timestamp}.xlsx"
                filepath = os.path.join(app.config['REPORT_PATH'], filename)
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Report', index=False)
                logger.info(f"📊 Generated Excel report: {filename}")
                return filepath, filename
            
            else:  # pdf
                filename = f"report_{schedule['name'].replace(' ', '_')}_{timestamp}.pdf"
                filepath = os.path.join(app.config['REPORT_PATH'], filename)
                
                # Create PDF
                doc = SimpleDocTemplate(filepath, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()
                
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    textColor=colors.HexColor('#667eea'),
                    alignment=TA_CENTER,
                    spaceAfter=30
                )
                elements.append(Paragraph(f"Sales Report: {schedule['name']}", title_style))
                elements.append(Spacer(1, 20))
                
                # Add summary
                summary_data = [
                    ['Metric', 'Value'],
                    ['Date Range', f"{start_date} to {end_date}"],
                    ['Total Sales', f"${df['total_sales'].sum():,.2f}" if 'total_sales' in df.columns else 'N/A'],
                    ['Total Records', str(len(df))]
                ]
                
                summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(summary_table)
                elements.append(Spacer(1, 30))
                
                # Add data table preview (first 20 rows)
                if len(df) > 0:
                    preview_df = df.head(20)
                    table_data = [preview_df.columns.tolist()] + preview_df.values.tolist()
                    data_table = Table(table_data)
                    data_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                    ]))
                    elements.append(data_table)
                
                doc.build(elements)
                logger.info(f"📑 Generated PDF report: {filename}")
                return filepath, filename
                
        except Exception as e:
            logger.error(f"❌ Error generating report file: {str(e)}")
            return None, None
    
    def send_report_now(self, schedule_id, user_id, recipients_override=None):
        """Send a report immediately (manual send) - legacy method"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        return self.send_report_now_with_dates(schedule_id, user_id, start_date, end_date, recipients_override)
    
    def send_report_now_with_dates(self, schedule_id, user_id, start_date, end_date, recipients_override=None, custom_report_type=None, custom_category=None, custom_region=None, custom_format=None):
        """Send a report immediately with custom date range and optional overrides"""
        logger.info(f"📧 Manually sending report for schedule ID: {schedule_id} with dates {start_date} to {end_date}")
        try:
            with get_db_connection() as conn:
                schedule = conn.execute('''
                    SELECT * FROM scheduled_reports WHERE id = ?
                ''', (schedule_id,)).fetchone()
                
                if not schedule:
                    logger.warning(f"⚠️ Schedule {schedule_id} not found")
                    return False, "Schedule not found"
                
                # Check permissions
                is_admin = session.get('role') == 'admin' if 'session' in globals() else False
                if schedule['created_by'] != user_id and not is_admin:
                    logger.warning(f"⚠️ User {user_id} denied access to schedule {schedule_id}")
                    return False, "You don't have permission to send this report"
            
            # Use custom parameters or defaults from schedule
            report_type = custom_report_type if custom_report_type else schedule['report_type']
            file_format = custom_format if custom_format else schedule.get('format', 'csv')
            
            # Generate report file with custom parameters
            filepath, filename = self.generate_report_file_with_filters(
                dict(schedule), start_date, end_date, report_type, custom_category, custom_region, file_format
            )
            
            if not filepath:
                logger.warning(f"⚠️ No data available for report {schedule['name']} in period {start_date} to {end_date}")
                return False, f"No data available for period {start_date} to {end_date}"
            
            recipients = recipients_override if recipients_override else (
                json.loads(schedule['recipients']) if isinstance(schedule['recipients'], str) else [schedule['recipients']]
            )
            
            # Create email with attachment
            subject = f"Sales Report: {schedule['name']}"
            
            body = f"""
            <html>
            <head><style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
            </style></head>
            <body>
                <div class="header">
                    <h1>Sales Forecasting System</h1>
                    <p>{schedule['name']} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                <div class="content">
                    <p>Dear User,</p>
                    <p>Your requested report "{schedule['name']}" is attached to this email.</p>
                    <p><strong>Report Details:</strong><br>
                    - Type: {report_type}<br>
                    - Period: {start_date} to {end_date}<br>
                    - Frequency: Manual Request</p>
                    <p>Best regards,<br>Sales Forecasting AI System</p>
                </div>
            </body>
            </html>
            """
            
            # Send email with attachment
            msg = MIMEMultipart()
            msg['From'] = self.email_service.config['sender']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            with open(filepath, 'rb') as f:
                attachment = MIMEApplication(f.read(), Name=filename)
                attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
                msg.attach(attachment)
            
            if self.email_service.config['use_ssl']:
                server = smtplib.SMTP_SSL(self.email_service.config['server'], self.email_service.config['port'])
            else:
                server = smtplib.SMTP(self.email_service.config['server'], self.email_service.config['port'])
                if self.email_service.config['use_tls']:
                    server.starttls()
            
            server.login(self.email_service.config['username'], self.email_service.config['password'])
            server.send_message(msg)
            server.quit()
            
            # Clean up file
            try:
                os.remove(filepath)
            except:
                pass
            
            logger.info(f"✅ Report sent successfully to {', '.join(recipients)}")
            return True, f"Report sent successfully for period {start_date} to {end_date}"
            
        except Exception as e:
            logger.error(f"❌ Error sending report now: {str(e)}")
            return False, str(e)
    
    def generate_report_file_with_filters(self, schedule, start_date, end_date, report_type, category=None, region=None, file_format='csv'):
        """Generate report file with custom filters - includes ALL records, not limited"""
        try:
            with get_db_connection() as conn:
                if report_type == 'sales':
                    query = """
                        SELECT date, product_category, region, units_sold, unit_price, total_sales,
                               promotion_flag, holiday_flag, discount_percent, stock_level
                        FROM sales 
                        WHERE date BETWEEN ? AND ?
                    """
                    params = [start_date.isoformat(), end_date.isoformat()]
                    
                    if category:
                        query += " AND product_category = ?"
                        params.append(category)
                    if region:
                        query += " AND region = ?"
                        params.append(region)
                    
                    query += " ORDER BY date"
                    df = pd.read_sql_query(query, conn, params=params)
                    
                elif report_type == 'forecast':
                    query = """
                        SELECT prediction_date, forecast_date, product_category, region,
                               predicted_sales, predicted_units, confidence_interval_lower,
                               confidence_interval_upper, model_type
                        FROM predictions 
                        WHERE forecast_date BETWEEN ? AND ? AND is_primary = 1
                    """
                    params = [start_date.isoformat(), end_date.isoformat()]
                    
                    if category:
                        query += " AND product_category = ?"
                        params.append(category)
                    if region:
                        query += " AND region = ?"
                        params.append(region)
                    
                    query += " ORDER BY forecast_date"
                    df = pd.read_sql_query(query, conn, params=params)
                    
                elif report_type == 'trends':
                    query = """
                        SELECT trend_type, trend_value, trend_direction, start_date, end_date, created_at
                        FROM trends 
                        WHERE start_date >= ? AND end_date <= ?
                    """
                    params = [start_date.isoformat(), end_date.isoformat()]
                    
                    query += " ORDER BY created_at DESC"
                    df = pd.read_sql_query(query, conn, params=params)
                    
                else:
                    # Default to sales
                    query = """
                        SELECT date, product_category, region, units_sold, unit_price, total_sales
                        FROM sales 
                        WHERE date BETWEEN ? AND ?
                    """
                    params = [start_date.isoformat(), end_date.isoformat()]
                    if category:
                        query += " AND product_category = ?"
                        params.append(category)
                    if region:
                        query += " AND region = ?"
                        params.append(region)
                    query += " ORDER BY date"
                    df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning(f"⚠️ No data for report in period {start_date} to {end_date}")
                return None, None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_format = file_format.lower()
            
            if report_format == 'csv':
                filename = f"report_{schedule['name'].replace(' ', '_')}_{timestamp}.csv"
                filepath = os.path.join(app.config['REPORT_PATH'], filename)
                df.to_csv(filepath, index=False)
                logger.info(f"📄 Generated CSV report: {filename} with {len(df)} records")
                return filepath, filename
            
            elif report_format == 'excel':
                filename = f"report_{schedule['name'].replace(' ', '_')}_{timestamp}.xlsx"
                filepath = os.path.join(app.config['REPORT_PATH'], filename)
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Report', index=False)
                logger.info(f"📊 Generated Excel report: {filename} with {len(df)} records")
                return filepath, filename
            
            else:  # pdf
                filename = f"report_{schedule['name'].replace(' ', '_')}_{timestamp}.pdf"
                filepath = os.path.join(app.config['REPORT_PATH'], filename)
                
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib.enums import TA_CENTER
                from reportlab.lib import colors
                
                doc = SimpleDocTemplate(filepath, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()
                
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    textColor=colors.HexColor('#667eea'),
                    alignment=TA_CENTER,
                    spaceAfter=30
                )
                elements.append(Paragraph(f"Report: {schedule['name']}", title_style))
                elements.append(Spacer(1, 20))
                
                # Add summary
                summary_data = [
                    ['Metric', 'Value'],
                    ['Report Type', report_type],
                    ['Date Range', f"{start_date} to {end_date}"],
                    ['Total Records', str(len(df))],
                    ['Generated On', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                ]
                
                if 'total_sales' in df.columns:
                    summary_data.append(['Total Sales', f"${df['total_sales'].sum():,.2f}"])
                if 'units_sold' in df.columns:
                    summary_data.append(['Total Units', f"{df['units_sold'].sum():,}"])
                
                summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(summary_table)
                elements.append(Spacer(1, 30))
                
                # Add data table - paginate for large datasets (all records included across multiple pages)
                elements.append(Paragraph("Data Details", styles['Heading2']))
                elements.append(Spacer(1, 10))
                
                # Convert dataframe to list of lists for table
                if len(df) > 0:
                    # Get column names
                    headers = list(df.columns)
                    # Convert all values to strings and handle None
                    data_rows = []
                    for _, row in df.iterrows():
                        row_data = []
                        for col in headers:
                            val = row[col]
                            if pd.isna(val):
                                row_data.append('')
                            elif isinstance(val, (int, float)):
                                if col in ['total_sales', 'predicted_sales', 'confidence_interval_lower', 'confidence_interval_upper', 'unit_price']:
                                    row_data.append(f"${val:,.2f}")
                                else:
                                    row_data.append(str(val))
                            else:
                                row_data.append(str(val))
                        data_rows.append(row_data)
                    
                    # Create table with all rows - ReportLab handles pagination automatically
                    table_data = [headers] + data_rows
                    
                    # Calculate column widths based on content
                    available_width = 7.5 * inch  # letter width minus margins
                    col_count = len(headers)
                    col_width = available_width / col_count if col_count > 0 else 1 * inch
                    col_widths = [col_width] * col_count
                    
                    data_table = Table(table_data, colWidths=col_widths, repeatRows=1)
                    data_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 8),
                        ('FONTSIZE', (0, 1), (-1, -1), 7),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    elements.append(data_table)
                
                doc.build(elements)
                logger.info(f"📑 Generated PDF report: {filename} with {len(df)} records")
                return filepath, filename
                
        except Exception as e:
            logger.error(f"❌ Error generating report file with filters: {str(e)}")
            return None, None
    
    def generate_scheduled_report(self, schedule):
        """Generate and send scheduled report"""
        logger.info(f"📅 Generating scheduled report: {schedule['name']}")
        
        end_date = datetime.now().date()
        
        # Determine date range based on frequency
        freq = schedule['frequency']
        if freq == 'daily':
            start_date = end_date - timedelta(days=1)
        elif freq == 'weekly':
            start_date = end_date - timedelta(days=7)
        elif freq == 'monthly':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)
        
        try:
            filepath, filename = self.generate_report_file(schedule, start_date, end_date)
            
            if not filepath:
                logger.warning(f"⚠️ No data for scheduled report {schedule['name']}")
                # Still update next_run to avoid constant retries
                with get_db_connection() as conn:
                    conn.execute('''
                        UPDATE scheduled_reports 
                        SET last_run = ?
                        WHERE id = ?
                    ''', (datetime.now(), schedule['id']))
                return
            
            # Read file content
            with open(filepath, 'rb') as f:
                file_content = f.read()
            
            recipients = json.loads(schedule['recipients']) if isinstance(schedule['recipients'], str) else [schedule['recipients']]
            
            # Create email with attachment
            subject = f"Sales Report: {schedule['name']}"
            
            body = f"""
            <html>
            <head><style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
            </style></head>
            <body>
                <div class="header">
                    <h1>Sales Forecasting System</h1>
                    <p>{schedule['name']} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
                <div class="content">
                    <p>Dear User,</p>
                    <p>Your scheduled report "{schedule['name']}" is attached to this email.</p>
                    <p><strong>Report Details:</strong><br>
                    - Type: {schedule['report_type']}<br>
                    - Period: {start_date} to {end_date}<br>
                    - Frequency: {schedule['frequency']}</p>
                    <p>Best regards,<br>Sales Forecasting AI System</p>
                </div>
            </body>
            </html>
            """
            
            # Send email with attachment
            msg = MIMEMultipart()
            msg['From'] = self.email_service.config['sender']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            attachment = MIMEApplication(file_content, Name=filename)
            attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
            msg.attach(attachment)
            
            if self.email_service.config['use_ssl']:
                server = smtplib.SMTP_SSL(self.email_service.config['server'], self.email_service.config['port'])
            else:
                server = smtplib.SMTP(self.email_service.config['server'], self.email_service.config['port'])
                if self.email_service.config['use_tls']:
                    server.starttls()
            
            server.login(self.email_service.config['username'], self.email_service.config['password'])
            server.send_message(msg)
            server.quit()
            
            # Clean up file
            try:
                os.remove(filepath)
            except:
                pass
            
            # Update last_run and next_run
            with get_db_connection() as conn:
                next_run = self.calculate_next_run(schedule['frequency'], schedule.get('day'), schedule['time'])
                conn.execute('''
                    UPDATE scheduled_reports 
                    SET last_run = ?, next_run = ?
                    WHERE id = ?
                ''', (datetime.now(), next_run, schedule['id']))
            
            logger.info(f"✅ Scheduled report sent successfully: {schedule['name']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send scheduled report: {str(e)}")
    
    def calculate_next_run(self, frequency, day, time_str):
        """Calculate next run time based on frequency"""
        try:
            now = datetime.now()
            time_parts = time_str.split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            
            if frequency == 'daily':
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run = next_run + timedelta(days=1)
                    
            elif frequency == 'weekly':
                day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                          'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                target_day = day_map.get(day, 0)
                days_ahead = (target_day - now.weekday() + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)
                
            elif frequency == 'monthly':
                try:
                    target_day = int(day) if day else 1
                    if target_day > 28:
                        target_day = 28
                except:
                    target_day = 1
                
                next_run = now.replace(day=target_day, hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    if next_run.month == 12:
                        next_run = next_run.replace(year=next_run.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=next_run.month + 1)
            else:
                next_run = now + timedelta(days=1)
            
            return next_run
            
        except Exception as e:
            logger.error(f"Error calculating next run: {str(e)}")
            return datetime.now() + timedelta(days=1)
    
    def run_scheduler(self):
        """Run the email scheduler"""
        logger.info("📧 Email scheduler started")
        
        while self.running:
            try:
                with get_db_connection() as conn:
                    schedules = conn.execute('''
                        SELECT * FROM scheduled_reports 
                        WHERE active = 1 AND (next_run IS NULL OR next_run <= datetime('now'))
                        ORDER BY next_run
                    ''').fetchall()
                
                for schedule in schedules:
                    schedule_dict = dict(schedule)
                    if schedule_dict['frequency'] in ['daily', 'weekly', 'monthly']:
                        self.generate_scheduled_report(schedule_dict)
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Error in email scheduler: {str(e)}")
                time.sleep(60)
    
    def start_scheduler(self):
        """Start the email scheduler in a background thread"""
        thread = threading.Thread(target=self.run_scheduler, daemon=True)
        thread.start()

# ============================================================================
# BATCH TRAINER
# ============================================================================

class BatchTrainer:
    def __init__(self):
        pass
    
    def get_models_to_train(self):
        """Get list of category-region combinations that need training"""
        try:
            with get_db_connection() as conn:
                combinations = conn.execute('''
                    SELECT DISTINCT product_category, region, COUNT(*) as count
                    FROM sales 
                    GROUP BY product_category, region
                    HAVING count >= 100
                ''').fetchall()
            return combinations
        except:
            return []
    
    def train_all_models(self):
        """Train models for all combinations"""
        combinations = self.get_models_to_train()
        
        logger.info(f"🚀 Starting batch training: {len(combinations)} combinations found")
        
        forecaster = LSTMForecaster()
        trained = 0
        failed = 0
        
        for combo in combinations:
            category = combo['product_category']
            region = combo['region']
            
            try:
                logger.info(f"🏋️ Training model for {category} / {region} ({combo['count']} records)")
                model, message = forecaster.train_model(category, region)
                if model:
                    trained += 1
                    logger.info(f"✅ Successfully trained {category} / {region}")
                else:
                    failed += 1
                    logger.error(f"❌ Failed to train {category} / {region}: {message}")
            except Exception as e:
                failed += 1
                logger.error(f"❌ Error training {category} / {region}: {str(e)}")
        
        logger.info(f"🏁 Batch training completed: {trained} trained, {failed} failed")
        return trained, failed

# ============================================================================
# AUTHENTICATION DECORATORS
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def api_login_required(f):
    """Decorator for API routes - returns JSON instead of redirect"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Session expired. Please login again.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

llm_explainer = LLMExplainer()
lstm_forecaster = LSTMForecaster()
baseline_models = BaselineModels()
anomaly_detector = AnomalyDetector()
report_generator = ReportGenerator()
email_service = EmailService(EMAIL_CONFIG)
model_scheduler = ModelScheduler()
email_scheduler = EmailScheduler(email_service)
batch_trainer = BatchTrainer()

# ============================================================================
# COMPANION MODEL HELPER
# ============================================================================

def _save_companion_models(category, region, days, hist_df, primary_session_id, primary_model_type):
    """
    Run the other two models and save their predictions under the same
    forecast_session_id with is_primary=0, so View Details can always
    show a complete 3-model comparison.
    """
    companion_map = {
        'lstm': ['arima', 'exponential_smoothing'],
        'arima': ['exponential_smoothing'],
        'exponential_smoothing': ['arima'],
    }
    companions = companion_map.get(primary_model_type.lower(), ['arima', 'exponential_smoothing'])
    # Always save all three — exclude the primary
    all_models = ['arima', 'exponential_smoothing', 'lstm']
    companions = [m for m in all_models if m != primary_model_type.lower()]

    if hist_df is None or len(hist_df) < 10:
        return

    sales_values = hist_df['total_sales'].values
    avg_price = (hist_df['total_sales'].sum() / hist_df['units_sold'].sum()) \
        if 'units_sold' in hist_df.columns and hist_df['units_sold'].sum() > 0 else 50

    last_date = pd.to_datetime(hist_df['date'].iloc[-1])
    future_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(days)]
    pred_date = datetime.now().date().isoformat()

    for model_name in companions:
        try:
            if model_name == 'arima':
                raw, msg = baseline_models.arima_forecast(sales_values, days)
                if raw is None:
                    continue
                preds = [max(0, float(v)) for v in raw['forecast']]

            elif model_name == 'exponential_smoothing':
                raw, msg = baseline_models.exponential_smoothing_forecast(sales_values, days)
                if raw is None:
                    continue
                preds = [max(0, float(v)) for v in raw['forecast']]

            elif model_name == 'lstm':
                # Call predict() — it saves its own rows with is_primary=1 under a NEW session.
                # We capture that session_id, then move those rows to is_primary=0 under
                # the primary_session_id, deleting the spurious session entirely.
                lstm_res, msg = lstm_forecaster.predict(category, region, days, force_use_available=True)
                if not lstm_res:
                    continue
                preds = [max(0, float(v)) for v in lstm_res['predictions']]

                # Find the auto-saved session that predict() just created (most recent LSTM session
                # for this cat/region that is NOT the primary session).
                with get_db_connection() as conn:
                    spurious = conn.execute('''
                        SELECT DISTINCT forecast_session_id FROM predictions
                        WHERE product_category = ? AND region = ?
                          AND model_type = 'LSTM'
                          AND forecast_session_id != ?
                          AND is_primary = 1
                        ORDER BY created_at DESC LIMIT 1
                    ''', (category, region, primary_session_id)).fetchone()

                    if spurious:
                        spurious_sid = spurious['forecast_session_id']
                        # Re-tag those rows: move them under primary session with is_primary=0
                        conn.execute('''
                            UPDATE predictions
                            SET forecast_session_id = ?, is_primary = 0
                            WHERE forecast_session_id = ?
                              AND product_category = ? AND region = ?
                        ''', (primary_session_id, spurious_sid, category, region))
                        logger.info(f"✅ Companion LSTM rows moved from session {spurious_sid} → {primary_session_id} (is_primary=0)")
                    else:
                        # Rows already gone or couldn't find — insert manually below
                        pass

                logger.info(f"✅ Companion model saved: LSTM for session {primary_session_id}")
                continue   # skip the generic insert below; rows are already handled

            else:
                continue

            units = [int(p / avg_price) for p in preds]
            lower = [p * 0.90 for p in preds]
            upper = [p * 1.10 for p in preds]

            batch = [
                (pred_date, future_dates[i], category, region,
                 preds[i], units[i], lower[i], upper[i],
                 model_name.upper(), primary_session_id, days, 0)  # is_primary=0
                for i in range(len(future_dates))
            ]

            with get_db_connection() as conn:
                conn.executemany('''
                    INSERT INTO predictions
                    (prediction_date, forecast_date, product_category, region,
                     predicted_sales, predicted_units, confidence_interval_lower,
                     confidence_interval_upper, model_type, forecast_session_id,
                     forecast_days, is_primary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)

            logger.info(f"✅ Companion model saved: {model_name.upper()} for session {primary_session_id}")

        except Exception as e:
            logger.warning(f"⚠️ Companion model {model_name} failed: {e}")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        ip_address = request.remote_addr
        
        logger.info(f"🔐 Login attempt: username='{username}', IP={ip_address}")
        
        try:
            with get_db_connection() as conn:
                user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                session.permanent = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                
                logger.info(f"✅ Login successful: user='{username}', role='{user['role']}', IP={ip_address}")
                
                # Log activity
                with get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO user_activity (user_id, action, details, ip_address)
                        VALUES (?, ?, ?, ?)
                    ''', (user['id'], 'login', 'User logged in successfully', ip_address))
                
                return redirect(url_for('dashboard'))
            else:
                logger.warning(f"❌ Login failed: Invalid credentials for username='{username}', IP={ip_address}")
                return render_template('login.html', error='Invalid username or password')
        except Exception as e:
            logger.error(f"❌ Login error: {str(e)}")
            return render_template('login.html', error='Database error. Please try again.')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        ip_address = request.remote_addr
        
        logger.info(f"📝 Registration attempt: username='{username}', email='{email}', IP={ip_address}")
        
        if password != confirm_password:
            logger.warning(f"❌ Registration failed: Passwords don't match for username='{username}'")
            return render_template('register.html', error='Passwords do not match')
        
        if len(password) < 6:
            logger.warning(f"❌ Registration failed: Password too short for username='{username}'")
            return render_template('register.html', error='Password must be at least 6 characters')
        
        try:
            with get_db_connection() as conn:
                existing_user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
                if existing_user:
                    logger.warning(f"❌ Registration failed: Username '{username}' already exists")
                    return render_template('register.html', error='Username already exists')
                
                existing_email = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
                if existing_email:
                    logger.warning(f"❌ Registration failed: Email '{email}' already registered")
                    return render_template('register.html', error='Email already registered')
                
                password_hash = generate_password_hash(password)
                conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                            (username, email, password_hash))
            
            logger.info(f"✅ Registration successful: user='{username}', email='{email}', IP={ip_address}")
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"❌ Registration error: {str(e)}")
            return render_template('register.html', error=f'Registration error: {str(e)}')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown')
    ip_address = request.remote_addr
    user_id = session.get('user_id')
    
    logger.info(f"🚪 User logout: '{username}', IP={ip_address}")
    
    # Log activity
    if user_id:
        try:
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO user_activity (user_id, action, details, ip_address)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, 'logout', 'User logged out', ip_address))
        except:
            pass
    
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    logger.info(f"📊 Dashboard accessed: user='{session.get('username')}'")
    return render_template('dashboard.html')

@app.route('/api/user-info')
@api_login_required
def get_user_info():
    """Return current session user info for frontend role checks"""
    return jsonify({
        'user_id': session.get('user_id'),
        'username': session.get('username'),
        'role': session.get('role', 'user')
    })

@app.route('/api/dashboard-stats')
@api_login_required
def get_dashboard_stats():
    try:
        with get_db_connection() as conn:
            total_sales = conn.execute('SELECT SUM(total_sales) as total FROM sales').fetchone()
            total_sales = total_sales['total'] if total_sales and total_sales['total'] else 0
            
            avg_daily_sales = conn.execute('SELECT AVG(total_sales) as avg FROM sales').fetchone()
            avg_daily_sales = avg_daily_sales['avg'] if avg_daily_sales and avg_daily_sales['avg'] else 0
            
            total_units = conn.execute('SELECT SUM(units_sold) as total FROM sales').fetchone()
            total_units = total_units['total'] if total_units and total_units['total'] else 0
            
            total_predictions = conn.execute('SELECT COUNT(*) as count FROM predictions WHERE COALESCE(is_primary, 1) = 1').fetchone()
            total_predictions = total_predictions['count'] if total_predictions and total_predictions['count'] else 0
            
            today = date.today()
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            sales_week = conn.execute('SELECT SUM(total_sales) as total FROM sales WHERE date >= ?', (week_ago.isoformat(),)).fetchone()
            sales_week = sales_week['total'] if sales_week and sales_week['total'] else 0
            
            sales_month = conn.execute('SELECT SUM(total_sales) as total FROM sales WHERE date >= ?', (month_ago.isoformat(),)).fetchone()
            sales_month = sales_month['total'] if sales_month and sales_month['total'] else 0
            
            monthly_growth = 0
            if sales_month > 0 and total_sales > sales_month:
                monthly_growth = ((total_sales - sales_month) / sales_month) * 100
            
            categories = conn.execute('''
                SELECT product_category, SUM(total_sales) as total
                FROM sales GROUP BY product_category ORDER BY total DESC LIMIT 5
            ''').fetchall()
            
            regions = conn.execute('''
                SELECT region, SUM(total_sales) as total
                FROM sales GROUP BY region ORDER BY total DESC
            ''').fetchall()
            
            # Only show PRIMARY forecasts (is_primary=1) in the dashboard
            # Each forecast session (forecast_session_id) is a single forecast run
            # We only show the model that the user actually ran (is_primary=1)
            # Companion models (is_primary=0) are hidden from the dashboard list
            recent_preds = conn.execute('''
                SELECT 
                    MIN(id) as id,
                    product_category,
                    region,
                    SUM(predicted_sales) as predicted_sales,
                    MIN(forecast_date) as forecast_date,
                    MAX(forecast_date) as forecast_end_date,
                    model_type,
                    MIN(confidence_interval_lower) as confidence_interval_lower,
                    MAX(confidence_interval_upper) as confidence_interval_upper,
                    SUM(predicted_units) as predicted_units,
                    COALESCE(forecast_session_id, prediction_date || '|' || product_category || '|' || region || '|' || model_type) as session_id,
                    COUNT(*) as forecast_days_count,
                    MAX(COALESCE(forecast_days, 0)) as forecast_days,
                    prediction_date,
                    is_primary
                FROM predictions
                WHERE is_primary = 1
                GROUP BY COALESCE(forecast_session_id, prediction_date || '|' || product_category || '|' || region || '|' || model_type)
                ORDER BY MAX(created_at) DESC
                LIMIT 10
            ''').fetchall()
        
        return jsonify({
            'stats': {
                'total_sales': float(total_sales),
                'avg_daily_sales': float(avg_daily_sales),
                'total_units': int(total_units),
                'total_predictions': int(total_predictions),
                'monthly_growth': float(monthly_growth),
                'anomaly_count': 0
            },
            'categories': [{'category': row['product_category'], 'total': float(row['total'])} for row in categories],
            'regions': [{'region': row['region'], 'total': float(row['total'])} for row in regions],
            'recent_predictions': [{
                'id': row['id'],
                'category': row['product_category'],
                'region': row['region'],
                'predicted_sales': float(row['predicted_sales']),
                'date': row['forecast_date'],
                'forecast_end_date': row['forecast_end_date'],
                'model_type': row['model_type'],
                'lower_bound': float(row['confidence_interval_lower']) if row['confidence_interval_lower'] else None,
                'upper_bound': float(row['confidence_interval_upper']) if row['confidence_interval_upper'] else None,
                'predicted_units': int(row['predicted_units']) if row['predicted_units'] else 0,
                'session_id': row['session_id'],
                'forecast_days': int(row['forecast_days']) if row['forecast_days'] else int(row['forecast_days_count']),
                'prediction_date': row['prediction_date']
            } for row in recent_preds]
        })
    except Exception as e:
        logger.error(f"❌ Dashboard stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ADDITIONAL ROUTE HANDLERS
# ============================================================================

@app.route('/data-management')
@login_required
def data_management():
    """Data management page"""
    logger.info(f"📁 Data management page accessed: user='{session.get('username')}'")
    return render_template('data_management.html')

@app.route('/forecast')
@login_required
def forecast():
    """Forecast generation page"""
    logger.info(f"🔮 Forecast page accessed: user='{session.get('username')}'")
    return render_template('forecast.html')

@app.route('/model-training')
@login_required
def model_training():
    """Model training page"""
    logger.info(f"🏋️ Model training page accessed: user='{session.get('username')}'")
    return render_template('model_training.html')

@app.route('/trends-analysis')
@login_required
def trends_analysis():
    """Trends analysis page"""
    logger.info(f"📈 Trends analysis page accessed: user='{session.get('username')}'")
    return render_template('trends_analysis.html')

@app.route('/reports')
@login_required
def reports():
    """Reports page"""
    logger.info(f"📄 Reports page accessed: user='{session.get('username')}'")
    return render_template('reports.html')

@app.route('/anomalies')
@login_required
def anomalies():
    """Anomalies page"""
    logger.info(f"🔍 Anomalies page accessed: user='{session.get('username')}'")
    return render_template('anomalies.html')

@app.route('/what-if')
@login_required
def what_if():
    """What-if analysis page"""
    logger.info(f"❓ What-if analysis page accessed: user='{session.get('username')}'")
    return render_template('what_if.html')

@app.route('/admin/users')
@login_required
def admin_users():
    """Admin user management page"""
    if session.get('role') != 'admin':
        logger.warning(f"⚠️ Unauthorized admin page access by user='{session.get('username')}'")
        return redirect(url_for('dashboard'))
    logger.info(f"👥 Admin users page accessed: user='{session.get('username')}'")
    return render_template('admin_users.html')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/sales-data')
@api_login_required
def get_sales_data():
    """Get sales data with optional filters"""
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM sales WHERE 1=1"
            params = []
            
            category = request.args.get('category')
            if category:
                query += " AND product_category = ?"
                params.append(category)
            
            region = request.args.get('region')
            if region:
                query += " AND region = ?"
                params.append(region)
            
            start_date = request.args.get('start_date')
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            end_date = request.args.get('end_date')
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC"
            
            limit = request.args.get('limit', 100)
            offset = request.args.get('offset', 0)
            query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor = conn.execute(query, params)
            data = [dict(row) for row in cursor.fetchall()]
        
        logger.debug(f"📊 Retrieved {len(data)} sales records")  # Changed to debug to avoid clutter
        return jsonify(data)
    except Exception as e:
        logger.error(f"❌ Sales data error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sales-count')
@api_login_required
def get_sales_count():
    """Get total count of sales records with filters (for pagination)"""
    try:
        with get_db_connection() as conn:
            query = "SELECT COUNT(*) as count FROM sales WHERE 1=1"
            params = []
            
            category = request.args.get('category')
            if category:
                query += " AND product_category = ?"
                params.append(category)
            
            region = request.args.get('region')
            if region:
                query += " AND region = ?"
                params.append(region)
            
            start_date = request.args.get('start_date')
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            end_date = request.args.get('end_date')
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            cursor = conn.execute(query, params)
            result = cursor.fetchone()
            
        return jsonify({'count': result['count'] if result else 0})
    except Exception as e:
        logger.error(f"❌ Sales count error: {str(e)}")
        return jsonify({'count': 0, 'error': str(e)}), 500


@app.route('/api/sales-summary')
@api_login_required
def get_sales_summary():
    """Get sales summary with filters for reports page"""
    try:
        with get_db_connection() as conn:
            query = """
                SELECT 
                    SUM(total_sales) as total_sales,
                    SUM(units_sold) as total_units,
                    AVG(total_sales) as avg_daily_sales
                FROM sales WHERE 1=1
            """
            params = []
            
            start_date = request.args.get('start_date')
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            end_date = request.args.get('end_date')
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            category = request.args.get('category')
            if category:
                query += " AND product_category = ?"
                params.append(category)
            
            region = request.args.get('region')
            if region:
                query += " AND region = ?"
                params.append(region)
            
            cursor = conn.execute(query, params)
            result = cursor.fetchone()
            
            # Get category breakdown
            cat_query = """
                SELECT product_category as category, SUM(total_sales) as total
                FROM sales WHERE 1=1
            """
            cat_params = []
            
            if start_date:
                cat_query += " AND date >= ?"
                cat_params.append(start_date)
            if end_date:
                cat_query += " AND date <= ?"
                cat_params.append(end_date)
            if category:
                cat_query += " AND product_category = ?"
                cat_params.append(category)
            if region:
                cat_query += " AND region = ?"
                cat_params.append(region)
            
            cat_query += " GROUP BY product_category ORDER BY total DESC LIMIT 5"
            
            categories = conn.execute(cat_query, cat_params).fetchall()
        
        return jsonify({
            'total_sales': float(result['total_sales']) if result and result['total_sales'] else 0,
            'total_units': int(result['total_units']) if result and result['total_units'] else 0,
            'avg_daily_sales': float(result['avg_daily_sales']) if result and result['avg_daily_sales'] else 0,
            'categories': [{'category': row['category'], 'total': float(row['total'])} for row in categories]
        })
    except Exception as e:
        logger.error(f"❌ Sales summary error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sales-trends')
@api_login_required
def get_sales_trends():
    """Get sales trends data"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT date, SUM(total_sales) as total
                FROM sales 
                WHERE date >= date('now', '-30 days')
                GROUP BY date
                ORDER BY date
            ''')
            daily = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({'daily': daily})
    except Exception as e:
        logger.error(f"❌ Sales trends error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends')
@api_login_required
def get_trends():
    """Get trends analysis data"""
    try:
        with get_db_connection() as conn:
            monthly = conn.execute('''
                SELECT strftime('%Y-%m', date) as month,
                       SUM(total_sales) as total_sales,
                       COUNT(*) as transaction_count
                FROM sales
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12
            ''').fetchall()
            
            categories = conn.execute('''
                SELECT product_category,
                       SUM(total_sales) as total_sales,
                       SUM(units_sold) as total_units
                FROM sales
                GROUP BY product_category
                ORDER BY total_sales DESC
            ''').fetchall()
        
        return jsonify({
            'monthly': [dict(row) for row in monthly],
            'categories': [dict(row) for row in categories]
        })
    except Exception as e:
        logger.error(f"❌ Trends error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trends-analysis-data')
@api_login_required
def get_trends_analysis_data():
    """Get trends analysis data with filters for reports page"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        category = request.args.get('category')
        region = request.args.get('region')
        
        with get_db_connection() as conn:
            # Base query for trends from database
            query = """
                SELECT * FROM trends 
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND start_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND end_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY created_at DESC"
            
            cursor = conn.execute(query, params)
            trends = [dict(row) for row in cursor.fetchall()]
            
            # Get sales data for decomposition
            sales_query = """
                SELECT date, total_sales, product_category, region
                FROM sales 
                WHERE 1=1
            """
            sales_params = []
            
            if start_date:
                sales_query += " AND date >= ?"
                sales_params.append(start_date)
            if end_date:
                sales_query += " AND date <= ?"
                sales_params.append(end_date)
            if category:
                sales_query += " AND product_category = ?"
                sales_params.append(category)
            if region:
                sales_query += " AND region = ?"
                sales_params.append(region)
            
            sales_query += " ORDER BY date"
            
            df = pd.read_sql_query(sales_query, conn, params=sales_params)
            
            decomposition = None
            if len(df) > 30:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df_daily = df.groupby(df['date'].dt.date)['total_sales'].sum().reset_index()
                    df_daily.columns = ['date', 'total_sales']
                    df_daily['date'] = pd.to_datetime(df_daily['date'])
                    df_daily.set_index('date', inplace=True)
                    
                    decomposition = {
                        'dates': df_daily.index.strftime('%Y-%m-%d').tolist(),
                        'trend': df_daily['total_sales'].rolling(window=7, min_periods=1).mean().tolist(),
                        'seasonal': [0] * len(df_daily),
                        'residual': [0] * len(df_daily)
                    }
                except Exception as e:
                    logger.warning(f"Could not generate decomposition: {e}")
            
        return jsonify({
            'trends': trends,
            'decomposition': decomposition
        })
    except Exception as e:
        logger.error(f"❌ Trends analysis data error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
@api_login_required
def get_predictions():
    """Get recent predictions"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM predictions 
                ORDER BY forecast_date DESC 
                LIMIT 50
            ''')
            predictions = [dict(row) for row in cursor.fetchall()]
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"❌ Predictions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-sales-record', methods=['POST'])
@api_login_required
def add_sales_record():
    """Add a single sales record manually"""
    try:
        data = request.json
        
        required = ['date', 'category', 'region', 'units', 'price']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'message': f'Missing field: {field}'}), 400
        
        total_sales = data['units'] * data['price']
        
        logger.info(f"📝 Adding sales record: date={data['date']}, category={data['category']}, region={data['region']}, units={data['units']}, price=${data['price']}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO sales 
                (date, product_category, region, units_sold, unit_price, total_sales,
                 promotion_flag, holiday_flag, discount_percent, stock_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['date'],
                data['category'],
                data['region'],
                data['units'],
                data['price'],
                total_sales,
                data.get('promotion', 0),
                data.get('holiday', 0),
                data.get('discount', 0),
                data.get('stock_level', 100)
            ))
        
        logger.info(f"✅ Sales record added: ${total_sales:,.2f}")
        return jsonify({'success': True, 'message': 'Sales record added successfully'})
    except Exception as e:
        logger.error(f"❌ Add sales record error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/upload-sales', methods=['POST'])
@api_login_required
def upload_sales():
    """Upload sales data from CSV file with intelligent category mapping"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'message': 'Please upload a CSV file'}), 400
        
        logger.info(f"📁 CSV upload started: filename='{file.filename}', user='{session.get('username')}'")
        
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_data = csv.DictReader(stream)
        
        category_mapping = {
            "apple": "Produce (Fruits & Vegetables)", "apples": "Produce (Fruits & Vegetables)",
            "banana": "Produce (Fruits & Vegetables)", "bananas": "Produce (Fruits & Vegetables)",
            "orange": "Produce (Fruits & Vegetables)", "oranges": "Produce (Fruits & Vegetables)",
            "milk": "Dairy & Eggs", "cheese": "Dairy & Eggs", "eggs": "Dairy & Eggs",
            "bread": "Bakery", "cake": "Bakery", "muffin": "Bakery",
            "soda": "Beverages", "juice": "Beverages", "water": "Beverages",
            "beef": "Meat & Seafood", "chicken": "Meat & Seafood", "fish": "Meat & Seafood",
            "frozen": "Frozen Foods", "ice cream": "Frozen Foods"
        }
        
        valid_categories = [
            "Grocery (Dry & Canned Goods)", "Produce (Fruits & Vegetables)", "Dairy & Eggs",
            "Meat & Seafood", "Frozen Foods", "Bakery", "Beverages", "Deli & Prepared Foods",
            "Household & Cleaning", "Health, Beauty & Personal Care", "General Merchandise / Non-Food",
            "Specialty & Wellness", "Floral", "Others"
        ]
        
        def map_category_to_valid(category_input):
            if not category_input:
                return "Others"
            category_lower = category_input.strip().lower()
            for valid_cat in valid_categories:
                if valid_cat.lower() == category_lower:
                    return valid_cat
            for keyword, mapped_category in category_mapping.items():
                if keyword in category_lower or category_lower in keyword:
                    return mapped_category
            for valid_cat in valid_categories:
                if valid_cat.lower() in category_lower or category_lower in valid_cat.lower():
                    return valid_cat
            return "Others"
        
        with get_db_connection() as conn:
            success_count = 0
            error_count = 0
            warning_count = 0
            errors = []
            warnings = []
            
            for row_num, row in enumerate(csv_data, start=2):
                try:
                    required_fields = ['date', 'product_category', 'region', 'units_sold', 'unit_price']
                    missing_fields = [f for f in required_fields if f not in row or not row[f]]
                    
                    if missing_fields:
                        error_count += 1
                        errors.append(f"Row {row_num}: Missing required fields: {', '.join(missing_fields)}")
                        continue
                    
                    try:
                        datetime.strptime(row['date'], '%Y-%m-%d')
                    except ValueError:
                        error_count += 1
                        errors.append(f"Row {row_num}: Invalid date format '{row['date']}'. Use YYYY-MM-DD")
                        continue
                    
                    try:
                        units = int(row['units_sold'])
                        if units <= 0:
                            raise ValueError("Units must be positive")
                    except ValueError as e:
                        error_count += 1
                        errors.append(f"Row {row_num}: Invalid units_sold '{row['units_sold']}': {str(e)}")
                        continue
                    
                    try:
                        price = float(row['unit_price'])
                        if price <= 0:
                            raise ValueError("Price must be positive")
                    except ValueError as e:
                        error_count += 1
                        errors.append(f"Row {row_num}: Invalid unit_price '{row['unit_price']}': {str(e)}")
                        continue
                    
                    original_category = row['product_category'].strip()
                    mapped_category = map_category_to_valid(original_category)
                    
                    if mapped_category != original_category:
                        warning_count += 1
                        warnings.append(f"Row {row_num}: Category '{original_category}' mapped to '{mapped_category}'")
                    
                    total_sales = units * price
                    promotion_flag = 1 if str(row.get('promotion_flag', '0')).strip() in ['1', 'true', 'True', 'yes', 'Yes'] else 0
                    holiday_flag = 1 if str(row.get('holiday_flag', '0')).strip() in ['1', 'true', 'True', 'yes', 'Yes'] else 0
                    
                    try:
                        discount_percent = float(row.get('discount_percent', 0))
                    except ValueError:
                        discount_percent = 0
                    
                    try:
                        stock_level = int(row.get('stock_level', 100))
                    except ValueError:
                        stock_level = 100
                    
                    conn.execute('''
                        INSERT INTO sales 
                        (date, product_category, region, units_sold, unit_price, total_sales,
                         promotion_flag, holiday_flag, discount_percent, stock_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['date'], mapped_category, row['region'].strip(), units, price, total_sales,
                        promotion_flag, holiday_flag, discount_percent, stock_level
                    ))
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append(f"Row {row_num}: {str(e)}")
        
        logger.info(f"✅ CSV upload completed: {success_count} inserted, {warning_count} warnings, {error_count} errors")
        
        message_parts = []
        if success_count > 0:
            message_parts.append(f"✅ Uploaded {success_count} records successfully")
        if warning_count > 0:
            message_parts.append(f"⚠️ {warning_count} categories were automatically mapped")
        if error_count > 0:
            message_parts.append(f"❌ {error_count} errors occurred")
        
        message = " | ".join(message_parts)
        
        response_data = {
            'success': True,
            'message': message,
            'success_count': success_count,
            'warning_count': warning_count,
            'error_count': error_count,
            'errors': errors[:20]
        }
        
        if warnings:
            response_data['warnings'] = warnings[:20]
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        return jsonify({'success': False, 'message': f"Upload error: {str(e)}"}), 500

@app.route('/api/delete-sales/<int:sale_id>', methods=['DELETE'])
@api_login_required
def delete_sales(sale_id):
    """Delete a sales record"""
    try:
        logger.info(f"🗑️ Deleting sales record ID={sale_id}, user='{session.get('username')}'")
        with get_db_connection() as conn:
            conn.execute('DELETE FROM sales WHERE id = ?', (sale_id,))
        logger.info(f"✅ Sales record {sale_id} deleted")
        return jsonify({'success': True, 'message': 'Record deleted successfully'})
    except Exception as e:
        logger.error(f"❌ Delete error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
@api_login_required
def train_model():
    """Train a model for specific category and region"""
    try:
        data = request.json
        category = data.get('category')
        region = data.get('region')
        
        if not category or not region:
            return jsonify({'success': False, 'message': 'Category and region are required'}), 400
        
        logger.info(f"🚀 Training model request: category='{category}', region='{region}', user='{session.get('username')}'")
        
        forecaster = LSTMForecaster()
        model, message = forecaster.train_model(category, region)
        
        if model:
            logger.info(f"✅ Model training completed: {category}/{region}")
            return jsonify({'success': True, 'message': message})
        else:
            logger.error(f"❌ Model training failed: {category}/{region} - {message}")
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        logger.error(f"❌ Train model error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/bulk-train', methods=['POST'])
@api_login_required
def bulk_train():
    """Train models for all combinations"""
    try:
        logger.info(f"🚀 Bulk training requested by user='{session.get('username')}'")
        trained, failed = batch_trainer.train_all_models()
        return jsonify({
            'success': True,
            'message': f'Trained {trained} models, {failed} failed'
        })
    except Exception as e:
        logger.error(f"❌ Bulk train error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/generate-forecast', methods=['POST'])
@api_login_required
def generate_forecast():
    """Generate sales forecast and save to database"""
    try:
        data = request.json
        category = data.get('category')
        region = data.get('region')
        days = data.get('days', 30)
        model_type = data.get('model_type', 'lstm')
        force_use_available = data.get('force_use_available', False)
        
        if not category or not region:
            return jsonify({'success': False, 'message': 'Category and region are required'}), 400
        
        logger.info(f"🔮 Forecast request: category='{category}', region='{region}', days={days}, model='{model_type}', force={force_use_available}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            count_result = conn.execute('''
                SELECT COUNT(*) as count FROM sales 
                WHERE product_category = ? AND region = ?
            ''', (category, region)).fetchone()
        
        record_count = count_result['count'] if count_result else 0
        
        if record_count < 10:
            logger.warning(f"⚠️ Insufficient data for forecast: {record_count} records for {category}/{region}")
            return jsonify({
                'success': False,
                'message': f'Insufficient data: {record_count} records found. Add at least 10 sales records for this category/region.'
            }), 400

        if model_type == 'lstm' and record_count < 90 and not force_use_available:
            logger.info(f"⚠️ LSTM needs more data: {record_count} records (90+ recommended)")
            return jsonify({
                'success': False,
                'message': f'Only {record_count} records available. 90+ records give the best accuracy. Check "Use available records" to forecast now with reduced accuracy.',
                'record_count': record_count,
                'can_force': True
            }), 400
        
        start_time = time.time()
        
        # Get historical data summary for insight table
        with get_db_connection() as conn:
            hist_df = pd.read_sql_query('''
                SELECT date, total_sales, units_sold FROM sales 
                WHERE product_category = ? AND region = ?
                ORDER BY date
            ''', conn, params=(category, region))
        
        historical_data = {
            'avg_sales': float(hist_df['total_sales'].mean()) if len(hist_df) > 0 else 0,
            'trend': 'increasing' if len(hist_df) >= 60 and hist_df['total_sales'].iloc[-30:].mean() > hist_df['total_sales'].iloc[:30].mean() else 'stable' if len(hist_df) >= 60 else 'stable',
            'seasonality': 'strong' if len(hist_df) > 90 else 'moderate' if len(hist_df) > 30 else 'weak'
        }
        
        if model_type == 'lstm':
            result, message = lstm_forecaster.predict(category, region, days, force_use_available)
        elif model_type in ('arima', 'exponential_smoothing'):
            df = hist_df
            if len(df) < 10:
                return jsonify({'success': False, 'message': f'Insufficient data for {model_type} (need at least 10 records)'}), 400

            if model_type == 'arima':
                raw, message = baseline_models.arima_forecast(df['total_sales'].values, days)
            else:
                raw, message = baseline_models.exponential_smoothing_forecast(df['total_sales'].values, days)

            if raw is None:
                return jsonify({'success': False, 'message': message}), 400

            last_date = pd.to_datetime(df['date'].iloc[-1])
            future_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(days)]
            predictions = [max(0, float(v)) for v in raw['forecast']]

            avg_price = (df['total_sales'].sum() / df['units_sold'].sum()) if df['units_sold'].sum() > 0 else 50
            predicted_units = [int(p / avg_price) for p in predictions]

            lower_bound = [p * 0.90 for p in predictions]
            upper_bound = [p * 1.10 for p in predictions]
            
            # Calculate growth rate
            growth_rate = ((predictions[-1] / predictions[0]) - 1) * 100 if len(predictions) > 0 and predictions[0] > 0 else 0
            avg_confidence = 85.0  # Default for classical models

            result = {
                'dates': future_dates,
                'predictions': predictions,
                'predicted_units': predicted_units,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'avg_price': float(avg_price),
                'trend_analysis': {},
                'growth_rate': growth_rate,
                'avg_confidence': avg_confidence,
                'llm_explanation': f"ARIMA model forecast for {category} in {region} over {days} days. Total predicted sales: ${sum(predictions):,.2f} with {growth_rate:.1f}% expected growth.",
                'metrics': raw.get('metrics', {}),
            }
            message = 'Success'
            
            # SAVE PREDICTIONS TO DATABASE FOR ARIMA/EXPONENTIAL SMOOTHING
            arima_session_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
            with get_db_connection() as conn:
                for i in range(len(future_dates)):
                    conn.execute('''
                        INSERT INTO predictions 
                        (prediction_date, forecast_date, product_category, region, 
                         predicted_sales, predicted_units, confidence_interval_lower, 
                         confidence_interval_upper, model_type, forecast_session_id, forecast_days,
                         is_primary)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().date().isoformat(),
                        future_dates[i],
                        category,
                        region,
                        float(predictions[i]),
                        int(predicted_units[i]),
                        float(lower_bound[i]),
                        float(upper_bound[i]),
                        model_type.upper(),
                        arima_session_id,
                        days,
                        1  # is_primary
                    ))
            
        else:
            return jsonify({'success': False, 'message': 'Invalid model type'}), 400
        
        if result:
            elapsed = time.time() - start_time
            logger.info(f"✅ Forecast generated in {elapsed:.2f}s: {category}/{region} - Total predicted: ${sum(result['predictions']):,.2f}")

            # ----------------------------------------------------------------
            # Save companion models (is_primary=0) so View Details can always
            # show a 3-model comparison without re-running anything.
            # We retrieve the session_id that was just saved.
            # ----------------------------------------------------------------
            try:
                primary_session_id = result.get('_session_id')  # set below per model
                if not primary_session_id:
                    # Look up the most-recently-inserted session for this cat/region
                    with get_db_connection() as conn:
                        row = conn.execute('''
                            SELECT forecast_session_id FROM predictions
                            WHERE product_category = ? AND region = ? AND is_primary = 1
                              AND forecast_session_id IS NOT NULL
                            ORDER BY created_at DESC LIMIT 1
                        ''', (category, region)).fetchone()
                        primary_session_id = row['forecast_session_id'] if row else None

                if primary_session_id:
                    _save_companion_models(category, region, days, hist_df, primary_session_id, model_type)
            except Exception as _ce:
                logger.warning(f"⚠️ Could not save companion models: {_ce}")

            result['historical_data'] = historical_data
            return jsonify({'success': True, 'data': result})
        else:
            logger.error(f"❌ Forecast failed: {message}")
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        logger.error(f"❌ Generate forecast error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/compare-models', methods=['POST'])
@api_login_required
def compare_models():
    """Compare different forecasting models"""
    try:
        data = request.json
        category = data.get('category')
        region = data.get('region')
        days = data.get('days', 30)
        
        if not category or not region:
            return jsonify({'success': False, 'message': 'Category and region are required'}), 400
        
        logger.info(f"📊 Model comparison request: category='{category}', region='{region}', days={days}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            df = pd.read_sql_query('''
                SELECT date, total_sales FROM sales 
                WHERE product_category = ? AND region = ?
                ORDER BY date
            ''', conn, params=(category, region))
        
        if len(df) < 10:
            return jsonify({'success': False, 'message': 'Insufficient data for comparison (need at least 10 records)'}), 400
        
        sales_data = df['total_sales'].values
        
        results = {}
        
        arima_result, _ = baseline_models.arima_forecast(sales_data, days)
        if arima_result:
            results['arima'] = arima_result
        
        exp_result, _ = baseline_models.exponential_smoothing_forecast(sales_data, days)
        if exp_result:
            results['exponential_smoothing'] = exp_result
        
        try:
            lstm_result, _ = lstm_forecaster.predict(category, region, days, force_use_available=True)
            if lstm_result:
                results['lstm'] = lstm_result
                # predict() auto-saved these rows with is_primary=1 — delete them so the
                # compare-models call doesn't pollute the dashboard with ghost entries.
                with get_db_connection() as conn:
                    spurious = conn.execute('''
                        SELECT DISTINCT forecast_session_id FROM predictions
                        WHERE product_category = ? AND region = ?
                          AND model_type = 'LSTM' AND is_primary = 1
                        ORDER BY created_at DESC LIMIT 1
                    ''', (category, region)).fetchone()
                    if spurious:
                        conn.execute('''
                            DELETE FROM predictions
                            WHERE forecast_session_id = ?
                        ''', (spurious['forecast_session_id'],))
        except Exception as e:
            logger.warning(f"⚠️ LSTM comparison failed: {str(e)}")
        
        last_date = pd.to_datetime(df['date'].iloc[-1])
        dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        
        comparison = {}
        if 'lstm' in results and 'arima' in results:
            lstm_mae = results['lstm'].get('metrics', {}).get('mae', 0)
            arima_mae = results['arima'].get('metrics', {}).get('mae', 1)
            if arima_mae > 0:
                comparison['lstm_vs_arima'] = ((arima_mae - lstm_mae) / arima_mae) * 100
        
        logger.info(f"✅ Model comparison completed: {len(results)} models compared")
        
        if not results:
            return jsonify({'success': False, 'message': 'Could not generate any comparison results'}), 400

        return jsonify({
            'success': True,
            'results': results,
            'dates': dates,
            'comparison': comparison
        })
        
    except Exception as e:
        logger.error(f"❌ Model comparison error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/model-metrics')
@api_login_required
def get_model_metrics():
    """Get model training metrics"""
    try:
        limit = request.args.get('limit', 10)
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM model_metrics 
                ORDER BY training_date DESC 
                LIMIT ?
            ''', (limit,))
            metrics = [dict(row) for row in cursor.fetchall()]
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"❌ Model metrics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trend-analysis')
@api_login_required
def trend_analysis():
    """Get trend analysis data"""
    try:
        with get_db_connection() as conn:
            sales_data = conn.execute('''
                SELECT date, SUM(total_sales) as total_sales
                FROM sales
                GROUP BY date
                ORDER BY date
                LIMIT 500
            ''').fetchall()
            
            trends = conn.execute('''
                SELECT * FROM trends 
                ORDER BY created_at DESC 
                LIMIT 20
            ''').fetchall()
        
        decomposition = None
        if len(sales_data) > 30:
            try:
                df = pd.DataFrame([dict(row) for row in sales_data])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                decomposition = {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'trend': df['total_sales'].rolling(window=7, min_periods=1).mean().tolist(),
                    'seasonal': [0] * len(df),
                    'residual': [0] * len(df)
                }
            except:
                pass
        
        try:
            trends_data = {
                'growth_rate': 5.2,
                'most_volatile': 'Electronics',
                'best_day': 'Saturday',
                'seasonal_strength': 25.5,
                'peak_periods': 'Q4 and holidays'
            }
            llm_explanation = llm_explainer.explain_trends(trends_data, 'last 30 days')
        except:
            llm_explanation = "Unable to generate AI explanation at this time."
        
        return jsonify({
            'decomposition': decomposition,
            'trends': [dict(row) for row in trends],
            'llm_explanation': llm_explanation
        })
        
    except Exception as e:
        logger.error(f"❌ Trend analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-anomalies')
@api_login_required
def detect_anomalies():
    """Detect anomalies in sales data with AI-powered business explanation"""
    try:
        category = request.args.get('category', '')
        region = request.args.get('region', '')
        
        logger.info(f"🔍 Anomaly detection request: category='{category or 'All'}', region='{region or 'All'}', user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            query = '''
                SELECT * FROM sales 
                ORDER BY date DESC 
                LIMIT 500
            '''
            
            # If category or region filters are provided, apply them
            if category or region:
                filter_parts = []
                params = []
                if category:
                    filter_parts.append("product_category = ?")
                    params.append(category)
                if region:
                    filter_parts.append("region = ?")
                    params.append(region)
                
                if filter_parts:
                    query = f'''
                        SELECT * FROM sales 
                        WHERE {' AND '.join(filter_parts)}
                        ORDER BY date DESC 
                        LIMIT 500
                    '''
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
            else:
                df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logger.info("📭 No sales data available for anomaly detection")
            return jsonify({
                'success': True,
                'business_summary': "📭 No sales data available to analyze.",
                'what_it_means': "Upload sales data first to enable anomaly detection.",
                'recommended_actions': ["Upload CSV files or add sales records manually in Data Management"],
                'priority': 'LOW',
                'raw_anomalies': {'iqr': [], 'zscore': [], 'isolation_forest': []},
                'total_count': 0,
                'explanation': 'No sales data available. Upload data first to detect anomalies.'
            })
        
        anomalies = anomaly_detector.detect_anomalies(df)
        
        serialized = {}
        for method, items in anomalies.items():
            serialized[method] = []
            for a in items[:15]:  # Limit to 15 per method for performance
                item = {}
                for k, v in a.items():
                    if hasattr(v, 'isoformat'):
                        item[k] = v.isoformat()[:10]
                    elif hasattr(v, 'item'):
                        item[k] = float(v)
                    else:
                        item[k] = v
                serialized[method].append(item)
        
        # Generate business-friendly explanation
        context = {'category': category or 'All', 'region': region or 'All'}
        business_explanation = anomaly_detector.generate_business_explanation(anomalies, df, context)
        
        total_count = sum(len(v) for v in anomalies.values())
        technical_explanation = anomaly_detector.get_anomaly_explanation(anomalies)
        
        logger.info(f"✅ Anomaly detection completed: {total_count} anomalies found")
        
        return jsonify({
            'success': True,
            'business_summary': business_explanation.get('summary', f"Found {total_count} anomalies in your sales data."),
            'what_it_means': business_explanation.get('what_it_means', "Unusual sales patterns detected that may require investigation."),
            'recommended_actions': business_explanation.get('actions', ["Review the anomalies listed below", "Verify data accuracy for the affected dates"]),
            'priority': business_explanation.get('priority', 'MEDIUM'),
            'raw_anomalies': serialized,
            'total_count': total_count,
            'explanation': technical_explanation
        })
        
    except Exception as e:
        logger.error(f"❌ Anomaly detection error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/generate-report', methods=['GET', 'POST'])
@api_login_required
def generate_report():
    """Generate and download report"""
    try:
        if request.method == 'GET':
            report_type = request.args.get('type', 'sales')
            format = request.args.get('format', 'csv')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            send_email = False
            email_recipients = []
        else:
            data = request.json or {}
            report_type = data.get('type', 'sales')
            format = data.get('format', 'csv')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            send_email = data.get('send_email', False)
            email_recipients = data.get('email_recipients', [])
        
        logger.info(f"📄 Report generation: type='{report_type}', format='{format}', period={start_date} to {end_date}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            if report_type == 'sales':
                query = '''
                    SELECT * FROM sales 
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            elif report_type == 'forecast':
                query = '''
                    SELECT * FROM predictions 
                    WHERE forecast_date BETWEEN ? AND ?
                    ORDER BY forecast_date
                '''
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            elif report_type == 'trends':
                query = '''
                    SELECT * FROM trends 
                    WHERE start_date >= ? AND end_date <= ?
                    ORDER BY start_date
                '''
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            else:
                query = '''
                    SELECT * FROM sales 
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        if df.empty:
            logger.warning(f"⚠️ No data found for report period {start_date} to {end_date}")
            return jsonify({'success': False, 'message': 'No data found for selected period'}), 404
        
        if format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            logger.info(f"✅ CSV report generated: {len(df)} records")
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'report_{report_type}_{start_date}_{end_date}.csv'
            )
            
        elif format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=report_type.capitalize(), index=False)
            output.seek(0)
            
            logger.info(f"✅ Excel report generated: {len(df)} records")
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'report_{report_type}_{start_date}_{end_date}.xlsx'
            )
            
        elif format == 'pdf':
            report_data = {
                'summary': {
                    'total_sales': float(df['total_sales'].sum()) if 'total_sales' in df.columns else 0,
                    'total_units': int(df['units_sold'].sum()) if 'units_sold' in df.columns else 0,
                },
                'daily': df.to_dict('records') if len(df) <= 100 else df.head(100).to_dict('records')
            }
            
            date_range = {'start': start_date, 'end': end_date}
            filepath, filename = report_generator.generate_sales_report(report_data, date_range, 'pdf')
            
            logger.info(f"✅ PDF report generated: {filename}")
            return send_file(
                filepath,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        
        if send_email and email_recipients:
            email_service.send_report(email_recipients, f"{report_type.capitalize()} Report", {}, format)
        
        return jsonify({'success': True, 'message': 'Report generated successfully'})
        
    except Exception as e:
        logger.error(f"❌ Report generation error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/export-data')
@api_login_required
def export_data():
    """Export data to CSV"""
    try:
        data_type = request.args.get('type', 'sales')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        logger.info(f"📊 Data export: type='{data_type}', user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            if data_type == 'sales':
                if start_date and end_date:
                    df = pd.read_sql_query('''
                        SELECT * FROM sales 
                        WHERE date BETWEEN ? AND ?
                        ORDER BY date
                    ''', conn, params=(start_date, end_date))
                else:
                    df = pd.read_sql_query('SELECT * FROM sales ORDER BY date', conn)
                    
            elif data_type == 'predictions':
                df = pd.read_sql_query('SELECT * FROM predictions ORDER BY forecast_date', conn)
            elif data_type == 'metrics':
                df = pd.read_sql_query('SELECT * FROM model_metrics ORDER BY training_date', conn)
            else:
                df = pd.read_sql_query('SELECT * FROM sales ORDER BY date', conn)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        logger.info(f"✅ Data exported: {filename} with {len(df)} records")
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"❌ Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule-report', methods=['POST'])
@api_login_required
def schedule_report():
    """Schedule a recurring report with retry logic"""
    try:
        data = request.json
        
        logger.info(f"📅 Scheduling report: name='{data['name']}', frequency='{data['frequency']}', user='{session.get('username')}'")
        
        next_run = datetime.now().replace(
            hour=int(data['time'].split(':')[0]),
            minute=int(data['time'].split(':')[1]),
            second=0
        )
        
        for attempt in range(3):
            try:
                with get_db_connection() as conn:
                    conn.execute('''
                        INSERT INTO scheduled_reports 
                        (name, report_type, frequency, day, time, recipients, format, created_by, next_run)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data['name'],
                        data.get('type', 'sales'),
                        data['frequency'],
                        data.get('day'),
                        data['time'],
                        json.dumps(data['email_recipients']),
                        data.get('format', 'pdf'),
                        session.get('user_id'),
                        next_run.isoformat()
                    ))
                logger.info(f"✅ Report scheduled: {data['name']}")
                return jsonify({'success': True, 'message': 'Report scheduled successfully'})
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        
    except Exception as e:
        logger.error(f"❌ Schedule report error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/scheduled-reports')
@api_login_required
def get_scheduled_reports():
    """Get all scheduled reports"""
    try:
        with get_db_connection() as conn:
            is_admin = 1 if session.get('role') == 'admin' else 0
            cursor = conn.execute('''
                SELECT * FROM scheduled_reports 
                WHERE created_by = ? OR ? = 1
                ORDER BY next_run
            ''', (session.get('user_id'), is_admin))
            
            reports = []
            for row in cursor.fetchall():
                report = dict(row)
                try:
                    report['recipients'] = json.loads(report['recipients'])
                except:
                    pass
                reports.append(report)
        
        return jsonify(reports)
        
    except Exception as e:
        logger.error(f"❌ Get scheduled reports error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle-schedule/<int:schedule_id>', methods=['POST'])
@api_login_required
def toggle_schedule(schedule_id):
    """Activate/deactivate a scheduled report"""
    try:
        data = request.json
        active = data.get('active', True)
        
        logger.info(f"🔄 Toggling schedule ID={schedule_id} to active={active}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            is_admin = 1 if session.get('role') == 'admin' else 0
            conn.execute('''
                UPDATE scheduled_reports 
                SET active = ? 
                WHERE id = ? AND (created_by = ? OR ? = 1)
            ''', (active, schedule_id, session.get('user_id'), is_admin))
        
        logger.info(f"✅ Schedule {schedule_id} toggled to {active}")
        return jsonify({'success': True, 'message': 'Schedule updated'})
        
    except Exception as e:
        logger.error(f"❌ Toggle schedule error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/delete-schedule/<int:schedule_id>', methods=['DELETE'])
@api_login_required
def delete_schedule(schedule_id):
    """Delete a scheduled report"""
    try:
        logger.info(f"🗑️ Deleting schedule ID={schedule_id}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            is_admin = 1 if session.get('role') == 'admin' else 0
            conn.execute('''
                DELETE FROM scheduled_reports 
                WHERE id = ? AND (created_by = ? OR ? = 1)
            ''', (schedule_id, session.get('user_id'), is_admin))
        
        logger.info(f"✅ Schedule {schedule_id} deleted")
        return jsonify({'success': True, 'message': 'Schedule deleted'})
        
    except Exception as e:
        logger.error(f"❌ Delete schedule error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/api/send-report-now/<int:schedule_id>', methods=['POST'])
@api_login_required
def send_report_now(schedule_id):
    """Send a scheduled report immediately with custom date range and report type"""
    try:
        data = request.json or {}
        recipients_override = data.get('recipients')
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        custom_report_type = data.get('report_type')
        custom_category = data.get('category')
        custom_region = data.get('region')
        custom_format = data.get('format')
        
        email_scheduler = EmailScheduler(email_service)
        
        # If custom date range provided, use it
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({'success': False, 'message': 'Invalid date format'}), 400
        else:
            # Default to last 30 days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
        
        success, message = email_scheduler.send_report_now_with_dates(
            schedule_id, 
            session.get('user_id'),
            start_date,
            end_date,
            recipients_override,
            custom_report_type,
            custom_category,
            custom_region,
            custom_format
        )
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        logger.error(f"❌ Send report now error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/update-schedule/<int:schedule_id>', methods=['PUT'])
@api_login_required
def update_schedule(schedule_id):
    """Update a scheduled report"""
    try:
        data = request.json
        
        logger.info(f"📝 Updating schedule ID={schedule_id}, user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            # Check permissions
            is_admin = 1 if session.get('role') == 'admin' else 0
            schedule = conn.execute('''
                SELECT * FROM scheduled_reports 
                WHERE id = ? AND (created_by = ? OR ? = 1)
            ''', (schedule_id, session.get('user_id'), is_admin)).fetchone()
            
            if not schedule:
                logger.warning(f"⚠️ Schedule {schedule_id} not found or access denied")
                return jsonify({'success': False, 'message': 'Schedule not found or access denied'}), 404
            
            # Update schedule
            conn.execute('''
                UPDATE scheduled_reports 
                SET name = ?, report_type = ?, frequency = ?, day = ?, 
                    time = ?, recipients = ?, format = ?
                WHERE id = ?
            ''', (
                data.get('name'),
                data.get('report_type', 'sales'),
                data.get('frequency'),
                data.get('day'),
                data.get('time'),
                json.dumps(data.get('email_recipients', [])),
                data.get('format', 'csv'),
                schedule_id
            ))
        
        logger.info(f"✅ Schedule {schedule_id} updated")
        return jsonify({'success': True, 'message': 'Schedule updated successfully'})
        
    except Exception as e:
        logger.error(f"❌ Update schedule error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/schedule-details/<int:schedule_id>', methods=['GET'])
@api_login_required
def get_schedule_details(schedule_id):
    """Get details of a specific scheduled report"""
    try:
        with get_db_connection() as conn:
            is_admin = 1 if session.get('role') == 'admin' else 0
            schedule = conn.execute('''
                SELECT * FROM scheduled_reports 
                WHERE id = ? AND (created_by = ? OR ? = 1)
            ''', (schedule_id, session.get('user_id'), is_admin)).fetchone()
            
            if not schedule:
                return jsonify({'success': False, 'message': 'Schedule not found'}), 404
            
            result = dict(schedule)
            try:
                result['recipients'] = json.loads(result['recipients'])
            except:
                pass
        
        return jsonify({'success': True, 'schedule': result})
        
    except Exception as e:
        logger.error(f"❌ Get schedule details error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/check-data-availability')
@api_login_required
def check_data_availability():
    """Check how many records are available for a category/region combination"""
    try:
        category = request.args.get('category')
        region = request.args.get('region')
        
        if not category or not region:
            return jsonify({'success': False, 'message': 'Category and region required'}), 400
        
        with get_db_connection() as conn:
            result = conn.execute('''
                SELECT COUNT(*) as count, 
                       MIN(date) as earliest_date,
                       MAX(date) as latest_date
                FROM sales 
                WHERE product_category = ? AND region = ?
            ''', (category, region)).fetchone()
        
        record_count = result['count'] if result else 0
        
        if record_count >= 90:
            status = 'optimal'
            message = f'Excellent data availability ({record_count} records)'
            recommendation = 'High-quality forecasts ready to generate'
        elif record_count >= 50:
            status = 'good'
            message = f'Good data availability ({record_count} records)'
            recommendation = 'Forecasts should be reasonably accurate'
        elif record_count >= 30:
            status = 'limited'
            message = f'Limited data ({record_count} records)'
            recommendation = 'Forecasting possible; accuracy improves as more data is added'
        elif record_count >= 10:
            status = 'minimal'
            message = f'Minimal data ({record_count} records)'
            recommendation = 'Forecasting available with reduced accuracy. Keep adding data for better results.'
        else:
            status = 'insufficient'
            message = f'Insufficient data ({record_count} records)'
            recommendation = 'Need at least 10 records for basic forecasting. Please add sales data first.'
        
        logger.info(f"📊 Data availability check: {category}/{region} - {record_count} records ({status})")
        
        return jsonify({
            'success': True,
            'record_count': record_count,
            'earliest_date': result['earliest_date'] if result else None,
            'latest_date': result['latest_date'] if result else None,
            'status': status,
            'message': message,
            'recommendation': recommendation,
            'can_predict': record_count >= 10,
            'needs_force': record_count < 90 and record_count >= 10
        })
        
    except Exception as e:
        logger.error(f"❌ Data availability check error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/categories-regions')
@api_login_required
def get_categories_regions():
    """Get all distinct product categories and regions from sales data"""
    try:
        with get_db_connection() as conn:
            categories = conn.execute(
                'SELECT DISTINCT product_category FROM sales ORDER BY product_category'
            ).fetchall()
            regions = conn.execute(
                'SELECT DISTINCT region FROM sales ORDER BY region'
            ).fetchall()
        return jsonify({
            'categories': [row['product_category'] for row in categories],
            'regions': [row['region'] for row in regions]
        })
    except Exception as e:
        logger.error(f"❌ Get categories/regions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/currency-settings', methods=['GET'])
@api_login_required
def get_currency_settings():
    """Get currency preference for the current user (stored in session)"""
    currency = session.get('currency', 'USD')
    rate = session.get('currency_rate', 1.0)
    symbols = {'USD': '$', 'KES': 'KSh'}
    return jsonify({'success': True, 'currency': currency, 'rate': float(rate), 'symbol': symbols.get(currency, currency + ' ')})

@app.route('/api/currency-settings', methods=['POST'])
@api_login_required
def set_currency_settings():
    """Set currency preference for the current user"""
    data = request.json or {}
    currency = data.get('currency', 'USD')
    try:
        rate = float(data.get('rate', 1.0))
    except (ValueError, TypeError):
        rate = 1.0
    if rate <= 0:
        return jsonify({'success': False, 'message': 'Rate must be positive'}), 400
    session['currency'] = currency
    session['currency_rate'] = rate
    session.modified = True
    symbols = {'USD': '$', 'KES': 'KSh'}
    logger.info(f"Currency changed to {currency} (rate: {rate}) for user '{session.get('username')}'")
    return jsonify({'success': True, 'currency': currency, 'rate': rate, 'symbol': symbols.get(currency, currency + ' ')})

@app.route('/api/users')
@api_login_required
def get_users():
    """Admin: list all users"""
    if session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    try:
        with get_db_connection() as conn:
            users = conn.execute(
                'SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC'
            ).fetchall()
        return jsonify({'success': True, 'users': [dict(u) for u in users]})
    except Exception as e:
        logger.error(f"❌ Get users error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update-user-role', methods=['POST'])
@api_login_required
def update_user_role():
    """Admin: change a user role"""
    if session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    try:
        data = request.json
        user_id = data.get('user_id')
        new_role = data.get('role')
        
        logger.info(f"👥 Updating user role: user_id={user_id} to '{new_role}', admin='{session.get('username')}'")
        
        if new_role not in ('admin', 'user'):
            return jsonify({'success': False, 'message': 'Invalid role'}), 400
        with get_db_connection() as conn:
            conn.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
        
        logger.info(f"✅ User {user_id} role updated to {new_role}")
        return jsonify({'success': True, 'message': 'Role updated'})
    except Exception as e:
        logger.error(f"❌ Update user role error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/delete-user/<int:user_id>', methods=['DELETE'])
@api_login_required
def delete_user(user_id):
    """Admin: delete a user"""
    if session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Admin access required'}), 403
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'message': 'Cannot delete yourself'}), 400
    try:
        logger.info(f"🗑️ Deleting user ID={user_id}, admin='{session.get('username')}'")
        with get_db_connection() as conn:
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        logger.info(f"✅ User {user_id} deleted")
        return jsonify({'success': True, 'message': 'User deleted'})
    except Exception as e:
        logger.error(f"❌ Delete user error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/api/prediction-details')
@api_login_required
def get_prediction_details():
    """Get detailed prediction for a specific forecast session"""
    try:
        category = request.args.get('category')
        region = request.args.get('region')
        session_id = request.args.get('session_id')
        forecast_date = request.args.get('forecast_date')  # legacy fallback

        if not category or not region:
            return jsonify({'success': False, 'message': 'Category and region required'}), 400

        with get_db_connection() as conn:
            # Resolve session_id: use provided, or find most recent session for this combo
            if session_id:
                resolved_session = session_id
            elif forecast_date:
                # Legacy: find the session that contains this forecast_date
                row = conn.execute('''
                    SELECT forecast_session_id as sid
                    FROM predictions
                    WHERE product_category = ? AND region = ? AND forecast_date = ?
                      AND COALESCE(is_primary, 1) = 1
                    ORDER BY created_at DESC LIMIT 1
                ''', (category, region, forecast_date)).fetchone()
                resolved_session = row['sid'] if row else None
            else:
                # Most recent session for this combo
                row = conn.execute('''
                    SELECT forecast_session_id as sid
                    FROM predictions
                    WHERE product_category = ? AND region = ?
                      AND COALESCE(is_primary, 1) = 1
                      AND forecast_session_id IS NOT NULL
                    ORDER BY created_at DESC LIMIT 1
                ''', (category, region)).fetchone()
                resolved_session = row['sid'] if row else None

            if not resolved_session:
                return jsonify({'success': False, 'message': 'No prediction found for this category/region'}), 404

            # Get all primary rows for this specific session
            all_preds = conn.execute('''
                SELECT forecast_date, predicted_sales, predicted_units,
                       confidence_interval_lower, confidence_interval_upper, model_type,
                       COALESCE(forecast_days, 0) as forecast_days
                FROM predictions
                WHERE forecast_session_id = ?
                  AND COALESCE(is_primary, 1) = 1
                ORDER BY forecast_date ASC
            ''', (resolved_session,)).fetchall()

            if not all_preds:
                return jsonify({'success': False, 'message': 'Session not found'}), 404

            # forecast_days: prefer the stored value, fall back to row count
            stored_days = int(all_preds[0]['forecast_days']) if all_preds[0]['forecast_days'] else 0
            actual_days = stored_days if stored_days > 0 else len(all_preds)

            # Comparison: fetch companion model rows saved under the same session (is_primary=0)
            comparison = {}
            current_model_type = all_preds[0]['model_type']  # e.g. 'LSTM'

            # Include the primary model itself
            comparison[current_model_type.lower()] = {
                'predictions': [float(p['predicted_sales']) for p in all_preds],
                'dates': [p['forecast_date'] for p in all_preds]
            }

            # Fetch companion models saved under the same session
            companion_rows = conn.execute('''
                SELECT model_type, forecast_date, predicted_sales
                FROM predictions
                WHERE forecast_session_id = ? AND is_primary = 0
                ORDER BY model_type, forecast_date ASC
            ''', (resolved_session,)).fetchall()

            from itertools import groupby as _groupby
            companion_rows_sorted = sorted(companion_rows, key=lambda r: r['model_type'])
            for mt, group in _groupby(companion_rows_sorted, key=lambda r: r['model_type']):
                rows_list = list(group)
                key = mt.lower()
                comparison[key] = {
                    'predictions': [float(r['predicted_sales']) for r in rows_list],
                    'dates': [r['forecast_date'] for r in rows_list]
                }

            result_predictions = [float(p['predicted_sales']) for p in all_preds]
            result_dates = [p['forecast_date'] for p in all_preds]

        # Generate LLM explanation outside the DB context
        try:
            total_pred = sum(result_predictions)
            avg_pred = total_pred / len(result_predictions) if result_predictions else 0
            growth = ((result_predictions[-1] / result_predictions[0]) - 1) * 100 \
                if len(result_predictions) > 1 and result_predictions[0] > 0 else 0

            # Build model comparison text for the prompt
            cmp_lines = []
            for k, v in comparison.items():
                if k != current_model_type.lower() and v.get('predictions'):
                    cmp_total = sum(v['predictions'])
                    cmp_lines.append(f"- {k.upper()} total: ${cmp_total:,.2f}")
            cmp_text = "\n".join(cmp_lines) if cmp_lines else "No other models available for comparison."

            prompt = (
                f"As a retail sales analyst, briefly explain this {current_model_type} sales forecast "
                f"for {category} in {region}.\n\n"
                f"Forecast period: {result_dates[0]} to {result_dates[-1]} ({actual_days} days)\n"
                f"Total predicted sales: ${total_pred:,.2f}\n"
                f"Average daily sales: ${avg_pred:,.2f}\n"
                f"Expected growth: {growth:.1f}%\n\n"
                f"Other model forecasts for same period:\n{cmp_text}\n\n"
                f"In 3-4 sentences, explain what these numbers mean for inventory and business decisions."
            )
            response = requests.get(f"{POLLINATIONS_API_URL}{prompt}", timeout=20)
            llm_explanation = response.text if response.status_code == 200 else None
        except Exception:
            llm_explanation = None

        result = {
            'dates': result_dates,
            'predictions': result_predictions,
            'predicted_units': [int(p['predicted_units']) for p in all_preds],
            'lower_bound': [float(p['confidence_interval_lower']) if p['confidence_interval_lower'] else None for p in all_preds],
            'upper_bound': [float(p['confidence_interval_upper']) if p['confidence_interval_upper'] else None for p in all_preds],
            'model_type': current_model_type,
            'forecast_days': actual_days,
            'avg_price': None,
            'trend_analysis': {},
            'llm_explanation': llm_explanation
        }

        return jsonify({
            'success': True,
            'prediction': result,
            'comparison': comparison
        })

    except Exception as e:
        logger.error(f"❌ Prediction details error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/trends-llm-analysis', methods=['POST'])
@api_login_required
def trends_llm_analysis():
    """Generate LLM analysis for trends based on selected filters"""
    try:
        data = request.json
        category = data.get('category', '')
        region = data.get('region', '')
        metric = data.get('metric', 'sales')
        time_period = data.get('time_period', '30d')
        
        logger.info(f"📈 Trend analysis request: category='{category or 'All'}', region='{region or 'All'}', period='{time_period}', user='{session.get('username')}'")
        
        with get_db_connection() as conn:
            query = "SELECT date, product_category, region, units_sold, unit_price, total_sales, promotion_flag FROM sales WHERE 1=1"
            params = []
            
            if category:
                query += " AND product_category = ?"
                params.append(category)
            
            if region:
                query += " AND region = ?"
                params.append(region)
            
            cutoff_date = get_cutoff_date_db(time_period)
            if cutoff_date:
                query += " AND date >= ?"
                params.append(cutoff_date)
            
            query += " ORDER BY date"
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            logger.warning("⚠️ No data available for trend analysis with selected filters")
            return jsonify({
                'success': True,
                'analysis': "No data available for the selected filters. Please adjust your filters and try again."
            })
        
        df['date'] = pd.to_datetime(df['date'])
        
        total_sales = float(df['total_sales'].sum())
        avg_daily_sales = float(df.groupby(df['date'].dt.date)['total_sales'].sum().mean())
        total_units = int(df['units_sold'].sum())
        avg_unit_price = float(df['unit_price'].mean())
        
        df_daily = df.groupby(df['date'].dt.date)['total_sales'].sum().reset_index()
        if len(df_daily) >= 14:
            first_week = df_daily['total_sales'].iloc[:7].mean()
            last_week = df_daily['total_sales'].iloc[-7:].mean()
            growth_rate = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0
        else:
            growth_rate = 0
        
        df['day_of_week'] = df['date'].dt.day_name()
        day_pattern = df.groupby('day_of_week')['total_sales'].mean().to_dict()
        
        promo_sales = df[df['promotion_flag'] == 1]['total_sales'].mean() if len(df[df['promotion_flag'] == 1]) > 0 else 0
        non_promo_sales = df[df['promotion_flag'] == 0]['total_sales'].mean() if len(df[df['promotion_flag'] == 0]) > 0 else total_sales / len(df)
        promo_impact = ((promo_sales - non_promo_sales) / non_promo_sales) * 100 if non_promo_sales > 0 else 0
        
        top_dates = df.nlargest(5, 'total_sales')[['date', 'total_sales']].to_dict('records')
        for item in top_dates:
            item['date'] = item['date'].strftime('%Y-%m-%d')
            item['total_sales'] = float(item['total_sales'])
        
        daily_totals = df.groupby(df['date'].dt.date)['total_sales'].sum()
        volatility = float(daily_totals.std() / daily_totals.mean() * 100) if daily_totals.mean() > 0 else 0
        
        period_map = {
            '7d': 'the last 7 days', '30d': 'the last 30 days', '90d': 'the last 90 days',
            '6m': 'the last 6 months', '1y': 'the last year', 'all': 'all available time'
        }
        period_desc = period_map.get(time_period, 'the selected period')
        
        prompt = f"""As a senior business analyst, analyze the following sales data and provide clear, actionable insights.

FILTERS:
- Time Period: {period_desc}
- Category: {category if category else 'All'}
- Region: {region if region else 'All'}

SUMMARY:
- Total Records: {len(df)}
- Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
- Total Sales: ${total_sales:,.2f}
- Average Daily Sales: ${avg_daily_sales:,.2f}
- Total Units: {total_units:,}
- Growth Rate: {growth_rate:.1f}%
- Volatility: {volatility:.1f}%
- Promotion Impact: {promo_impact:+.1f}%

Provide:
1. Executive Summary
2. Key Trends
3. Actionable Recommendations
4. Risk Factors
5. Opportunities

Use plain text without asterisks or hash symbols."""
        
        try:
            response = requests.get(f"{POLLINATIONS_API_URL}{prompt}", timeout=30)
            if response.status_code == 200:
                analysis = response.text
                logger.info("✅ AI-powered trend analysis generated")
            else:
                analysis = generate_fallback_analysis(df, total_sales, avg_daily_sales, growth_rate, volatility, promo_impact, category, region, period_desc)
                logger.info("📊 Using fallback trend analysis (AI unavailable)")
        except Exception as e:
            logger.warning(f"⚠️ LLM trend analysis failed: {str(e)}")
            analysis = generate_fallback_analysis(df, total_sales, avg_daily_sales, growth_rate, volatility, promo_impact, category, region, period_desc)
        
        return jsonify({'success': True, 'analysis': analysis})
        
    except Exception as e:
        logger.error(f"❌ Trend analysis error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

def get_cutoff_date_db(period):
    """Get cutoff date for database query based on period"""
    if period == 'all':
        return None
    
    today = datetime.now().date()
    if period == '7d':
        return (today - timedelta(days=7)).isoformat()
    elif period == '30d':
        return (today - timedelta(days=30)).isoformat()
    elif period == '90d':
        return (today - timedelta(days=90)).isoformat()
    elif period == '6m':
        return (today - timedelta(days=180)).isoformat()
    elif period == '1y':
        return (today - timedelta(days=365)).isoformat()
    return None

def generate_fallback_analysis(df, total_sales, avg_daily_sales, growth_rate, volatility, promo_impact, category, region, period_desc):
    """Generate a fallback analysis when LLM is unavailable"""
    category_text = f" for {category}" if category else ""
    region_text = f" in {region}" if region else ""
    
    analysis = f"""
SALES PERFORMANCE ANALYSIS{category_text}{region_text} - {period_desc.upper()}

Executive Summary:
This analysis covers {len(df)} sales records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}. 
Total sales reached ${total_sales:,.2f} with an average of ${avg_daily_sales:,.2f} per day.

Key Trends:
• Sales are showing a {growth_rate:+.1f}% growth rate over the analyzed period.
• Daily sales volatility is {volatility:.1f}%, indicating {'stable' if volatility < 30 else 'moderate' if volatility < 60 else 'highly variable'} performance.
• Promotional activities {'increase' if promo_impact > 0 else 'decrease'} sales by {abs(promo_impact):.1f}% on average.

Recommendations:
1. Inventory Planning: {'Increase stock levels to meet growing demand' if growth_rate > 5 else 'Maintain current inventory levels with cautious optimism' if growth_rate > -5 else 'Review and potentially reduce inventory to minimize holding costs'}.
2. Promotional Strategy: {'Continue current promotional strategies as they show positive impact' if promo_impact > 10 else 'Consider testing new promotional approaches to improve effectiveness' if promo_impact > 0 else 'Re-evaluate promotional strategy as current approach may not be optimal'}.
3. Risk Management: Focus on {'stabilizing daily sales performance' if volatility > 50 else 'maintaining current operational efficiency' if volatility < 20 else 'balancing growth with consistency'}.

This analysis is based on the selected filters and available historical data. For more detailed insights, consider narrowing your focus to specific product categories or regions.
"""
    return analysis

# ============================================================================
# TEMPLATE FILTERS
# ============================================================================

@app.template_filter('currency')
def currency_filter(value):
    try:
        return f"${float(value):,.2f}"
    except:
        return value

@app.template_filter('datetime')
def datetime_filter(value, format='%Y-%m-%d %H:%M'):
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return value.strftime(format)
    except:
        return value

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("📊 SALES FORECASTING SYSTEM - ACTIVITY LOGS")
    print("="*60)
    print("📍 System startup - Logging initialized")
    print("="*60 + "\n")
    
    print("[1/3] Initializing database...")
    if init_db():
        print("[2/3] Starting background services...")
        
        model_scheduler.start_scheduler()
        email_scheduler.start_scheduler()
        
        print("[3/3] System ready!")
        print("\n" + "="*60)
        print("🌐 Access the application at: http://localhost:5005")
        print("🔐 Login with: admin / admin123")
        print("📝 All activities will be logged below")
        print("="*60 + "\n")
        
        logger.info("🚀 Sales Forecasting System started successfully")
        logger.info(f"🌐 Server running on http://localhost:5005")
        
        app.run(debug=False, port=5005)  # debug=False to reduce noise
    else:
        print("\n✗ Failed to initialize database. Please check permissions and try again.")
        logger.error("❌ Database initialization failed - system cannot start")
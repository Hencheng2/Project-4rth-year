Deep Learning-Based Sales Forecasting System

Author: Henry Ochieng Onyango
Reg No: SCT221-0262/2022
Institution: Jomo Kenyatta University of Agriculture and Technology (JKUAT)
Course: Bachelor of Science in Information Technology

===============================================

VIDEO DEMONSTRATION

Click the image below to watch the complete system demonstration on YouTube:

https://youtu.be/eTZLiRP0TPM

===============================================

PROJECT OVERVIEW

A Deep Learning-based Sales Forecasting and Trend Prediction System for retail business optimization. The system uses Long Short-Term Memory (LSTM) neural networks to predict future sales based on historical data.

PROBLEM STATEMENT

Retailers relying on spreadsheets or simple moving averages experience:

Stockouts during peak seasons (10-20% revenue loss)

Overstock tying up capital and increasing storage costs

Inability to capture seasonal patterns and promotional impacts

SOLUTION

LSTM deep learning captures complex, non-linear patterns in time-series sales data, outperforming traditional methods like ARIMA and Exponential Smoothing.

===============================================

KEY FEATURES

LSTM Forecasting: 30-90 day forecasts with 95% Monte Carlo Dropout confidence intervals

Model Comparison: Side-by-side LSTM vs ARIMA vs Exponential Smoothing

AI Explanations: LLM-generated natural language business insights

Anomaly Detection: IQR, Z-Score, and Isolation Forest with AI analysis

Interactive Dashboard: Real-time KPIs, trend charts, category breakdowns

What-If Analysis: Simulate price changes, promotion impacts, seasonal factors

Automated Reports: Schedule and email CSV/Excel/PDF reports

Role-Based Access: Admin and User roles with protected functions

===============================================

TECHNOLOGY STACK

Backend: Python 3.11, Flask
Deep Learning: TensorFlow / Keras (LSTM, Bidirectional LSTM, Conv1D)
Data Processing: Pandas, NumPy, Scikit-learn
Statistical Models: Statsmodels (ARIMA, Exponential Smoothing)
Database: SQLite3 with Write-Ahead Logging (WAL)
Frontend: HTML5, CSS3, JavaScript, Plotly.js
Reporting: ReportLab (PDF), openpyxl (Excel)
AI Integration: Pollinations AI API

===============================================

INSTALLATION AND SETUP

Prerequisites:

Python 3.8 or higher

pip package manager

Step 1: Clone the Repository

git clone https://github.com/Hencheng2/Project-4rth-year.git
cd Project-4rth-year

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Run the Application

python app.py

Step 4: Access the System

Open your browser and go to: http://localhost:5005

Default Login Credentials:

Admin: username "admin", password "admin123"
User: create your own via Register page

===============================================

RESEARCH OBJECTIVES

Analyze historical sales data and identify patterns - ACHIEVED

Implement classical statistical models (ARIMA, Exponential Smoothing) - ACHIEVED

Develop LSTM-based deep learning model - ACHIEVED

Compare LSTM accuracy against classical methods - ACHIEVED

===============================================

PROJECT STRUCTURE

Project-4rth-year/
├── app.py # Main Flask application
├── config.py # Configuration settings
├── requirements.txt # Python dependencies
├── sales_data_generator.html # Synthetic data tool
├── templates/ # HTML templates
│ ├── dashboard.html
│ ├── forecast.html
│ ├── data_management.html
│ ├── trends_analysis.html
│ ├── anomalies.html
│ ├── reports.html
│ ├── what_if.html
│ ├── model_training.html
│ ├── admin_users.html
│ ├── login.html
│ └── register.html
├── documents/ # Project documentation
└── media/ # Images for README

===============================================

SUBMISSION NOTES

This project is submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Information Technology at Jomo Kenyatta University of Agriculture and Technology.

Supervisor: [Insert your supervisor's name]

===============================================

CONTACT

Henry Ochieng Onyango
Email: hochieng86@gmail.com
GitHub: https://github.com/Hencheng2

===============================================

ACKNOWLEDGEMENTS

My supervisor for guidance throughout this project

Department of Information Technology, JKUAT

Open-source libraries: TensorFlow, Flask, Scikit-learn, Statsmodels, and others

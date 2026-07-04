import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'banking_secret_secure_key_credit_card_approval'

VALID_EMAIL = 'admin@apexbank.com'
VALID_PASSWORD = 'admin123'

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')

USER_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')
DEFAULT_USER_EMAIL = 'admin@apexbank.com'
DEFAULT_USER_PASSWORD = 'admin123'

# Cache for ML assets
model_cache = {
    'model': None,
    'scaler': None,
    'encoder': None,
    'loaded': False,
    'error': None
}

MODEL_NAME = 'XGBoost Classifier'

CATEGORICAL_COLS = [
    'Gender', 'Income Type', 'Education Level', 'Marital Status',
    'Occupation', 'Housing Type', 'Credit History', 'Employment Status'
]
NUMERICAL_COLS = [
    'Age', 'Annual Income', 'Employment Duration', 'Number of Children',
    'Loan Amount', 'Existing Loan Balance', 'Number of Credit Inquiries', 'Monthly Income'
]

INPUT_COLUMNS = CATEGORICAL_COLS + NUMERICAL_COLS

REQUIRED_FIELDS = [
    'gender', 'age', 'annual_income', 'income_type', 'employment_duration',
    'education_level', 'marital_status', 'occupation', 'num_children',
    'housing_type', 'credit_history', 'loan_amount', 'existing_loan_balance',
    'credit_inquiries', 'monthly_income', 'employment_status'
]


def load_ml_assets():
    if model_cache['loaded']:
        return True
    
    try:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
            model_cache['error'] = "Machine learning model files are missing. Please run 'train_model.py' to train and serialize the model."
            return False
        
        model_cache['model'] = joblib.load(MODEL_PATH)
        model_cache['scaler'] = joblib.load(SCALER_PATH)
        model_cache['encoder'] = joblib.load(ENCODER_PATH)
        model_cache['loaded'] = True
        model_cache['error'] = None
        return True
    except Exception as e:
        model_cache['error'] = f"Failed to load machine learning assets: {str(e)}"
        return False


def load_users():
    if not os.path.exists(USER_STORE_PATH):
        return {}
    try:
        with open(USER_STORE_PATH, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (json.JSONDecodeError, IOError):
        return {}


def save_users(users):
    os.makedirs(os.path.dirname(USER_STORE_PATH), exist_ok=True)
    with open(USER_STORE_PATH, 'w', encoding='utf-8') as file:
        json.dump(users, file, indent=2)


def authenticate_user(email, password):
    users = load_users()
    user = users.get(email)
    if not user:
        return False
    return check_password_hash(user.get('password_hash', ''), password)


def register_user(name, email, password):
    users = load_users()
    users[email] = {
        'name': name,
        'email': email,
        'password_hash': generate_password_hash(password),
        'created_at': datetime.datetime.now().isoformat()
    }
    save_users(users)


def ensure_default_user():
    users = load_users()
    if DEFAULT_USER_EMAIL not in users:
        users[DEFAULT_USER_EMAIL] = {
            'name': 'Apex Bank Admin',
            'email': DEFAULT_USER_EMAIL,
            'password_hash': generate_password_hash(DEFAULT_USER_PASSWORD),
            'created_at': datetime.datetime.now().isoformat()
        }
        save_users(users)


def validate_form_data(form):
    values = {}
    errors = []
    for field in REQUIRED_FIELDS:
        raw_value = form.get(field)
        if raw_value is None or raw_value.strip() == '':
            errors.append(f"{field.replace('_', ' ').capitalize()} is a required field.")
        else:
            values[field] = raw_value.strip()
    return values, errors


def parse_numeric_values(values):
    errors = []
    parsed = {}
    try:
        parsed['age'] = float(values['age'])
        parsed['annual_income'] = float(values['annual_income'])
        parsed['employment_duration'] = float(values['employment_duration'])
        parsed['num_children'] = int(values['num_children'])
        parsed['loan_amount'] = float(values['loan_amount'])
        parsed['existing_loan_balance'] = float(values['existing_loan_balance'])
        parsed['credit_inquiries'] = int(values['credit_inquiries'])
        parsed['monthly_income'] = float(values['monthly_income'])
    except ValueError:
        errors.append("Invalid numerical values entered. Please check all numeric input fields.")
        return parsed, errors

    if parsed['age'] < 18 or parsed['age'] > 100:
        errors.append("Age must be between 18 and 100.")
    if parsed['annual_income'] < 0:
        errors.append("Annual income cannot be negative.")
    if parsed['employment_duration'] < 0 or parsed['employment_duration'] > parsed['age'] - 15:
        errors.append("Employment duration must be a realistic number of years and cannot exceed working age.")
    if parsed['num_children'] < 0:
        errors.append("Number of children cannot be negative.")
    if parsed['loan_amount'] < 0:
        errors.append("Loan amount cannot be negative.")
    if parsed['existing_loan_balance'] < 0:
        errors.append("Existing loan balance cannot be negative.")
    if parsed['credit_inquiries'] < 0:
        errors.append("Number of credit inquiries cannot be negative.")
    if parsed['monthly_income'] < 0:
        errors.append("Monthly income cannot be negative.")

    return parsed, errors


def build_input_dataframe(values, parsed):
    return pd.DataFrame({
        'Gender': [values['gender']],
        'Age': [parsed['age']],
        'Annual Income': [parsed['annual_income']],
        'Income Type': [values['income_type']],
        'Employment Duration': [parsed['employment_duration']],
        'Education Level': [values['education_level']],
        'Marital Status': [values['marital_status']],
        'Occupation': [values['occupation']],
        'Number of Children': [parsed['num_children']],
        'Housing Type': [values['housing_type']],
        'Credit History': [values['credit_history']],
        'Loan Amount': [parsed['loan_amount']],
        'Existing Loan Balance': [parsed['existing_loan_balance']],
        'Number of Credit Inquiries': [parsed['credit_inquiries']],
        'Monthly Income': [parsed['monthly_income']],
        'Employment Status': [values['employment_status']]
    })


def preprocess_input(input_df):
    encoded = model_cache['encoder'].transform(input_df[CATEGORICAL_COLS])
    scaled = model_cache['scaler'].transform(input_df[NUMERICAL_COLS])
    return np.hstack((scaled, encoded))


def format_currency(value):
    return f"${value:,.2f}"


def create_result_details(values, parsed, prediction_val, confidence, explanation):
    formatted_inputs = {
        'Applicant Name': values.get('applicant_name') or 'Unnamed Applicant',
        'Gender': values['gender'],
        'Age': parsed['age'],
        'Annual Income': format_currency(parsed['annual_income']),
        'Income Type': values['income_type'],
        'Employment Duration': f"{parsed['employment_duration']} years",
        'Education Level': values['education_level'],
        'Marital Status': values['marital_status'],
        'Occupation': values['occupation'],
        'Number of Children': parsed['num_children'],
        'Housing Type': values['housing_type'],
        'Credit History': f"{values['credit_history']} Credit",
        'Loan Amount': format_currency(parsed['loan_amount']),
        'Existing Loan Balance': format_currency(parsed['existing_loan_balance']),
        'Credit Inquiries': parsed['credit_inquiries'],
        'Monthly Income': format_currency(parsed['monthly_income']),
        'Employment Status': values['employment_status']
    }

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'status': prediction_val,
        'model': MODEL_NAME,
        'confidence': f"{confidence:.1f}%",
        'explanation': explanation,
        'timestamp': timestamp,
        'prediction_time': timestamp,
        'inputs': formatted_inputs
    }

# Custom context processor to inject navigation state
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now(datetime.timezone.utc)}

@app.before_request
def protect_routes():
    if request.endpoint in {'static'}:
        return None
    if request.endpoint in {'login', 'register'}:
        return None
    if not session.get('logged_in'):
        return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if authenticate_user(email, password):
            users = load_users()
            user = users.get(email, {})
            session['logged_in'] = True
            session['user_email'] = email
            session['user_name'] = user.get('name', email.split('@')[0])
            flash('Login successful!', 'success')
            return redirect(url_for('home'))

        flash('Invalid email or password', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('logged_in'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not name or not email or not password or not confirm_password:
            flash('All fields are required to create an account.', 'danger')
            return render_template('register.html', form_data=request.form)

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html', form_data=request.form)

        users = load_users()
        if email in users:
            flash('An account with that email already exists.', 'danger')
            return render_template('register.html', form_data=request.form)

        register_user(name, email, password)
        flash('Account created successfully. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    values, errors = validate_form_data(request.form)
    if errors:
        for err in errors:
            flash(err, 'danger')
        return render_template('predict.html', form_data=request.form)

    parsed, numeric_errors = parse_numeric_values(values)
    if numeric_errors:
        for err in numeric_errors:
            flash(err, 'danger')
        return render_template('predict.html', form_data=request.form)

    success = load_ml_assets()
    if not success:
        flash(model_cache['error'], 'danger')
        return render_template('predict.html', form_data=request.form)
    
    try:
        input_df = build_input_dataframe(values, parsed)
        X_input = preprocess_input(input_df)

        prediction_val = int(model_cache['model'].predict(X_input)[0])
        prediction_probs = model_cache['model'].predict_proba(X_input)[0]
        confidence = float(prediction_probs[prediction_val] * 100)

        if prediction_val == 1:
            if values['credit_history'] == 'Good' and (parsed['existing_loan_balance'] / parsed['annual_income'] < 0.8):
                explanation = "Approval granted based on an excellent credit history, favorable debt-to-income ratio, and stable employment profile."
            else:
                explanation = "Approved. The applicant demonstrates solid monthly income levels and appropriate credit worthiness scores."
        else:
            if values['credit_history'] == 'Bad':
                explanation = "Application rejected primarily due to a historical record of outstanding debts or past defaults."
            elif (parsed['existing_loan_balance'] / parsed['annual_income'] > 1.2):
                explanation = "Application rejected due to an excessively high debt-to-income ratio. Existing loan balances exceed stable thresholds."
            elif parsed['credit_inquiries'] > 4:
                explanation = "Application rejected due to too many recent credit inquiries, which signals high financial risk."
            else:
                explanation = "Application rejected. The applicant's monthly income does not sufficiently cover the requested loan amount alongside existing debts."

        result_details = create_result_details(values, parsed, prediction_val, confidence, explanation)
        session['last_result'] = result_details

        session.setdefault('history', [])
        history_entry = {
            'date': result_details['timestamp'],
            'name': result_details['inputs']['Applicant Name'],
            'status': result_details['status'],
            'loan_amount': result_details['inputs']['Loan Amount'],
            'income': result_details['inputs']['Annual Income'],
            'confidence': result_details['confidence']
        }
        session['history'].insert(0, history_entry)
        session['history'] = session['history'][:10]
        session.modified = True

        return redirect(url_for('result'))
    except Exception as e:
        flash(f"An error occurred during model prediction: {str(e)}", 'danger')
        return render_template('predict.html', form_data=request.form)

@app.route('/result')
def result():
    last_result = session.get('last_result')
    if not last_result:
        flash("No active prediction result found. Please submit a new credit application.", "info")
        return redirect(url_for('predict'))
    return render_template('result.html', result=last_result)

@app.route('/history')
def history():
    history_list = session.get('history', [])
    return render_template('history.html', history=history_list)

@app.route('/clear_history')
def clear_history():
    session.pop('history', None)
    session.pop('last_result', None)
    flash("Prediction history and results successfully cleared.", "success")
    return redirect(url_for('history'))

if __name__ == '__main__':
    # Try loading ML assets at startup (log error if fail, but let app run)
    load_ml_assets()
    app.run(debug=True, host='0.0.0.0', port=5000)

import os
import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'banking_secret_secure_key_credit_card_approval'

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')

# Cache for ML assets
model_cache = {
    'model': None,
    'scaler': None,
    'encoder': None,
    'loaded': False,
    'error': None
}

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

# Custom context processor to inject navigation state
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now(datetime.timezone.utc)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    # POST - Form Submission & Inference
    # 1. Gather form fields
    fields = [
        'gender', 'age', 'annual_income', 'income_type', 'employment_duration',
        'education_level', 'marital_status', 'occupation', 'num_children',
        'housing_type', 'credit_history', 'loan_amount', 'existing_loan_balance',
        'credit_inquiries', 'monthly_income', 'employment_status'
    ]
    
    # Check if any field is missing
    form_data = {}
    errors = []
    
    for f in fields:
        val = request.form.get(f)
        if val is None or val.strip() == '':
            errors.append(f"{f.replace('_', ' ').capitalize()} is a required field.")
        else:
            form_data[f] = val.strip()
            
    if errors:
        for err in errors:
            flash(err, 'danger')
        return render_template('predict.html', form_data=request.form)
    
    # Try parsing numerical inputs
    try:
        age = float(form_data['age'])
        annual_income = float(form_data['annual_income'])
        employment_duration = float(form_data['employment_duration'])
        num_children = int(form_data['num_children'])
        loan_amount = float(form_data['loan_amount'])
        existing_loan_balance = float(form_data['existing_loan_balance'])
        credit_inquiries = int(form_data['credit_inquiries'])
        monthly_income = float(form_data['monthly_income'])
        
        # Validations
        if age < 18 or age > 100:
            errors.append("Age must be between 18 and 100.")
        if annual_income < 0:
            errors.append("Annual income cannot be negative.")
        if employment_duration < 0 or employment_duration > age - 15:
            errors.append("Employment duration must be a realistic number of years and cannot exceed working age.")
        if num_children < 0:
            errors.append("Number of children cannot be negative.")
        if loan_amount < 0:
            errors.append("Loan amount cannot be negative.")
        if existing_loan_balance < 0:
            errors.append("Existing loan balance cannot be negative.")
        if credit_inquiries < 0:
            errors.append("Number of credit inquiries cannot be negative.")
        if monthly_income < 0:
            errors.append("Monthly income cannot be negative.")
            
    except ValueError as e:
        flash("Invalid numerical values entered. Please check all numeric input fields.", 'danger')
        return render_template('predict.html', form_data=request.form)

    if errors:
        for err in errors:
            flash(err, 'danger')
        return render_template('predict.html', form_data=request.form)

    # 2. Make prediction
    success = load_ml_assets()
    if not success:
        flash(model_cache['error'], 'danger')
        return render_template('predict.html', form_data=request.form)
    
    try:
        # Create input DataFrame matching columns in training
        # Training Columns:
        # 'Gender', 'Age', 'Annual Income', 'Income Type', 'Employment Duration', 
        # 'Education Level', 'Marital Status', 'Occupation', 'Number of Children', 
        # 'Housing Type', 'Credit History', 'Loan Amount', 'Existing Loan Balance', 
        # 'Number of Credit Inquiries', 'Monthly Income', 'Employment Status'
        
        input_dict = {
            'Gender': [form_data['gender']],
            'Age': [age],
            'Annual Income': [annual_income],
            'Income Type': [form_data['income_type']],
            'Employment Duration': [employment_duration],
            'Education Level': [form_data['education_level']],
            'Marital Status': [form_data['marital_status']],
            'Occupation': [form_data['occupation']],
            'Number of Children': [num_children],
            'Housing Type': [form_data['housing_type']],
            'Credit History': [form_data['credit_history']],
            'Loan Amount': [loan_amount],
            'Existing Loan Balance': [existing_loan_balance],
            'Number of Credit Inquiries': [credit_inquiries],
            'Monthly Income': [monthly_income],
            'Employment Status': [form_data['employment_status']]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Preprocessing matching train_model.py
        categorical_cols = [
            'Gender', 'Income Type', 'Education Level', 'Marital Status', 
            'Occupation', 'Housing Type', 'Credit History', 'Employment Status'
        ]
        numerical_cols = [
            'Age', 'Annual Income', 'Employment Duration', 'Number of Children', 
            'Loan Amount', 'Existing Loan Balance', 'Number of Credit Inquiries', 'Monthly Income'
        ]
        
        # Transform
        input_cat = model_cache['encoder'].transform(input_df[categorical_cols])
        input_num = model_cache['scaler'].transform(input_df[numerical_cols])
        
        # Combine
        X_input = np.hstack((input_num, input_cat))
        
        # Run Predict
        prediction_val = int(model_cache['model'].predict(X_input)[0])
        prediction_probs = model_cache['model'].predict_proba(X_input)[0]
        confidence = float(prediction_probs[prediction_val] * 100)
        
        # Generate prediction explanation
        explanation = ""
        if prediction_val == 1:
            if form_data['credit_history'] == 'Good' and (existing_loan_balance / annual_income < 0.8):
                explanation = "Approval granted based on an excellent credit history, favorable debt-to-income ratio, and stable employment profile."
            else:
                explanation = "Approved. The applicant demonstrates solid monthly income levels and appropriate credit worthiness scores."
        else:
            if form_data['credit_history'] == 'Bad':
                explanation = "Application rejected primarily due to a historical record of outstanding debts or past defaults."
            elif (existing_loan_balance / annual_income > 1.2):
                explanation = "Application rejected due to an excessively high debt-to-income ratio. Existing loan balances exceed stable thresholds."
            elif credit_inquiries > 4:
                explanation = "Application rejected due to too many recent credit inquiries, which signals high financial risk."
            else:
                explanation = "Application rejected. The applicant's monthly income does not sufficiently cover the requested loan amount alongside existing debts."
        
        # Format current results
        result_details = {
            'status': prediction_val, # 1 for Approved, 0 for Rejected
            'confidence': f"{confidence:.1f}%",
            'explanation': explanation,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'inputs': {
                'Applicant Name': request.form.get('applicant_name', 'Unnamed Applicant') or 'Unnamed Applicant',
                'Gender': form_data['gender'],
                'Age': age,
                'Annual Income': f"${annual_income:,.2f}",
                'Income Type': form_data['income_type'],
                'Employment Duration': f"{employment_duration} years",
                'Education Level': form_data['education_level'],
                'Marital Status': form_data['marital_status'],
                'Occupation': form_data['occupation'],
                'Number of Children': num_children,
                'Housing Type': form_data['housing_type'],
                'Credit History': f"{form_data['credit_history']} Credit",
                'Loan Amount': f"${loan_amount:,.2f}",
                'Existing Loan Balance': f"${existing_loan_balance:,.2f}",
                'Credit Inquiries': credit_inquiries,
                'Monthly Income': f"${monthly_income:,.2f}",
                'Employment Status': form_data['employment_status']
            }
        }
        
        # Save in session for the result page
        session['last_result'] = result_details
        
        # Append to prediction history in session
        if 'history' not in session:
            session['history'] = []
            
        history_entry = {
            'date': result_details['timestamp'],
            'name': result_details['inputs']['Applicant Name'],
            'status': result_details['status'],
            'loan_amount': result_details['inputs']['Loan Amount'],
            'income': result_details['inputs']['Annual Income'],
            'confidence': result_details['confidence']
        }
        
        # Insert at the beginning of the list
        history_list = session['history']
        history_list.insert(0, history_entry)
        session['history'] = history_list[:10] # limit to last 10
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

@app.route('/about')
def about():
    return render_template('about.html')

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

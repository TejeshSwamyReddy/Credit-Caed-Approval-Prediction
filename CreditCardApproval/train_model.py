import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

def generate_synthetic_data(num_samples=1500):
    np.random.seed(42)
    
    # Define categories
    genders = ['Male', 'Female']
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
    education_levels = ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    marital_statuses = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    occupations = [
        'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
        'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 
        'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 
        'Waiters/barmen staff', 'Secretaries', 'HR staff', 'IT staff', 'Realty agents'
    ]
    housing_types = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
    credit_histories = ['Good', 'Bad']
    employment_statuses = ['Employed', 'Unemployed', 'Retired']
    
    # Generate columns
    gender = np.random.choice(genders, size=num_samples, p=[0.48, 0.52])
    age = np.random.uniform(18, 75, size=num_samples)
    
    # Incomes depend slightly on age and occupation level
    base_income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples) # approx 50k median
    annual_income = np.clip(base_income, 15000, 600000)
    
    # Income Type
    income_type = np.random.choice(income_types, size=num_samples, p=[0.60, 0.20, 0.12, 0.07, 0.01])
    
    # Employment duration (depends on age and income type)
    employment_duration = []
    for idx, age_val in enumerate(age):
        i_type = income_type[idx]
        if i_type == 'Pensioner' or age_val > 65:
            duration = 0.0
        else:
            max_dur = max(0, age_val - 18)
            duration = np.random.uniform(0, min(max_dur, 40))
        employment_duration.append(duration)
    employment_duration = np.array(employment_duration)
    
    education_level = np.random.choice(education_levels, size=num_samples, p=[0.25, 0.60, 0.10, 0.04, 0.01])
    marital_status = np.random.choice(marital_statuses, size=num_samples, p=[0.65, 0.18, 0.08, 0.06, 0.03])
    
    occupation = np.random.choice(occupations, size=num_samples)
    num_children = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.55, 0.25, 0.14, 0.05, 0.01])
    housing_type = np.random.choice(housing_types, size=num_samples, p=[0.85, 0.07, 0.03, 0.03, 0.01, 0.01])
    
    # Credit History (Good/Bad)
    credit_history = np.random.choice(credit_histories, size=num_samples, p=[0.82, 0.18])
    
    # Loan Amount (correlated with income)
    loan_amount = annual_income * np.random.uniform(0.1, 2.5, size=num_samples)
    loan_amount = np.clip(loan_amount, 5000, 800000)
    
    # Existing Loan Balance
    existing_loan_balance = []
    for idx, loan in enumerate(loan_amount):
        # some percentage of loan amount
        balance = np.random.uniform(0, 1.2) * loan if np.random.rand() > 0.3 else 0.0
        existing_loan_balance.append(balance)
    existing_loan_balance = np.array(existing_loan_balance)
    
    # Inquiries
    credit_inquiries = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=num_samples, p=[0.45, 0.25, 0.15, 0.07, 0.04, 0.02, 0.01, 0.01])
    
    # Monthly Income
    monthly_income = annual_income / 12.0
    
    # Employment Status
    employment_status = []
    for idx, dur in enumerate(employment_duration):
        i_type = income_type[idx]
        if dur > 0:
            status = 'Employed'
        elif i_type == 'Pensioner' or age[idx] > 60:
            status = 'Retired'
        else:
            status = 'Unemployed'
        employment_status.append(status)
    employment_status = np.array(employment_status)
    
    # Let's define the approval rule (deterministic base with some random noise)
    # Score starts at 50
    # Add points for good credit history (+40), high income relative to loan (-30 if balance/income > 1.5)
    # Deduct points for bad credit history (-40)
    # Deduct points for excessive inquiries (-5 per inquiry over 2)
    # Deduct points for unemployed/no duration (-15)
    # Add points for higher education (+10)
    
    approved = []
    for idx in range(num_samples):
        score = 60 # Base score
        
        # Credit history is the strongest signal
        if credit_history[idx] == 'Good':
            score += 35
        else:
            score -= 45
            
        # Debt-to-income
        debt_to_income = existing_loan_balance[idx] / annual_income[idx]
        if debt_to_income > 1.2:
            score -= 25
        elif debt_to_income > 0.5:
            score -= 10
            
        # Loan-to-income ratio for this request
        loan_to_income = loan_amount[idx] / annual_income[idx]
        if loan_to_income > 3.0:
            score -= 20
        elif loan_to_income > 1.5:
            score -= 10
            
        # Inquiries
        inq = credit_inquiries[idx]
        if inq > 4:
            score -= 25
        elif inq > 2:
            score -= 10
            
        # Employment Status
        if employment_status[idx] == 'Unemployed':
            score -= 20
        elif employment_status[idx] == 'Retired' and age[idx] < 50:
            score -= 10 # suspicious young retired
            
        # Education Level
        edu = education_level[idx]
        if edu in ['Higher education', 'Academic degree']:
            score += 15
        elif edu == 'Lower secondary':
            score -= 10
            
        # Add noise
        noise = np.random.normal(0, 8)
        final_score = score + noise
        
        # Threshold at 65
        approved.append(1 if final_score >= 65 else 0)
        
    approved = np.array(approved)
    
    df = pd.DataFrame({
        'Gender': gender,
        'Age': np.round(age, 1),
        'Annual Income': np.round(annual_income, 2),
        'Income Type': income_type,
        'Employment Duration': np.round(employment_duration, 1),
        'Education Level': education_level,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Number of Children': num_children,
        'Housing Type': housing_type,
        'Credit History': credit_history,
        'Loan Amount': np.round(loan_amount, 2),
        'Existing Loan Balance': np.round(existing_loan_balance, 2),
        'Number of Credit Inquiries': credit_inquiries,
        'Monthly Income': np.round(monthly_income, 2),
        'Employment Status': employment_status,
        'Approved': approved
    })
    
    return df

def train_and_save():
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(1500)
    
    # Save the raw dataset
    os.makedirs('dataset', exist_ok=True)
    dataset_path = os.path.join('dataset', 'credit_card_data.csv')
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")
    print(f"Approval rate in synthetic dataset: {df['Approved'].mean() * 100:.2f}%")
    
    # Define columns
    categorical_cols = [
        'Gender', 'Income Type', 'Education Level', 'Marital Status', 
        'Occupation', 'Housing Type', 'Credit History', 'Employment Status'
    ]
    numerical_cols = [
        'Age', 'Annual Income', 'Employment Duration', 'Number of Children', 
        'Loan Amount', 'Existing Loan Balance', 'Number of Credit Inquiries', 'Monthly Income'
    ]
    
    X = df.drop(columns=['Approved'])
    y = df['Approved']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    print("Fitting Encoder and Scaler...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()
    
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    
    X_test_cat = encoder.transform(X_test[categorical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])
    
    # Combine processed features
    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_test_processed = np.hstack((X_test_num, X_test_cat))
    
    # Initial training for baseline models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    }

    print("Training baseline models...")
    results = []
    for model_name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        results.append((model_name, model, acc))
        print(f"{model_name} Validation Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred))

    # Tune XGBoost with randomized search over parameter space
    print("Tuning XGBoost model...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    search_space = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.75, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.75, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5]
    }

    randomized_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=search_space,
        n_iter=24,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    randomized_search.fit(X_train_processed, y_train)
    best_xgb = randomized_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test_processed)
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Best Validation Accuracy: {xgb_acc * 100:.2f}%")
    print("Best XGBoost parameters:", randomized_search.best_params_)
    print(classification_report(y_test, y_pred_xgb))

    # Add XGBoost to result set
    results.append(('XGBoost', best_xgb, xgb_acc))

    # Determine best model by accuracy
    results.sort(key=lambda item: item[2], reverse=True)
    best_model_name, best_model, best_accuracy = results[0]
    print(f"Best model selected for deployment: {best_model_name} with accuracy {best_accuracy * 100:.2f}%")

    # Save the models
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model, os.path.join('model', 'model.pkl'))
    joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
    joblib.dump(encoder, os.path.join('model', 'encoder.pkl'))
    print("Serialized best model, scaler, and encoder successfully exported to 'model/' directory.")

if __name__ == '__main__':
    train_and_save()

# Credit Card Approval Prediction

A Flask web application that predicts credit card approval based on applicant details. The project includes a synthetic data generator, model training pipeline, and a web interface for submitting applications and viewing prediction results.

## Features

- Login-protected web dashboard
- Form-based credit approval prediction
- XGBoost model with pre-processing (scaler + one-hot encoder)
- Model training script that generates synthetic data and saves model artifacts
- Static templates for login, prediction, and results

## Project Structure

- `app.py` - Flask application and prediction endpoint
- `train_model.py` - Generates synthetic dataset, trains models, selects the best model, and saves artifacts
- `requirements.txt` - Python dependencies
- `dataset/` - Generated dataset file (created by `train_model.py`)
- `model/` - Saved `model.pkl`, `scaler.pkl`, `encoder.pkl`
- `static/` - CSS, JavaScript, and image assets
- `templates/` - Flask HTML templates
- `tests/` - Test files

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

The app expects trained model artifacts in the `model/` folder:

- `model/model.pkl`
- `model/scaler.pkl`
- `model/encoder.pkl`

Run the training script to generate these files:

```bash
python train_model.py
```

This will also create a synthetic dataset at `dataset/credit_card_data.csv`.

## Running the App

Start the Flask application:

```bash
python app.py
```

Then open your browser at:

```text
http://127.0.0.1:5000
```

## Login Credentials

Use the following credentials to access the app:

- Email: `admin@apexbank.com`
- Password: `admin123`

## Tests

Run available tests with:

```bash
python -m pytest
```

## Notes

- If the model files are missing, the app will prompt you to run `train_model.py`.
- The prediction form validates required fields and numeric inputs before scoring.
- The current model is based on synthetic data and should be retrained with real data for production use.

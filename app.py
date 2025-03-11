# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bootstrap import Bootstrap
from werkzeug.security import generate_password_hash, check_password_hash   
import pandas as pd
import numpy as np
import joblib
from database import init_db, add_user, get_user, get_user_by_id, save_prediction, get_user_predictions, get_all_predictions
from utils import simulate_email, generate_report
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
Bootstrap(app)

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role
@login_manager.user_loader
def load_user(user_id):
    user = get_user_by_id(int(user_id))
    if user:
        return User(user[0], user[1], user[3])
    return None

# Load models
scaler = joblib.load('models/scaler.pkl')
reg_models = {
    'LinearRegression': joblib.load('models/LinearRegression.pkl'),
    'RandomForestRegressor': joblib.load('models/RandomForestRegressor.pkl')
}
clf_models = {
    'RandomForestClassifier': joblib.load('models/RandomForestClassifier.pkl'),
    'XGBoost': joblib.load('models/XGBoost.pkl'),
    'EnsembleClassifier': joblib.load('models/EnsembleClassifier.pkl')
}

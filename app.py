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

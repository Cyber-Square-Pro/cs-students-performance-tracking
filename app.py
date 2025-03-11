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

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1], user[3]))
            return redirect(url_for('predict'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            add_user(username, password)
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        studytime = float(request.form['studytime'])
        absences = float(request.form['absences'])
        g1 = float(request.form['g1'])
        g2 = float(request.form['g2'])
        failures = float(request.form['failures'])
        reg_model_name = request.form['reg_model']
        clf_model_name = request.form['clf_model']
        
        input_data = np.array([[studytime, absences, g1, g2, failures]])
        input_scaled = scaler.transform(input_data)
        
        predicted_grade = reg_models[reg_model_name].predict(input_scaled)[0]
        pass_fail = clf_models[clf_model_name].predict(input_scaled)[0]
        
        save_prediction(current_user.id, studytime, absences, g1, g2, failures, 
                       reg_model_name, predicted_grade, clf_model_name, pass_fail)
        
        simulate_email(current_user.id, {'grade': predicted_grade, 'pass_fail': pass_fail})
        
        return render_template('predict.html', 
                             grade=round(predicted_grade, 2), 
                             pass_fail='Pass' if pass_fail == 1 else 'Fail',
                             reg_model=reg_model_name, clf_model=clf_model_name)
    return render_template('predict.html', reg_models=reg_models.keys(), clf_models=clf_models.keys())

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = get_user_predictions(current_user.id)
    plot_html = generate_report(predictions)
    return render_template('dashboard.html', plot_html=plot_html)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('predict'))
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file, sep=';')
            X, y_reg, y_clf = load_data(file_path=None)  # Use uploaded data
            train_models(X, y_reg, y_clf)
            flash('Models retrained with uploaded data!')
    
    all_predictions = get_all_predictions()
    plot_html = generate_report(all_predictions)
    return render_template('admin.html', plot_html=plot_html)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file, sep=';')
            # Predict for all rows
            X = df[['studytime', 'absences', 'G1', 'G2', 'failures']]
            X_scaled = scaler.transform(X)
            grades = reg_models['RandomForestRegressor'].predict(X_scaled)
            pass_fails = clf_models['EnsembleClassifier'].predict(X_scaled)
            
            df['predicted_grade'] = grades
            df['pass_fail'] = pass_fails
            output = io.BytesIO()
            df.to_excel(output, index=False)
            output.seek(0)
            return send_file(output, as_attachment=True, download_name='predictions.xlsx')
    return render_template('upload.html')

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
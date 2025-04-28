from flask import Flask, request, render_template, redirect, url_for, session, flash
import joblib
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly
import json
import webbrowser
import threading
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')

DATABASE = 'users.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            gender TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    # Insert a default user if not exists
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if cursor.fetchone() is None:
        cursor.execute('INSERT INTO users (username, email, phone, gender, password_hash) VALUES (?, ?, ?, ?, ?)',
                       ('admin', 'admin@example.com', '0000000000', 'Male', generate_password_hash('admin123')))
    conn.commit()
    conn.close()


def initialize():
    init_db()

@app.before_request
def before_request():
    if not hasattr(app, 'db_initialized'):
        initialize()
        app.db_initialized = True


# ------------------ Routes ------------------
@app.route('/')
def serve_index():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user'] = username
            return redirect(url_for('serve_index'))
        else:
            flash('Invalid username or password')
            return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form.get('gender')
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username already exists')
            conn.close()
            return render_template('register.html')

        password_hash = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, email, phone, gender, password_hash) VALUES (?, ?, ?, ?, ?)',
                       (username, email, phone, gender, password_hash))
        conn.commit()
        conn.close()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    else:
        return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/customer')
def serve_customer():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('customer.html')

@app.route('/aboutus')
def serve_aboutus():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('aboutus.html')

@app.route('/contactus')
def serve_contactus():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('contactus.html')

# ------------------ Load Model ------------------
model = joblib.load("rf_churn_model.joblib")
explainer = joblib.load("shap_explainer.joblib")

# ------------------ Encoding Maps ------------------
geography_map = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_map = {'Female': 0, 'Male': 1}

# ------------------ Predict Route ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('user'):
        return redirect(url_for('login'))
    form = request.form

    data = {
        'CreditScore': float(form['CreditScore']),
        'Geography': geography_map[form['Geography']],
        'Gender': gender_map[form['Gender']],
        'Age': float(form['Age']),
        'Tenure': float(form.get('Tenure', 0)),
        'Balance': float(form['Balance']),
        'NumOfProducts': float(form['NumOfProducts']),
        'HasCrCard': int('HasCrCard' in form),
        'IsActiveMember': int('IsActiveMember' in form),
        'EstimatedSalary': float(form.get('EstimatedSalary', 0))
    }

    df = pd.DataFrame([data])
    df = df[model.feature_names_in_]

    prob = model.predict_proba(df)[0][1]
    confidence = max(model.predict_proba(df)[0])
    risk_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'

    shap_vals_all = explainer.shap_values(df)
    shap_vals = shap_vals_all[1][0] if isinstance(shap_vals_all, list) else shap_vals_all[0]
    shap_vals = shap_vals.flatten()

    expected_features = model.feature_names_in_
    importance = {f: abs(v) for f, v in zip(expected_features, shap_vals)}
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    top_features = [
        {'feature': f, 'contribution': float(shap_vals[list(expected_features).index(f)])}
        for f, _ in sorted_feats[:3]
    ]

    # Charts
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=list(expected_features),
        y=df.iloc[0].values,
        mode='lines+markers',
        line=dict(color='lightgreen')
    ))
    line_fig.update_layout(template='plotly_dark', height=500)

    pie_fig = go.Figure(data=[go.Pie(
        labels=[f for f, _ in sorted_feats[:5]],
        values=[v for _, v in sorted_feats[:5]],
        hole=0.3
    )])
    pie_fig.update_layout(template='plotly_dark', height=500)

    bar_data = {
        'Feature': [f for f, _ in sorted_feats[:5]],
        'Value': [v for _, v in sorted_feats[:5]]
    }
    bar_fig = px.bar(
        bar_data, x='Feature', y='Value',
        title='Feature Importance Bar Chart',
        color_discrete_sequence=['#a889f4']
    )
    bar_fig.update_layout(template='plotly_dark', height=500)

    return render_template("result.html",
        churn_prob=f"{prob * 100:.1f}",
        confidence=f"{confidence * 100:.1f}",
        risk_level=risk_level,
        insight="Limited product engagement increases churn likelihood.",
        top_features=top_features,
        line_chart_json=json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder),
        pie_chart_json=json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder),
        bar_chart_json=json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder),
    )

# ------------------ Open Browser Automatically ------------------
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

import os

def main():
    initialize()
    threading.Timer(1, open_browser).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    main()

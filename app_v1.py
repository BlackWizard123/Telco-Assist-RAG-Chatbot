from flask import Flask, redirect, url_for, request, render_template, flash, session
from flask import Flask     
import sqlite3, os
# from cryptography.fernet import Fernet
import base64

from embedding import *
from retriever import *
from convRetrChain import *
from chat_v1 import chat_bp

app = Flask(__name__)            
app.secret_key = 'secret_key' 
# ENCRYPTION_KEY = Fernet.generate_key()
# cipher = Fernet(ENCRYPTION_KEY)

app.register_blueprint(chat_bp,  url_prefix='/chat')

def encrypt_api_key(api_key):
    return cipher.encrypt(api_key.encode())

def decrypt_api_key(encrypted_api_key):
    return cipher.decrypt(encrypted_api_key).decode()

def fetchUserAPIKeys(username):
        # Connect to the database and check credentials
        conn = sqlite3.connect('TAUserDB.db')
        c = conn.cursor()
        c.execute("SELECT google_api, cohere_api, huggingface_api1, huggingface_api2 FROM users WHERE username = ?", (username,))
        apikeys = c.fetchone()
        conn.close()
        if apikeys:
            google_api, cohere_api, huggingfaceapi1, huggingfaceapi2 = apikeys
            session['HFAPI1'] = huggingfaceapi1
            session['HFAPI2'] = huggingfaceapi2
            session['GGLEAPI'] = google_api
            session['CHREAPI'] = cohere_api
            # session['HFAPI1'] = encrypt_api_key(huggingfaceapi1)
            # session['HFAPI2'] = encrypt_api_key(huggingfaceapi2)
            # session['GGLEAPI'] = encrypt_api_key(google_api)
            # session['CHREAPI'] = encrypt_api_key(cohere_api)
        else:
            # Handle case where the user does not exist or has no API keys
            print("No API keys found for the user.")
        
def init_db():
    conn = sqlite3.connect('TAUserDB.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    userid INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT,
                    last_name TEXT
                    email TEXT,
                    phone TEXT,
                    username TEXT UNIQUE,
                    password TEXT,
                    google_api TEXT,
                    huggingface_api1 TEXT,
                    huggingface_api1 TEXT,
                    cohere_api TEXT
                )''')
    conn.commit()
    conn.close()

# ------------------------------------------------

@app.route("/")                   
def main_home():                      
    return render_template("Home.html")  

@app.route('/sign_in')
def login_home():
    return render_template('Login.html')

@app.route('/sign_up')
def registration_1():
    return render_template('Register1.html')

@app.route('/register', methods=['POST'])
def register1():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        phonenum = request.form['phonenum']
        password = request.form['password']
        re_password = request.form['re_password']

        if password != re_password:
            flash("Password doesn't match")
            return redirect(url_for('registration_1'))

        try:
            conn = sqlite3.connect('TAUserDB.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (first_name, last_name, email, phone, username, password) VALUES (?, ?, ?, ?, ?, ?)",
                      (first_name, last_name, email, phonenum, username, password))
            conn.commit()
            conn.close()

            # Create User folder
            os.makedirs(f"Userdata/{username}")
            os.makedirs(f"Userdata/{username}/Vector Store")
            os.makedirs(f"Userdata/{username}/Chat History")

            return redirect(url_for('register2', username = username))
        except sqlite3.IntegrityError:
            flash("Username already taken. Try a different one.")
    
    return render_template('register1.html')

@app.route('/registerkeys/<username>', methods=['GET','POST'])
def register2(username):
    if request.method == 'POST':
        googleapi = request.form['googleapi1']
        hfapi1 = request.form['hfapi1']
        hfapi2 = request.form['hfapi2']
        cohereapi = request.form['cohereapi']

        conn = sqlite3.connect('TAUserDB.db')
        c = conn.cursor()
        c.execute("UPDATE users SET google_api = ?, huggingface_api1 = ?, huggingface_api2 = ?, cohere_api = ? WHERE username = ?",
                  (googleapi, hfapi1, hfapi2, cohereapi, username))
        conn.commit()
        conn.close()

        return redirect(url_for('success_register'))
    
    return render_template('Register2.html', username = username)

@app.route('/registration_success')
def success_register():
    return render_template('Register_success.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Connect to the database and check credentials
        conn = sqlite3.connect('TAUserDB.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = username
            apiKeys = fetchUserAPIKeys(username)
            return redirect(url_for('chat_bp.chat_index'))
        else:
            flash('Invalid username or password. Please try again.')
            return redirect(url_for('login_home'))  

    return render_template('Login.html')

@app.route('/get_profile', methods=['GET'])
def get_profile():
    username = session.get('username')
    if not username:
        return jsonify({"error": "User not logged in"}), 401

    conn = sqlite3.connect('TAUserDB.db')
    c = conn.cursor()
    c.execute("SELECT first_name, last_name, phone, email, google_api, cohere_api, huggingface_api1, huggingface_api2 FROM users WHERE username = ?", (username,))
    user_data = c.fetchone()
    conn.close()

    if user_data:
        return jsonify({
            "first_name": user_data[0],
            "last_name": user_data[1],
            "phone": user_data[2],
            "email": user_data[3],
            "google_api": user_data[4],
            "cohere_api": user_data[5],
            "huggingface_api1": user_data[6],
            "huggingface_api2": user_data[7],
            "username": username
        })
    return jsonify({"error": "User not found"}), 404

@app.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.json
    username = session.get('username')
    if not username:
        return jsonify({"error": "User not logged in"}), 401

    fields = ['first_name', 'last_name', 'phone', 'email', 'google_api', 'cohere_api', 'huggingface_api1', 'huggingface_api2']
    updates = {field: data[field] for field in fields if field in data}

    if not updates:
        return jsonify({"error": "No fields to update"}), 400

    placeholders = ', '.join([f"{field} = ?" for field in updates.keys()])
    values = list(updates.values()) + [username]

    conn = sqlite3.connect('TAUserDB.db')
    c = conn.cursor()
    c.execute(f"UPDATE users SET {placeholders} WHERE username = ?", values)
    conn.commit()
    conn.close()

    return jsonify({"success": "Profile updated successfully"})




# -------------------------------------------------------

if __name__ == "__main__":        
    app.run()

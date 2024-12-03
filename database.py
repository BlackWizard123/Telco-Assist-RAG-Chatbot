import sqlite3
def init_db():
    conn = sqlite3.connect('TAUserDB.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    userid INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT,
                    last_name TEXT,
                    email TEXT,
                    phone TEXT,
                    username TEXT UNIQUE,
                    password TEXT,
                    google_api TEXT,
                    huggingface_api1 TEXT,
                    huggingface_api2 TEXT,
                    cohere_api TEXT
                )''')
    conn.commit()
    conn.close()

def addEntry():
    first_name = 123
    last_name = 1230
    email=123
    phonenum=123
    username=123
    password=123
    conn = sqlite3.connect('TAUserDB.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (first_name, last_name, email, phone, username, password) VALUES (?, ?, ?, ?, ?, ?)",
                (first_name, last_name, email, phonenum, username, password))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    #init_db()
    addEntry()
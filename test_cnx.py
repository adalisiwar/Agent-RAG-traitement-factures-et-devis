import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)

cursor = conn.cursor()

cursor.execute("SELECT version();")

print(cursor.fetchone())

cursor.close()
conn.close()

print("✅ Connexion Neon réussie")
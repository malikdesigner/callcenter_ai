import sqlite3
import os

DB_PATH = "data/appointments.db"

if not os.path.exists(DB_PATH):
    print(f"Database {DB_PATH} not found. No migration needed.")
else:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if transcript column exists
        cursor.execute("PRAGMA table_info(appointment)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "transcript" not in columns:
            print("Adding 'transcript' column to 'appointment' table...")
            cursor.execute("ALTER TABLE appointment ADD COLUMN transcript TEXT")
            conn.commit()
            print("Migration successful.")
        else:
            print("'transcript' column already exists.")
            
        conn.close()
    except Exception as e:
        print(f"Error during migration: {e}")

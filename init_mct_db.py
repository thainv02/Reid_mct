
import os
import psycopg2
from database.db_config import DatabaseConfig

def run_sql_file(filename):
    print(f"Executing SQL file: {filename}...")
    
    # Read SQL content
    with open(filename, 'r') as f:
        sql_content = f.read()
        
    # Connect to DB
    conn = DatabaseConfig.get_connection()
    conn.autocommit = True
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql_content)
        print("✅ Successfully executed SQL script.")
        print("   - Tables created/verified: mct_sessions, mct_face_recognition, mct_position_tracking")
        print("   - Indexes created/verified.")
        
    except Exception as e:
        print(f"❌ Error executing SQL: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Ensure we are in the right directory or use absolute path
    sql_file = "database/create_mct_tables.sql"
    if os.path.exists(sql_file):
        run_sql_file(sql_file)
    else:
        print(f"❌ File not found: {sql_file}")

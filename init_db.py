#!/usr/bin/env python3
"""Initialize database and check schema with debug output."""

from database.db import Database
from pathlib import Path
import sqlite3
import traceback

try:
    db_path = Path('visioncare.db')
    before_size = db_path.stat().st_size if db_path.exists() else 0
    print(f'DB file before: {before_size} bytes')

    # Delete the empty file if it exists
    if db_path.exists() and before_size == 0:
        db_path.unlink()
        print('Removed empty database file')

    # Initialize database
    print('Initializing database...')
    db = Database()
    print(f'DB file after init: {db_path.stat().st_size} bytes')

    # Check tables
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    table_list = [t[0] for t in tables]
    print(f'Tables created: {table_list}')
    
    # Check scans table columns
    if 'scans' in table_list:
        cursor.execute("PRAGMA table_info(scans)")
        cols = cursor.fetchall()
        print(f'Scans table columns: {[c[1] for c in cols]}')
    
    conn.close()
    print('Database initialization successful!')
    
except Exception as e:
    print(f'Error: {e}')
    traceback.print_exc()

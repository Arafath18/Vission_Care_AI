#!/usr/bin/env python3
"""Check the actual database."""

import sqlite3
from pathlib import Path

db_path = Path('visioncare.db')
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
print('Tables:')
for t in tables:
    print(f'  - {t[0]}')

# Check specific tables
has_reports = any(t[0] == 'reports' for t in tables)
if has_reports:
    cursor.execute('SELECT COUNT(*) FROM reports')
    count = cursor.fetchone()[0]
    print(f'\nReports table: {count} records')
    
    if count > 0:
        cursor.execute('SELECT scan_id, language, pdf_path FROM reports LIMIT 5')
        for row in cursor.fetchall():
            print(f'  Scan {row[0]}: {row[1]} - {row[2]}')
else:
    print('\nReports table: NOT FOUND')

# Check scans table
cursor.execute('SELECT COUNT(*) FROM scans')
scan_count = cursor.fetchone()[0]
print(f'\nScans table: {scan_count} records')

# Check if total_affected_pct exists
cursor.execute("PRAGMA table_info(scans)")
cols = cursor.fetchall()
col_names = [c[1] for c in cols]
print(f'\nScans columns include total_affected_pct: {"total_affected_pct" in col_names}')

conn.close()

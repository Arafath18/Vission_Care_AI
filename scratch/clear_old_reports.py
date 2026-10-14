import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
db_path = root / 'database' / 'visioncare.db'
reports_dir = root / 'outputs' / 'reports'

print("Database path:", db_path)
print("Reports directory:", reports_dir)

# 1. Clear database reports table
if db_path.exists():
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reports")
        conn.commit()
        print(f"Cleared {cursor.rowcount} rows from the 'reports' table in the database.")
        conn.close()
    except Exception as e:
        print("Error clearing database table:", e)
else:
    print("Database file does not exist.")

# 2. Delete old PDF files on disk
if reports_dir.exists():
    deleted_files = []
    for f in reports_dir.glob('*.pdf'):
        try:
            f.unlink()
            deleted_files.append(f.name)
        except Exception as e:
            print(f"Error deleting file {f.name}:", e)
    print("Deleted PDF files:", deleted_files)
else:
    print("Reports directory does not exist.")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
print('ROOT OK')
import time
start = time.time()
try:
    import utils.inference as ui
    print('IMPORT OK', time.time() - start)
except Exception as exc:
    print('IMPORT FAIL', exc)

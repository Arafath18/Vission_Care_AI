from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from utils.inference import get_predictor

patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
root = Path('datasets/kermany/test/NORMAL')
images = []
for p in patterns:
    images.extend(root.glob(p))
images = sorted(images)
print('FOUND', len(images), 'NORMAL images')
if not images:
    raise SystemExit('No sample images found')

print('LOADING PREDICTOR')
predictor = get_predictor()
print('PREDICTOR LOADED')
image_path = images[0]
img = cv2.imread(str(image_path))
print('IMAGE PATH', image_path)
print('IMG READ', img is not None)
if img is None:
    raise SystemExit('Failed to load sample')
print('RUNNING PREDICT')
result = predictor.predict(img)
print('PREDICT DONE')
print('label:', result['label'])
print('confidence:', result['confidence'])
print('probabilities:', result['probabilities'])

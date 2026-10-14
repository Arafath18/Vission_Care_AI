from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print('ROOT OK')
from utils.inference import get_predictor
print('IMPORT OK')
from pathlib import Path
import cv2

images = list(Path('visioncare/datasets/kermany/test/NORMAL').glob('*.jpeg')) + list(Path('visioncare/datasets/kermany/test/NORMAL').glob('*.jpg'))
print('FOUND', len(images), 'images')
if images:
    img = cv2.imread(str(images[0]))
    print('IMG READ', img is not None)
    if img is not None:
        predictor = get_predictor()
        print('MODEL LOADED')
        res = predictor.predict(img)
        print('PREDICTION RESULT:', res['label'], res['confidence'])

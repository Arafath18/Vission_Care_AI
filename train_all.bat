@echo off
cd /d "%~dp0"
title VisionCare Training

echo === VisionCare: dataset + model training ===
python scripts\validate_dataset.py
if errorlevel 1 (
  echo Dataset validation failed.
  pause
  exit /b 1
)

echo.
echo Training classifiers (EfficientNet + MobileNet)...
echo This may take 30-90 minutes on CPU with 400 images/class.
python -u models\train.py --data datasets\kermany --epochs 8 --batch 16 --lr 0.0001 --patience 4 --max-per-class 400 --test-max-per-class 200

echo.
echo Training segmentation (Attention U-Net)...
python -u models\train_segmentation.py

echo.
echo Setup database and demo user...
python scripts\setup_stack.py

echo.
echo Training complete. Start app with run.bat
pause

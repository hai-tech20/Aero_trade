@echo off
echo ===============================
echo  HYBRID AI TRADING BOT
echo ===============================
echo.

echo 1. Checking Python...
py --version
if errorlevel 1 (
    echo ❌ Python not found
    pause
    exit /b 1
)

echo 2. Installing requirements...
py -m pip install -r requirements.txt

echo 3. Testing imports...
py -c "print('Testing imports...'); from utils.logger import logger; print('✅ Logger works!')"

echo 4. Training AI models...
py train_models.py

echo 5. Starting bot...
echo Press Ctrl+C to stop
py main.py

pause
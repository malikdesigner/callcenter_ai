@echo off
echo ============================================================
echo  AI Hospital Receptionist - First Time Setup
echo ============================================================
echo.

echo [1/4] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo [2/4] Installing dependencies (this takes a few minutes)...
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo.
echo [3/4] Creating folders...
mkdir voices 2>nul
mkdir data   2>nul
mkdir static 2>nul

echo.
echo [4/4] Pulling Ollama model (requires Ollama installed)...
ollama pull llama3.2:3b

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Place your voice sample at:  voices\receptionist.wav
echo    2. Run Ollama:                  ollama serve
echo    3. Start the agent:             python main.py
echo    4. Open browser:                http://localhost:8000
echo ============================================================
pause

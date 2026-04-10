@echo off
echo Starting AI Hospital Receptionist...
call venv\Scripts\activate
start "" ollama serve
timeout /t 2 /nobreak >nul
python main.py

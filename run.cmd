@echo off
setlocal
cd /d "%~dp0"

echo == Morphology Demo setup ^& run ==

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found. Please install Python 3.10+ then rerun.
  exit /b 1
)

if not exist ".venv\" (
  echo Creating venv...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies (this may take a while)...
python -m pip install -r requirements.txt

echo Starting Streamlit. Open http://localhost:8501
streamlit run app.py --server.port 8501


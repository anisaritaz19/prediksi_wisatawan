@echo off
REM Sekali jalan: buat venv, pasang library, latih model, jalankan aplikasi.
REM Syarat: file BPS (.csv/.xlsx) sudah ada di folder data\raw\
cd /d %~dp0
if not exist venv ( python -m venv venv )
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
python -m src.train || goto :err
echo.
echo Selesai. Aplikasi berjalan di http://127.0.0.1:5000  (Ctrl+C untuk berhenti)
python app.py
goto :eof
:err
echo Terjadi galat. Baca pesan di atas.
pause

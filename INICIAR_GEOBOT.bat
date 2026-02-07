@echo off
echo ============================================
echo   GeoBot v1.0 - Inicializador
echo ============================================
echo.

REM Verifica se venv existe
if not exist "venv\" (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo.
    echo Execute primeiro:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [1/2] Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo [2/2] Iniciando GeoBot...
echo.
echo Acesse no navegador: http://localhost:8501
echo.
echo Para encerrar: Ctrl+C
echo.

streamlit run geobot.py

pause

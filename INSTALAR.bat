@echo off
echo ============================================
echo   GeoBot v1.0 - Instalador
echo ============================================
echo.

echo [1/3] Verificando Python 3.11...
python --version | findstr "3.11" > nul
if errorlevel 1 (
    echo [AVISO] Python 3.11 nao detectado!
    echo Recomenda-se Python 3.11.9
    echo.
    pause
)

echo [2/3] Criando ambiente virtual...
if exist "venv\" (
    echo Ambiente virtual ja existe. Pulando...
) else (
    python -m venv venv
    echo Ambiente virtual criado!
)

echo [3/3] Instalando dependencias...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo   Instalacao Concluida!
echo ============================================
echo.
echo Proximos passos:
echo   1. Execute: INICIAR_GEOBOT.bat
echo   2. Acesse: http://localhost:8501
echo   3. Insira sua API Key Groq
echo.
pause

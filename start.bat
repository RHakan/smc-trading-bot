@echo off
chcp 65001 >nul
title Bot Futures

cd /d "%~dp0"

echo.
echo  ======================================
echo   Bot Futures - Inicializando...
echo  ======================================
echo.

:: Verifica se o .env existe
if not exist ".env" (
    echo  [AVISO] Arquivo .env nao encontrado!
    echo.
    echo  Crie o .env antes de iniciar:
    echo    copy .env.example .env
    echo    Edite o .env com suas chaves da Binance.
    echo.
    pause
    exit /b 1
)

:: Verifica se o Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERRO] Python nao encontrado no PATH.
    echo  Instale em: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Verifica se as dependencias estao instaladas
python -c "import fastapi, ccxt, apscheduler" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Instalando dependencias...
    pip install -r requirements.txt
    echo.
)

echo  [OK] Ambiente pronto.
echo  [OK] Iniciando servidor na porta 8080...
echo.
echo  Dashboard: http://localhost:8080
echo  Pressione Ctrl+C para parar o bot.
echo.

:: Abre o browser apos 2 segundos (em paralelo)
start "" cmd /c "timeout /t 2 >nul && start http://localhost:8080"

:: Inicia o servidor
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080

echo.
echo  [INFO] Servidor encerrado.
pause

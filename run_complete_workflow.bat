@echo off
setlocal

REM Ejecuta el workflow interactivo completo
python "%~dp0complete_workflow.py"

if errorlevel 1 (
    echo.
    echo [ERROR] La ejecucion termino con errores.
    exit /b 1
)

echo.
echo [OK] Workflow finalizado.
exit /b 0

@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo [1/2] 打包前端...
cd frontend
call npm run build
if errorlevel 1 (
  echo [错误] 前端打包失败，请检查 frontend 依赖（npm install）后重试。
  exit /b 1
)
cd ..

echo [2/2] 后台启动后端：http://localhost:8000
cd backend
start "Agent Lab Backend" python -m uvicorn app.main:app --port 8000
endlocal

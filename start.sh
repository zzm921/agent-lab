#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"

echo "[1/2] 打包前端..."
(cd frontend && npm run build)

echo "[2/2] 后台启动后端：http://localhost:8000（日志：/tmp/agent-lab-backend.log）"
cd backend
nohup python -m uvicorn app.main:app --port 8000 > /tmp/agent-lab-backend.log 2>&1 &
echo "后端 PID: $!"

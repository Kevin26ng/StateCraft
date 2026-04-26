#!/bin/bash
set -e

echo "Starting Crisis Governance Simulator..."

# Start FastAPI backend on port 5000
cd /app
uvicorn api.server:app --host 0.0.0.0 --port 5000 &
echo "FastAPI started on :5000"

# Start Next.js frontend on port 3000
cd /app/frontend
npm start -- -p 3000 &
echo "Next.js started on :3000"

# Wait for services to come up
sleep 3

# Start nginx (foreground, keeps container alive)
echo "Starting nginx on :7860"
nginx -g 'daemon off;'

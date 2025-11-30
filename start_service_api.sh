#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
PATH=$PATH:/Users/yifeichen/.local/bin

# Kill existing service on port 8099
lsof -t -i:8099 | xargs kill -9 2>/dev/null

echo "Starting Service API..."
poetry run python -m uvicorn service_api:app --host 0.0.0.0 --port 8099 > service_api.log 2>&1 &
SERVICE_PID=$!
echo "Service API PID: $SERVICE_PID"

sleep 5

if ps -p $SERVICE_PID > /dev/null; then
   echo "Service API is running."
else
   echo "Service API failed to start. Log:"
   cat service_api.log
fi


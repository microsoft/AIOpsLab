#!/bin/bash

echo "=== Quick Functionality Test ==="
echo ""

# 1. Check if API is running
echo "1. Checking API health..."
curl -s http://localhost:8000/health | jq '.' || echo "API not running"

echo ""
echo "2. Checking internal workers status..."
curl -s http://localhost:8000/api/v1/workers/internal/status | jq '.'

echo ""
echo "3. Submitting a test task..."
TASK_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "test-problem-001",
    "parameters": {"max_steps": 10},
    "priority": 5
  }')

TASK_ID=$(echo $TASK_RESPONSE | jq -r '.id')
echo "Task created: $TASK_ID"

echo ""
echo "4. Checking task status..."
sleep 3
curl -s "http://localhost:8000/api/v1/tasks/$TASK_ID" | jq '.status'

echo ""
echo "5. Listing all workers..."
curl -s http://localhost:8000/api/v1/workers | jq '.total'

echo ""
echo "=== Test Complete ==="
# Quickstart Guide: AIOpsLab Task Execution API

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Docker (for Kind clusters)
- Kind CLI installed
- kubectl configured

## Installation

### 1. Database Setup

Create the PostgreSQL database and tables:

```bash
# Create database
createdb aiopslab_tasks

# Run migrations
psql aiopslab_tasks < migrations/001_create_tables.sql
```

### 2. API Server Setup

```bash
# Install dependencies
cd api/
pip install -r requirements.txt

# Configure database connection
export DATABASE_URL="postgresql://user:pass@localhost/aiopslab_tasks"

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Worker Setup

```bash
# Install dependencies
cd workers/
pip install -r requirements.txt

# Create Kind cluster for worker
export WORKER_ID="001"
kind create cluster --name worker-${WORKER_ID}-kind

# Start worker process
python worker.py --id worker-${WORKER_ID}-kind --backend-type default
```

## Basic Usage

### 1. Submit a Task

```bash
# Create a new task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "misconfig_app_hotel_res-detection-1",
    "agent_config": {
      "model": "gpt-4",
      "temperature": 0.7
    },
    "max_steps": 30,
    "timeout_minutes": 30
  }'

# Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_id": "misconfig_app_hotel_res-detection-1",
  "status": "pending",
  "created_at": "2025-09-12T10:00:00Z",
  "updated_at": "2025-09-12T10:00:00Z"
}
```

### 2. Check Task Status

```bash
# Get task details
curl http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000

# Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_id": "misconfig_app_hotel_res-detection-1",
  "status": "running",
  "worker_id": "worker-001-kind",
  "started_at": "2025-09-12T10:00:05Z",
  "parameters": {
    "agent_config": {"model": "gpt-4", "temperature": 0.7},
    "max_steps": 30,
    "timeout_minutes": 30
  }
}
```

### 3. List All Tasks

```bash
# List tasks with filters
curl "http://localhost:8000/tasks?status=pending&limit=10"

# Response
{
  "tasks": [...],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

### 4. Get Task Results

```bash
# Get completed task with results
curl "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000?include_logs=true"

# Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "logs": ["Starting problem execution...", "Problem solved successfully"],
    "metrics": {
      "duration_seconds": 120,
      "steps_taken": 15
    },
    "output": {
      "solution": "Misconfiguration detected in deployment.yaml"
    }
  },
  "completed_at": "2025-09-12T10:02:05Z"
}
```

### 5. Monitor Workers

```bash
# List active workers
curl http://localhost:8000/workers

# Response
{
  "workers": [
    {
      "id": "worker-001-kind",
      "backend_type": "default",
      "status": "busy",
      "current_task_id": "550e8400-e29b-41d4-a716-446655440000",
      "last_heartbeat": "2025-09-12T10:02:00Z",
      "tasks_completed": 10,
      "tasks_failed": 2
    }
  ],
  "total": 1
}
```

## Testing the System

### End-to-End Test Scenario

```python
import requests
import time
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_task_lifecycle():
    """Test complete task lifecycle from creation to completion."""
    
    # 1. Create a task
    task_data = {
        "problem_id": "misconfig_app_hotel_res-detection-1",
        "agent_config": {"model": "gpt-4"},
        "max_steps": 30
    }
    
    response = requests.post(f"{BASE_URL}/tasks", json=task_data)
    assert response.status_code == 201
    task = response.json()
    task_id = task["id"]
    print(f"Created task: {task_id}")
    
    # 2. Verify task is pending
    response = requests.get(f"{BASE_URL}/tasks/{task_id}")
    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "pending"
    print("Task is pending")
    
    # 3. Wait for worker to pick up task
    max_wait = 30  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        task = response.json()
        if task["status"] != "pending":
            break
        time.sleep(1)
    
    assert task["status"] == "running"
    assert task["worker_id"] is not None
    print(f"Task picked up by worker: {task['worker_id']}")
    
    # 4. Wait for task completion
    max_wait = 300  # 5 minutes
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        task = response.json()
        if task["status"] in ["completed", "failed", "timeout"]:
            break
        time.sleep(5)
    
    # 5. Verify task completed
    assert task["status"] == "completed"
    assert task["result"] is not None
    print(f"Task completed successfully")
    
    # 6. Get task logs
    response = requests.get(f"{BASE_URL}/tasks/{task_id}/logs")
    assert response.status_code == 200
    logs = response.json()
    print(f"Task generated {len(logs['logs'])} log entries")
    
    return task

if __name__ == "__main__":
    test_task_lifecycle()
    print("✅ All tests passed!")
```

### Load Testing

```bash
# Submit multiple tasks concurrently
for i in {1..10}; do
  curl -X POST http://localhost:8000/tasks \
    -H "Content-Type: application/json" \
    -d '{
      "problem_id": "misconfig_app_hotel_res-detection-'$i'",
      "priority": '$((RANDOM % 10))'
    }' &
done
wait

# Monitor task processing
watch -n 1 'curl -s http://localhost:8000/tasks?limit=100 | jq ".tasks | group_by(.status) | map({status: .[0].status, count: length})"'
```

## Scaling Workers

### Add More Workers

```bash
# Start additional workers
for i in {002..005}; do
  # Create Kind cluster
  kind create cluster --name worker-${i}-kind
  
  # Start worker
  python worker.py --id worker-${i}-kind --backend-type default &
done
```

### Monitor Worker Health

```bash
# Check worker status
curl http://localhost:8000/workers | jq '.workers | map({id: .id, status: .status, tasks: .tasks_completed})'
```

## Troubleshooting

### Common Issues

1. **Task stuck in pending**: Check if workers are running and sending heartbeats
2. **Task timeout**: Increase `timeout_minutes` parameter or optimize problem execution
3. **Worker offline**: Check worker logs and Kind cluster status
4. **Database connection errors**: Verify PostgreSQL is running and credentials are correct

### Debug Commands

```bash
# Check API logs
tail -f api/logs/api.log

# Check worker logs
tail -f workers/logs/worker-001.log

# Check Kind cluster status
kind get clusters
kubectl get pods --all-namespaces

# Database queries
psql aiopslab_tasks -c "SELECT id, status, worker_id FROM tasks WHERE status='running';"
psql aiopslab_tasks -c "SELECT id, status, last_heartbeat FROM workers;"
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Next Steps

1. Configure multiple workers for parallel execution
2. Set up monitoring dashboards
3. Implement callback webhooks for task completion
4. Add custom problem types to workers
5. Set up production deployment with Docker Compose or Kubernetes
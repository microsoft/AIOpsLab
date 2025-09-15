# Quick Start Guide - AIOpsLab Task Executor

## Prerequisites

- Docker & Docker Compose (for PostgreSQL database)
- Python 3.11+
- Poetry (Python package manager)

## Setup

### 1. Start the Database
```bash
make db-up
```
This starts PostgreSQL on port 5432.

### 2. Start the API Server
```bash
make api-dev
```
The API server will start on http://localhost:8000 with:
- 3 integrated workers automatically running
- Auto-reload enabled for development
- OpenAPI docs at http://localhost:8000/docs

## Running Experiments

### Basic Task Execution

1. **Create a task:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "k8s_target_port-misconfig-detection-1",
    "parameters": {
      "max_steps": 30,
      "timeout": 1800
    }
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_id": "k8s_target_port-misconfig-detection-1",
  "status": "pending",
  ...
}
```

2. **Check task status:**
```bash
# Replace with your task ID
TASK_ID="550e8400-e29b-41d4-a716-446655440000"
curl http://localhost:8000/api/v1/tasks/$TASK_ID
```

3. **View statistics:**
```bash
curl http://localhost:8000/api/v1/tasks/stats
```

### Batch Experiments

Create multiple tasks to test throughput:
```bash
# Create 10 tasks with different problem types
PROBLEMS=("k8s_target_port-misconfig-detection-1"
          "auth_miss_mongodb-detection-1"
          "pod_failure-localization-1"
          "network_delay-analysis-1"
          "cart_service_failure-mitigation-1")

for problem in "${PROBLEMS[@]}"; do
  curl -X POST http://localhost:8000/api/v1/tasks \
    -H "Content-Type: application/json" \
    -d "{
      \"problem_id\": \"$problem\",
      \"parameters\": {\"max_steps\": 30}
    }"
done
```

### Monitor System

1. **Check health:**
```bash
curl http://localhost:8000/health
```

2. **View worker status:**
```bash
curl http://localhost:8000/api/v1/workers/internal/status
```

3. **Scale workers:**
```bash
# Increase to 5 workers
curl -X POST http://localhost:8000/api/v1/workers/internal/scale \
  -H "Content-Type: application/json" \
  -d '{"num_workers": 5}'
```

## Testing

Run the test suite:
```bash
make test-all
```

## Useful Commands

| Command | Description |
|---------|-------------|
| `make api-dev` | Start API with auto-reload |
| `make db-up` | Start PostgreSQL |
| `make db-reset` | Reset database |
| `make test` | Run quick tests |
| `make stop` | Stop all services |
| `make clean` | Clean up files |

## API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Detailed API Guide**: See `api/API_DOCUMENTATION.md`

## Architecture Notes

- **Workers are integrated**: No need to start separate worker processes
- **Database as queue**: PostgreSQL serves as both task queue and storage
- **Auto-scaling**: Workers scale automatically based on load
- **No auth required**: API is open for development use
- **30-minute timeout**: Default for all tasks

## Quick Experiment Script

Save this as `experiment.sh`:
```bash
#!/bin/bash

# Start services
echo "Starting database..."
make db-up
sleep 3

echo "Starting API server..."
make api-dev &
API_PID=$!
sleep 5

# Run experiment with real problems
echo "Creating AIOpsLab tasks..."
PROBLEMS=("k8s_target_port-misconfig-detection-1"
          "mongodb_auth_miss-localization-1"
          "pod_kill-analysis-1"
          "network_loss-mitigation-1"
          "ad_service_high_cpu-detection-1")

for problem in "${PROBLEMS[@]}"; do
  curl -s -X POST http://localhost:8000/api/v1/tasks \
    -H "Content-Type: application/json" \
    -d "{\"problem_id\": \"$problem\", \"parameters\": {}}" \
    | jq -r '.id'
done

echo "Waiting for completion..."
sleep 10

echo "Checking statistics..."
curl -s http://localhost:8000/api/v1/tasks/stats | jq '.'

# Cleanup
echo "Stopping services..."
kill $API_PID
make db-down
```

## Troubleshooting

**Database connection issues:**
```bash
# Reset database
make db-reset
```

**Port already in use:**
```bash
# Kill existing processes
lsof -ti:8000 | xargs kill -9
lsof -ti:5432 | xargs kill -9
```

**Check logs:**
```bash
# API logs are in console output
# Database logs
docker-compose logs postgres
```

## Available Problem IDs

AIOpsLab includes various problem scenarios across different task types:

### Detection Tasks
- `k8s_target_port-misconfig-detection-1` - Kubernetes target port misconfiguration
- `auth_miss_mongodb-detection-1` - MongoDB authentication missing
- `revoke_auth_mongodb-detection-1` - MongoDB authentication revoked
- `ad_service_failure-detection-1` - Advertisement service failure
- `ad_service_high_cpu-detection-1` - Advertisement service high CPU usage

### Localization Tasks
- `k8s_target_port-misconfig-localization-1` - Locate K8s port misconfiguration
- `auth_miss_mongodb-localization-1` - Locate MongoDB auth issues
- `pod_failure-localization-1` - Locate pod failures
- `network_loss-localization-1` - Locate network packet loss

### Analysis Tasks
- `k8s_target_port-misconfig-analysis-1` - Analyze K8s port misconfiguration
- `pod_kill-analysis-1` - Analyze pod termination issues
- `network_delay-analysis-1` - Analyze network latency problems
- `cart_service_failure-analysis-1` - Analyze cart service failures

### Mitigation Tasks
- `k8s_target_port-misconfig-mitigation-1` - Fix K8s port misconfiguration
- `auth_miss_mongodb-mitigation-1` - Fix MongoDB authentication
- `cart_service_failure-mitigation-1` - Fix cart service issues
- `payment_service_failure-mitigation-1` - Fix payment service failures

## Next Steps

1. Modify `task-executor/api/src/workers/executor.py` to implement actual task logic
2. Connect to actual AIOpsLab orchestrator for real problem execution
3. Configure worker capabilities for different problem types
4. Set up monitoring with Prometheus metrics endpoint
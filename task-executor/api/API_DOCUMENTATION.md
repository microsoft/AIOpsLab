# Task Executor API Documentation

## Overview

The Task Executor API provides a RESTful interface for managing and executing AIOpsLab tasks. It features integrated worker management, automatic task execution, and comprehensive monitoring capabilities.

## Base URL

```
http://localhost:8000
```

## API Version

All endpoints are prefixed with `/api/v1` for versioning.

## Architecture

- **API Server**: FastAPI-based REST service
- **Database**: PostgreSQL for task queue and persistent storage
- **Workers**: Internal background workers managed by the API
- **Task Queue**: Database-backed queue with atomic task claiming

## Authentication

No authentication is required for the current version.

## Endpoints

### Health & Monitoring

#### Health Check
```http
GET /health
```

Returns the overall health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-14T10:30:00Z",
  "database": {
    "connected": true,
    "latency_ms": 5.2
  },
  "workers": {
    "total": 3,
    "idle": 2,
    "busy": 1,
    "offline": 0
  },
  "queue": {
    "pending": 5,
    "running": 1,
    "completed": 42
  }
}
```

#### Readiness Check
```http
GET /ready
```

Indicates if the service is ready to accept requests.

**Response:**
```json
{
  "ready": true,
  "database": true,
  "workers": true
}
```

#### Prometheus Metrics
```http
GET /metrics
```

Returns metrics in Prometheus format.

### Task Management

#### Create Task
```http
POST /api/v1/tasks
```

Creates a new task for execution.

**Request Body:**
```json
{
  "problem_id": "test-problem-001",
  "parameters": {
    "max_steps": 30,
    "timeout": 1800,
    "agent_config": {
      "model": "gpt-4",
      "temperature": 0.7
    }
  }
}
```

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_id": "test-problem-001",
  "status": "pending",
  "parameters": {...},
  "created_at": "2025-01-14T10:30:00Z",
  "updated_at": "2025-01-14T10:30:00Z"
}
```

#### Get Task
```http
GET /api/v1/tasks/{task_id}
```

Retrieves details of a specific task.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_id": "test-problem-001",
  "status": "completed",
  "parameters": {...},
  "worker_id": "worker-001-kind",
  "result": {
    "success": true,
    "output": "Task completed successfully",
    "metrics": {
      "duration_seconds": 45.2,
      "steps_taken": 12
    }
  },
  "created_at": "2025-01-14T10:30:00Z",
  "updated_at": "2025-01-14T10:31:00Z",
  "started_at": "2025-01-14T10:30:05Z",
  "completed_at": "2025-01-14T10:30:50Z"
}
```

#### List Tasks
```http
GET /api/v1/tasks
```

Lists tasks with optional filtering.

**Query Parameters:**
- `status` (string): Filter by status (pending, running, completed, failed, timeout, cancelled)
- `problem_id` (string): Filter by problem ID
- `worker_id` (string): Filter by worker ID
- `limit` (integer): Maximum number of results (default: 100)
- `offset` (integer): Pagination offset (default: 0)

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "problem_id": "test-problem-001",
    "status": "completed",
    "worker_id": "worker-001-kind",
    "created_at": "2025-01-14T10:30:00Z"
  }
]
```

#### Cancel Task
```http
POST /api/v1/tasks/{task_id}/cancel
```

Cancels a pending or running task.

**Response:**
```json
{
  "message": "Task cancelled successfully",
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Get Task Logs
```http
GET /api/v1/tasks/{task_id}/logs
```

Retrieves execution logs for a task.

**Query Parameters:**
- `level` (string): Filter by log level (debug, info, warning, error, critical)
- `limit` (integer): Maximum number of log entries

**Response:**
```json
[
  {
    "timestamp": "2025-01-14T10:30:05Z",
    "level": "info",
    "message": "Task execution started",
    "context": {
      "step": 1,
      "action": "initialize"
    }
  }
]
```

### Worker Management

#### List Workers
```http
GET /api/v1/workers
```

Lists all registered workers.

**Query Parameters:**
- `status` (string): Filter by status (idle, busy, offline)
- `backend_type` (string): Filter by backend type

**Response:**
```json
[
  {
    "id": "worker-001-kind",
    "backend_type": "kind",
    "status": "idle",
    "last_heartbeat": "2025-01-14T10:30:00Z",
    "capabilities": {
      "max_parallel_tasks": 5,
      "supported_problems": ["test-*"]
    },
    "tasks_completed": 42,
    "tasks_failed": 3
  }
]
```

#### Register Worker
```http
POST /api/v1/workers/register
```

Registers a new worker.

**Request Body:**
```json
{
  "worker_id": "worker-002-kind",
  "backend_type": "kind",
  "capabilities": {
    "max_parallel_tasks": 5,
    "supported_problems": ["test-*", "benchmark-*"]
  }
}
```

**Response (201 Created):**
```json
{
  "id": "worker-002-kind",
  "backend_type": "kind",
  "status": "idle",
  "registered_at": "2025-01-14T10:30:00Z"
}
```

#### Worker Heartbeat
```http
POST /api/v1/workers/{worker_id}/heartbeat
```

Updates worker status and heartbeat.

**Request Body:**
```json
{
  "status": "idle",
  "current_task_id": null
}
```

**Response:**
```json
{
  "message": "Heartbeat received",
  "timestamp": "2025-01-14T10:30:00Z"
}
```

#### Claim Task (For Workers)
```http
POST /api/v1/workers/{worker_id}/claim
```

Worker claims the next available task.

**Response (200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "problem_id": "test-problem-001",
  "status": "running",
  "parameters": {...}
}
```

**Response (204 No Content):**
No tasks available.

### Internal Worker Control

#### Get Internal Workers Status
```http
GET /api/v1/workers/internal/status
```

Gets the status of internal worker processes.

**Response:**
```json
{
  "running": true,
  "num_workers": 3,
  "workers": [
    {
      "id": "worker-001-kind",
      "status": "idle",
      "current_task": null
    }
  ]
}
```

#### Scale Internal Workers
```http
POST /api/v1/workers/internal/scale
```

Adjusts the number of internal workers.

**Request Body:**
```json
{
  "num_workers": 5
}
```

**Response:**
```json
{
  "message": "Workers scaled successfully",
  "previous_count": 3,
  "new_count": 5
}
```

#### Stop Internal Workers
```http
POST /api/v1/workers/internal/stop
```

Stops all internal workers.

**Response:**
```json
{
  "message": "Workers stopped",
  "stopped_count": 3
}
```

#### Start Internal Workers
```http
POST /api/v1/workers/internal/start
```

Starts internal workers.

**Response:**
```json
{
  "message": "Workers started",
  "worker_count": 3
}
```

## Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `204 No Content`: Request successful, no content to return
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Operation conflicts with current state
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Task States

Tasks progress through the following states:

1. **pending**: Task created, waiting to be processed
2. **running**: Task being executed by a worker
3. **completed**: Task finished successfully
4. **failed**: Task execution failed
5. **timeout**: Task exceeded time limit
6. **cancelled**: Task was cancelled

## Error Responses

Error responses follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong",
  "type": "error_type",
  "request_id": "req_123abc"
}
```

## Examples

### Complete Task Lifecycle

1. **Create a task:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "example-problem",
    "parameters": {
      "max_steps": 30,
      "timeout": 1800
    }
  }'
```

2. **Monitor task status:**
```bash
TASK_ID="550e8400-e29b-41d4-a716-446655440000"
curl http://localhost:8000/api/v1/tasks/$TASK_ID
```

3. **Get task logs:**
```bash
curl http://localhost:8000/api/v1/tasks/$TASK_ID/logs
```

4. **Cancel if needed:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks/$TASK_ID/cancel
```

### Worker Operations

1. **Check worker status:**
```bash
curl http://localhost:8000/api/v1/workers
```

2. **Scale workers:**
```bash
curl -X POST http://localhost:8000/api/v1/workers/internal/scale \
  -H "Content-Type: application/json" \
  -d '{"num_workers": 5}'
```

3. **Monitor health:**
```bash
curl http://localhost:8000/health
```

## WebSocket Support (Future)

WebSocket support for real-time task updates is planned for future versions:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tasks/550e8400-e29b-41d4-a716-446655440000');
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Task update:', update);
};
```

## Rate Limiting

Currently no rate limiting is implemented. In production, consider adding rate limiting based on:
- Requests per minute per IP
- Concurrent tasks per client
- Total system load

## Best Practices

1. **Polling**: When monitoring task status, use exponential backoff:
   - Start with 1-second intervals
   - Double the interval up to 30 seconds
   - Stop after task reaches terminal state

2. **Error Handling**: Always handle potential errors:
   - Network failures
   - Task timeouts
   - Worker unavailability

3. **Resource Management**:
   - Set appropriate timeouts for tasks
   - Cancel tasks that are no longer needed
   - Monitor worker health regularly

4. **Idempotency**: Task creation should include unique identifiers to prevent duplicates

## Environment Variables

Configure the API using these environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `NUM_INTERNAL_WORKERS`: Number of internal workers (default: 3)
- `AUTO_START_WORKERS`: Auto-start workers on startup (default: true)
- `DEFAULT_TIMEOUT_MINUTES`: Default task timeout (default: 30)
- `DEFAULT_MAX_STEPS`: Default maximum steps (default: 30)
- `LOG_LEVEL`: Logging level (debug, info, warning, error)
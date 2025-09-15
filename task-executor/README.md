# Task Executor

RESTful API for scalable AIOpsLab task execution with integrated worker management.

## Architecture

```
task-executor/
├── api/                    # RESTful API Server with integrated workers
│   ├── src/               # Source code
│   │   ├── api/          # API endpoints
│   │   ├── models/       # Database models
│   │   ├── services/     # Business logic
│   │   ├── schemas/      # Pydantic schemas
│   │   ├── lib/          # Libraries (task queue)
│   │   └── workers/      # Internal worker management
│   ├── tests/            # Test suites
│   └── pyproject.toml    # Python dependencies (Poetry)
```

## Key Changes from Previous Version

✨ **Workers are now integrated into the API process** - No need to manually start separate worker processes. The API automatically manages internal workers as background tasks.

## Components

### API Server with Integrated Workers
- **Framework**: FastAPI with async support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Task Queue**: Database-backed queue using SELECT FOR UPDATE SKIP LOCKED
- **Workers**: Internal background tasks managed by the API
- **Monitoring**: Prometheus metrics and structured logging

## Key Features

- **Integrated Workers**: Workers run as background tasks within the API process
- **Auto-scaling**: Dynamic worker scaling via API endpoints
- **Atomic Task Claiming**: SELECT FOR UPDATE SKIP LOCKED prevents race conditions
- **Comprehensive Monitoring**: Metrics, logs, and real-time status
- **Flexible Configuration**: Environment variables for all settings
- **Automatic Recovery**: Timeout handling and error recovery

## Quick Start

```bash
# 1. Start PostgreSQL
docker-compose up -d

# 2. Install dependencies
cd api
poetry install

# 3. Start API (workers start automatically!)
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

That's it! The API will automatically start 3 internal workers by default.

## Configuration

Create a `.env` file in the `api` directory:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://aiopslab:aiopslab@localhost:5432/aiopslab

# Worker Settings
NUM_INTERNAL_WORKERS=3      # Number of workers to start
AUTO_START_WORKERS=true      # Auto-start workers on API startup

# Task Settings
DEFAULT_TIMEOUT_MINUTES=30   # Task timeout
DEFAULT_MAX_STEPS=30         # Max steps for task execution
```

## API Endpoints

### Tasks
- `POST /api/v1/tasks` - Create new task
- `GET /api/v1/tasks` - List tasks with filtering
- `GET /api/v1/tasks/{id}` - Get task details
- `PUT /api/v1/tasks/{id}/cancel` - Cancel task
- `GET /api/v1/tasks/{id}/logs` - Get task logs
- `GET /api/v1/tasks/stats` - Queue statistics

### Workers
- `GET /api/v1/workers` - List all workers
- `GET /api/v1/workers/{id}` - Get worker details
- `GET /api/v1/workers/{id}/stats` - Worker statistics

### Internal Worker Control
- `GET /api/v1/workers/internal/status` - Internal worker status
- `POST /api/v1/workers/internal/scale?num_workers=N` - Scale workers
- `POST /api/v1/workers/internal/stop` - Stop all workers
- `POST /api/v1/workers/internal/start` - Start workers

### Monitoring
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Testing the System

Run the integrated test script:

```bash
cd task-executor
python test_integrated.py
```

This will:
1. Check API health
2. Verify workers are running
3. Submit test tasks
4. Monitor task completion
5. Test worker scaling

## Task Submission Example

```python
import aiohttp
import asyncio

async def submit_task():
    async with aiohttp.ClientSession() as session:
        task_data = {
            "problem_id": "test-problem-001",
            "parameters": {
                "max_steps": 10,
                "agent_config": {"name": "test-agent"}
            },
            "priority": 5
        }

        async with session.post(
            "http://localhost:8000/api/v1/tasks",
            json=task_data
        ) as resp:
            task = await resp.json()
            print(f"Task created: {task['id']}")

asyncio.run(submit_task())
```

## Worker Scaling Examples

```bash
# Scale to 10 workers
curl -X POST "http://localhost:8000/api/v1/workers/internal/scale?num_workers=10"

# Check worker status
curl "http://localhost:8000/api/v1/workers/internal/status"

# Stop all workers
curl -X POST "http://localhost:8000/api/v1/workers/internal/stop"
```

## Development

```bash
# Run tests
make test

# TDD mode
make tdd

# Format code
poetry run black src/

# Type check
poetry run pyright src/
```

## Database Schema

- **tasks**: Task queue and execution results
- **workers**: Worker registration and status
- **task_logs**: Detailed task execution logs

## Architecture Benefits

The integrated worker approach provides:

1. **Simplicity**: Single process to manage
2. **Efficiency**: Shared memory and resources
3. **Control**: Direct API control over workers
4. **Monitoring**: Unified logs and metrics
5. **Deployment**: Easier containerization and deployment
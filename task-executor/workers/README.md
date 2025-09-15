# Task Workers

Worker processes that poll the task queue and execute AIOpsLab tasks.

## Structure

```
workers/
├── src/
│   ├── __init__.py
│   ├── worker.py           # Main worker process
│   ├── executor.py         # Task executor
│   └── orchestrator_client.py  # Client for AIOpsLab orchestrator
├── tests/
│   └── test_worker.py
├── requirements.txt
└── README.md
```

## Usage

```bash
# Start a worker
python src/worker.py --id worker-001-kind --backend-type default

# Start multiple workers
for i in 001 002 003; do
    python src/worker.py --id worker-$i-kind --backend-type default &
done
```

## Worker Lifecycle

1. **Registration**: Worker registers with API server
2. **Polling**: Continuously polls for available tasks
3. **Execution**: Claims task and executes using orchestrator
4. **Reporting**: Reports results back to API server
5. **Heartbeat**: Sends periodic heartbeats to maintain status
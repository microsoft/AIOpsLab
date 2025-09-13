# Data Model: AIOpsLab Task Execution API

## Core Entities

### Task
Primary entity representing an AIOpsLab problem execution request.

**Fields**:
- `id` (UUID): Unique identifier, auto-generated
- `problem_id` (String): AIOpsLab problem identifier (required)
- `status` (Enum): Current task state
  - Values: `pending`, `running`, `completed`, `failed`, `timeout`
  - Default: `pending`
- `parameters` (JSONB): Task execution parameters
  - `agent_config` (Object): Agent configuration settings
  - `max_steps` (Integer): Maximum execution steps (default: 30)
  - `timeout_minutes` (Integer): Custom timeout (default: 30)
  - `callback_url` (String): Optional webhook for completion
  - `metadata` (Object): User-defined metadata
  - `priority` (Integer): Task priority (0-10, default: 5)
- `worker_id` (String): Assigned worker's cluster name
- `result` (JSONB): Execution results
  - `logs` (Array): Execution log entries
  - `metrics` (Object): Performance metrics
  - `output` (Object): Problem-specific output
- `error_details` (Text): Error message if failed
- `created_at` (Timestamp): Task creation time
- `updated_at` (Timestamp): Last modification time
- `started_at` (Timestamp): Execution start time
- `completed_at` (Timestamp): Execution end time

**Constraints**:
- `problem_id` must be non-empty
- `status` transitions: pending → running → {completed|failed|timeout}
- `worker_id` set only when status = running
- `timeout_minutes` must be between 1 and 180

**Indexes**:
- Primary: `id`
- Secondary: `status`, `worker_id`, `created_at`, `problem_id`

### Worker
Represents a worker process that executes tasks.

**Fields**:
- `id` (String): Worker identifier (cluster name)
- `backend_type` (String): Backend configuration type
- `status` (Enum): Worker status
  - Values: `idle`, `busy`, `offline`
  - Default: `idle`
- `last_heartbeat` (Timestamp): Last health check time
- `current_task_id` (UUID): Currently executing task
- `capabilities` (JSONB): Worker capabilities
  - `max_parallel_tasks` (Integer): Concurrency limit
  - `supported_problems` (Array): Problem types supported
- `metadata` (JSONB): Worker metadata
  - `host` (String): Host machine
  - `version` (String): Worker version
  - `kind_cluster` (String): Kind cluster name
- `registered_at` (Timestamp): Registration time
- `tasks_completed` (Integer): Total completed tasks
- `tasks_failed` (Integer): Total failed tasks

**Constraints**:
- `id` must match pattern `worker-XXX-kind`
- `last_heartbeat` must be within 60 seconds for `idle`/`busy` status
- `current_task_id` set only when status = busy

**Indexes**:
- Primary: `id`
- Secondary: `status`, `backend_type`, `last_heartbeat`

### TaskLog
Detailed execution logs for tasks.

**Fields**:
- `id` (UUID): Log entry identifier
- `task_id` (UUID): Associated task (foreign key)
- `timestamp` (Timestamp): Log entry time
- `level` (Enum): Log level
  - Values: `debug`, `info`, `warning`, `error`, `critical`
- `message` (Text): Log message
- `context` (JSONB): Additional context
  - `step` (Integer): Execution step number
  - `action` (String): Action being performed
  - `details` (Object): Action-specific details

**Constraints**:
- `task_id` must reference existing task
- `timestamp` must be between task's `started_at` and `completed_at`

**Indexes**:
- Primary: `id`
- Secondary: `task_id`, `timestamp`, `level`

## State Transitions

### Task State Machine
```
pending → running → completed
         ↓       ↓
       failed  timeout
```

**Transition Rules**:
1. `pending` → `running`: Worker claims task
2. `running` → `completed`: Successful execution
3. `running` → `failed`: Execution error
4. `running` → `timeout`: Exceeds timeout_minutes
5. No transitions from terminal states (completed/failed/timeout)

### Worker State Machine
```
offline ←→ idle ←→ busy
```

**Transition Rules**:
1. `offline` → `idle`: Worker sends heartbeat
2. `idle` → `busy`: Worker claims task
3. `busy` → `idle`: Task completes
4. `idle`/`busy` → `offline`: Heartbeat timeout (60s)

## Data Relationships

```
Task (1) ←→ (0..1) Worker
Task (1) ←→ (0..*) TaskLog
```

## Validation Rules

### Task Creation
- `problem_id` must be provided and non-empty
- `parameters` optional but must be valid JSON if provided
- `max_steps` if provided must be 1-1000
- `timeout_minutes` if provided must be 1-180
- `priority` if provided must be 0-10
- `callback_url` if provided must be valid HTTP(S) URL

### Worker Registration
- `id` must follow pattern `worker-XXX-kind`
- `backend_type` must be provided
- `capabilities` must include at least `max_parallel_tasks`

### State Updates
- Only `pending` tasks can transition to `running`
- Only `running` tasks can transition to terminal states
- Worker must be `idle` to claim a task
- Worker must send heartbeat every 30 seconds

## Query Patterns

### Common Queries
```sql
-- Get next pending task for worker
SELECT * FROM tasks 
WHERE status = 'pending' 
ORDER BY priority DESC, created_at ASC 
LIMIT 1 
FOR UPDATE SKIP LOCKED;

-- Get worker status
SELECT id, status, last_heartbeat, current_task_id 
FROM workers 
WHERE last_heartbeat > NOW() - INTERVAL '60 seconds';

-- Get task with logs
SELECT t.*, tl.* 
FROM tasks t 
LEFT JOIN task_logs tl ON t.id = tl.task_id 
WHERE t.id = ? 
ORDER BY tl.timestamp;

-- Get task statistics
SELECT 
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE status = 'completed') as completed,
  COUNT(*) FILTER (WHERE status = 'failed') as failed,
  AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration
FROM tasks 
WHERE created_at > NOW() - INTERVAL '24 hours';
```

## Data Retention

- Tasks: Permanent retention (no deletion)
- TaskLogs: Permanent retention (linked to tasks)
- Workers: Soft delete after 7 days offline
- Partitioning: Monthly partitions on `tasks.created_at`

## Performance Considerations

- JSONB columns indexed for common query paths
- Partial indexes on status fields for active records
- Connection pooling for concurrent worker access
- Batch inserts for task logs
- Vacuum scheduling for deleted worker records
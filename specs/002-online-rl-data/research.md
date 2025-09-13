# Research Findings: AIOpsLab Task Execution API

## Task Queue Implementation Pattern

**Decision**: PostgreSQL-based task queue with worker polling
**Rationale**: 
- Simple implementation without additional message queue infrastructure
- Leverages existing database for persistence
- Supports ACID transactions for task state management
- Easy debugging and monitoring through SQL queries

**Alternatives considered**:
- Celery with Redis/RabbitMQ: More complex, requires additional infrastructure
- AWS SQS/Azure Service Bus: Cloud vendor lock-in, external dependency
- In-memory queue: No persistence, loses tasks on restart

## Worker Polling Strategy

**Decision**: SELECT FOR UPDATE SKIP LOCKED pattern with 5-second polling interval
**Rationale**:
- Prevents race conditions between workers
- Built-in PostgreSQL feature, no external locks needed
- Efficient for moderate task volumes (thousands per day)
- Workers can dynamically join/leave without coordination

**Alternatives considered**:
- LISTEN/NOTIFY: More complex, requires persistent connections
- Distributed locks (Redis/Zookeeper): Additional infrastructure
- Timestamp-based claiming: Race condition prone

## Database Schema Design

**Decision**: Single tasks table with JSONB for flexible parameters
**Rationale**:
- Simple schema evolution without migrations for new parameters
- PostgreSQL JSONB provides indexing and querying capabilities
- Status enum field for clear state machine
- Audit trail through created_at/updated_at timestamps

**Schema**:
```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    parameters JSONB,
    worker_id VARCHAR(255),
    result JSONB,
    error_details TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_worker ON tasks(worker_id);
CREATE INDEX idx_tasks_created ON tasks(created_at);
```

**Alternatives considered**:
- Separate tables for parameters/results: Over-normalization
- NoSQL database: Loses ACID guarantees
- Multiple status tables: Complex joins

## AIOpsLab Orchestrator Integration

**Decision**: Direct Python import of existing Orchestrator class
**Rationale**:
- Reuses existing, tested code
- No API translation layer needed
- Direct access to all Orchestrator features
- Maintains compatibility with current AIOpsLab patterns

**Integration approach**:
```python
from aiopslab.orchestrator import Orchestrator

class AIOpsLabClient:
    def __init__(self):
        self.orchestrator = Orchestrator()
    
    def execute_problem(self, problem_id, **kwargs):
        self.orchestrator.init_problem(problem_id)
        # ... execution logic
```

**Alternatives considered**:
- REST API wrapper: Additional complexity, performance overhead
- Subprocess execution: Difficult error handling, resource management
- Microservice architecture: Over-engineering for current scale

## Kind Cluster Isolation

**Decision**: Cluster name prefix with worker ID (e.g., worker-001-kind)
**Rationale**:
- Clear ownership identification
- Prevents resource conflicts
- Easy cleanup on worker failure
- Supports concurrent execution

**Management strategy**:
```bash
# Create cluster for worker
kind create cluster --name worker-${WORKER_ID}-kind

# Set context for worker
kubectl config use-context kind-worker-${WORKER_ID}-kind

# Cleanup on termination
kind delete cluster --name worker-${WORKER_ID}-kind
```

**Alternatives considered**:
- Shared cluster with namespaces: Resource contention
- Docker-in-Docker: Complex networking
- VMs per worker: Resource intensive

## API Framework Selection

**Decision**: FastAPI with Pydantic models
**Rationale**:
- Automatic OpenAPI documentation generation
- Built-in request/response validation
- Async support for better performance
- Strong typing with Pydantic
- Active community and ecosystem

**Alternatives considered**:
- Flask: Requires additional libraries for same features
- Django REST: Heavyweight for task queue API
- aiohttp: Lower level, more boilerplate

## Worker Process Management

**Decision**: Standalone Python processes with systemd/supervisor
**Rationale**:
- Simple deployment model
- Easy to scale horizontally
- Independent failure domains
- Standard logging and monitoring

**Alternatives considered**:
- Kubernetes Jobs: Complex for long-running workers
- Docker Swarm: Additional orchestration layer
- Threading: GIL limitations, shared memory issues

## Data Retention Strategy

**Decision**: Permanent storage with partitioned tables by month
**Rationale**:
- Meets requirement for permanent retention
- Partitioning enables future archival if needed
- Query performance maintained over time
- Easy backup and recovery

**Alternatives considered**:
- Archive to S3 after X days: Additional complexity
- No partitioning: Performance degradation over time
- Separate archive database: Operational overhead

## Error Handling and Resilience

**Decision**: Exponential backoff with max retries configurable per task
**Rationale**:
- Handles transient failures gracefully
- Prevents thundering herd on recovery
- Configurable per problem type
- Clear failure states for debugging

**Alternatives considered**:
- Immediate failure: Poor user experience
- Infinite retries: Resource waste, unclear state
- Fixed retry intervals: Not adaptive to load

## Monitoring and Observability

**Decision**: Structured logging with correlation IDs
**Rationale**:
- Traces task execution across API and workers
- Integration with existing logging infrastructure
- Easy debugging and audit trail
- Performance metrics collection

**Implementation**:
```python
import structlog

logger = structlog.get_logger()
logger = logger.bind(task_id=task_id, worker_id=worker_id)
logger.info("task.started", problem_id=problem_id)
```

**Alternatives considered**:
- OpenTelemetry: Over-complex for initial version
- Custom metrics system: Reinventing the wheel
- Plain text logs: Difficult to parse and analyze

## Summary

All technical decisions have been made with focus on:
1. **Simplicity**: Using proven patterns and existing tools
2. **Reliability**: ACID guarantees, clear failure modes
3. **Scalability**: Horizontal scaling through worker addition
4. **Maintainability**: Clear separation of concerns, standard patterns
5. **Compatibility**: Direct integration with existing AIOpsLab code

The architecture avoids unnecessary complexity while meeting all functional requirements from the specification.
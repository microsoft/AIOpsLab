# Tasks: AIOpsLab Task Execution API

**Input**: Design documents from `/specs/002-online-rl-data/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **API Project**: `api/src/`, `api/tests/`
- **Worker Project**: `workers/src/`, `workers/tests/`
- Paths shown below follow the plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create project structure for API and Workers per implementation plan
- [ ] T002 Initialize API project with FastAPI, SQLAlchemy, PostgreSQL dependencies in api/requirements.txt
- [ ] T003 Initialize Worker project with AIOpsLab integration dependencies in workers/requirements.txt
- [ ] T004 [P] Configure pytest and testing tools in api/pytest.ini and workers/pytest.ini
- [ ] T005 [P] Set up database migrations with Alembic in api/alembic.ini
- [ ] T006 [P] Create Docker Compose for PostgreSQL development in docker-compose.yml
- [ ] T007 [P] Configure logging with structlog in api/src/config/logging.py and workers/src/config/logging.py

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (API)
- [ ] T008 [P] Contract test POST /tasks in api/tests/contract/test_create_task.py
- [ ] T009 [P] Contract test GET /tasks in api/tests/contract/test_list_tasks.py
- [ ] T010 [P] Contract test GET /tasks/{id} in api/tests/contract/test_get_task.py
- [ ] T011 [P] Contract test DELETE /tasks/{id} in api/tests/contract/test_cancel_task.py
- [ ] T012 [P] Contract test GET /tasks/{id}/logs in api/tests/contract/test_get_task_logs.py
- [ ] T013 [P] Contract test GET /workers in api/tests/contract/test_list_workers.py
- [ ] T014 [P] Contract test GET /workers/{id} in api/tests/contract/test_get_worker.py
- [ ] T015 [P] Contract test POST /workers/heartbeat in api/tests/contract/test_worker_heartbeat.py
- [ ] T016 [P] Contract test POST /workers/register in api/tests/contract/test_register_worker.py

### Integration Tests
- [ ] T017 [P] Integration test task lifecycle (create→poll→complete) in api/tests/integration/test_task_lifecycle.py
- [ ] T018 [P] Integration test worker registration and heartbeat in api/tests/integration/test_worker_management.py
- [ ] T019 [P] Integration test task queueing with multiple workers in api/tests/integration/test_task_queue.py
- [ ] T020 [P] Integration test task timeout handling in api/tests/integration/test_task_timeout.py
- [ ] T021 [P] Integration test worker failure recovery in workers/tests/integration/test_worker_recovery.py
- [ ] T022 [P] Integration test concurrent task execution in workers/tests/integration/test_concurrent_tasks.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Database Models
- [ ] T023 [P] Task model with SQLAlchemy in api/src/models/task.py
- [ ] T024 [P] Worker model with SQLAlchemy in api/src/models/worker.py
- [ ] T025 [P] TaskLog model with SQLAlchemy in api/src/models/task_log.py
- [ ] T026 [P] Database session management in api/src/models/database.py
- [ ] T027 Create initial database migration for all models

### Libraries
- [ ] T028 [P] Task queue library with polling logic in api/src/lib/task_queue/__init__.py
- [ ] T029 [P] Task queue CLI commands in api/src/lib/task_queue/cli.py
- [ ] T030 [P] Worker manager library in workers/src/lib/worker_manager/__init__.py
- [ ] T031 [P] Worker manager CLI in workers/src/lib/worker_manager/cli.py
- [ ] T032 [P] AIOpsLab client wrapper in workers/src/lib/aiopslab_client/__init__.py
- [ ] T033 [P] AIOpsLab client CLI in workers/src/lib/aiopslab_client/cli.py

### Services
- [ ] T034 [P] Task service with business logic in api/src/services/task_service.py
- [ ] T035 [P] Worker service with registration/heartbeat in api/src/services/worker_service.py
- [ ] T036 [P] Task execution service in workers/src/services/execution_service.py
- [ ] T037 [P] Kind cluster management in workers/src/services/cluster_service.py

### API Endpoints
- [ ] T038 POST /tasks endpoint in api/src/api/tasks.py
- [ ] T039 GET /tasks endpoint with filtering in api/src/api/tasks.py
- [ ] T040 GET /tasks/{id} endpoint in api/src/api/tasks.py
- [ ] T041 DELETE /tasks/{id} cancel endpoint in api/src/api/tasks.py
- [ ] T042 GET /tasks/{id}/logs endpoint in api/src/api/tasks.py
- [ ] T043 GET /workers endpoint in api/src/api/workers.py
- [ ] T044 GET /workers/{id} endpoint in api/src/api/workers.py
- [ ] T045 POST /workers/heartbeat endpoint in api/src/api/workers.py
- [ ] T046 POST /workers/register endpoint in api/src/api/workers.py

### Worker Implementation
- [ ] T047 Main worker process with polling loop in workers/src/worker.py
- [ ] T048 Task claim with SELECT FOR UPDATE SKIP LOCKED in workers/src/worker.py
- [ ] T049 Task execution with Orchestrator integration in workers/src/worker.py
- [ ] T050 Heartbeat sender with threading in workers/src/worker.py
- [ ] T051 Error handling and retry logic in workers/src/worker.py

### Validation & Error Handling
- [ ] T052 Pydantic schemas for request/response in api/src/schemas/
- [ ] T053 Custom exceptions and error handlers in api/src/exceptions.py
- [ ] T054 Input validation middleware in api/src/middleware/validation.py

## Phase 3.4: Integration

### Database & Connections
- [ ] T055 Connection pooling configuration in api/src/config/database.py
- [ ] T056 Transaction management for task claiming in api/src/services/task_service.py
- [ ] T057 Database health check endpoint in api/src/api/health.py

### Middleware & Logging
- [ ] T058 Request ID middleware for tracing in api/src/middleware/request_id.py
- [ ] T059 Structured logging middleware in api/src/middleware/logging.py
- [ ] T060 Error tracking middleware in api/src/middleware/error_tracking.py

### Configuration
- [ ] T061 Environment configuration with pydantic-settings in api/src/config/settings.py
- [ ] T062 Worker configuration management in workers/src/config/settings.py
- [ ] T063 FastAPI app initialization in api/src/main.py

## Phase 3.5: Polish

### Unit Tests
- [ ] T064 [P] Unit tests for task state machine in api/tests/unit/test_task_states.py
- [ ] T065 [P] Unit tests for worker state transitions in api/tests/unit/test_worker_states.py
- [ ] T066 [P] Unit tests for priority queue logic in api/tests/unit/test_priority_queue.py
- [ ] T067 [P] Unit tests for timeout calculations in api/tests/unit/test_timeout.py
- [ ] T068 [P] Unit tests for validation schemas in api/tests/unit/test_schemas.py

### Performance & Monitoring
- [ ] T069 Performance test for 100 concurrent tasks in api/tests/performance/test_load.py
- [ ] T070 Add Prometheus metrics export in api/src/monitoring/metrics.py
- [ ] T071 Health check endpoints for workers in workers/src/api/health.py

### Documentation
- [ ] T072 [P] API documentation with examples in docs/api.md
- [ ] T073 [P] Worker deployment guide in docs/deployment.md
- [ ] T074 [P] Troubleshooting guide in docs/troubleshooting.md
- [ ] T075 [P] Update README.md with setup instructions

### Final Validation
- [ ] T076 Run end-to-end test from quickstart.md
- [ ] T077 Load test with 10 workers and 100 tasks
- [ ] T078 Verify all contract tests pass
- [ ] T079 Code cleanup and remove TODOs
- [ ] T080 Security review for SQL injection and input validation

## Dependencies
- Setup (T001-T007) must complete first
- All tests (T008-T022) before any implementation (T023-T054)
- Models (T023-T027) before services (T034-T037)
- Services before endpoints (T038-T046)
- Libraries (T028-T033) can run parallel with models
- Worker implementation (T047-T051) after libraries
- Integration (T055-T063) after core implementation
- Polish (T064-T080) after everything else

## Parallel Execution Examples

### Launch all contract tests together (T008-T016):
```
Task: "Contract test POST /tasks in api/tests/contract/test_create_task.py"
Task: "Contract test GET /tasks in api/tests/contract/test_list_tasks.py"
Task: "Contract test GET /tasks/{id} in api/tests/contract/test_get_task.py"
Task: "Contract test DELETE /tasks/{id} in api/tests/contract/test_cancel_task.py"
Task: "Contract test GET /tasks/{id}/logs in api/tests/contract/test_get_task_logs.py"
Task: "Contract test GET /workers in api/tests/contract/test_list_workers.py"
Task: "Contract test GET /workers/{id} in api/tests/contract/test_get_worker.py"
Task: "Contract test POST /workers/heartbeat in api/tests/contract/test_worker_heartbeat.py"
Task: "Contract test POST /workers/register in api/tests/contract/test_register_worker.py"
```

### Launch all models together (T023-T025):
```
Task: "Task model with SQLAlchemy in api/src/models/task.py"
Task: "Worker model with SQLAlchemy in api/src/models/worker.py"
Task: "TaskLog model with SQLAlchemy in api/src/models/task_log.py"
```

### Launch all libraries together (T028-T033):
```
Task: "Task queue library with polling logic in api/src/lib/task_queue/__init__.py"
Task: "Task queue CLI commands in api/src/lib/task_queue/cli.py"
Task: "Worker manager library in workers/src/lib/worker_manager/__init__.py"
Task: "Worker manager CLI in workers/src/lib/worker_manager/cli.py"
Task: "AIOpsLab client wrapper in workers/src/lib/aiopslab_client/__init__.py"
Task: "AIOpsLab client CLI in workers/src/lib/aiopslab_client/cli.py"
```

### Launch unit tests together (T064-T068):
```
Task: "Unit tests for task state machine in api/tests/unit/test_task_states.py"
Task: "Unit tests for worker state transitions in api/tests/unit/test_worker_states.py"
Task: "Unit tests for priority queue logic in api/tests/unit/test_priority_queue.py"
Task: "Unit tests for timeout calculations in api/tests/unit/test_timeout.py"
Task: "Unit tests for validation schemas in api/tests/unit/test_schemas.py"
```

## Notes
- **[P] tasks** = different files, no shared dependencies
- **TDD Enforcement**: Tests MUST fail before implementation
- **Commit frequency**: After each task completion
- **Avoid**: Tasks that modify the same file in parallel
- **Database**: Ensure PostgreSQL is running before T027
- **Workers**: Each worker needs unique cluster name (worker-XXX-kind)
- **Timeouts**: Default 30 minutes, configurable per task

## Validation Checklist
- ✅ All 9 API endpoints have contract tests (T008-T016)
- ✅ All 3 entities have models (T023-T025)
- ✅ All endpoints have implementations (T038-T046)
- ✅ Worker polling and execution covered (T047-T051)
- ✅ Integration tests for key scenarios (T017-T022)
- ✅ Libraries have CLI interfaces (T029, T031, T033)
- ✅ Performance and monitoring included (T069-T071)
- ✅ Total tasks: 80 (comprehensive coverage)
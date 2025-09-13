# Implementation Plan: AIOpsLab Task Execution API

**Branch**: `002-online-rl-data` | **Date**: 2025-09-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-online-rl-data/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Build a RESTful API for executing AIOpsLab tasks through a scalable worker-task queue pattern. The system allows users to submit problem execution requests via API, stores them in a database queue, and processes them through multiple independent workers, each with their own Orchestrator and Kind cluster.

## Technical Context
**Language/Version**: Python 3.11 (matches AIOpsLab requirement)
**Primary Dependencies**: FastAPI (REST framework), SQLAlchemy (ORM), PostgreSQL (database), Celery/APScheduler (worker polling)
**Storage**: PostgreSQL for task queue, results, and permanent data storage
**Testing**: pytest (consistent with AIOpsLab)
**Target Platform**: Linux server (Docker containers for workers)
**Project Type**: web (API server + worker processes)
**Performance Goals**: Handle 100+ concurrent tasks, 30-minute task timeout
**Constraints**: No authentication required, permanent data retention, worker isolation via Kind clusters
**Scale/Scope**: Multiple workers (5-10), thousands of tasks per day

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 2 (api, workers) ✓
- Using framework directly? Yes - FastAPI, SQLAlchemy directly ✓
- Single data model? Yes - shared between API and workers ✓
- Avoiding patterns? Yes - no unnecessary abstractions ✓

**Architecture**:
- EVERY feature as library? Yes - task queue lib, worker lib, api lib
- Libraries listed:
  - `task_queue`: Task management and database operations
  - `worker_manager`: Worker polling and task execution
  - `aiopslab_client`: Interface to existing Orchestrator
- CLI per library: Yes - each library exposes CLI commands
- Library docs: llms.txt format planned ✓

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes ✓
- Git commits show tests before implementation? Will enforce ✓
- Order: Contract→Integration→E2E→Unit strictly followed? Yes ✓
- Real dependencies used? Yes - real PostgreSQL, real Kind clusters ✓
- Integration tests for: new libraries, contract changes, shared schemas? Yes ✓
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:
- Structured logging included? Yes ✓
- Frontend logs → backend? N/A (API only)
- Error context sufficient? Yes - task failures stored with full context ✓

**Versioning**:
- Version number assigned? 1.0.0
- BUILD increments on every change? Yes ✓
- Breaking changes handled? Will maintain backward compatibility ✓

## Project Structure

### Documentation (this feature)
```
specs/002-online-rl-data/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── openapi.yaml     # API specification
│   └── schemas.json     # Request/response schemas
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 2: Web application (API + Workers)
api/
├── src/
│   ├── models/          # SQLAlchemy models
│   ├── services/        # Business logic
│   ├── api/             # FastAPI endpoints
│   └── lib/
│       └── task_queue/  # Task queue library
└── tests/
    ├── contract/
    ├── integration/
    └── unit/

workers/
├── src/
│   ├── worker.py        # Main worker process
│   ├── lib/
│   │   ├── worker_manager/  # Worker management library
│   │   └── aiopslab_client/ # Orchestrator interface
│   └── cli/
└── tests/
    ├── integration/
    └── unit/
```

**Structure Decision**: Option 2 (Web application) - API server with separate worker processes

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - Best practices for task queue implementation in Python
   - Worker polling patterns vs message queues
   - Database schema for task lifecycle management
   - Integration patterns with existing AIOpsLab Orchestrator
   - Kind cluster management for multiple workers

2. **Generate and dispatch research agents**:
   ```
   Task: "Research task queue patterns for Python REST APIs"
   Task: "Find best practices for worker polling with PostgreSQL"
   Task: "Research database schema patterns for job queues"
   Task: "Investigate AIOpsLab Orchestrator integration approaches"
   Task: "Research Kind cluster isolation strategies"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Task: id, problem_id, status, created_at, updated_at, worker_id, parameters
   - Worker: id, cluster_name, status, last_heartbeat, backend_type
   - TaskResult: id, task_id, logs, metrics, error_details, completed_at
   - TaskRequest: problem_id, agent_config, max_steps, timeout_minutes, callback_url, metadata, priority

2. **Generate API contracts** from functional requirements:
   - POST /tasks - Create new task
   - GET /tasks/{id} - Get task status and results
   - GET /tasks - List tasks with filters
   - GET /workers - List registered workers
   - POST /workers/heartbeat - Worker heartbeat endpoint
   - Output OpenAPI specification to `/contracts/openapi.yaml`

3. **Generate contract tests** from contracts:
   - test_create_task.py - Schema validation for task creation
   - test_get_task.py - Response format validation
   - test_list_tasks.py - Pagination and filtering tests
   - test_worker_endpoints.py - Worker management tests

4. **Extract test scenarios** from user stories:
   - Task submission and queueing scenario
   - Worker polling and execution scenario
   - Task status tracking scenario
   - Failure handling scenario
   - Timeout handling scenario

5. **Update agent file incrementally**:
   - Add FastAPI, SQLAlchemy, PostgreSQL specifics
   - Include task queue patterns
   - Document worker architecture

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md updates

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each API endpoint → contract test task [P]
- Each entity → model creation task [P]
- Each worker component → implementation task
- Integration test tasks for complete workflows

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Models → Services → API → Workers
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 30-35 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No violations - architecture follows constitutional principles*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
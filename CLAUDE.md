# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIOpsLab is a holistic framework for designing, developing, and evaluating autonomous AIOps agents. It provides a benchmarking suite with standardized problems to test AIOps agents in interactive cloud environments. The framework can deploy microservice applications, inject faults, generate workloads, and export telemetry data.

### Current Development: Task Execution API
Building a RESTful API for scalable AIOpsLab task execution using:
- **API Framework**: FastAPI with Pydantic validation
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Task Queue**: Database-backed queue with worker polling (SELECT FOR UPDATE SKIP LOCKED)
- **Workers**: Independent processes with isolated Kind clusters
- **Architecture**: API server + multiple worker processes, each with own Orchestrator

## Common Development Commands

### Environment Setup
```bash
# Install dependencies with Poetry (Python 3.11+ required)
poetry install
poetry shell

# Activate virtual environment
poetry shell

# Install shell plugin for better CLI experience
poetry self add poetry-plugin-shell
```

### Cluster Management
```bash
# Create local Kind cluster (x86)
kind create cluster --config kind/kind-config-x86.yaml

# Create local Kind cluster (ARM)
kind create cluster --config kind/kind-config-arm.yaml

# Monitor cluster status
k9s  # if installed
```

### Running AIOpsLab
```bash
# Run interactive CLI
python3 cli.py

# Run GPT-4 baseline agent
python3 clients/gpt.py

# Run vLLM agent
./clients/launch_vllm.sh

# Start AIOpsLab service (for remote execution)
SERVICE_HOST=<HOST> SERVICE_PORT=<PORT> SERVICE_WORKERS=<WORKERS> python service.py
```

### Code Quality
```bash
# Format code with Black
black aiopslab/

# Type checking with Pyright
pyright aiopslab/
```

### Testing
```bash
# Run tests (based on test structure found)
python -m pytest tests/
```

## High-Level Architecture

### Core Components

1. **Orchestrator** (`aiopslab/orchestrator/`)
   - Central orchestration engine that manages the interaction between agents and environment
   - Handles problem initialization, agent registration, and session management
   - Parses agent responses and evaluates solutions
   - Key class: `Orchestrator` in `orchestrator.py`

2. **Session Management** (`aiopslab/session.py`)
   - Manages individual problem-solving sessions
   - Tracks agent interactions, execution traces, and results
   - Persists session data to `data/results/`

3. **Problem Registry** (`aiopslab/orchestrator/problems/registry.py`)
   - Central registry of all available problems
   - Each problem combines an application, task type, fault, workload, and evaluator
   - Problems are organized by category (e.g., misconfig, stress, delay)

4. **Applications** (`aiopslab/service/apps/`)
   - Microservice applications that can be deployed to Kubernetes
   - Base `Application` class provides deployment/deletion interface
   - Applications use Helm charts for deployment (metadata in `aiopslab/service/metadata/`)

5. **Tasks** (`aiopslab/orchestrator/tasks/`)
   - Four AIOps task types: Detection, Localization, Analysis, Mitigation
   - Each task defines specific objectives and evaluation criteria
   - Tasks provide instructions and available actions to agents

6. **Fault Injection** (`aiopslab/generators/fault/`)
   - Fault generators organized by injection level (app, virtual, hardware)
   - Support for misconfigurations, resource stress, network delays, etc.
   - Critical section handling to ensure fault recovery on exit

7. **Observability** (`aiopslab/observer/`)
   - Prometheus for metrics collection
   - Filebeat/Logstash for log aggregation
   - APIs for accessing metrics, logs, and traces

8. **Agent Interface**
   - Agents must implement `async def get_action(self, state: str) -> str`
   - Agents interact through standardized APIs for telemetry and actions
   - Support for multiple agent frameworks (OpenAI, Autogen, vLLM, etc.)

### Data Flow

1. **Problem Initialization**:
   - Orchestrator deploys application via Helm
   - Fault injection applied to application
   - Workload generation started

2. **Agent Interaction Loop**:
   - Agent receives state information
   - Agent performs actions (query metrics, execute commands, etc.)
   - Orchestrator parses and validates agent actions
   - Session records all interactions

3. **Evaluation**:
   - Agent submits solution
   - Problem-specific evaluator checks correctness
   - Results saved with metrics (duration, steps, success rate)

### Key Directories

- `aiopslab/orchestrator/problems/`: Problem definitions
- `aiopslab/service/apps/`: Application interfaces
- `aiopslab/generators/`: Fault and workload generators
- `clients/`: Pre-built agent implementations
- `data/results/`: Session results and traces
- `specs/`: Feature specifications (when using /specify workflow)

## Working with the Specify Workflow

The repository includes a `/specify` command workflow for feature development:
- Creates feature branches following pattern: `###-feature-name`
- Specifications stored in `specs/###-feature-name/spec.md`
- Templates in `.specify/templates/`
- Scripts in `.specify/scripts/`

## API Keys and Configuration

- Store API keys in `.env` file (gitignored)
- Configure cluster connection in `aiopslab/config.yml`
- Key environment variables:
  - `OPENAI_API_KEY`: For GPT-based agents
  - `USE_WANDB`: Enable Weights & Biases logging
  - `no_proxy=localhost`: When using proxy with local vLLM

## Adding New Components

### New Problem
1. Create problem class in `aiopslab/orchestrator/problems/`
2. Implement `start_workload()`, `inject_fault()`, `eval()` methods
3. Register in `aiopslab/orchestrator/problems/registry.py`

### New Application
1. Add metadata JSON in `aiopslab/service/metadata/`
2. Create app class extending `Application` in `aiopslab/service/apps/`
3. Include Helm chart configuration if using Helm deployment

### New Agent
1. Implement `async def get_action(self, state: str) -> str`
2. Register with orchestrator using `orch.register_agent(agent)`
3. See `clients/` directory for examples

## Important Notes

- Python 3.11+ is required
- Kubernetes cluster (Kind or remote) needed for running problems
- Problems automatically deploy/cleanup applications
- Fault recovery handled via atexit handlers
- Session data persisted for analysis and debugging
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIOpsLab is a holistic framework for designing, developing, and evaluating autonomous AIOps agents. It deploys microservice cloud environments, injects faults, generates workloads, and provides interfaces for agent interaction and evaluation.

## Development Environment Setup

### Prerequisites
- Python >= 3.11
- Poetry for dependency management
- Helm for Kubernetes deployments
- kubectl configured for cluster access

### Installation
```bash
poetry env use python3.11
poetry install
poetry shell
```

### Configuration
Copy and configure `config.yml`:
```bash
cp aiopslab/config.yml.example aiopslab/config.yml
```

## Common Development Commands

### Running Tests
```bash
python -m unittest discover tests/
```

### Running AIOpsLab Locally

Interactive CLI mode:
```bash
python cli.py
```

Run specific agent:
```bash
python clients/gpt.py
```

Launch vLLM server:
```bash
./clients/launch_vllm.sh
```

### Remote Service Mode
```bash
SERVICE_HOST=<HOST> SERVICE_PORT=<PORT> SERVICE_WORKERS=<WORKERS> python service.py
```

### Kubernetes Cluster Setup

Local cluster with Kind:
```bash
# x86 machines
kind create cluster --config kind/kind-config-x86.yaml

# ARM machines  
kind create cluster --config kind/kind-config-arm.yaml
```

## Architecture and Key Components

### Core Modules

**Orchestrator** (`aiopslab/orchestrator/`)
- Main orchestration engine managing agent-environment interaction
- Problem registry with 100+ pre-defined problems
- Task types: Detection, Localization, Analysis, Mitigation
- Response parser for agent actions

**Service Layer** (`aiopslab/service/`)
- Application deployments via Helm charts
- kubectl and shell interfaces for cluster interaction
- Telemetry collection (logs, metrics, traces)

**Generators** (`aiopslab/generators/`)
- Fault injection at multiple levels (app, virtual, hardware)
- Workload generation using wrk2

**Observer** (`aiopslab/observer/`)
- Prometheus for metrics
- Filebeat/Logstash for logs
- APIs for telemetry data access

**Clients** (`clients/`)
- Agent implementations (GPT, vLLM, Qwen, etc.)
- LLM utilities and templates
- Agent registry for dynamic loading

### Problem Structure

Problems combine 5 components:
1. Application (e.g., social-network, hotel-reservation)
2. Task (Detection/Localization/Analysis/Mitigation)
3. Fault injection
4. Workload generation
5. Evaluation metrics

Problems are registered in `aiopslab/orchestrator/problems/registry.py`

### Session Management

Each agent-problem interaction creates a Session tracking:
- Agent actions and environment responses
- Telemetry data
- Evaluation results
- Session IDs for tracking

Results stored in `data/results/`

## Key Development Patterns

### Adding New Problems

1. Create problem class inheriting from task type
2. Implement `start_workload()`, `inject_fault()`, `eval()`
3. Register in `aiopslab/orchestrator/problems/registry.py`

### Adding New Applications

1. Add metadata JSON in `aiopslab/service/metadata/`
2. Create app class extending `Application` base
3. Include Helm chart configuration if applicable

### Agent Development

Agents must implement:
```python
async def get_action(self, state: str) -> str:
    # Agent logic here
```

Register with orchestrator:
```python
orch = Orchestrator()
orch.register_agent(agent)
```

## API Keys and Environment Variables

Create `.env` file:
```
OPENAI_API_KEY=<key>
QWEN_API_KEY=<key>
DEEPSEEK_API_KEY=<key>
USE_WANDB=true  # Optional
```

## Testing Approach

- Unit tests using Python unittest framework
- Test files in `tests/` directory following `test_*.py` pattern
- Tests cover parser, shell execution, registry, and judge components
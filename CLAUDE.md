# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AIOpsLab is a holistic framework for evaluating autonomous AIOps agents in interactive cloud environments. It orchestrates microservice deployments, fault injection, workload generation, and telemetry collection to benchmark AI agents on detection, localization, analysis, and mitigation tasks.

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry (recommended, requires Python >= 3.11)
poetry env use python3.11
poetry install
eval $(poetry env activate)

# Alternative: pip install
pip install -e .
```

### Running Agents Locally
```bash
# Interactive CLI with human agent
python3 cli.py
# In the REPL:
(aiopslab) $ start <problem_id>  # e.g., misconfig_app_hotel_res-detection-1
(aiopslab) $ submit("Yes")

# Run baseline agents (GPT, Qwen, DeepSeek, etc.)
python3 clients/gpt.py
python3 clients/qwen.py
python3 clients/deepseek.py
python3 clients/vllm.py

# Configure API keys in .env file
cp .env.example .env
# Edit .env to add OPENAI_API_KEY, QWEN_API_KEY, etc.
```

### Running AIOpsLab as a Service
```bash
# Start FastAPI service on remote machine
SERVICE_HOST=0.0.0.0 SERVICE_PORT=1818 SERVICE_WORKERS=1 python service.py

# Test endpoints
curl http://<host>:<port>/health
curl http://<host>:<port>/problems
curl http://<host>:<port>/agents

# Run simulation via API
curl -X POST http://<host>:<port>/simulate \
  -H "Content-Type: application/json" \
  -d '{"problem_id": "misconfig_app_hotel_res-mitigation-1", "agent_name": "vllm", "max_steps": 10}'
```

### Testing
```bash
# Run unit tests
python -m unittest discover tests/

# Run specific test files
python -m unittest tests.parser.test_parser
python -m unittest tests.registry.test_get_actions
```

### Cluster Setup

#### Option 1: Local simulated cluster (kind)
```bash
kind create cluster --config kind/kind-config-x86.yaml   # For x86
kind create cluster --config kind/kind-config-arm.yaml   # For ARM

# Configure AIOpsLab
cd aiopslab
cp config.yml.example config.yml
# Edit config.yml: set k8s_host to localhost or kind
```

#### Option 2: Azure VMs with Terraform + Ansible
See section: [Azure Deployment with Terraform + Ansible](#azure-deployment-with-terraform--ansible)

### Cluster Management
```bash
# Use k9s for monitoring (recommended)
k9s

# Or use kubectl directly
kubectl get pods -n <namespace>
kubectl logs <pod-name> -n <namespace>
```

## Core Architecture

### Orchestrator Architecture

**Central Coordination Flow:**
```
Orchestrator
  ├── init_problem() → deploys app, injects fault, starts workload
  ├── start_problem() → agent-environment interaction loop
  │     ├── ask_agent() → get next action from agent
  │     ├── ask_env() → execute action via session.problem
  │     └── returns observation to agent
  └── Evaluates results and saves session
```

**Key Components:**
- **Orchestrator** (`aiopslab/orchestrator/orchestrator.py`): Central coordinator managing lifecycle from initialization to evaluation
- **Session** (`aiopslab/session.py`): Tracks agent-environment interaction history, timing, results; supports JSON export and W&B logging
- **ResponseParser** (`aiopslab/orchestrator/parser.py`): Extracts API calls from agent responses using AST parsing for complex arguments

### Problem Structure

Problems inherit from Task subclasses (Detection, Localization, Analysis, Mitigation) and define:

- `inject_fault()`: Injects problem using fault injector
- `recover_fault()`: Cleans up injected fault
- `start_workload()`: Initiates load generation (can be sync or async)
- `eval(soln, trace, duration)`: Evaluates agent solution
- `get_task_description()`: Human/LLM-readable description
- `get_instructions()`: API usage instructions
- `get_available_actions()`: Dict of available APIs with docstrings

**Problem ID Format:** `<problem_type>-<task_type>-<variant>`
- Example: `pod_failure_hotel_res-detection-1`
- Example: `k8s_target_port-misconfig-mitigation-2`

**Problem Registry** (`aiopslab/orchestrator/problems/registry.py`):
- Centralized mapping of problem IDs to instances
- 60+ problems across multiple categories
- Access via `orch.probs.get_problem_ids()` or list at `/problems` endpoint

### Agent Integration

**Required Agent Interface:**
```python
class YourAgent:
    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        """Initialize agent with problem context"""
        pass

    async def get_action(self, observation: str) -> str:
        """Return next action as markdown code block"""
        return "Action:\n```\napi_name(args)\n```"
```

**Agent Lifecycle:**
1. Create agent instance with model parameters
2. Register with orchestrator: `orch.register_agent(agent, name="agent-name")`
3. Initialize problem: `problem_desc, instructs, apis = orch.init_problem(problem_id)`
4. Set agent context: `agent.init_context(problem_desc, instructs, apis)`
5. Start problem: `await orch.start_problem(max_steps=30)`

**Agent Registry** (`clients/registry.py`):
- Maps agent names to implementations
- Supported: GPT, Qwen, DeepSeek, vLLM, OpenRouter, Groq
- Access via `AgentRegistry().get_agent_ids()`

### Action System

**Action Categories by Task:**
- **Detection**: `get_logs`, `get_metrics`, `get_traces`, `exec_shell`, `submit`
- **Localization**: Same as detection + targeted analysis
- **Mitigation**: All above + scaling, config patching, restarts
- **Analysis**: Comprehensive debugging capabilities

**Action Decorators:**
```python
@action  # Standard action
@read    # Read-only (no state change)
@write   # State-modifying action
```

Actions are dynamically discovered via these decorators and exposed to agents with auto-generated documentation from docstrings.

**Action Execution:**
- Agent response parsed by `ResponseParser`
- Routed via `Task.perform_action(api_name, *args, **kwargs)`
- Returns string observation or error
- Invalid actions raise `InvalidActionError`

### Service Layer

**Application Base** (`aiopslab/service/apps/base.py`):
- Abstract interface for all applications
- Loads metadata from JSON (`aiopslab/service/metadata/`)
- Key attributes: `namespace`, `helm_configs`, `k8s_deploy_path`, `docker_deploy_path`
- Methods: `load_app_json()`, `get_app_summary()`, `create_namespace()`, `cleanup()`

**Available Applications:**
- HotelReservation: Microservice hotel booking
- SocialNetwork: Social media app
- AstronomyShop: OpenTelemetry e-commerce demo
- FlightTicket, TrainTicket, TiDBCluster, Flower (federated learning)

**Helm Integration** (`aiopslab/service/helm.py`):
- Static methods for chart install/uninstall
- Automatic dependency resolution
- Namespace creation and extra args support

**Kubectl Wrapper** (`aiopslab/service/kubectl.py`):
- Abstracts K8s API calls
- Pod management, namespace ops, config maps, logs
- Container runtime detection (Docker/Containerd)

### Fault Injection

**Fault Injector Hierarchy:**
```
FaultInjector (base)
  ├── SymptomFaultInjector (Chaos Mesh-based) ← Most common
  ├── ApplicationFaultInjector (App-level)
  ├── OSFaultInjector (OS-level)
  ├── HardwareFaultInjector
  ├── OperatorFaultInjector (K8s operator)
  ├── OTelFaultInjector (OpenTelemetry)
  ├── VirtualFaultInjector (Mock faults)
  └── NoOpFaultInjector (Testing)
```

**SymptomFaultInjector** (`aiopslab/generators/fault/inject_symp.py`):
- Uses Chaos Mesh v2.6.2 for fault injection
- Installed via Helm in `chaos-mesh` namespace
- Supported fault types:
  - `inject_pod_failure()`: Kill pods
  - `inject_network_delay()`: Add latency
  - `inject_network_loss()`: Drop packets
  - Container kills, resource exhaustion, etc.
- Pattern: `inject_<fault_type>()` and `recover_<fault_type>()`
- Creates temporary YAML for each experiment

**Workload Generation** (`aiopslab/generators/workload/wrk.py`):
- Wraps wrk2 load generator
- Configurable: rate, connections, threads, duration, distribution (normal/exponential/uniform)
- Supports Lua script payloads via ConfigMap
- Launches as K8s Job for distributed load

### Observability

**Telemetry Collection** (`aiopslab/observer/observe.py`):
- Multi-threaded collection of traces, logs, metrics
- Time-window based (configurable start/end)
- Saves to: `telemetry_data_YYYYMMDD_HHMMSS/`

**Metric Collection** (`aiopslab/observer/metric_api.py`):
- `PrometheusAPI` queries Prometheus
- Collects: CPU, memory, network I/O per pod/container
- Container limits, Istio metrics (latency, throughput)
- Exports as CSV/DataFrame

**Log Collection** (`aiopslab/observer/log_api.py`):
- `LogAPI` connects to Elasticsearch
- Filters by namespace, pod, timestamp range
- Exports as CSV

**Trace Collection** (`aiopslab/observer/trace_api.py`):
- `TraceAPI` queries Jaeger for distributed traces
- Service-level trace analysis

### Evaluation

**Evaluation Flow:**
1. Agent submits solution via `submit(answer)`
2. Problem's `eval(soln, trace, duration)` is called
3. Task base class provides default metrics:
   - **Detection**: Accuracy (Yes/No), TTD (Time to Detection)
   - **Localization**: Accuracy (service name match), TTL (Time to Localization)
   - **Analysis**: Root cause accuracy, TTA (Time to Analysis)
   - **Mitigation**: Success rate, TTM (Time to Mitigation)
4. Custom metrics added via `self.add_result(metric_name, value)`
5. Results saved to `data/results/` as JSON
6. Optional W&B logging (set `USE_WANDB=true` in .env)

**LLM-as-Judge** (optional):
- Set `qualitative_eval: true` in `config.yml`
- Evaluates reasoning quality using LLM
- Prompts in `aiopslab/orchestrator/evaluators/prompts.py`

## Code Patterns and Conventions

### Async/Await Support
- `start_workload()` can be sync or async
- Orchestrator detects via `inspect.iscoroutinefunction()`
- All agent `get_action()` methods are async

### Template Method Pattern
Task base class defines contract, subclasses override:
- `get_task_description()`
- `get_instructions()`
- `perform_action()`
- `eval()`

### Factory Pattern
- `ProblemRegistry` creates problems on demand via lambdas
- `AgentRegistry` instantiates agents by name
- Enables parameterized problem variants

### Context Manager Pattern
- `CriticalSection` for thread-safe fault recovery
- `atexit` handlers ensure cleanup on unexpected exit
- Fault recovery registered during `inject_fault()`

### Configuration Management
**`aiopslab/config.yml`** (copy from `config.yml.example`):
- `k8s_host`: Control plane hostname (localhost/kind/<hostname>)
- `k8s_user`: Username on control plane
- `ssh_key_path`: Path to SSH key
- `data_dir`: Where telemetry/results are stored
- `qualitative_eval`: Enable LLM-as-Judge evaluation
- `print_session`: Print session trace after completion

**Environment Variables** (`.env`):
- API keys: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `GROQ_API_KEY`
- W&B: `USE_WANDB=true`

## Adding New Components

### Adding a New Problem

1. **Define Problem Class** (in `aiopslab/orchestrator/problems/<your_problem>/`):
```python
from aiopslab.orchestrator.tasks.localization import LocalizationTask
from aiopslab.service.apps.myapp import MyApp

class MyProblem(LocalizationTask):
    def __init__(self):
        self.app = MyApp()

    def start_workload(self):
        # Workload generation logic
        pass

    def inject_fault(self):
        # Fault injection logic
        pass

    def eval(self, soln, trace, duration):
        super().eval(soln, trace, duration)  # Default metrics
        # Add custom metrics
        self.add_result("custom_metric", value)
        return self.results
```

2. **Register Problem** (in `aiopslab/orchestrator/problems/registry.py`):
```python
from aiopslab.orchestrator.problems.my_problem import MyProblem

class ProblemRegistry:
    def __init__(self):
        self.PROBLEM_REGISTRY = {
            # ... existing problems ...
            "my_problem-localization-1": MyProblem,
        }
```

### Adding a New Application

1. **Create Metadata JSON** (`aiopslab/service/metadata/myapp.json`):
```json
{
  "name": "MyApp",
  "description": "Description of the app",
  "namespace": "test-myapp",
  "Helm Config": {
    "release_name": "myapp-release",
    "chart_path": "path/to/helm/chart",
    "namespace": "test-myapp"
  }
}
```

2. **Create Application Class** (`aiopslab/service/apps/myapp.py`):
```python
from aiopslab.service.apps.base import Application

class MyApp(Application):
    def __init__(self):
        super().__init__("path/to/metadata/myapp.json")
```

### Adding a New Agent

1. **Implement Agent** (`clients/myagent.py`):
```python
class MyAgent:
    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        self.problem_desc = problem_desc
        self.instructions = instructions
        self.apis = apis

    async def get_action(self, observation: str) -> str:
        # Your agent logic
        return f"Action:\n```\n{api_call}\n```"
```

2. **Register Agent** (`clients/registry.py`):
```python
from clients.myagent import MyAgent

class AgentRegistry:
    def __init__(self):
        self.AGENT_REGISTRY = {
            # ... existing agents ...
            "myagent": MyAgent,
        }
```

## Important Implementation Notes

### Shell Command Restrictions
- No interactive commands: `kubectl edit`, `docker logs -f`
- Use specific APIs instead: `get_logs()`, `get_metrics()`, `get_traces()`

### Problem Execution Lifecycle
1. **Initialization**: Deploy app, create storage (OpenEBS for K8s)
2. **Fault Injection**: Deploy Chaos Mesh experiment, register recovery
3. **Workload Start**: Launch wrk2 job at specified rate
4. **Agent Loop**: Up to max_steps iterations of agent-environment interaction
5. **Evaluation**: Measure correctness and efficiency
6. **Cleanup**: Recover fault, delete namespace, remove storage

### Fault Recovery
- Always registered with `atexit` during `inject_fault()`
- Thread-safe via `CriticalSection` context manager
- Cleanup happens even on unexpected termination

### Testing Strategy
- Unit tests in `tests/` organized by functionality
- Parser tests: API extraction, argument parsing, context extraction
- Registry tests: Action discovery via decorators
- Shell tests: Command execution validation
- Use mocks where possible to avoid K8s dependencies

## Key Files Reference

- `cli.py`: Interactive CLI for human agents
- `service.py`: FastAPI service for remote execution
- `assessment.py`: Batch evaluation script
- `aiopslab/orchestrator/orchestrator.py`: Main orchestration engine
- `aiopslab/session.py`: Session tracking and persistence
- `aiopslab/orchestrator/parser.py`: Action parsing from agent responses
- `aiopslab/orchestrator/problems/registry.py`: Problem definitions
- `clients/registry.py`: Agent implementations
- `aiopslab/service/apps/`: Application interfaces
- `aiopslab/generators/fault/`: Fault injection implementations
- `aiopslab/generators/workload/wrk.py`: Workload generation
- `aiopslab/observer/`: Telemetry collection (logs, metrics, traces)

## Common Workflows

### Evaluating an Agent on a Problem
```python
from aiopslab.orchestrator import Orchestrator
from clients.gpt import GPTAgent

# Create and register agent
agent = GPTAgent()
orch = Orchestrator()
orch.register_agent(agent, name="gpt-agent")

# Initialize and start problem
problem_desc, instructs, apis = orch.init_problem("pod_failure_hotel_res-detection-1")
agent.init_context(problem_desc, instructs, apis)
await orch.start_problem(max_steps=30)

# Results saved to data/results/<session_id>.json
```

### Batch Evaluation
```python
# Use assessment.py for batch evaluation
python assessment.py
# Configure problems and agents in the script
```

### Debugging Agent Responses
- Enable session printing: `print_session: true` in `config.yml`
- Check parsed actions in session trace
- Use `ResponseParser` directly for testing:
  ```python
  from aiopslab.orchestrator.parser import ResponseParser
  parser = ResponseParser()
  result = parser.parse(agent_response)
  ```

---

## Azure Deployment with Terraform + Ansible

### Deployment Modes

| Mode | AIOpsLab Runs On | K8s Cluster | Use Case |
|------|------------------|-------------|----------|
| **Mode A** | Controller VM (inside cluster) | Same machine | Production, full fault injection support |
| **Mode B** | Your laptop (remote kubectl) | Azure VMs | Development, debugging |

**Note:** `VirtualizationFaultInjector` requires Docker on the machine running AIOpsLab. Use official Poetry installer, not `apt install python3-poetry`.

**Tested on:** WSL2 (Ubuntu 22.04) on Windows 11 with Azure VMs (Ubuntu 22.04 LTS, amd64). The `deploy.py` auto-install targets Linux/amd64; macOS and native Windows are not currently supported.

### Quick Start (Mode B - Laptop)

```bash
# Single command: provisions VMs, runs Ansible, installs tools, configures AIOpsLab
python3 scripts/terraform/deploy.py --apply --resource-group <your-rg> --workers 2 --mode B

# After deploy completes, start AIOpsLab:
eval $(poetry env activate)
python3 cli.py
```

### Quick Start (Mode A - Controller VM)

```bash
# Clone mode: git clones the repo on the controller
python3 scripts/terraform/deploy.py --apply --resource-group <your-rg> --workers 2 --mode A

# Dev mode: rsync local code to the controller
python3 scripts/terraform/deploy.py --apply --resource-group <your-rg> --workers 2 --mode A --dev

# After deploy, SSH to controller:
ssh -i ~/.ssh/id_rsa azureuser@<controller-ip>
cd ~/AIOpsLab && eval $(poetry env activate)
python3 cli.py
```

### Other deploy.py commands
```bash
# Dry-run:
python3 scripts/terraform/deploy.py --plan --resource-group <your-rg> --workers 2

# Re-run setup without reprovisioning VMs (e.g., after code changes):
python3 scripts/terraform/deploy.py --setup-only --mode A --dev

# Restrict NSG access (SSH + K8s API) to a service tag or CIDR:
python3 scripts/terraform/deploy.py --apply --resource-group <your-rg> --allowed-ips CorpNetPublic

# Destroy infrastructure:
python3 scripts/terraform/deploy.py --destroy --resource-group <your-rg>
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/terraform/deploy.py` | Single-command deployment (Terraform + Ansible + AIOpsLab setup) |
| `scripts/terraform/main.tf` | Azure VM provisioning (controller + workers) |
| `scripts/terraform/variables.tf` | Configurable parameters (VM size, count, etc.) |
| `scripts/terraform/generate_inventory.py` | Creates Ansible inventory from Terraform output |
| `scripts/ansible/setup_common.yml` | Installs Docker, K8s packages on all nodes |
| `scripts/ansible/remote_setup_controller_worker.yml` | Initializes K8s cluster, joins workers |
| `scripts/ansible/setup_aiopslab.yml` | Mode A: installs Python 3.11, Poetry, Helm, clones/rsyncs repo, runs poetry install |
| `scripts/ansible/templates/config.yml.j2` | Mode A: Jinja2 template for aiopslab/config.yml |
| `scripts/ansible/inventory.yml` | Generated inventory (don't edit manually) |

### Important Configuration

**Ansible Inventory Variables** (`inventory.yml`):
- `k8s_user`: SSH username (e.g., `azureuser`)
- `user_home_base`: Home directory base path
  - `/home` for cloud VMs (Azure, AWS, GCP)
  - `/users` for Emulab testbed
- `private_ip`: Internal IP for K8s cluster communication
- `ansible_host`: Public IP for SSH access

**AIOpsLab config.yml** (for Mode B):
```yaml
k8s_host: <controller-public-ip>  # e.g., 20.150.145.167
k8s_user: azureuser
ssh_key_path: ~/.ssh/id_rsa
```

### Common Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `conntrack not found` | Missing package | Added to `setup_common.yml` prerequisites |
| Certificate error with kubectl | Cert missing public IP | Playbook adds `--apiserver-cert-extra-sans` |
| Kubeconfig uses private IP | Can't reach from laptop | Playbook auto-updates to public IP |
| Helm chart not found | Submodules not cloned | Run `git submodule update --init --recursive` |
| Submodule init fails in WSL | Worktree `.git` file has Windows paths | Run from Git Bash, not WSL |
| `poetry shell` not found | Removed in Poetry 2.0 | Use `eval $(poetry env activate)` instead |
| Poetry "not supported" Python | System python too old | `poetry env use python3.11 && poetry install` |
| Path `/users/` not found | Wrong home base for cloud | Set `user_home_base: /home` in inventory |

### NSG (Network Security Group) Rules

The Terraform config creates NSG rules for:
- **SSH (22)**: Open to all (`*`) by default. Restrict via `--allowed-ips` flag or `nsg_allowed_source` variable.
- **K8s API (6443)**: Open to all (`*`) by default. Restrict via `--allowed-ips` flag or `nsg_allowed_source` variable.

To allow access from other IPs, modify `main.tf` or add rules via Azure CLI.

### Destroying Infrastructure

```bash
cd scripts/terraform
terraform destroy -var="resource_group_name=<your-rg>"
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Laptop (WSL)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Terraform  │→ │   Ansible   │→ │     AIOpsLab        │ │
│  │  (Azure VMs)│  │ (K8s setup) │  │ (kubectl + cli.py)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Azure (VNet 10.0.0.0/16)                 │
│  ┌─────────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Controller    │  │  Worker 1   │  │  Worker N   │     │
│  │  (K8s control)  │  │             │  │             │     │
│  │  Public + Priv  │  │ Private IP  │  │ Private IP  │     │
│  └─────────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

# AIOpsLab Development Instructions

AIOpsLab is a Python-based AI operations lab for evaluating AI agents in DevOps scenarios. It uses Poetry for dependency management, Kubernetes (via kind) for cluster operations, and Helm for application deployments.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Repository Setup and Dependencies
Bootstrap, build, and test the repository:
- Install Poetry if not available: `pip3 install poetry`
- Export Poetry to PATH: `export PATH="$HOME/.local/bin:$PATH"`
- Configure Python environment: `poetry env use python3.12` (takes ~1 second)
- Update dependencies: `poetry lock` (takes ~45 seconds, NEVER CANCEL - dependency resolution is intensive)
- Install dependencies: `poetry install` (takes ~2-3 minutes, NEVER CANCEL. Set timeout to 300+ seconds)
- Copy config file: `cp aiopslab/config.yml.example aiopslab/config.yml`

### Testing and Validation
- Run linting: `poetry run black --check --diff . --exclude=aiopslab-applications` (takes ~3 seconds)
- Format code: `poetry run black . --exclude=aiopslab-applications` 
- Type checking: `poetry run pyright` (requires network access for initial setup)
- **CRITICAL**: Tests require external network access and will fail in restricted environments
- **CRITICAL**: CLI requires network access to download OpenAI tiktoken encodings on first run

### Kubernetes Cluster Setup
For local development and testing:
- **NEVER CANCEL**: Cluster creation takes 2-5 minutes depending on image download
- Create kind cluster: `kind create cluster --config kind/kind-config-x86.yaml` (for x86 systems)
- For ARM systems: `kind create cluster --config kind/kind-config-arm.yaml`
- Configure aiopslab for kind: Edit `aiopslab/config.yml` and set `k8s_host: kind` and `k8s_user: kind`
- Cleanup: `kind delete cluster` (takes ~10 seconds)

### Running AIOpsLab
- **CRITICAL**: All CLI operations require network access and a configured Kubernetes cluster
- Run CLI: `poetry run python cli.py`
- Example usage: Start a problem with `start misconfig_app_hotel_res-detection-1` in the CLI
- The CLI will fail without proper network access due to tiktoken dependency

## Validation

### Manual Testing Requirements
**ALWAYS manually validate any new code by running these validation steps after making changes:**

1. **Environment Setup Validation**:
   - Verify Poetry installation works: `poetry --version`
   - Verify dependencies install cleanly: `poetry install`
   - Verify basic Python imports work: `poetry run python -c "print('Dependencies OK')"`

2. **Code Quality Validation**:
   - **ALWAYS run before committing**: `poetry run black --check --diff .`
   - Fix any formatting issues: `poetry run black .`
   - Verify no new linting errors are introduced

3. **Kubernetes Functionality**:
   - If making K8s-related changes, create a test cluster: `kind create cluster --config kind/kind-config-x86.yaml`
   - Verify cluster connectivity: `kubectl cluster-info`
   - Clean up after testing: `kind delete cluster`

### Timing Expectations
- **NEVER CANCEL** these operations - they require significant time:
  - `poetry lock`: 45 seconds (dependency resolution)
  - `poetry install`: 2-3 minutes (package installation)
  - `kind create cluster`: 2-5 minutes (image download and cluster setup)
- Quick operations (< 30 seconds):
  - `poetry run black --check`: ~3 seconds
  - `kind delete cluster`: ~10 seconds
  - Basic Python imports: ~1 second

## Common Tasks

### Repository Structure
```
.
├── README.md              # Main documentation
├── pyproject.toml        # Poetry configuration and dependencies  
├── cli.py               # Command-line interface
├── aiopslab/            # Main Python package
│   ├── config.yml.example # Configuration template
│   ├── orchestrator/    # Core orchestration logic
│   ├── generators/      # Fault and workload generators
│   └── service/         # Kubernetes and service management
├── tests/               # Test suite (requires network access)
├── kind/                # Kubernetes-in-Docker configurations
└── scripts/             # Setup and deployment scripts
```

### Key Configuration Files
- `pyproject.toml`: Poetry dependencies and project metadata
- `aiopslab/config.yml`: Runtime configuration (copy from .example)
- `kind/kind-config-x86.yaml`: Kind cluster configuration for x86
- `kind/kind-config-arm.yaml`: Kind cluster configuration for ARM

### Important Dependencies
- **Python >= 3.11** (uses 3.12 in development)
- **Poetry**: Dependency management
- **Helm**: Kubernetes package manager
- **kubectl**: Kubernetes CLI
- **kind**: Kubernetes-in-Docker for local clusters
- **Docker**: Required for kind clusters

### Network Dependencies
**WARNING**: The following operations require internet access and will fail in restricted environments:
- Initial `poetry install` (downloads packages)
- Running tests (downloads OpenAI tiktoken encodings)
- Running CLI (downloads OpenAI tiktoken encodings)
- `kind create cluster` (may download container images)

### Development Workflow
1. **ALWAYS start with**: `export PATH="$HOME/.local/bin:$PATH"`
2. **For new environments**: Follow complete setup process above
3. **For code changes**: 
   - Make changes
   - Run `poetry run black .` to format
   - Test basic imports: `poetry run python -c "import aiopslab"`
   - If touching K8s code, test with kind cluster
4. **Before committing**: Run linting validation

### Troubleshooting Common Issues
- **"poetry: command not found"**: Add Poetry to PATH with `export PATH="$HOME/.local/bin:$PATH"`
- **"config.yml not found"**: Copy `aiopslab/config.yml.example` to `aiopslab/config.yml`
- **Network errors in tests/CLI**: Tests and CLI require internet access for OpenAI tiktoken downloads
- **Kind cluster issues**: Ensure Docker is running and user has Docker permissions
- **Import errors**: Verify `poetry install` completed successfully without errors

### Performance Notes
- Repository uses submodules - clone with `git clone --recurse-submodules`
- Poetry resolves complex dependencies - dependency operations take significant time
- Kind clusters require Docker image downloads on first use
- Black formatter processes ~180 files - formatting operations may take several seconds

## Project Architecture
AIOpsLab evaluates AI agents performing DevOps tasks in Kubernetes environments. The system includes:
- **Orchestrator**: Manages evaluation sessions and agent interactions
- **Problems**: Defines specific DevOps scenarios for evaluation
- **Generators**: Creates faults and workloads for testing
- **Observers**: Collects telemetry and logs
- **Service**: Provides Kubernetes and application management APIs

Changes should maintain the evaluation framework's integrity while following the established patterns for problem definitions, agent interfaces, and validation procedures.
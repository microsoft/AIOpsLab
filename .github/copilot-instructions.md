<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AIOpsLab Project Instructions

This is the AIOpsLab project - a holistic framework for designing, developing, and evaluating autonomous AIOps agents.

## Project Context
- **Framework**: AIOpsLab is designed for building reproducible, standardized, interoperable and scalable benchmarks for AIOps agents
- **Language**: Python 3.11+ with Poetry for dependency management
- **Key Features**: 
  - Deploy microservice cloud environments
  - Inject faults for testing
  - Generate workloads
  - Export telemetry data
  - Orchestrate components
  - Provide interfaces for agent interaction and evaluation

## Code Style Guidelines
- Follow Python PEP 8 standards
- Use type hints where appropriate
- Maintain consistent docstring format
- Use the existing project structure and patterns

## Key Components
- `aiopslab/`: Core framework code
- `aiopslab/generators/`: Fault injection and workload generation
- `aiopslab/observer/`: Monitoring and telemetry
- `aiopslab/orchestrator/`: Main orchestration logic
- `aiopslab/service/`: Service management utilities
- `clients/`: AI/ML client implementations
- `tests/`: Test suites

## Dependencies
- Uses Poetry for dependency management
- Requires Python >= 3.11, < 3.13
- Key dependencies include Kubernetes, OpenAI, Pydantic, Rich, Prometheus API client

## Development Notes
- This is an active research project for AIOps agent evaluation
- Focus on maintaining compatibility with the existing benchmark suite
- When adding new features, consider the impact on reproducibility and scalability

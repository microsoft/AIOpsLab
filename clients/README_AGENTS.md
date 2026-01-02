# Custom AIOps Agents for AIOpsLab

This directory contains custom agents designed for the AIOpsLab framework. These agents demonstrate different approaches to solving AIOps problems with varying levels of complexity and capabilities.

## Available Agents

### 1. Custom Agent (`custom_agent.py`)
**Advanced agent with comprehensive problem-solving capabilities**

**Features:**
- Systematic problem analysis framework
- Multi-step reasoning with hypothesis validation
- Telemetry data interpretation
- Iterative solution refinement
- Detailed analysis tracking and reporting

**Best for:**
- Complex problems requiring deep analysis
- Scenarios where you need detailed reasoning traces
- Problems with multiple potential root causes
- When you want comprehensive documentation of the solution process

**Usage:**
```python
from clients.custom_agent import CustomAgent

agent = CustomAgent()
orchestrator.register_agent(agent, name="custom-aiops-agent")
```

### 2. Simple Agent (`simple_agent.py`)
**Streamlined agent focused on efficiency**

**Features:**
- Clear, step-by-step problem analysis
- Systematic troubleshooting approach
- Effective solution implementation
- Minimal overhead with focused reasoning

**Best for:**
- Straightforward problems with clear symptoms
- When you need quick, effective solutions
- Learning and understanding the AIOpsLab framework
- Production environments where efficiency is key

**Usage:**
```python
from clients.simple_agent import SimpleAgent

agent = SimpleAgent()
orchestrator.register_agent(agent, name="simple-aiops-agent")
```

## Getting Started

### Prerequisites
1. Ensure AIOpsLab is properly set up and running
2. Configure your OpenAI API key for the LLM backend
3. Have a Kubernetes cluster available (local or remote)

### Quick Start

1. **Test the Simple Agent:**
   ```bash
   cd c:\workspace
   python clients\test_agents.py --agent simple
   ```

2. **Test the Custom Agent:**
   ```bash
   python clients\test_agents.py --agent custom --steps 15
   ```

3. **List Available Problems:**
   ```bash
   python clients\test_agents.py --list-problems
   ```

4. **Test with a specific problem:**
   ```bash
   python clients\test_agents.py --agent custom --problem misconfig_app_hotel_res-mitigation-1 --steps 12
   ```

### Configuration

Edit `clients/configs/agent_config.yml` to customize agent behavior:

```yaml
agent_config:
  custom_agent:
    max_steps: 15
    analysis_depth: "comprehensive"
  simple_agent:
    max_steps: 10
    analysis_depth: "focused"
```

## Agent Architecture

Both agents follow the standard AIOpsLab agent interface:

```python
class Agent:
    def init_context(self, problem_desc: str, instructions: str, apis: str):
        """Initialize agent with problem context and available APIs"""
        pass
    
    async def get_action(self, input_data: str) -> str:
        """Process input and return the next action"""
        pass
```

### Key Components

1. **Problem Context Initialization**
   - Parse problem description and instructions
   - Categorize available APIs (telemetry, shell, submit)
   - Set up the reasoning framework

2. **Action Generation**
   - Process environmental input
   - Apply reasoning methodology
   - Generate structured responses

3. **Response Formatting**
   - Follow consistent output patterns
   - Provide clear reasoning traces
   - Include actionable next steps

## Customization

### Creating Your Own Agent

1. **Inherit from base patterns:**
   ```python
   class MyAgent:
       def __init__(self):
           self.history = []
           self.llm = GPT4Turbo()
       
       def init_context(self, problem_desc, instructions, apis):
           # Your initialization logic
           pass
       
       async def get_action(self, input_data):
           # Your action generation logic
           pass
   ```

2. **Add custom reasoning:**
   - Implement domain-specific analysis
   - Add specialized API handling
   - Create custom response formats

3. **Register with orchestrator:**
   ```python
   orchestrator.register_agent(agent, name="my-custom-agent")
   ```

### Extending Existing Agents

You can extend the provided agents by:

1. **Overriding methods:**
   ```python
   class EnhancedCustomAgent(CustomAgent):
       def _process_input(self, input_data):
           # Add custom input processing
           return super()._process_input(input_data)
   ```

2. **Adding new capabilities:**
   ```python
   class SpecializedAgent(SimpleAgent):
       def __init__(self):
           super().__init__()
           self.specialized_tools = []
       
       def add_specialized_analysis(self, data):
           # Your specialized logic
           pass
   ```

## API Reference

### Available APIs in AIOpsLab

**Telemetry APIs:**
- Prometheus metrics queries
- Log analysis
- Trace data access

**Shell API:**
- Execute commands in the target environment
- File system operations
- Service management

**Submit API:**
- Submit solution for evaluation
- Provide final analysis report

### Response Format

Agents should structure their responses as:

```
Thought: <reasoning about the current situation>
Action: <specific action to take>
```

For the custom agent, use the enhanced format:

```
Analysis: <systematic analysis of the problem>
Hypothesis: <current hypothesis about root cause>
Action: <specific action to take>
Rationale: <why this action addresses the hypothesis>
Expected_Outcome: <what you expect to happen>
```

## Testing and Validation

### Running Tests

1. **Unit Tests:**
   ```bash
   python -m pytest tests/ -v
   ```

2. **Integration Tests:**
   ```bash
   python clients\test_agents.py --agent simple --problem test_problem
   ```

3. **Performance Tests:**
   ```bash
   python clients\test_agents.py --agent custom --steps 20
   ```

### Debugging

Enable detailed logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor agent behavior:
```python
# For custom agent
summary = agent.get_analysis_summary()
print(f"Steps taken: {summary['total_steps']}")
```

## Best Practices

1. **Problem Analysis**
   - Always start with understanding the problem scope
   - Gather baseline data before making changes
   - Validate hypotheses with evidence

2. **Solution Implementation**
   - Make incremental changes
   - Test each change before proceeding
   - Document your reasoning

3. **Error Handling**
   - Gracefully handle API failures
   - Provide meaningful error messages
   - Implement retry mechanisms

4. **Performance**
   - Minimize unnecessary API calls
   - Cache frequently accessed data
   - Use appropriate timeouts

## Troubleshooting

### Common Issues

1. **Agent not responding:**
   - Check OpenAI API key configuration
   - Verify network connectivity
   - Review error logs

2. **API errors:**
   - Ensure Kubernetes cluster is accessible
   - Check API permissions
   - Verify service endpoints

3. **Performance issues:**
   - Reduce max_steps if needed
   - Optimize API usage
   - Check resource constraints

## Contributing

To contribute new agents or improvements:

1. Fork the repository
2. Create a new agent following the established patterns
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## License

This code is licensed under the MIT License. See LICENSE.txt for details.

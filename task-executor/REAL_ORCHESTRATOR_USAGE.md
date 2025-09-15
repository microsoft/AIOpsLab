# Using Real AIOpsLab Orchestrator with Task Executor

This document explains how to use the real AIOpsLab orchestrator with Kind clusters instead of mocked LLM conversations.

## Overview

The Task Executor API uses the **Real AIOpsLab Orchestrator** with Kind clusters and real LLM agents for all task execution.

## Prerequisites

For real orchestrator execution, you need:

1. **Docker** installed and running
2. **Kind** (Kubernetes in Docker) installed
3. **kubectl** configured
4. **API Keys** for LLM providers (choose one):
   - `OPENROUTER_API_KEY` for OpenRouter (supports 100+ models)
   - `OPENAI_API_KEY` for direct OpenAI
   - `ANTHROPIC_API_KEY` for direct Claude

## Configuration

### Environment Variables

#### Option 1: OpenRouter (Recommended - Access to Many Models)
```bash
export OPENROUTER_API_KEY="sk-or-v1-xxx"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"  # Optional
```

#### Option 2: Direct Provider APIs
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Task Parameters

When creating a task, use these parameters to control execution:

```json
{
  "problem_id": "k8s_target_port-misconfig-detection-1",
  "parameters": {
    "max_steps": 30,
    "timeout": 1800,
    "agent_config": {
      "model": "gpt-4",        // See model options below
      "temperature": 0.7,
      "max_tokens": 4000,
      "use_openrouter": true   // Optional: force OpenRouter even if other keys exist
    }
  }
}
```

#### Supported Models with OpenRouter

When using OpenRouter, you can access 100+ models:

- **OpenAI**: `openai/gpt-4`, `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`, `anthropic/claude-3-haiku`
- **Meta**: `meta-llama/llama-3-70b-instruct`, `meta-llama/llama-3-8b-instruct`
- **Mistral**: `mistralai/mixtral-8x7b-instruct`, `mistralai/mistral-7b-instruct`
- **Google**: `google/gemini-pro`, `google/palm-2-chat-bison`
- **And many more...**

You can also use shorthand names like `"gpt-4"` which will be mapped automatically.

## Architecture

### Real Orchestrator Flow

```
1. Task Created → Worker Claims Task
2. Worker Creates Kind Cluster (aiopslab-worker-xxx)
3. Worker Initializes Orchestrator with Cluster
4. Worker Creates LLM Agent (GPT/Claude)
5. Worker Wraps Agent with LLMLoggingAgent
6. Orchestrator Deploys Application to Cluster
7. Orchestrator Injects Fault
8. Agent Interacts with Environment
9. All LLM Conversations Logged to Database
10. Results Returned with Real Solution
```

### Components

- **OrchestratorExecutor** (`orchestrator_executor.py`)
  - Manages Kind clusters
  - Initializes AIOpsLab orchestrator
  - Creates real LLM agents
  - Captures conversation history

- **LLMLoggingAgent** (`llm_logging_agent.py`)
  - Wraps real agents (GPT, Claude)
  - Intercepts all agent interactions
  - Logs conversations to database
  - Preserves agent functionality

## API Examples

### Create Task with Real Orchestrator

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "k8s_target_port-misconfig-detection-1",
    "parameters": {
      "use_orchestrator": true,
      "agent_config": {
        "model": "gpt-4",
        "temperature": 0.7
      }
    }
  }'
```

### Retrieve Real LLM Conversation

```bash
# Get task status
curl http://localhost:8000/api/v1/tasks/{task_id}

# Get conversation details
curl http://localhost:8000/api/v1/llm-conversations/{conversation_id}

# Get actual messages
curl http://localhost:8000/api/v1/llm-conversations/{conversation_id}/messages
```

## Execution Details

| Feature | Real Orchestrator |
|---------|-----------------|
| Execution Time | 30-300 seconds |
| Kubernetes Cluster | Real Kind cluster |
| Application Deployment | Real Helm deployment |
| Fault Injection | Real fault injection |
| LLM API Calls | Real OpenAI/Anthropic API via OpenRouter |
| Cost | Real API costs |
| Conversation Data | Real agent interactions |
| Problem Solving | May fail based on agent performance |

## Supported Problems

All problems from AIOpsLab are supported:

- **Misconfiguration Detection**: `k8s_target_port-misconfig-detection-*`
- **Misconfiguration Mitigation**: `*-misconfig-mitigation-*`
- **Stress Analysis**: `*-stress-analysis-*`
- **Network Delay**: `*-delay-*`
- **And more...**

## Database Schema

Real conversations are stored with:

```sql
llm_conversations:
  - id: UUID
  - task_id: References task
  - model_name: "gpt-4", "claude-3", etc.
  - messages: JSONB array of all interactions
  - total_tokens: Actual token usage
  - total_cost: Estimated cost
  - conversation_metadata: {
      "cluster_name": "aiopslab-worker-001",
      "problem_id": "...",
      "success": true/false
    }
```

## Monitoring

### View Worker Clusters

```bash
kind get clusters
# Output: aiopslab-worker-001-internal, etc.
```

### View Deployed Applications

```bash
kubectl get pods --all-namespaces
kubectl get services --all-namespaces
```

### Clean Up Clusters

```bash
# Delete specific cluster
kind delete cluster --name aiopslab-worker-001-internal

# Delete all worker clusters
kind get clusters | grep aiopslab-worker | xargs -I {} kind delete cluster --name {}
```

## Cost Considerations

Using real orchestrator incurs:

1. **LLM API Costs**:
   - GPT-4: ~$0.03-0.06 per 1K tokens
   - GPT-3.5: ~$0.002 per 1K tokens
   - Claude: ~$0.01-0.02 per 1K tokens

2. **Typical Problem Costs**:
   - Simple detection: $0.10-0.50
   - Complex mitigation: $0.50-2.00

3. **Resource Usage**:
   - Each Kind cluster: ~500MB RAM
   - Deployed apps: ~100-500MB per app

## Troubleshooting

### Issue: Cluster creation fails
```bash
# Check Docker is running
docker ps

# Check Kind is installed
kind version
```

### Issue: LLM API errors
```bash
# Verify API keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### Issue: Import errors
```bash
# Ensure AIOpsLab is in Python path
export PYTHONPATH=/path/to/AIOpsLab:$PYTHONPATH
```

## Future Enhancements

1. **Cluster Pooling**: Pre-create clusters for faster execution
2. **Cost Tracking**: Real-time cost monitoring per task
3. **Agent Comparison**: Run multiple agents on same problem
4. **Distributed Workers**: Run workers on multiple machines
5. **Result Caching**: Cache solutions for repeated problems
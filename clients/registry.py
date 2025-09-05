"""Agent registry for AIOpsLab."""

from clients.gpt import GPTAgent
from clients.qwen import QwenAgent
from clients.deepseek import DeepSeekAgent
from clients.vllm import vLLMAgent
from clients.openrouter import OpenRouterAgent

class AgentRegistry:
    """Registry for agent implementations."""
    
    def __init__(self):
        self.AGENT_REGISTRY = {
            "gpt": GPTAgent,
            "qwen": QwenAgent,
            "deepseek": DeepSeekAgent,
            "vllm": vLLMAgent,
            "openrouter": OpenRouterAgent,
        }
    
    def register(self, name, agent_cls):
        """Register an agent implementation."""
        self.agents[name] = agent_cls
        return agent_cls

    def get_agent(self, agent: str):
        """Get an agent implementation."""
        return self.AGENT_REGISTRY.get(agent)
    
    def get_agent_ids(self):
        return list(self.AGENT_REGISTRY.keys())

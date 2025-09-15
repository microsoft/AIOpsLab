"""Agent wrapper to capture LLM conversations."""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import LLMConversation, MessageRole
from ..config.logging import get_logger

logger = get_logger(__name__)


class LLMLoggingAgent:
    """Wrapper agent that logs all LLM interactions to database."""

    def __init__(self, base_agent: Any, conversation: LLMConversation, session: AsyncSession):
        """Initialize the logging wrapper.

        Args:
            base_agent: The actual agent (GPT, Claude, etc.)
            conversation: The LLMConversation record to log to
            session: Database session for committing logs
        """
        self.base_agent = base_agent
        self.conversation = conversation
        self.session = session
        self.step_count = 0

    async def get_action(self, state: str) -> str:
        """Get action from agent while logging the conversation.

        This method is called by the orchestrator with the current state
        and returns the agent's action.
        """
        self.step_count += 1

        try:
            # Log the state/observation sent to the agent
            await self._log_message(
                MessageRole.USER,
                state,
                metadata={
                    "step": self.step_count,
                    "type": "observation"
                }
            )

            # Get the actual response from the base agent
            # Handle both sync and async agents
            if hasattr(self.base_agent.get_action, '__call__'):
                # Synchronous agent
                response = self.base_agent.get_action(state)
            else:
                # Async agent
                response = await self.base_agent.get_action(state)

            # Parse the response to extract any tool calls
            tool_calls = self._extract_tool_calls(response)

            # Log the agent's response
            if tool_calls:
                # Log with tool call information
                for tool_call in tool_calls:
                    await self._log_message(
                        MessageRole.ASSISTANT,
                        tool_call.get("description", response),
                        function_name=tool_call.get("function"),
                        function_args=tool_call.get("args"),
                        metadata={
                            "step": self.step_count,
                            "type": "action",
                            "tool": tool_call.get("function")
                        }
                    )
            else:
                # Log as regular assistant message
                await self._log_message(
                    MessageRole.ASSISTANT,
                    response,
                    metadata={
                        "step": self.step_count,
                        "type": "action"
                    }
                )

            logger.debug(
                "llm_logging.interaction",
                step=self.step_count,
                conversation_id=str(self.conversation.id),
                response_length=len(response)
            )

            return response

        except Exception as e:
            logger.error(
                "llm_logging.error",
                step=self.step_count,
                error=str(e)
            )

            # Log the error
            await self._log_message(
                MessageRole.SYSTEM,
                f"Error during agent interaction: {str(e)}",
                metadata={
                    "step": self.step_count,
                    "type": "error",
                    "error": str(e)
                }
            )

            # Re-raise the error
            raise

    def _extract_tool_calls(self, response: str) -> list:
        """Extract tool/function calls from agent response.

        Parses the response to identify API calls like:
        - exec_bash
        - get_logs
        - query_metrics
        - etc.
        """
        tool_calls = []

        # Common patterns for tool calls in responses
        # This is a simplified parser - should be enhanced based on actual response format
        if "exec_bash" in response:
            # Extract bash command
            import re
            bash_match = re.search(r'exec_bash\s*\((.*?)\)', response, re.DOTALL)
            if bash_match:
                tool_calls.append({
                    "function": "exec_bash",
                    "args": {"command": bash_match.group(1).strip().strip('"\'')},
                    "description": "Executing bash command"
                })

        elif "get_logs" in response:
            # Extract log query parameters
            tool_calls.append({
                "function": "get_logs",
                "args": {},
                "description": "Fetching logs"
            })

        elif "query_metrics" in response:
            # Extract metrics query
            tool_calls.append({
                "function": "query_metrics",
                "args": {},
                "description": "Querying metrics"
            })

        return tool_calls

    async def _log_message(
        self,
        role: MessageRole,
        content: str,
        function_name: Optional[str] = None,
        function_args: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Log a message to the conversation."""
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role.value,
            "content": content
        }

        if function_name:
            message["function_name"] = function_name
        if function_args:
            message["function_args"] = function_args
        if metadata:
            message["metadata"] = metadata

        # Create a new list to trigger SQLAlchemy's change detection
        messages_copy = list(self.conversation.messages)
        messages_copy.append(message)
        self.conversation.messages = messages_copy
        self.conversation.total_messages += 1

        # Commit to database
        await self.session.commit()

    # Pass through other agent methods if they exist
    def __getattr__(self, name):
        """Pass through any other methods to the base agent."""
        return getattr(self.base_agent, name)
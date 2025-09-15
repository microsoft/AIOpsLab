"""Pydantic schemas for LLM conversation data."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


class MessageRole(str, Enum):
    """Role of the message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class LLMMessage(BaseModel):
    """Individual message in an LLM conversation."""
    timestamp: datetime
    role: MessageRole
    content: str
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = None
    function_result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMConversationBase(BaseModel):
    """Base schema for LLM conversation."""
    model_name: Optional[str] = Field(None, description="LLM model used")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMConversationCreate(LLMConversationBase):
    """Schema for creating an LLM conversation."""
    task_id: UUID = Field(..., description="Associated task ID")


class LLMConversationUpdate(BaseModel):
    """Schema for updating an LLM conversation."""
    ended_at: Optional[datetime] = None
    total_tokens: Optional[int] = None
    total_cost: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMConversationResponse(LLMConversationBase):
    """Schema for LLM conversation response."""
    model_config: ConfigDict = ConfigDict(from_attributes=True)

    id: UUID
    task_id: UUID
    session_id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    total_messages: int
    total_tokens: Optional[int] = None
    total_cost: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    llm_config: Dict[str, Any] = Field(default_factory=dict)


class LLMConversationSummary(BaseModel):
    """Summary of an LLM conversation."""
    model_config: ConfigDict = ConfigDict(from_attributes=True)

    id: UUID
    task_id: UUID
    session_id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    model_name: Optional[str] = None
    total_messages: int
    total_tokens: Optional[int] = None
    success: Optional[bool] = None


class LLMConversationListResponse(BaseModel):
    """Response for listing LLM conversations."""
    conversations: List[LLMConversationSummary]
    total: int
    page: int
    page_size: int
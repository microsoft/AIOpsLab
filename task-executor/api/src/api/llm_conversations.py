"""API endpoints for LLM conversation history."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import Optional, List
from uuid import UUID

from ..models import get_db, LLMConversation, Task
from ..schemas.llm_conversation import (
    LLMConversationResponse,
    LLMConversationSummary,
    LLMConversationListResponse
)
from ..config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/llm-conversations", tags=["llm-conversations"])


@router.get("", response_model=LLMConversationListResponse)
async def list_conversations(
    task_id: Optional[UUID] = Query(None, description="Filter by task ID"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    session: AsyncSession = Depends(get_db)
):
    """List all LLM conversations with filtering and pagination."""
    try:
        # Build query
        query = select(LLMConversation)

        # Apply filters
        if task_id:
            query = query.where(LLMConversation.task_id == task_id)
        if model_name:
            query = query.where(LLMConversation.model_name == model_name)

        # Add ordering
        query = query.order_by(LLMConversation.started_at.desc())

        # Count total
        count_query = select(func.count()).select_from(LLMConversation)
        if task_id:
            count_query = count_query.where(LLMConversation.task_id == task_id)
        if model_name:
            count_query = count_query.where(LLMConversation.model_name == model_name)

        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # Execute query
        result = await session.execute(query)
        conversations = result.scalars().all()

        # Convert to summary
        summaries = []
        for conv in conversations:
            summary = LLMConversationSummary(
                id=conv.id,
                task_id=conv.task_id,
                session_id=conv.session_id,
                started_at=conv.started_at,
                ended_at=conv.ended_at,
                model_name=conv.model_name,
                total_messages=conv.total_messages,
                total_tokens=conv.total_tokens,
                success=conv.conversation_metadata.get("success") if conv.conversation_metadata else None
            )
            summaries.append(summary)

        return LLMConversationListResponse(
            conversations=summaries,
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error("llm_conversations.list.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=LLMConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_db)
):
    """Get a specific LLM conversation with full message history."""
    try:
        # Get conversation
        result = await session.execute(
            select(LLMConversation).where(LLMConversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return LLMConversationResponse(
            id=conversation.id,
            task_id=conversation.task_id,
            session_id=conversation.session_id,
            started_at=conversation.started_at,
            ended_at=conversation.ended_at,
            model_name=conversation.model_name,
            llm_config=conversation.model_config or {},
            total_messages=conversation.total_messages,
            total_tokens=conversation.total_tokens,
            total_cost=conversation.total_cost,
            messages=conversation.messages or [],
            metadata=conversation.conversation_metadata or {}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "llm_conversations.get.error",
            conversation_id=str(conversation_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}/conversations", response_model=List[LLMConversationSummary])
async def get_task_conversations(
    task_id: UUID,
    session: AsyncSession = Depends(get_db)
):
    """Get all LLM conversations for a specific task."""
    try:
        # Check if task exists
        task_result = await session.execute(
            select(Task).where(Task.id == task_id)
        )
        task = task_result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Get conversations for this task
        result = await session.execute(
            select(LLMConversation)
            .where(LLMConversation.task_id == task_id)
            .order_by(LLMConversation.started_at.desc())
        )
        conversations = result.scalars().all()

        # Convert to summaries
        summaries = []
        for conv in conversations:
            summary = LLMConversationSummary(
                id=conv.id,
                task_id=conv.task_id,
                session_id=conv.session_id,
                started_at=conv.started_at,
                ended_at=conv.ended_at,
                model_name=conv.model_name,
                total_messages=conv.total_messages,
                total_tokens=conv.total_tokens,
                success=conv.conversation_metadata.get("success") if conv.conversation_metadata else None
            )
            summaries.append(summary)

        return summaries

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "llm_conversations.task.error",
            task_id=str(task_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: UUID,
    role: Optional[str] = Query(None, description="Filter by message role"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum messages to return"),
    session: AsyncSession = Depends(get_db)
):
    """Get messages from a specific conversation with optional filtering."""
    try:
        # Get conversation
        result = await session.execute(
            select(LLMConversation).where(LLMConversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Filter messages
        messages = conversation.messages or []
        if role:
            messages = [msg for msg in messages if msg.get("role") == role]

        # Apply limit
        messages = messages[:limit]

        return {
            "conversation_id": str(conversation_id),
            "total_messages": len(messages),
            "messages": messages
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "llm_conversations.messages.error",
            conversation_id=str(conversation_id),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/summary")
async def get_conversation_stats(
    session: AsyncSession = Depends(get_db)
):
    """Get statistics about LLM conversations."""
    try:
        # Total conversations
        total_result = await session.execute(
            select(func.count(LLMConversation.id))
        )
        total_conversations = total_result.scalar() or 0

        # Conversations by model
        model_result = await session.execute(
            select(
                LLMConversation.model_name,
                func.count(LLMConversation.id).label("count")
            )
            .group_by(LLMConversation.model_name)
        )
        conversations_by_model = {
            row.model_name: row.count for row in model_result if row.model_name
        }

        # Average messages per conversation
        avg_messages_result = await session.execute(
            select(func.avg(LLMConversation.total_messages))
        )
        avg_messages = avg_messages_result.scalar() or 0

        # Total tokens consumed
        total_tokens_result = await session.execute(
            select(func.sum(LLMConversation.total_tokens))
        )
        total_tokens = total_tokens_result.scalar() or 0

        # Success rate (from metadata)
        success_count_result = await session.execute(
            select(func.count(LLMConversation.id))
            .where(
                LLMConversation.conversation_metadata["success"].astext == "true"
            )
        )
        success_count = success_count_result.scalar() or 0
        success_rate = (success_count / total_conversations) if total_conversations > 0 else 0

        return {
            "total_conversations": total_conversations,
            "conversations_by_model": conversations_by_model,
            "avg_messages_per_conversation": float(avg_messages),
            "total_tokens_consumed": total_tokens,
            "success_rate": success_rate,
            "successful_conversations": success_count
        }

    except Exception as e:
        logger.error("llm_conversations.stats.error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
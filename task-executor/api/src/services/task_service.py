"""Task service for business logic."""

from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Task, TaskStatus, TaskLog, LogLevel
from ..lib.task_queue import TaskQueue
from ..schemas.task import TaskCreate, TaskStatusUpdate
from ..config.logging import get_logger

logger = get_logger(__name__)


class TaskService:
    """Service layer for task operations."""

    def __init__(self, session: AsyncSession):
        """Initialize task service with database session."""
        self.session = session
        self.queue = TaskQueue(session)

    async def create_task(self, task_data: TaskCreate) -> Task:
        """Create a new task and add to queue."""
        task = await self.queue.create_task(
            problem_id=task_data.problem_id,
            parameters=task_data.parameters
        )

        await self._log_task_event(
            task.id,
            LogLevel.INFO,
            "Task created",
            {
                "problem_id": task_data.problem_id,
                "parameters": task_data.parameters
            }
        )

        logger.info(
            "task.created",
            task_id=str(task.id),
            problem_id=task_data.problem_id
        )

        return task

    async def get_task(self, task_id: UUID) -> Optional[Task]:
        """Get task by ID."""
        return await self.queue.get_task(task_id)

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        problem_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at"
    ) -> Tuple[List[Task], int]:
        """List tasks with filtering and pagination."""
        offset = (page - 1) * page_size

        tasks, total = await self.queue.list_tasks(
            status=status,
            problem_id=problem_id,
            worker_id=worker_id,
            limit=page_size,
            offset=offset,
            sort=sort
        )

        return tasks, total

    async def update_task_status(
        self,
        task_id: UUID,
        update_data: TaskStatusUpdate
    ) -> Task:
        """Update task status (internal use by workers)."""
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if update_data.status == TaskStatus.COMPLETED:
            if not update_data.result:
                raise ValueError("Result required for completed status")

            task = await self.queue.complete_task(
                task_id,
                update_data.result
            )

            await self._log_task_event(
                task_id,
                LogLevel.INFO,
                "Task completed successfully",
                {"result": update_data.result}
            )

        elif update_data.status == TaskStatus.FAILED:
            if not update_data.error_details:
                raise ValueError("Error details required for failed status")

            task = await self.queue.fail_task(
                task_id,
                update_data.error_details
            )

            await self._log_task_event(
                task_id,
                LogLevel.ERROR,
                "Task failed",
                {"error": update_data.error_details}
            )

        elif update_data.status == TaskStatus.TIMEOUT:
            task = await self.queue.timeout_task(task_id)

            await self._log_task_event(
                task_id,
                LogLevel.WARNING,
                "Task timed out",
                {}
            )

        elif update_data.status == TaskStatus.CANCELLED:
            task = await self.queue.cancel_task(task_id)

            await self._log_task_event(
                task_id,
                LogLevel.INFO,
                "Task cancelled",
                {}
            )

        else:
            raise ValueError(f"Cannot update to status {update_data.status}")

        return task

    async def cancel_task(self, task_id: UUID) -> Task:
        """Cancel a pending or running task."""
        task = await self.queue.cancel_task(task_id)

        await self._log_task_event(
            task_id,
            LogLevel.INFO,
            "Task cancelled by user",
            {}
        )

        return task

    async def get_task_logs(
        self,
        task_id: UUID,
        level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[TaskLog]:
        """Get logs for a specific task."""
        from sqlalchemy import select, and_

        query = select(TaskLog).where(TaskLog.task_id == task_id)

        if level:
            query = query.where(TaskLog.level == level)

        query = query.order_by(TaskLog.timestamp.desc()).limit(limit)

        result = await self.session.execute(query)
        logs = result.scalars().all()

        return list(reversed(logs))

    async def add_task_log(
        self,
        task_id: UUID,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskLog:
        """Add a log entry for a task."""
        return await self._log_task_event(
            task_id,
            level,
            message,
            context or {}
        )

    async def get_queue_stats(self) -> Dict[str, int]:
        """Get task queue statistics."""
        return await self.queue.get_queue_stats()

    async def check_timeouts(self) -> List[Task]:
        """Check and timeout expired tasks."""
        timeout_tasks = await self.queue.check_timeouts()

        for task in timeout_tasks:
            await self._log_task_event(
                task.id,
                LogLevel.WARNING,
                "Task timed out automatically",
                {
                    "timeout_minutes": task.parameters.get("timeout_minutes", 30)
                }
            )

        return timeout_tasks

    async def _log_task_event(
        self,
        task_id: UUID,
        level: LogLevel,
        message: str,
        context: Dict[str, Any]
    ) -> TaskLog:
        """Internal method to log task events."""
        log_entry = TaskLog(
            task_id=task_id,
            level=level,
            message=message,
            context=context
        )

        self.session.add(log_entry)
        await self.session.commit()
        await self.session.refresh(log_entry)

        return log_entry

    async def get_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        from sqlalchemy import func, case, select

        # Count tasks by status
        status_counts = await self.session.execute(
            select(
                Task.status,
                func.count(Task.id).label("count")
            ).group_by(Task.status)
        )

        status_dict = {row.status.value: row.count for row in status_counts}

        # Calculate average execution time for completed tasks
        avg_time_result = await self.session.execute(
            select(
                func.avg(
                    func.extract("epoch", Task.completed_at - Task.started_at)
                ).label("avg_time")
            ).where(
                Task.status == TaskStatus.COMPLETED,
                Task.started_at.isnot(None),
                Task.completed_at.isnot(None)
            )
        )
        avg_execution_time = avg_time_result.scalar()

        # Count tasks by problem_id
        problem_counts = await self.session.execute(
            select(
                Task.problem_id,
                func.count(Task.id).label("count")
            ).group_by(Task.problem_id)
        )
        tasks_by_problem = {row.problem_id: row.count for row in problem_counts}

        # Count tasks by worker_id
        worker_counts = await self.session.execute(
            select(
                Task.worker_id,
                func.count(Task.id).label("count")
            ).where(Task.worker_id.isnot(None))
            .group_by(Task.worker_id)
        )
        tasks_by_worker = {row.worker_id: row.count for row in worker_counts}

        # Calculate totals
        total_tasks = sum(status_dict.values())
        completed = status_dict.get("completed", 0)
        failed = status_dict.get("failed", 0)

        # Calculate success rate
        terminal_tasks = completed + failed + status_dict.get("timeout", 0)
        success_rate = (completed / terminal_tasks) if terminal_tasks > 0 else 0.0

        return {
            "total_tasks": total_tasks,
            "pending_tasks": status_dict.get("pending", 0),
            "running_tasks": status_dict.get("running", 0),
            "completed_tasks": completed,
            "failed_tasks": failed,
            "timeout_tasks": status_dict.get("timeout", 0),
            "cancelled_tasks": status_dict.get("cancelled", 0),
            "avg_execution_time": float(avg_execution_time) if avg_execution_time else None,
            "success_rate": success_rate,
            "tasks_by_problem": tasks_by_problem,
            "tasks_by_worker": tasks_by_worker
        }
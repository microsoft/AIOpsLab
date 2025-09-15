"""Main worker process with polling loop for task execution."""

import asyncio
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import argparse
import logging
import os

import httpx
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskData(BaseModel):
    """Task data from API."""
    id: str
    problem_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str
    worker_id: Optional[str] = None


class Worker:
    """Worker process that polls for and executes tasks."""

    def __init__(
        self,
        worker_id: str,
        backend_type: str = "default",
        api_url: str = "http://localhost:8000",
        poll_interval: int = 5,
        heartbeat_interval: int = 30
    ):
        """Initialize worker with configuration."""
        self.worker_id = worker_id
        self.backend_type = backend_type
        self.api_url = api_url.rstrip('/')
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval

        self.client = httpx.AsyncClient(timeout=30.0)
        self.running = False
        self.current_task: Optional[TaskData] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(f"Worker {worker_id} initialized with backend {backend_type}")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.shutdown_event.set()

    async def register(self) -> bool:
        """Register worker with API server."""
        try:
            response = await self.client.post(
                f"{self.api_url}/api/v1/workers/register",
                json={
                    "worker_id": self.worker_id,
                    "backend_type": self.backend_type,
                    "capabilities": {
                        "max_parallel_tasks": 1,
                        "supported_problems": []
                    },
                    "metadata": {
                        "host": os.uname().nodename,
                        "pid": os.getpid(),
                        "version": "1.0.0",
                        "kind_cluster": self.worker_id
                    }
                }
            )

            if response.status_code == 201 or response.status_code == 200:
                logger.info(f"Worker {self.worker_id} registered successfully")
                return True
            else:
                logger.error(f"Failed to register: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def _send_heartbeat(self):
        """Send periodic heartbeats in background thread."""
        while not self.shutdown_event.is_set():
            try:
                # Use synchronous httpx client in thread
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(
                        f"{self.api_url}/api/v1/workers/{self.worker_id}/heartbeat",
                        json={
                            "status": "busy" if self.current_task else "idle",
                            "current_task_id": self.current_task.id if self.current_task else None
                        }
                    )

                    if response.status_code != 200:
                        logger.warning(f"Heartbeat failed: {response.status_code}")

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # Wait for next heartbeat or shutdown
            self.shutdown_event.wait(self.heartbeat_interval)

    def start_heartbeat(self):
        """Start heartbeat thread."""
        if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(
                target=self._send_heartbeat,
                daemon=True
            )
            self.heartbeat_thread.start()
            logger.info("Heartbeat thread started")

    async def claim_task(self) -> Optional[TaskData]:
        """Claim next available task from queue."""
        try:
            response = await self.client.post(
                f"{self.api_url}/api/v1/workers/{self.worker_id}/claim"
            )

            if response.status_code == 200:
                data = response.json()
                if data:  # API returns null if no tasks
                    task = TaskData(**data)
                    logger.info(f"Claimed task {task.id}: {task.problem_id}")
                    return task

            elif response.status_code == 204:
                # No tasks available
                return None
            else:
                logger.warning(f"Failed to claim task: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error claiming task: {e}")
            return None

    async def execute_task(self, task: TaskData) -> Dict[str, Any]:
        """Execute task using AIOpsLab orchestrator."""
        logger.info(f"Executing task {task.id}: {task.problem_id}")

        try:
            # Import AIOpsLab components
            import sys
            import os

            # Add parent directory to path to import aiopslab
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            sys.path.insert(0, parent_dir)

            from aiopslab.orchestrator import Orchestrator
            from aiopslab.orchestrator.tasks import TaskType

            # Initialize orchestrator
            orchestrator = Orchestrator()

            # Set up session parameters
            max_steps = task.parameters.get("max_steps", 30)
            agent_config = task.parameters.get("agent_config", {})

            # Create a simple agent that returns empty actions
            class SimpleAgent:
                async def get_action(self, state: str) -> str:
                    # In real implementation, this would use the agent_config
                    # to create appropriate agent (GPT, Claude, etc.)
                    return '{"action": "finish", "message": "Task completed"}'

            agent = SimpleAgent()

            # Run the problem
            result = await orchestrator.start_problem(
                problem_id=task.problem_id,
                agent=agent,
                max_steps=max_steps
            )

            logger.info(f"Task {task.id} completed successfully")

            return {
                "success": True,
                "result": result,
                "execution_time": datetime.utcnow().isoformat()
            }

        except ImportError as e:
            logger.error(f"Failed to import AIOpsLab: {e}")
            return {
                "success": False,
                "error": f"AIOpsLab import error: {str(e)}",
                "execution_time": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": datetime.utcnow().isoformat()
            }

    async def complete_task(self, task: TaskData, result: Dict[str, Any]) -> bool:
        """Report task completion to API."""
        try:
            if result.get("success"):
                response = await self.client.post(
                    f"{self.api_url}/api/v1/workers/{self.worker_id}/tasks/{task.id}/complete",
                    json=result
                )
            else:
                response = await self.client.post(
                    f"{self.api_url}/api/v1/workers/{self.worker_id}/tasks/{task.id}/fail",
                    json={"error": result.get("error", "Unknown error")}
                )

            if response.status_code == 200:
                logger.info(f"Task {task.id} result reported successfully")
                return True
            else:
                logger.error(f"Failed to report result: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error reporting task result: {e}")
            return False

    async def run(self):
        """Main worker loop."""
        # Register with API
        if not await self.register():
            logger.error("Failed to register, exiting")
            return

        # Start heartbeat thread
        self.start_heartbeat()

        self.running = True
        consecutive_errors = 0
        max_consecutive_errors = 5

        logger.info(f"Worker {self.worker_id} starting main loop")

        while self.running:
            try:
                # Claim a task
                task = await self.claim_task()

                if task:
                    self.current_task = task
                    consecutive_errors = 0  # Reset error counter

                    # Execute the task
                    result = await self.execute_task(task)

                    # Report result
                    await self.complete_task(task, result)

                    self.current_task = None
                else:
                    # No task available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), shutting down")
                    self.running = False
                else:
                    # Wait longer after errors
                    await asyncio.sleep(self.poll_interval * 2)

        # Cleanup
        self.shutdown_event.set()
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        await self.client.aclose()

        logger.info(f"Worker {self.worker_id} stopped")


async def main():
    """Main entry point for worker process."""
    parser = argparse.ArgumentParser(description="AIOpsLab Task Worker")
    parser.add_argument(
        "--id",
        required=True,
        help="Worker ID (format: worker-XXX-kind)"
    )
    parser.add_argument(
        "--backend-type",
        default="default",
        help="Backend type for this worker"
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API server URL"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Polling interval in seconds"
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Heartbeat interval in seconds"
    )

    args = parser.parse_args()

    # Validate worker ID format
    import re
    if not re.match(r'^worker-\d{3}-kind$', args.id):
        logger.error("Worker ID must follow pattern: worker-XXX-kind")
        sys.exit(1)

    # Create and run worker
    worker = Worker(
        worker_id=args.id,
        backend_type=args.backend_type,
        api_url=args.api_url,
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
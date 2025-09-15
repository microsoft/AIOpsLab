#!/usr/bin/env python3
"""Test script for integrated API with internal workers."""

import asyncio
import aiohttp
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"


async def test_integrated_system():
    """Test the integrated API with internal workers."""
    async with aiohttp.ClientSession() as session:
        print("\n=== Testing Integrated Task Executor API ===\n")

        # 1. Check health
        print("1. Checking API health...")
        async with session.get(f"{API_BASE_URL}/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"   ✓ API is healthy: {data}")
            else:
                print(f"   ✗ Health check failed: {resp.status}")
                return

        # 2. Check internal workers status
        print("\n2. Checking internal workers status...")
        async with session.get(f"{API_BASE_URL}/api/v1/workers/internal/status") as resp:
            data = await resp.json()
            print(f"   Status: {json.dumps(data, indent=2)}")

        # 3. List all workers
        print("\n3. Listing all workers...")
        async with session.get(f"{API_BASE_URL}/api/v1/workers") as resp:
            data = await resp.json()
            print(f"   Total workers: {data['total']}")
            for worker in data['workers'][:3]:  # Show first 3
                print(f"   - {worker['id']}: {worker['status']}")

        # 4. Submit a test task
        print("\n4. Submitting a test task...")
        task_data = {
            "problem_id": "test-problem-001",
            "parameters": {
                "max_steps": 10,
                "agent_config": {
                    "name": "test-agent",
                    "version": "1.0"
                }
            },
            "priority": 5
        }
        async with session.post(
            f"{API_BASE_URL}/api/v1/tasks",
            json=task_data
        ) as resp:
            if resp.status == 201:
                task = await resp.json()
                task_id = task['id']
                print(f"   ✓ Task created: {task_id}")
                print(f"     Status: {task['status']}")
            else:
                print(f"   ✗ Failed to create task: {resp.status}")
                return

        # 5. Wait for task to be processed
        print("\n5. Waiting for task to be processed...")
        for i in range(10):  # Wait up to 10 seconds
            await asyncio.sleep(1)
            async with session.get(f"{API_BASE_URL}/api/v1/tasks/{task_id}") as resp:
                task = await resp.json()
                status = task['status']
                print(f"   [{i+1}s] Status: {status}")

                if status in ["completed", "failed"]:
                    print(f"\n   ✓ Task finished with status: {status}")
                    if status == "completed":
                        print(f"     Result: {json.dumps(task.get('result'), indent=2)}")
                    break
        else:
            print("   ⚠ Task still processing after 10 seconds")

        # 6. Submit multiple tasks
        print("\n6. Submitting multiple tasks to test concurrent processing...")
        task_ids = []
        for i in range(5):
            task_data = {
                "problem_id": f"test-problem-{i+1:03d}",
                "parameters": {
                    "max_steps": 5,
                    "task_number": i+1
                },
                "priority": 10 - i  # Different priorities
            }
            async with session.post(
                f"{API_BASE_URL}/api/v1/tasks",
                json=task_data
            ) as resp:
                if resp.status == 201:
                    task = await resp.json()
                    task_ids.append(task['id'])
                    print(f"   ✓ Task {i+1} created: {task['id']}")

        # 7. Check task queue stats
        print("\n7. Checking task queue statistics...")
        async with session.get(f"{API_BASE_URL}/api/v1/tasks/stats") as resp:
            stats = await resp.json()
            print(f"   Queue stats: {json.dumps(stats, indent=2)}")

        # 8. Wait for all tasks to complete
        print("\n8. Monitoring task completion...")
        completed = 0
        for _ in range(15):  # Wait up to 15 seconds
            await asyncio.sleep(1)

            completed_count = 0
            for task_id in task_ids:
                async with session.get(f"{API_BASE_URL}/api/v1/tasks/{task_id}") as resp:
                    task = await resp.json()
                    if task['status'] in ["completed", "failed"]:
                        completed_count += 1

            if completed_count != completed:
                completed = completed_count
                print(f"   Completed: {completed}/{len(task_ids)}")

            if completed == len(task_ids):
                print("   ✓ All tasks completed!")
                break

        # 9. Test scaling workers
        print("\n9. Testing worker scaling...")
        print("   Scaling to 5 workers...")
        async with session.post(
            f"{API_BASE_URL}/api/v1/workers/internal/scale?num_workers=5"
        ) as resp:
            data = await resp.json()
            print(f"   Result: {data}")

        await asyncio.sleep(1)

        print("   Scaling back to 3 workers...")
        async with session.post(
            f"{API_BASE_URL}/api/v1/workers/internal/scale?num_workers=3"
        ) as resp:
            data = await resp.json()
            print(f"   Result: {data}")

        # 10. Final status check
        print("\n10. Final system status...")
        async with session.get(f"{API_BASE_URL}/api/v1/workers/internal/status") as resp:
            data = await resp.json()
            print(f"   Workers running: {data['running']}")
            print(f"   Number of workers: {data['num_workers']}")

        print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    asyncio.run(test_integrated_system())
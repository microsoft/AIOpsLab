#!/usr/bin/env python3
import httpx
import time
import json

# Configuration
# We use localhost because we are running this script on the same machine as the service_api
SERVICE_IP = "127.0.0.1" 
SERVICE_PORT = 8099
MODEL_ECHO_IP = "14.103.221.215"

# Define tasks - Social Network problems
SOCIAL_NETWORK_TASKS = [
    "k8s_target_port-misconfig-detection-1",
    "k8s_target_port-misconfig-detection-2",
    "k8s_target_port-misconfig-detection-3",
    "auth_miss_mongodb-detection-1",
]

# Astronomy Shop problems (ClusterRole names now include namespace - can run in parallel!)
ASTRONOMY_SHOP_TASKS = [
    "astronomy_shop_ad_service_failure-detection-1",
    "astronomy_shop_ad_service_high_cpu-detection-1",
    "astronomy_shop_ad_service_manual_gc-detection-1",
]

# Test 3 astronomy_shop problems in parallel to verify ClusterRole fix
TASKS_TO_RUN = ASTRONOMY_SHOP_TASKS[:3]

payload = {
    "problems": [
        {"problem_id": task_id, "runs": 1, "max_steps": 12}
        for task_id in TASKS_TO_RUN
    ],
    "concurrency": 3,  # 3 astronomy-shop in parallel
    "chat": {
        "model": "/data1/xj/pred_model/Qwen2.5-Coder-14B-Instruct",
        "base_url": f"http://{MODEL_ECHO_IP}:18200/v1",
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 512
    },
    # "echo": {
    #     "url": f"http://{MODEL_ECHO_IP}:18209"
    # }
}

print(f"Sending {len(TASKS_TO_RUN)} tasks to {SERVICE_IP}:{SERVICE_PORT}...")
for i, task_id in enumerate(TASKS_TO_RUN, 1):
    print(f"  {i}. {task_id}")
print()

try:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"http://{SERVICE_IP}:{SERVICE_PORT}/echo/jobs",
            json=payload,
        )
        resp.raise_for_status()
        print("Job submitted successfully!")
        job_data = resp.json()
        print(json.dumps(job_data, indent=2))
        
        job_id = job_data.get("job_id")
        if job_id:
            print(f"\nPolling status for job {job_id}...")
            while True:
                status_resp = client.get(f"http://{SERVICE_IP}:{SERVICE_PORT}/echo/jobs/{job_id}")
                status_resp.raise_for_status()
                status = status_resp.json()
                print(f"Status: {status['status']} | Completed: {status['completed_runs']}/{status['total_runs']}")
                
                if status['status'] in ['succeeded', 'failed', 'cancelled']:
                    print("\nFinal Status:")
                    print(json.dumps(status, indent=2))
                    break
                time.sleep(5)

except Exception as e:
    print(f"Error: {e}")


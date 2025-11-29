import requests
import json

# Configuration
SERVICE_IP = "127.0.0.1"           # Localhost since we are running locally
SERVICE_PORT = 8099

# Define the tasks to run
ALL_TASKS = [
    "k8s_target_port-misconfig-detection-1",
    "k8s_target_port-misconfig-detection-2",
    "k8s_target_port-misconfig-detection-3",
    "auth_miss_mongodb-detection-1",
]

# Select specific tasks
TASKS_TO_RUN = ALL_TASKS[2:4]  # Indices 2 and 3

payload = {
    "problems": [
        {"problem_id": task_id, "runs": 1, "max_steps": 12}
        for task_id in TASKS_TO_RUN
    ],
    "concurrency": 2,  # Parallel execution
    "chat": {
        "model": "Qwen/Qwen2.5-Coder-14B-Instruct", 
        "base_url": "http://mock-model-server:18200/v1",  
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 512
    },
    # Removing echo config for now as requested to ignore callback
    # "echo": {
    #     "url": "http://mock-echo-server:18209"
    # }
}

print(f"Preparing to send {len(TASKS_TO_RUN)} tasks to {SERVICE_IP}:{SERVICE_PORT}:")
for i, task_id in enumerate(TASKS_TO_RUN, 1):
    print(f"  {i}. {task_id}")
print()

try:
    resp = requests.post(
        f"http://{SERVICE_IP}:{SERVICE_PORT}/echo/jobs",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    print("Job submitted successfully!")
    print(json.dumps(resp.json(), indent=2))
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response content: {e.response.text}")


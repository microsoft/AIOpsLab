# Echo Job Callback API

This document describes the FastAPI surface exposed by `Echo/server.py`.  The
server acts as a lightweight coordination point between rollout workers and the
RL training harness: workers declare which problems to run, stream lifecycle
events as episodes progress, and fetch status or stored trajectories for later
inspection.

Base URL for the examples below: `http://localhost:8098`.

## Authentication

The reference implementation does **not** enforce authentication.  When deployed
in shared environments you can layer transport security or bearer tokens in
front of the service (for example with an API gateway).  The orchestrator
components include optional `api_key` and `extra_headers` fields when they are
available.

## Job Lifecycle Overview

1. **Create** a job with `POST /jobs`.  Provide one or more problem identifiers
   and the desired episode count for each.
2. **Stream events** to `/jobs/{job_id}/events` as the rollout progresses.
   - The first `initializing` or `run_started` event moves the job into the
     `"running"` state.
   - Every `run_finished` event increments the completed-run counter.
   - `job_finished` moves the job to `"done"` and freezes the run summary.
   - `job_failed` transitions the job to `"failed"` and stores the failure
     reason/partial results if provided.
3. **Inspect** the job via `/jobs/{job_id}/status`, `/events`, or `/results`.
   Completed or failed jobs expose the final run payloads under `/results`.

Jobs are identified by a 32-character hex string generated server-side.

## Endpoint Summary

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/jobs` | Create a new job and receive callback URLs. |
| `POST` | `/jobs/{job_id}/events` | Submit lifecycle events (run started, run finished, etc.). |
| `GET` | `/jobs/{job_id}/status` | Fetch aggregate state and per-problem progress. |
| `GET` | `/jobs/{job_id}/events` | Retrieve the raw event log for debugging. |
| `GET` | `/jobs/{job_id}/results` | Retrieve the stored run payloads and failure metadata. |
| `POST` | `/callbacks/runs` | Report a finished rollout trajectory. |
| `GET` | `/callbacks/runs` | List recent trajectories kept in memory. |
| `POST` | `/callbacks/runs/export` | Persist the trajectory buffer to a JSONL file. |
| `DELETE` | `/callbacks/runs` | Clear the in-memory trajectory history. |

## Create a Job — `POST /jobs`

Request body:

```json
{
  "problems": [
    {"id": "astronomy_shop_payment_service_failure-detection-1", "runs": 2},
    {"id": "k8s_target_port-misconfig-detection-1", "runs": 1}
  ]
}
```

- Each `problem.id` must be unique within the job.
- `runs` must be a positive integer.

Response:

```json
{
  "job_id": "2c5d18bb6d7443028ce2d8a6c2f4af8c",
  "expected_runs": 3,
  "callbacks": {
    "status": "/jobs/2c5d18bb6d7443028ce2d8a6c2f4af8c/status",
    "events": "/jobs/2c5d18bb6d7443028ce2d8a6c2f4af8c/events"
  }
}
```

The client is responsible for POSTing events to the returned callback URL.

## Report Job Events — `POST /jobs/{job_id}/events`

Events drive the server-side state machine.  The accepted fields are:

- `event` (required): free-form string label. Case-insensitive helpers are
  built in for the known lifecycle events described below.
- `problem_id`: required when the event refers to a specific problem.
- `run_index`: zero-based index of the run within that problem.
- `payload`: arbitrary JSON document, often used to store per-run metadata.
- `reason`: optional string explaining failures.
- `partial`: optional list of rollback payloads for partial failures.

### Typical Event Sequence

```json
POST /jobs/{job_id}/events
{
  "event": "initializing"
}

POST /jobs/{job_id}/events
{
  "event": "run_started",
  "problem_id": "astronomy_shop_payment_service_failure-detection-1",
  "run_index": 0
}

POST /jobs/{job_id}/events
{
  "event": "run_finished",
  "problem_id": "astronomy_shop_payment_service_failure-detection-1",
  "run_index": 0,
  "payload": {"total_reward": 0.75, "steps": 18}
}

POST /jobs/{job_id}/events
{
  "event": "job_finished"
}
```

`run_finished` validates that the provided `run_index` is within the expected
range for that problem and adds the payload to the per-job results buffer.
Duplicate `run_finished` calls for the same index will overwrite the stored
payload but do **not** increment the completion counter a second time.

When the orchestration fails, send:

```json
POST /jobs/{job_id}/events
{
  "event": "job_failed",
  "reason": "kubelet not reachable after 5 attempts",
  "partial": [
    {
      "problem_id": "astronomy_shop_payment_service_failure-detection-1",
      "run_index": 0,
      "payload": {"total_reward": 0.0, "steps": 2}
    }
  ]
}
```

The job transitions to `"failed"` and the `reason` and `partial` payloads are
persisted.

Response for every event submission:

```json
{
  "status": "ok",
  "job": {
    "job_id": "2c5d18bb6d7443028ce2d8a6c2f4af8c",
    "state": "running",
    "total_runs": 3,
    "completed_runs": 1,
    "problems": [
      {"id": "astronomy_shop_payment_service_failure-detection-1", "runs_total": 2, "runs_done": 1},
      {"id": "k8s_target_port-misconfig-detection-1", "runs_total": 1, "runs_done": 0}
    ]
  }
}
```

## Inspect Job Status — `GET /jobs/{job_id}/status`

The status payload mirrors the structure returned from event submissions.  When
the job is `"done"` or `"failed"` the response also embeds the collected run
payloads under `results`.

## Retrieve Event Log — `GET /jobs/{job_id}/events`

Returns all raw events in the order they were received:

```json
{
  "job_id": "2c5d18bb6d7443028ce2d8a6c2f4af8c",
  "events": [
    {"event": "initializing", "problem_id": null, "run_index": null, "payload": null, "reason": null, "partial": null},
    ...
  ]
}
```

Use this endpoint to debug unexpected transitions or duplicate notifications.

## Fetch Final Results — `GET /jobs/{job_id}/results`

Produces the latest run payloads even if the job is still technically running.
If the job failed the failure metadata is included:

```json
{
  "job_id": "2c5d18bb6d7443028ce2d8a6c2f4af8c",
  "runs": [
    {"problem_id": "astronomy_shop_payment_service_failure-detection-1", "run_index": 0, "payload": {"total_reward": 0.75, "steps": 18}},
    {"problem_id": "k8s_target_port-misconfig-detection-1", "run_index": 0, "payload": {"total_reward": 0.0, "steps": 22}}
  ],
  "failure_reason": "kubelet not reachable after 5 attempts",
  "partial": [
    {"problem_id": "astronomy_shop_payment_service_failure-detection-1", "run_index": 1, "payload": {"total_reward": 0.0, "steps": 2}}
  ]
}
```

## Manage Run Callbacks

The callback buffer is independent of the job abstractions: it simply stores
rollout trajectories for inspection.

### `POST /callbacks/runs`

Provide a `RunCallbackPayload`:

```json
{
  "job_id": "2c5d18bb6d7443028ce2d8a6c2f4af8c",
  "problem_id": "astronomy_shop_payment_service_failure-detection-1",
  "run_index": 0,
  "env_id": "gym-uuid-123",
  "total_reward": 0.75,
  "done": true,
  "trajectory": [
    {"step": 0, "observation": {"cluster_state": "..."}},
    {"step": 1, "action": "kubectl get pods", "reward": 0.0}
  ]
}
```

Response:

```json
{"status": "ok", "stored_runs": 12}
```

Only the most recent 200 payloads are retained (configurable via `_MAX_HISTORY`).

### `GET /callbacks/runs?limit=N`

Retrieves the most recent `N` payloads (`limit` defaults to 20).

### `POST /callbacks/runs/export`

Body: `{"path": "/tmp/echo_runs.jsonl"}`  
Writes the in-memory payloads as JSONL and reports the full path and count.

### `DELETE /callbacks/runs`

Purges the callback buffer and returns how many entries were removed.

## Updating This Document

- The canonical API surface lives in `Echo/server.py`.  Regenerate or verify
  examples whenever the request/response models change.
- `scripts/run_mock_echo_harness.py` provides an end-to-end example of the Echo
  server coordinating with the RL environment.


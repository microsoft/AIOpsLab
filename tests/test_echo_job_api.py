from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from Echo.server import app


client = TestClient(app)


def _create_job() -> str:
    response = client.post(
        "/jobs",
        json={"problems": [{"id": "prob-a", "runs": 2}, {"id": "prob-b", "runs": 1}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["expected_runs"] == 3
    return payload["job_id"]


def test_job_lifecycle_success():
    job_id = _create_job()

    # Initial status
    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["state"] == "pending"
    assert status["completed_runs"] == 0

    # Drive basic event sequence
    assert client.post(f"/jobs/{job_id}/events", json={"event": "initializing"}).status_code == 200
    assert client.post(
        f"/jobs/{job_id}/events", json={"event": "run_started", "problem_id": "prob-a", "run_index": 0}
    ).status_code == 200
    payload_a0 = {"reward": 0.5}
    resp = client.post(
        f"/jobs/{job_id}/events",
        json={
            "event": "run_finished",
            "problem_id": "prob-a",
            "run_index": 0,
            "payload": payload_a0,
        },
    )
    assert resp.status_code == 200
    progress = resp.json()["job"]
    assert progress["completed_runs"] == 1

    # Second run for problem A
    payload_a1 = {"reward": 0.7}
    client.post(
        f"/jobs/{job_id}/events",
        json={
            "event": "run_finished",
            "problem_id": "prob-a",
            "run_index": 1,
            "payload": payload_a1,
        },
    )

    # Run for problem B
    payload_b0 = {"reward": 1.0}
    client.post(
        f"/jobs/{job_id}/events",
        json={
            "event": "run_finished",
            "problem_id": "prob-b",
            "run_index": 0,
            "payload": payload_b0,
        },
    )

    # Finalise job
    client.post(f"/jobs/{job_id}/events", json={"event": "job_finished"})

    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["state"] == "done"
    assert status["completed_runs"] == 3
    results = status["results"]
    assert {run["run_index"] for run in results["runs"]} == {0, 1}
    assert len(results["runs"]) == 3

    # Dedicated results endpoint mirrors bundle
    results_resp = client.get(f"/jobs/{job_id}/results").json()
    assert len(results_resp["runs"]) == 3
    assert {"problem_id": "prob-a", "run_index": 0, "payload": payload_a0} in results_resp["runs"]
    assert {"problem_id": "prob-a", "run_index": 1, "payload": payload_a1} in results_resp["runs"]
    assert {"problem_id": "prob-b", "run_index": 0, "payload": payload_b0} in results_resp["runs"]

    # Events API preserves chronological log
    events = client.get(f"/jobs/{job_id}/events").json()["events"]
    # initializing + run_started + 3 run_finished + job_finished
    assert len(events) == 6
    assert events[0]["event"] == "initializing"


def test_job_failure_records_reason_and_partial():
    response = client.post("/jobs", json={"problems": [{"id": "prob-fail", "runs": 1}]})
    job_id = response.json()["job_id"]

    failure_payload = {
        "event": "job_failed",
        "reason": "model crashed",
        "partial": [{"problem_id": "prob-fail", "run_index": 0, "payload": {"reward": 0}}],
    }
    client.post(f"/jobs/{job_id}/events", json=failure_payload)

    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["state"] == "failed"
    assert status["failure_reason"] == "model crashed"

    results = client.get(f"/jobs/{job_id}/results").json()
    assert results["failure_reason"] == "model crashed"
    assert results["partial"] == failure_payload["partial"]


def test_invalid_events_raise_errors():
    response = client.post("/jobs", json={"problems": [{"id": "prob-x", "runs": 1}]})
    job_id = response.json()["job_id"]

    # Unknown problem
    bad_resp = client.post(
        f"/jobs/{job_id}/events",
        json={"event": "run_finished", "problem_id": "prob-unknown", "run_index": 0},
    )
    assert bad_resp.status_code == 400

    # Out-of-range run index
    bad_resp = client.post(
        f"/jobs/{job_id}/events",
        json={"event": "run_finished", "problem_id": "prob-x", "run_index": 9},
    )
    assert bad_resp.status_code == 400

    # Unknown job id
    missing = client.get("/jobs/nonexistent/status")
    assert missing.status_code == 404

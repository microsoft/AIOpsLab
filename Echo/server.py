#!/usr/bin/env python3
"""
Minimal Echo callback receiver used during local RL experiments.

The RL training service can be configured with an ``EchoServerConfig`` that
points at this application.  Every time a rollout finishes the training
service will POST the trajectory payload to ``/callbacks/runs``.  We keep the
last N payloads in memory so that developers can inspect them via ``GET`` and
optionally export them to disk for further analysis.

Run locally with:
    uvicorn Echo.server:app --reload --port 8098
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path as PathParam
from pydantic import BaseModel, Field, field_validator


app = FastAPI(
    title="Mock Echo Callback Server",
    description="Receives rollout results from the RL training service.",
    version="0.1.0",
)

_RUN_PAYLOADS: List[Dict[str, Any]] = []
_MAX_HISTORY = 200
_JOBS: Dict[str, "JobRecord"] = {}
_JOBS_LOCK = Lock()


class RunCallbackPayload(BaseModel):
    job_id: str
    problem_id: str
    run_index: int
    env_id: str
    total_reward: float
    done: bool
    trajectory: List[Dict[str, Any]]


def _append_payload(payload: Dict[str, Any]) -> None:
    _RUN_PAYLOADS.append(payload)
    if len(_RUN_PAYLOADS) > _MAX_HISTORY:
        _RUN_PAYLOADS.pop(0)


@app.post("/callbacks/runs")
async def receive_run(payload: RunCallbackPayload) -> Dict[str, Any]:
    """Record a completed rollout pushed by the training service."""

    data = payload.model_dump()
    _append_payload(data)
    return {"status": "ok", "stored_runs": len(_RUN_PAYLOADS)}


@app.get("/callbacks/runs")
async def list_runs(limit: int = 20) -> Dict[str, Any]:
    """Return the most recent rollout payloads (default: 20)."""

    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    return {"runs": _RUN_PAYLOADS[-limit:]}


@app.post("/callbacks/runs/export")
async def export_runs(path: str) -> Dict[str, Any]:
    """Persist the current run history to a JSONL file."""

    target = Path(path).expanduser()
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for payload in _RUN_PAYLOADS:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"status": "ok", "written": len(_RUN_PAYLOADS), "path": str(target)}


@app.delete("/callbacks/runs")
async def clear_runs() -> Dict[str, Any]:
    """Clear the in-memory history."""

    count = len(_RUN_PAYLOADS)
    _RUN_PAYLOADS.clear()
    return {"status": "ok", "cleared": count}


class JobProblemRequest(BaseModel):
    id: str = Field(..., min_length=1)
    runs: int = Field(..., ge=1)


class JobCreateRequest(BaseModel):
    problems: List[JobProblemRequest]

    @field_validator("problems")
    @classmethod
    def validate_problems(cls, value: List[JobProblemRequest]) -> List[JobProblemRequest]:
        if not value:
            raise ValueError("At least one problem must be provided.")
        identifiers = [p.id for p in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Problem identifiers must be unique.")
        return value


class JobCreateResponse(BaseModel):
    job_id: str
    expected_runs: int
    callbacks: Dict[str, str]


class JobEventRequest(BaseModel):
    event: str = Field(..., min_length=1)
    problem_id: Optional[str] = None
    run_index: Optional[int] = Field(default=None, ge=0)
    payload: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    partial: Optional[List[Dict[str, Any]]] = None


class JobProblemState:
    __slots__ = ("problem_id", "runs_total", "runs_done", "runs_payloads")

    def __init__(self, problem_id: str, runs_total: int) -> None:
        self.problem_id = problem_id
        self.runs_total = runs_total
        self.runs_done = 0
        self.runs_payloads: Dict[int, Dict[str, Any]] = {}

    def mark_run_finished(self, run_index: Optional[int], payload: Optional[Dict[str, Any]]) -> None:
        if run_index is None:
            return
        if run_index in self.runs_payloads:
            # Duplicate notification; do not count twice but allow payload updates.
            if payload is not None:
                self.runs_payloads[run_index] = payload
            return

        if run_index >= self.runs_total:
            raise ValueError(f"Run index {run_index} exceeds declared runs ({self.runs_total}) for problem {self.problem_id}.")

        self.runs_done += 1
        self.runs_payloads[run_index] = payload or {}

    def as_status(self) -> Dict[str, Any]:
        return {
            "id": self.problem_id,
            "runs_total": self.runs_total,
            "runs_done": self.runs_done,
        }

    def iter_runs(self) -> List[Dict[str, Any]]:
        return [
            {
                "problem_id": self.problem_id,
                "run_index": run_idx,
                "payload": payload,
            }
            for run_idx, payload in sorted(self.runs_payloads.items())
        ]


class JobRecord:
    __slots__ = (
        "job_id",
        "state",
        "problems",
        "events",
        "created_runs",
        "last_results",
        "failure_reason",
        "failure_partial",
    )

    def __init__(self, job_id: str, problems: List[JobProblemRequest]) -> None:
        self.job_id = job_id
        self.state: str = "pending"
        self.problems: Dict[str, JobProblemState] = {
            item.id: JobProblemState(item.id, item.runs) for item in problems
        }
        self.events: List[Dict[str, Any]] = []
        self.created_runs: int = sum(item.runs for item in problems)
        self.last_results: Optional[List[Dict[str, Any]]] = None
        self.failure_reason: Optional[str] = None
        self.failure_partial: Optional[List[Dict[str, Any]]] = None

    def total_completed(self) -> int:
        return sum(problem.runs_done for problem in self.problems.values())

    def append_event(self, event_payload: Dict[str, Any]) -> None:
        self.events.append(event_payload)

    def mark_running(self) -> None:
        if self.state == "pending":
            self.state = "running"

    def mark_done(self) -> None:
        self.state = "done"
        self.last_results = self.collect_results()

    def mark_failed(self, reason: Optional[str], partial: Optional[List[Dict[str, Any]]]) -> None:
        self.state = "failed"
        self.failure_reason = reason
        self.failure_partial = partial
        self.last_results = self.collect_results()

    def collect_results(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for problem in self.problems.values():
            runs.extend(problem.iter_runs())
        return runs

    def status_payload(self, include_results: bool = False) -> Dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "state": self.state,
            "total_runs": self.created_runs,
            "completed_runs": self.total_completed(),
            "problems": [problem.as_status() for problem in self.problems.values()],
        }
        if self.state == "failed" and self.failure_reason:
            payload["failure_reason"] = self.failure_reason
        if include_results and self.last_results is not None:
            payload["results"] = {
                "job_id": self.job_id,
                "runs": self.last_results,
            }
        return payload


def _get_job(job_id: str) -> JobRecord:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(payload: JobCreateRequest) -> JobCreateResponse:
    job_id = uuid4().hex
    job_record = JobRecord(job_id, payload.problems)
    with _JOBS_LOCK:
        _JOBS[job_id] = job_record

    status_path = f"/jobs/{job_id}/status"
    event_path = f"/jobs/{job_id}/events"
    return JobCreateResponse(
        job_id=job_id,
        expected_runs=job_record.created_runs,
        callbacks={"status": status_path, "events": event_path},
    )


def _handle_job_event(job: JobRecord, data: JobEventRequest) -> None:
    event_name = data.event.lower()
    event_record = {
        "event": data.event,
        "problem_id": data.problem_id,
        "run_index": data.run_index,
        "payload": data.payload,
        "reason": data.reason,
        "partial": data.partial,
    }
    job.append_event(event_record)

    if event_name in {"initializing", "run_started"}:
        job.mark_running()
        return

    if event_name == "run_finished":
        if not data.problem_id or data.problem_id not in job.problems:
            raise HTTPException(status_code=400, detail="Unknown problem_id for run_finished event.")
        problem_state = job.problems[data.problem_id]
        try:
            problem_state.mark_run_finished(data.run_index, data.payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job.mark_running()
        return

    if event_name == "job_finished":
        job.mark_done()
        return

    if event_name == "job_failed":
        job.mark_failed(data.reason, data.partial)
        return

    # Other events are simply recorded; no state mutation.
    if job.state == "pending":
        job.mark_running()


@app.post("/jobs/{job_id}/events")
async def post_job_event(
    job_id: str = PathParam(..., description="Identifier returned by the job creation endpoint."),
    payload: JobEventRequest = ...,
) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _get_job(job_id)
        _handle_job_event(job, payload)
        status = job.status_payload(include_results=False)
    return {"status": "ok", "job": status}


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _get_job(job_id)
        include_results = job.state in {"done", "failed"}
        status = job.status_payload(include_results=include_results)
    return status


@app.get("/jobs/{job_id}/events")
async def get_job_events(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _get_job(job_id)
        events = list(job.events)
    return {"job_id": job.job_id, "events": events}


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        job = _get_job(job_id)
        results = job.last_results or job.collect_results()
        payload = {"job_id": job.job_id, "runs": results}
        if job.state == "failed" and job.failure_reason:
            payload["failure_reason"] = job.failure_reason
            if job.failure_partial is not None:
                payload["partial"] = job.failure_partial
    return payload


def load_runs(path: Optional[str] = None) -> None:
    """Utility invoked from scripts to preload run history."""

    if not path:
        return
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                _append_payload(payload)


if __name__ == "__main__":
    # Convenience entrypoint: python Echo/server.py --port 8098
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the mock Echo callback server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--load", help="Optional JSONL file to preload run history.")
    args = parser.parse_args()

    load_runs(args.load)
    uvicorn.run("Echo.server:app", host=args.host, port=args.port, reload=False)

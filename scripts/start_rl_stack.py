#!/usr/bin/env python3
"""
Convenience launcher for the RL service + Echo callback server combo.

The script:
  1. Starts ``service_api`` (FastAPI) via uvicorn.
  2. Starts the Echo callback server.
  3. Waits for both to become available.
  4. Optionally runs a lightweight smoke check to confirm the expected problem
     is available and that the Echo job lifecycle API accepts requests.

Example:
    ./scripts/start_rl_stack.py --problem-id container_kill-analysis-1

Use Ctrl+C to stop both services; the script will clean up the subprocesses.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional

import requests


DEFAULT_HOST = "127.0.0.1"
DEFAULT_SERVICE_PORT = 8099
DEFAULT_ECHO_PORT = 8098


@dataclass
class ProcessHandle:
    name: str
    popen: subprocess.Popen

    def terminate(self) -> None:
        if self.popen.poll() is not None:
            return
        self.popen.send_signal(signal.SIGINT)
        try:
            self.popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.popen.kill()


def _start_process(cmd: Iterable[str], *, name: str) -> ProcessHandle:
    popen = subprocess.Popen(list(cmd))
    return ProcessHandle(name=name, popen=popen)


def _wait_for_endpoint(url: str, *, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                return
        except Exception:
            time.sleep(0.5)
        else:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def _smoke_service(service_base: str, problem_id: Optional[str]) -> None:
    print("[smoke] Checking service_api /health ...")
    health = requests.get(f"{service_base}/health", timeout=5)
    health.raise_for_status()
    print(f"[smoke] health: {health.json()}")

    print("[smoke] Fetching available problems ...")
    problems_resp = requests.get(f"{service_base}/problems", timeout=10)
    problems_resp.raise_for_status()
    problems = problems_resp.json().get("problems", [])
    print(f"[smoke] problems returned: {len(problems)}")
    if problem_id and problem_id not in problems:
        print(
            f"[smoke][warn] problem '{problem_id}' not reported by service_api. "
            "Double-check the orchestrator assets.",
            file=sys.stderr,
        )

    if not problem_id:
        return

    payload = {
        "problem_id": problem_id,
        "max_steps": 2,
    }
    print(f"[smoke] Attempting /rl/reset for problem '{problem_id}' ...")
    reset_resp = requests.post(f"{service_base}/rl/reset", json=payload, timeout=30)
    reset_resp.raise_for_status()
    obs = reset_resp.json().get("observation", {})
    print(f"[smoke] reset ok; problem state snippet: {json.dumps(obs)[:160]}...")


def _smoke_echo(echo_base: str) -> None:
    print("[smoke] Creating dummy Echo job ...")
    create_resp = requests.post(
        f"{echo_base}/jobs",
        json={"problems": [{"id": "smoke-problem", "runs": 1}]},
        timeout=10,
    )
    create_resp.raise_for_status()
    job_id = create_resp.json().get("job_id")
    if not job_id:
        raise RuntimeError("Echo server did not return job_id during smoke test.")
    print(f"[smoke] echo job id: {job_id}")

    payload = {"event": "job_finished"}
    post_resp = requests.post(
        f"{echo_base}/jobs/{job_id}/events",
        json=payload,
        timeout=10,
    )
    post_resp.raise_for_status()
    print("[smoke] Echo event API responded OK.")


def _local_rl_check(problem_id: str, *, max_steps: int = 2) -> None:
    """Run a minimal ProblemRLEnvironment episode locally via service helpers."""
    try:
        import service  # Local module import
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[env-check][warn] Cannot import service module: {exc}", file=sys.stderr)
        return

    print(f"[env-check] Resetting local ProblemRLEnvironment for '{problem_id}' ...")
    handle = None
    try:
        handle = service.reset_rl_environment(problem_id, max_steps=max_steps)
        observation, info = service.get_rl_environment_state(handle.env_id)
        actions_left = observation.get("actions_left") if isinstance(observation, dict) else None
        print(f"[env-check] observation.actions_left={actions_left}")
        actions_catalog = info.get("actions") if isinstance(info, dict) else {}
        action_text = None
        if isinstance(actions_catalog, dict):
            if "exec_shell" in actions_catalog:
                action_text = '```python\nexec_shell("kubectl get pods -A")\n```'
            elif actions_catalog:
                first_api = next(iter(actions_catalog))
                action_text = f'```python\n{first_api}()\n```'

        if action_text:
            print(f"[env-check] Executing test action: {action_text.strip()}")
            step_result = service.step_rl_environment(
                handle.env_id,
                step=1,
                action=action_text,
                llm_response=action_text,
                llm_raw_response=action_text,
            )
            print(f"[env-check] step reward={step_result.reward}, actions_left={step_result.actions_left}")
        else:
            print("[env-check][warn] No suitable test action found; skipping step execution.")
    except Exception as exc:
        print(f"[env-check][warn] Problem environment check failed: {exc}", file=sys.stderr)
    finally:
        if handle is not None:
            try:
                service.close_rl_environment(handle.env_id)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick launcher for RL service + Echo callback server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host/interface for uvicorn listeners.")
    parser.add_argument("--service-port", type=int, default=DEFAULT_SERVICE_PORT, help="Port for service_api FastAPI.")
    parser.add_argument("--echo-port", type=int, default=DEFAULT_ECHO_PORT, help="Port for Echo callback server.")
    parser.add_argument(
        "--problem-id",
        help="Optional problem id to verify via /rl/reset smoke call.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke checks (only launch services).",
    )
    parser.add_argument(
        "--skip-echo",
        action="store_true",
        help="Do not launch the Echo callback server.",
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Print environment snippet (useful when running inside tmux/screen).",
    )
    parser.add_argument(
        "--local-env-check",
        action="store_true",
        help="Run a local ProblemRLEnvironment reset/step (requires --problem-id).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.env:
        print("[info] Selected environment variables:")
        for key in ("AIOPSLAB_GROUND_TRUTH_DIR", "OPENROUTER_API_KEY"):
            if key in os.environ:
                print(f"  {key}={os.environ[key]}")

    service_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "service_api:app",
        "--host",
        args.host,
        "--port",
        str(args.service_port),
    ]
    echo_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "Echo.server:app",
        "--host",
        args.host,
        "--port",
        str(args.echo_port),
    ]

    handles: list[ProcessHandle] = []
    try:
        print("[launch] Starting service_api ...")
        handles.append(_start_process(service_cmd, name="service_api"))
        _wait_for_endpoint(f"http://{args.host}:{args.service_port}/health")
        print(f"[ready] service_api at http://{args.host}:{args.service_port}")

        if not args.skip_echo:
            print("[launch] Starting Echo server ...")
            handles.append(_start_process(echo_cmd, name="echo_server"))
            _wait_for_endpoint(f"http://{args.host}:{args.echo_port}/callbacks/runs?limit=1")
            print(f"[ready] Echo server at http://{args.host}:{args.echo_port}")
        else:
            print("[info] Skipping Echo server launch.")

        if not args.skip_smoke:
            service_base = f"http://{args.host}:{args.service_port}"
            echo_base = f"http://{args.host}:{args.echo_port}"
            try:
                _smoke_service(service_base, args.problem_id)
            except Exception as exc:
                print(f"[smoke][warn] service_api smoke check failed: {exc}", file=sys.stderr)
            if not args.skip_echo:
                try:
                    _smoke_echo(echo_base)
                except Exception as exc:
                    print(f"[smoke][warn] Echo smoke check failed: {exc}", file=sys.stderr)

        if args.local_env_check:
            if not args.problem_id:
                print("[env-check][warn] --local-env-check requested but --problem-id missing.", file=sys.stderr)
            else:
                _local_rl_check(args.problem_id)

        print("[info] Servers running. Press Ctrl+C to stop.")
        while all(handle.popen.poll() is None for handle in handles):
            time.sleep(1.0)
        return 0
    except KeyboardInterrupt:
        print("\n[info] Ctrl+C received, stopping services ...")
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        for handle in handles:
            handle.terminate()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import requests


def _start_process(cmd: Iterable[str], *, cwd: Optional[Path] = None) -> subprocess.Popen:
    return subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
    )


def _wait_for_endpoint(url: str, timeout: float = 30.0) -> None:
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


def _terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch end-to-end RL rollout with mock Echo callback capture."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host used for local services.")
    parser.add_argument("--service-port", type=int, default=8099, help="Port for service_api.")
    parser.add_argument("--echo-port", type=int, default=8098, help="Port for Echo callback server.")
    parser.add_argument(
        "--chat-url",
        default="http://14.103.221.215:18200/v1/chat/completions",
        help="Chat completion endpoint used by the mock agent.",
    )
    parser.add_argument(
        "--chat-model",
        default="/data0/xj/lunwen/verl/save_model/new_model_save_vllm-GPTQ-Int4-detail",
        help="Model identifier to send to the chat completion endpoint.",
    )
    parser.add_argument(
        "--problem-id",
        default="container_kill-analysis-1",
        help="Orchestrator problem identifier for the rollout.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=8, help="Maximum number of turns for the rollout."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where callback payload snapshots will be saved.",
    )
    parser.add_argument(
        "--agent-script",
        type=Path,
        default=Path("scripts/mock_echo_agent.py"),
        help="Path to the driving agent script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip launching the agent and just verify services come up.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    print("Starting service_api...")
    service_proc = _start_process(service_cmd)
    try:
        _wait_for_endpoint(f"http://{args.host}:{args.service_port}/health")
        print("service_api is ready.")
    except Exception:
        _terminate_process(service_proc, "service_api")
        raise

    print("Starting Echo callback server...")
    echo_proc = _start_process(echo_cmd)
    try:
        _wait_for_endpoint(f"http://{args.host}:{args.echo_port}/callbacks/runs?limit=1")
        print("Echo callback server is ready.")
    except Exception:
        _terminate_process(echo_proc, "echo_server")
        _terminate_process(service_proc, "service_api")
        raise

    exit_code = 0
    try:
        job_id: str | None = None
        if not args.dry_run:
            job_id = f"harness-{int(time.time())}"
            agent_cmd = [
                sys.executable,
                str(args.agent_script),
                "--base",
                f"http://{args.host}:{args.service_port}",
                "--problem-id",
                args.problem_id,
                "--max-steps",
                str(args.max_steps),
                "--chat-url",
                args.chat_url,
                "--chat-model",
                args.chat_model,
                "--echo-url",
                f"http://{args.host}:{args.echo_port}",
                "--job-id",
                job_id,
            ]
            print("Running mock Echo agent...")
            result = subprocess.run(agent_cmd, check=False)
            exit_code = result.returncode
            if exit_code != 0:
                print(f"mock_echo_agent exited with {exit_code}", file=sys.stderr)
        else:
            print("Dry run requested; skipping agent execution.")

        args.log_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = args.log_dir / f"echo_job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            base = f"http://{args.host}:{args.echo_port}"
            job_id_lookup = job_id if not args.dry_run else None
            snapshot: dict[str, object] = {}
            if job_id_lookup:
                status_resp = requests.get(f"{base}/jobs/{job_id_lookup}/status", timeout=10)
                results_resp = requests.get(f"{base}/jobs/{job_id_lookup}/results", timeout=10)
                events_resp = requests.get(f"{base}/jobs/{job_id_lookup}/events", timeout=10)
                status_resp.raise_for_status()
                results_resp.raise_for_status()
                events_resp.raise_for_status()
                snapshot = {
                    "job_id": job_id_lookup,
                    "status": status_resp.json(),
                    "results": results_resp.json(),
                    "events": events_resp.json(),
                }
            else:
                snapshot = {"message": "dry-run; no job executed"}
        except Exception as exc:
            snapshot = {"error": f"Failed to fetch Echo job data: {exc}"}

        with snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        print(f"Saved Echo job snapshot to {snapshot_path}")
    finally:
        _terminate_process(echo_proc, "echo_server")
        _terminate_process(service_proc, "service_api")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

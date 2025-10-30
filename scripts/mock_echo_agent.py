#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests


def call_chat_api(url: str, model: str, system: str, user: str, *,
                  temperature: float = 0.2, max_tokens: int = 800,
                  api_key: str | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock Echo agent driving RL FastAPI with GPT")
    p.add_argument("--base", default="http://127.0.0.1:8099", help="service_api base URL")
    p.add_argument("--problem-id", default="container_kill-analysis-1")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.2)
    # OpenAI-compatible endpoint (Echo/vLLM, OpenRouter, etc.)
    p.add_argument("--chat-url", default=os.getenv("CHAT_URL", "http://14.103.221.215:18200/v1/chat/completions"))
    p.add_argument(
        "--chat-model",
        default=os.getenv(
            "CHAT_MODEL",
            "/data0/xj/lunwen/verl/save_model/new_model_save_vllm-GPTQ-Int4-detail",
        ),
    )
    p.add_argument(
        "--echo-url",
        help="Optional Echo server URL (base like http://127.0.0.1:8098 or legacy /callbacks/runs endpoint).",
    )
    p.add_argument("--job-id", default=None, help="Identifier used when posting to the Echo callback server.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # API key optional for self-hosted endpoints
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = logs_dir / f"mock_echo_{ts}.log"
    jsonl_path = logs_dir / f"mock_echo_{ts}.jsonl"

    def log(line: str) -> None:
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def req(method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{args.base}{path}"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(payload).encode() if payload is not None else None
        r = requests.request(method, url, data=data, headers=headers, timeout=600)
        r.raise_for_status()
        resp_json = r.json()
        # append full round-trip to JSONL
        with jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps({
                "ts": dt.datetime.utcnow().isoformat() + "Z",
                "url": url,
                "method": method,
                "request": payload,
                "response": resp_json,
                "status": r.status_code,
            }, ensure_ascii=False) + "\n")
        return resp_json

    system_prompt = (
        "You are an Echo-like SRE agent operating a Kubernetes cluster.\n"
        "You may ONLY respond with ONE API call in a Python fenced code block.\n"
        "Do NOT include chain-of-thought or <think> text.\n"
        "Available APIs:\n- exec_shell(\"<kubectl-or-shell>\")\n- submit({json_solution})\n\n"
        "Guidelines: Investigate systematically (pods -> logs -> services),\n"
        "then submit the diagnosis as soon as confident."
    )

    # 1) reset
    reset_payload = {
        "problem_id": args.problem_id,
        "max_steps": args.max_steps,
        "reward_config": {"command_match_multiplier": 0.1},
        "ground_truth_dir": str(Path(__file__).resolve().parents[1] / "ground_truth"),
    }
    log("=== RESET ===")
    reset = req("POST", "/rl/reset", reset_payload)
    env_id = reset.get("env_id", "")
    log(f"env_id: {env_id}")
    # Log reset observation + info fully
    log("RESET observation: " + json.dumps(reset.get("observation", {}), ensure_ascii=False))
    log("RESET info: " + json.dumps(reset.get("info", {}), ensure_ascii=False))

    trajectory: list[Dict[str, Any]] = [{
        "step": 0,
        "observation": reset.get("observation"),
        "info": reset.get("info"),
    }]
    total_reward = 0.0

    # Step 0
    log("=== STEP 0 ===")
    step0_payload = {"step": 0}
    log("STEP0 request: " + json.dumps(step0_payload, ensure_ascii=False))
    step0 = req("POST", f"/rl/{env_id}/step", step0_payload)
    log("STEP0 response: " + json.dumps(step0, ensure_ascii=False))
    state = step0.get("state") or ""
    actions_left = step0.get("actions_left", 0)
    total_reward += float(step0.get("reward", 0.0) or 0.0)
    done_flag = bool(step0.get("info", {}).get("environment", {}).get("done", False))
    log(f"actions_left: {actions_left}")
    trajectory.append({
        "step": 0,
        "state": state,
        "actions_left": actions_left,
        "actions": step0.get("actions"),
        "reward": step0.get("reward"),
        "info": step0.get("info"),
    })

    for step in range(1, args.max_steps + 1):
        log(f"=== STEP {step} ===")
        user_prompt = (
            f"Actions left: {actions_left}\n\nObservation:\n{state}\n\n"
            "Return exactly one API call wrapped in ```python code fence."
        )
        try:
            reply = call_chat_api(
                url=args.chat_url,
                model=(args.chat_model or args.model),
                system=system_prompt,
                user=user_prompt,
                temperature=args.temp,
                max_tokens=800,
                api_key=api_key,
            )
        except Exception as e:
            log(f"Chat endpoint error: {e}")
            return 3

        # Execute (log full payload)
        payload = {"step": step, "action": reply, "llm_response": reply, "llm_raw_response": reply}
        log("STEP request: " + json.dumps(payload, ensure_ascii=False))
        resp = req("POST", f"/rl/{env_id}/step", payload)
        log("STEP response: " + json.dumps(resp, ensure_ascii=False))
        reward = resp.get("reward", 0.0)
        done = resp.get("info", {}).get("environment", {}).get("done", False)
        done_flag = done_flag or bool(done)
        log(f"reward: {reward} done: {done}")
        total_reward += float(reward or 0.0)
        trajectory.append({
            "step": step,
            "action": reply,
            "state": resp.get("state"),
            "actions_left": resp.get("actions_left"),
            "reward": reward,
            "info": resp.get("info"),
            "llm_response": reply,
            "llm_raw_response": reply,
        })

        # Prepare next
        state = resp.get("state") or ""
        actions_left = resp.get("actions_left", 0)
        if done:
            break
        time.sleep(0.5)

    log(f"Saved log: {log_path}")

    if args.echo_url:
        echo_url = args.echo_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        try:
            if echo_url.endswith("/callbacks/runs"):
                # Legacy callback endpoint
                callback_payload = {
                    "job_id": args.job_id or f"mock-{env_id}",
                    "problem_id": args.problem_id,
                    "run_index": 0,
                    "env_id": env_id,
                    "total_reward": total_reward,
                    "done": bool(actions_left == 0 or done_flag),
                    "trajectory": trajectory,
                }
                log(f"Posting rollout to legacy Echo endpoint: {echo_url}")
                response = requests.post(echo_url, json=callback_payload, headers=headers, timeout=30)
                response.raise_for_status()
                log(f"Echo callback response: {response.status_code} {response.text}")
            else:
                base = echo_url
                job_id = args.job_id
                if not job_id:
                    create_payload = {"problems": [{"id": args.problem_id, "runs": 1}]}
                    log(f"Creating Echo job via {base}/jobs")
                    create_resp = requests.post(f"{base}/jobs", json=create_payload, headers=headers, timeout=30)
                    create_resp.raise_for_status()
                    job_id = create_resp.json().get("job_id")
                    if not job_id:
                        raise RuntimeError("Echo job creation response missing job_id.")
                    log(f"Created Echo job {job_id}")

                run_payload = {
                    "run_id": f"{env_id}-0",
                    "env_id": env_id,
                    "total_reward": total_reward,
                    "done": bool(actions_left == 0 or done_flag),
                    "steps": trajectory,
                }
                event_endpoint = f"{base}/jobs/{job_id}/events"
                run_started = {
                    "event": "run_started",
                    "problem_id": args.problem_id,
                    "run_index": 0,
                }
                run_finished = {
                    "event": "run_finished",
                    "problem_id": args.problem_id,
                    "run_index": 0,
                    "payload": run_payload,
                }
                job_finished = {"event": "job_finished"}

                log(f"Posting run events to Echo job {job_id} at {event_endpoint}")
                for payload in (run_started, run_finished, job_finished):
                    resp = requests.post(event_endpoint, json=payload, headers=headers, timeout=30)
                    resp.raise_for_status()
                log("Echo job events posted successfully.")
        except Exception as exc:
            log(f"Failed to notify Echo server: {exc}")
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

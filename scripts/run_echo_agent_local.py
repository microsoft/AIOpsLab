#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import requests

FAKE_NAMESPACE = "test-hotel-reservation"
FAKE_PODS = [
    {"name": "frontend-7c7fdc5cc8-abcde", "service": "frontend", "status": "Running"},
    {"name": "geo-7bf89d945f-fghij", "service": "geo", "status": "Running"},
    {"name": "profile-5f6d8c9bc-jklmn", "service": "profile", "status": "Running"},
]
FAKE_SERVICES = [
    {"name": "frontend", "cluster_ip": "10.0.0.10"},
    {"name": "geo", "cluster_ip": "10.0.0.11"},
    {"name": "profile", "cluster_ip": "10.0.0.12"},
]
FALLBACK_ACTIONS = [
    'exec_shell("kubectl get pods -n test-hotel-reservation")',
    'exec_shell("kubectl logs geo-7bf89d945f-fghij -n test-hotel-reservation")',
    'exec_shell("kubectl get pods -n test-hotel-reservation")',
    'submit({"system_level": "Application", "fault_type": "Dependency Problem"})',
]
_FALLBACK_INDEX = 0


def call_chat_api(url: str, model: str, system: str, user: str, *,
                  temperature: float = 0.2, max_tokens: int = 512,
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
    print("[chat] POST payload:", json.dumps(payload, ensure_ascii=False))
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.RequestException as exc:
        global _FALLBACK_INDEX
        print(f"[chat] request failed ({exc}); using fallback action.")
        action = FALLBACK_ACTIONS[min(_FALLBACK_INDEX, len(FALLBACK_ACTIONS) - 1)]
        _FALLBACK_INDEX += 1
        return f"```python\n{action}\n```"


def apply_dry_run_stubs() -> None:
    """Monkeypatch heavy external dependencies to no-op for local runs."""
    # Docker
    try:
        import aiopslab.service.dock as dock_mod

        class _NoopDocker:
            def compose_up(self, *a, **k):
                return ""
            def compose_down(self, *a, **k):
                return ""
            def cleanup(self, *a, **k):
                return ""
            def exec_command(self, *a, **k):
                return ""

        dock_mod.Docker = _NoopDocker  # type: ignore[attr-defined]
    except Exception:
        pass

    # KubeCtl
    try:
        import aiopslab.service.kubectl as kube_mod

        class _NoopKubeCtl:
            def __init__(self):
                from types import SimpleNamespace

                self._ns = {FAKE_NAMESPACE, "chaos-mesh", "openebs", "default"}
                self._configmaps: dict[tuple[str, str], dict] = {}
                self._pods = [
                    SimpleNamespace(
                        metadata=SimpleNamespace(name=pod["name"]),
                        status=SimpleNamespace(
                            container_statuses=[SimpleNamespace(ready=True)]
                        ),
                    )
                    for pod in FAKE_PODS
                ]
                self._services = {
                    svc["name"]: svc for svc in FAKE_SERVICES
                }

            # --- Introspection helpers -------------------------------------------------
            def _pod_table(self) -> str:
                header = "NAME READY STATUS RESTARTS AGE"
                rows = [
                    f"{pod['name']} 1/1 {pod['status']} 0 5m"
                    for pod in FAKE_PODS
                ]
                return "\n".join([header, *rows])

            # --- Core interface --------------------------------------------------------
            def exec_command(self, command: str, *_, **__) -> str:
                cmd = (command or "").strip()
                base = cmd.split("|", 1)[0].strip()

                if base.startswith("kubectl get namespace"):
                    parts = base.split()
                    namespace = parts[-1]
                    if namespace in self._ns:
                        return f"Name: {namespace}"
                    return f"Error from server (NotFound): namespaces \"{namespace}\" not found"

                if base.startswith("kubectl create namespace"):
                    ns = base.split()[-1]
                    self._ns.add(ns)
                    return f"namespace/{ns} created"

                if base.startswith("kubectl delete namespace"):
                    ns = base.split()[-1]
                    self._ns.discard(ns)
                    return f"namespace \"{ns}\" deleted"

                if "kubectl get pods" in base and FAKE_NAMESPACE in base:
                    return self._pod_table()

                if base.startswith("kubectl apply"):
                    return "configuration applied"

                if base.startswith("kubectl delete -f"):
                    return "configuration deleted"

                if "kubectl get pv" in base:
                    return ""

                if base.startswith("kubectl patch pv"):
                    return "patched"

                if base.startswith("kubectl delete pv"):
                    pv = base.split()[-1]
                    return f"persistentvolume \"{pv}\" deleted"

                if base.startswith("kubectl get service"):
                    parts = base.split()
                    name = parts[3]
                    svc = self._services.get(name)
                    if not svc:
                        return ""
                    return json.dumps(
                        {
                            "metadata": {"name": svc["name"]},
                            "spec": {"clusterIP": svc["cluster_ip"]},
                        }
                    )

                return ""

            def wait_for_ready(self, namespace: str, *_, **__) -> None:
                if namespace not in self._ns:
                    raise RuntimeError(f"Namespace {namespace} not found.")
                return None

            def delete_namespace(self, namespace: str, *_, **__) -> str:
                self._ns.discard(namespace)
                return ""

            def get_node_architectures(self):
                return ["arm64"]

            def get_cluster_ip(self, service_name: str, *_, **__) -> str:
                return self._services.get(service_name, {}).get("cluster_ip", "127.0.0.1")

            def create_or_update_configmap(self, name: str, namespace: str, data: dict):
                self._configmaps[(namespace, name)] = dict(data)
                return None

            def create_configmap(self, name: str, namespace: str, data: dict):
                return self.create_or_update_configmap(name, namespace, data)

            def apply_configs(self, namespace: str, *_a, **_k):
                self._ns.add(namespace)
                return None

            def delete_configs(self, namespace: str, *_a, **_k):
                return None

            def create_namespace_if_not_exist(self, namespace: str, *_, **__):
                self._ns.add(namespace)
                return None

            def get_container_runtime(self, *_, **__) -> str:
                return "containerd://1.6.0"

            def list_pods(self, namespace: str):
                from types import SimpleNamespace

                if namespace not in self._ns:
                    return SimpleNamespace(items=[])
                return SimpleNamespace(items=list(self._pods))

            def list_services(self, namespace: str):
                from types import SimpleNamespace

                services = [
                    SimpleNamespace(
                        metadata=SimpleNamespace(name=svc["name"]),
                        spec=SimpleNamespace(cluster_ip=svc["cluster_ip"]),
                    )
                    for svc in FAKE_SERVICES
                ]
                return SimpleNamespace(items=services)

        kube_mod.KubeCtl = _NoopKubeCtl  # type: ignore[attr-defined]
    except Exception:
        pass

    # Prometheus
    try:
        import aiopslab.service.telemetry.prometheus as prom_mod

        class _NoopProm:
            def __init__(self):
                pass
            def deploy(self):
                return None

        prom_mod.Prometheus = _NoopProm  # type: ignore[attr-defined]
    except Exception:
        pass

    # Fault injector (virtual/noop)
    try:
        import aiopslab.generators.fault.inject_virtual as inj_mod

        class _NoopInj:
            def __init__(self, *a, **k):
                pass
            def _inject(self, *a, **k):
                return None
            def _recover(self, *a, **k):
                return None

        inj_mod.VirtualizationFaultInjector = _NoopInj  # type: ignore[attr-defined]
    except Exception:
        pass

    # Application deploy/delete no-op at base level
    try:
        import aiopslab.service.apps.base as app_base

        def _noop(self, *a, **k):
            return None

        app_base.Application.deploy = _noop  # type: ignore[attr-defined]
        app_base.Application.delete = _noop  # type: ignore[attr-defined]
        app_base.Application.cleanup = _noop  # type: ignore[attr-defined]
    except Exception:
        pass

    # Helm operations no-op
    try:
        import aiopslab.service.helm as helm_mod

        class _NoopHelm:
            @staticmethod
            def install(**kwargs):
                return None
            @staticmethod
            def uninstall(**kwargs):
                return None
            @staticmethod
            def exists_release(*a, **k):
                return False
            @staticmethod
            def assert_if_deployed(namespace: str):
                return True
            @staticmethod
            def add_repo(*args, **kwargs):
                return None

        helm_mod.Helm.install = _NoopHelm.install  # type: ignore[attr-defined]
        helm_mod.Helm.uninstall = _NoopHelm.uninstall  # type: ignore[attr-defined]
        helm_mod.Helm.exists_release = _NoopHelm.exists_release  # type: ignore[attr-defined]
        helm_mod.Helm.assert_if_deployed = _NoopHelm.assert_if_deployed  # type: ignore[attr-defined]
        helm_mod.Helm.add_repo = _NoopHelm.add_repo  # type: ignore[attr-defined]
    except Exception:
        pass

    # Workload generator no-op
    try:
        import aiopslab.generators.workload.wrk as wrk_mod

        class _NoopWrk:
            def __init__(self, *a, **k):
                pass
            def start_workload(self, *a, **k):
                return None

        wrk_mod.Wrk = _NoopWrk  # type: ignore[attr-defined]
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local RL env with Echo endpoint as policy")
    p.add_argument("--problem-id", default="noop_detection_social_network-1")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--chat-url", default=os.getenv("CHAT_URL", "http://14.103.221.215:18200/v1/chat/completions"))
    p.add_argument("--chat-model", default=os.getenv("CHAT_MODEL", "/data0/xj/lunwen/verl/save_model/new_model_save_vllm-GPTQ-Int4-detail"))
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--ground-truth-dir", default=str(Path("ground_truth").resolve()))
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Stub external systems (Docker/K8s/Prometheus)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        apply_dry_run_stubs()

    # Build RL environment in-process
    from aiopslab.orchestrator.rl_env import ProblemRLEnvironment, RewardConfig

    env = ProblemRLEnvironment(
        max_steps=args.max_steps,
        reward_config=RewardConfig(command_match_multiplier=0.1),
        ground_truth_dir=args.ground_truth_dir,
    )

    system_prompt = (
        "You are an Echo-like SRE agent operating a Kubernetes cluster.\n"
        "You may ONLY respond with ONE API call in a Python fenced code block.\n"
        "Do NOT include chain-of-thought or <think> text.\n"
        "Available APIs:\n- exec_shell(\"<kubectl-or-shell>\")\n- submit({json_solution})\n\n"
        "Guidelines: Investigate systematically (pods -> logs -> services),\n"
        "then submit the diagnosis as soon as confident."
    )

    obs, info = env.reset(args.problem_id)
    state = obs.get("state", "")

    for step in range(1, args.max_steps + 1):
        user_prompt = (
            f"Actions left: {obs.get('actions_left', 0)}\n\nObservation:\n{state}\n\n"
            "Return exactly one API call wrapped in ```python code fence."
        )
        reply = call_chat_api(
            url=args.chat_url,
            model=args.chat_model,
            system=system_prompt,
            user=user_prompt,
            temperature=args.temperature,
            max_tokens=512,
        )
        obs, reward, done, info = env.step(reply)
        print(json.dumps({
            "step": step,
            "action": reply,
            "reward": reward,
            "done": done,
        }, ensure_ascii=False))
        state = obs.get("state", "")
        if done:
            break
        time.sleep(0.5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

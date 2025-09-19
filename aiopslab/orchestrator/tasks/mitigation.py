# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define and query information about an AIOps Mitigation task."""

import textwrap
import time
from typing import Any, Callable, Dict, Iterable

from aiopslab.orchestrator.actions.mitigation import MitigationActions
from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.tasks.variant_task import VariantProblemMixin
from aiopslab.service.apps.base import Application
from aiopslab.session import SessionItem
from aiopslab.utils.actions import get_actions
from aiopslab.utils.status import InvalidActionError


class MitigationTask(Task):
    """An AIOps anomaly mitigation task."""

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.helm_configs = self.app.helm_configs
        self.app_summary = self.app.get_app_summary()
        self.actions = MitigationActions()

        self.task_desc = """\
            You are an expert DevOps engineer assigned to mitigate anomalies in a deployed service.

            Service Details:
            {app_summary}

            You will begin by analyzing the service's state and telemetry, and then submit a solution that mitigates any detected anomalies.
            Your mitigation can be performed using any of the available APIs.
            """

        self.instructions = """\
            You will respond with one of the above APIs as your next action.
            Please respond in the following format in a markdown code block:
            ```\n<API_NAME>(<API_PARAM1>, <API_PARAM2> ...)\n```

            For instance, if you want to list files in current directory, your response must be exactly:
            
            ```\nexec_shell("ls -l")\n```

            Once your solution is complete and ready for evaluation, you must call:
            
            ```\nsubmit()\n```

            Note:
            - The submit() call for the mitigation task does not take any parameters.
            - A submission via submit() is considered valid if it is made, though this does not necessarily indicate that your solution is correct.

            Please respond with only a single API call (a.k.a., action) per turn without any additional words, labels, or prefixes.
            """

    def get_task_description(self):
        return textwrap.dedent(self.task_desc).format(app_summary=self.app_summary)

    def get_instructions(self):
        return textwrap.dedent(self.instructions)

    def get_available_actions(self):
        return get_actions(task="mitigation")

    def perform_action(self, action_name, *args, **kwargs):
        action_method = getattr(self.actions, action_name, None)

        if action_method is not None and callable(action_method):
            return action_method(*args, **kwargs)
        else:
            raise InvalidActionError(action_name)

    def _check_pods_ready(self, config: Any) -> bool:
        if isinstance(config, bool):
            return config
        if callable(config):
            return bool(config())
        if not isinstance(config, dict):
            return bool(config)

        namespaces = config.get("namespaces")
        if namespaces is None:
            namespace = config.get("namespace", getattr(self, "namespace", None))
            namespaces = [namespace] if namespace else []
        elif isinstance(namespaces, str):
            namespaces = [namespaces]
        else:
            namespaces = [ns for ns in namespaces if ns]

        if not namespaces:
            return True

        selectors = config.get("services") or config.get("names") or []
        if isinstance(selectors, str):
            selectors = [selectors]
        selector_set = {str(item) for item in selectors}

        timeout = config.get("timeout", 60)
        interval = config.get("interval", 5)
        include_all = config.get("include_all", not selector_set)

        deadline = time.time() + timeout

        while True:
            all_ready = True
            matched_total: set[str] = set()

            for namespace in namespaces:
                try:
                    pod_list = self.kubectl.list_pods(namespace)
                except Exception:
                    all_ready = False
                    break

                items = getattr(pod_list, "items", []) or []
                namespace_ready, matched = self._pods_ready_in_namespace(
                    items, selector_set, include_all
                )
                matched_total.update(matched)
                if not namespace_ready:
                    all_ready = False
                    break

            if all_ready and (not selector_set or selector_set.issubset(matched_total)):
                return True

            if time.time() >= deadline:
                return False

            time.sleep(interval)

    def _pods_ready_in_namespace(
        self,
        pods: Iterable[Any],
        selectors: set[str],
        include_all: bool,
    ) -> tuple[bool, set[str]]:
        matched: set[str] = set()

        if include_all or not selectors:
            for pod in pods:
                if not self._is_pod_ready(pod):
                    return False, matched
            return True, matched

        for selector in selectors:
            selector_ready = False
            for pod in pods:
                name = getattr(getattr(pod, "metadata", None), "name", "")
                if selector in name:
                    matched.add(selector)
                    if self._is_pod_ready(pod):
                        selector_ready = True
                    else:
                        return False, matched
            if not selector_ready:
                return False, matched

        return True, matched

    def _is_pod_ready(self, pod: Any) -> bool:
        status = getattr(pod, "status", None)
        container_statuses = getattr(status, "container_statuses", None)
        if not container_statuses:
            return False

        for container_status in container_statuses:
            state = getattr(container_status, "state", None)
            waiting = getattr(state, "waiting", None)
            if waiting and getattr(waiting, "reason", "") in {
                "CrashLoopBackOff",
                "Error",
                "ImagePullBackOff",
                "ErrImagePull",
            }:
                return False

            terminated = getattr(state, "terminated", None)
            if terminated and getattr(terminated, "reason", "") != "Completed":
                return False

            if not getattr(container_status, "ready", False):
                return False

        return True

    def _check_deployments(self, config: Any) -> bool:
        if isinstance(config, bool):
            return config
        if callable(config):
            return bool(config())

        deployments = config
        if isinstance(deployments, dict):
            deployments = [deployments]

        success = True

        for spec in deployments or []:
            if not isinstance(spec, dict):
                continue

            name = spec.get("name")
            namespace = spec.get("namespace", getattr(self, "namespace", None))
            expected = spec.get("expected_replicas")
            expected_available = spec.get("available_replicas", expected)

            if not name or not namespace:
                success = False
                break

            try:
                deployment = self.kubectl.get_deployment(name, namespace)
            except Exception:
                success = False
                break

            desired = getattr(getattr(deployment, "spec", None), "replicas", None)
            available = getattr(getattr(deployment, "status", None), "available_replicas", None)

            if expected is not None and desired != expected:
                success = False
                break

            if expected_available is not None and available != expected_available:
                success = False
                break

        return success

    def _run_callable_check(self, config: Any) -> bool:
        if config is None:
            return True
        if isinstance(config, (list, tuple, set)):
            return all(self._run_callable_check(item) for item in config)
        if callable(config):
            return bool(config())
        if isinstance(config, dict) and "callable" in config:
            func = config.get("callable")
            args = config.get("args", [])
            kwargs = config.get("kwargs", {})
            if callable(func):
                return bool(func(*args, **kwargs))
            return False
        return bool(config)

    def evaluate_variant_mitigation(self):
        if not isinstance(self, VariantProblemMixin):
            return None

        if "success" in self.results:
            return self.results.get("success")

        expectations = self.get_mitigation_expectations()
        if not expectations:
            return None

        check_results: Dict[str, bool] = {}

        if "pods_ready" in expectations:
            check_results["pods_ready"] = self._check_pods_ready(expectations["pods_ready"])

        if "deployments" in expectations:
            check_results["deployments"] = self._check_deployments(expectations["deployments"])

        if "workloads_resumed" in expectations:
            check_results["workloads_resumed"] = self._run_callable_check(
                expectations["workloads_resumed"]
            )

        if "cr_restored" in expectations:
            check_results["cr_restored"] = self._run_callable_check(expectations["cr_restored"])

        if "custom" in expectations:
            check_results["custom"] = self._run_callable_check(expectations["custom"])

        for check_name, passed in check_results.items():
            self.add_result(f"mitigation_{check_name}", passed)

        if check_results:
            success = all(check_results.values())
            self.results["success"] = success
            return success

        return None

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self.evaluate_variant_mitigation()
        self.add_result("TTM", duration)
        self.common_eval(trace)
        return self.results

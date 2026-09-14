# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define and query information about an AIOps Mitigation task."""

import os
import textwrap
import time
from typing import Any

from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.mitigation import MitigationActions
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

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self.add_result("TTM", duration)
        self.common_eval(trace)
        return self.results

    # ------------------------------------------------------------------
    # Recovery-health helpers shared by mitigation problems.
    #
    # Mitigation eval inspects live pod state. Done as a single snapshot the
    # instant submit() is called, it penalizes correct fixes whose effect is
    # not yet visible: app-level faults (e.g. database auth) leave dependent
    # pods in CrashLoopBackOff, whose exponential back-off (10s, 20s, 40s, ...)
    # routinely outlasts the moment of submission. Allowing a bounded settle
    # window makes eval reflect whether the system actually recovered, instead
    # of whether it happened to be healthy at one instant.
    # ------------------------------------------------------------------
    def pods_unhealthy_reasons(self, namespace: str) -> list[str]:
        """Return human-readable reasons for any unhealthy container in `namespace`.

        An empty list means every container is healthy (no CrashLoopBackOff,
        no abnormal termination, all ready).
        """
        pod_list = self.kubectl.list_pods(namespace)
        reasons: list[str] = []
        for pod in pod_list.items:
            if not pod.status.container_statuses:
                continue
            for container_status in pod.status.container_statuses:
                waiting = container_status.state.waiting
                terminated = container_status.state.terminated
                if waiting and waiting.reason == "CrashLoopBackOff":
                    reasons.append(
                        f"Container {container_status.name} is in CrashLoopBackOff"
                    )
                elif terminated and terminated.reason != "Completed":
                    reasons.append(
                        f"Container {container_status.name} is terminated with "
                        f"reason: {terminated.reason}"
                    )
                elif not container_status.ready:
                    reasons.append(f"Container {container_status.name} is not ready")
        return reasons

    def wait_until_pods_healthy(
        self,
        namespace: str,
        timeout: float | None = None,
        poll_interval: float = 5.0,
    ) -> bool:
        """Poll until all pods in `namespace` are healthy, or `timeout` elapses.

        Returns True as soon as the namespace is healthy. On timeout, prints the
        outstanding reasons (preserving the previous eval's diagnostic output)
        and returns False. The window is configurable via the
        ``AIOPSLAB_MITIGATION_SETTLE_SECONDS`` env var (default 120s); set it to
        0 to keep the old single-snapshot behavior.
        """
        if timeout is None:
            timeout = float(os.getenv("AIOPSLAB_MITIGATION_SETTLE_SECONDS", "120"))
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            reasons = self.pods_unhealthy_reasons(namespace)
            if not reasons:
                return True
            if time.monotonic() >= deadline:
                for reason in reasons:
                    print(reason)
                return False
            time.sleep(poll_interval)

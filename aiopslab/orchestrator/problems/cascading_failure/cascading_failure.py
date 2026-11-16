# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hub-service failure in the Astronomy Shop, with a measured blast radius.

One hub service (checkout, product-catalog or cart) is made unavailable with a
Chaos Mesh pod-failure. Besides the usual detection and localization tasks,
the blast-radius task asks the agent which other services lost successful
requests because of the failure. The answer is scored against a set measured
from the OpenTelemetry span metrics: every service whose rate of successful
requests (server and consumer spans) fell by at least `threshold` between the
baseline window before the fault and the fault window.
"""

import os
import socket
import subprocess
import textwrap
import time
from contextlib import contextmanager
from typing import Any

from aiopslab.generators.fault.inject_symp import SymptomFaultInjector
from aiopslab.orchestrator.evaluators.quantitative import *
from aiopslab.orchestrator.tasks import *
from aiopslab.service.apps.astronomy_shop import AstronomyShop
from aiopslab.service.kubectl import KubeCtl
from aiopslab.session import SessionItem
from aiopslab.utils.blast_radius import (
    SpanMetricsQuery,
    blast_radius,
    rate_drops,
    score_submission,
)

EXPERIMENT_NAME = "cascading-failure"


@contextmanager
def prometheus_url(namespace: str, service: str = "prometheus", port: int = 9090):
    """Yield a URL for the app's Prometheus.

    Uses AIOPSLAB_PROMETHEUS_URL when it is set (for example when AIOpsLab
    runs inside the cluster); otherwise port-forwards the service.
    """
    url = os.environ.get("AIOPSLAB_PROMETHEUS_URL")
    if url:
        yield url
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        local = s.getsockname()[1]
    proc = subprocess.Popen(
        ["kubectl", "port-forward", f"svc/{service}", f"{local}:{port}", "-n", namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", local)) == 0:
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"port-forward to {service} in {namespace} did not open")
        yield f"http://127.0.0.1:{local}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class CascadingFailureBaseTask:
    # Label the OpenTelemetry Demo chart puts on each component's pods.
    POD_LABEL = "opentelemetry.io/name"

    def __init__(self, faulty_service: str, threshold: float = 0.2):
        self.app = AstronomyShop()
        self.kubectl = KubeCtl()
        self.namespace = self.app.namespace
        self.faulty_service = faulty_service
        self.threshold = threshold
        self.baseline_seconds = 180
        self.settle_seconds = 30
        self.min_fault_seconds = 90
        # A service needs this many expected requests in the fault window to be
        # measured; fewer make a 20% drop indistinguishable from noise.
        self.min_expected_requests = 20
        self.injector = SymptomFaultInjector(namespace=self.namespace)
        self.baseline_window = None
        self.injected_at = None

    def start_workload(self):
        print("== Start Workload ==")
        print("Workload skipped since AstronomyShop has a built-in load generator.")

    def inject_fault(self):
        print("== Fault Injection ==")
        end = time.time()
        self.baseline_window = (end - self.baseline_seconds, end)
        experiment = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {"name": EXPERIMENT_NAME, "namespace": self.namespace},
            "spec": {
                "action": "pod-failure",
                "mode": "all",
                "duration": "1h",
                "selector": {
                    "namespaces": [self.namespace],
                    "labelSelectors": {self.POD_LABEL: self.faulty_service},
                },
            },
        }
        self.injector.create_chaos_experiment(experiment, EXPERIMENT_NAME)
        self.injected_at = time.time()
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self.injector.delete_chaos_experiment(EXPERIMENT_NAME)

    def measure_blast_radius(self) -> tuple[set[str], dict[str, float]]:
        """Measured set of affected services and each service's drop."""
        if self.injected_at is None:
            raise RuntimeError("inject_fault() must run before measuring")
        start = self.injected_at + self.settle_seconds
        wait = start + self.min_fault_seconds - time.time()
        if wait > 0:
            time.sleep(wait)
        end = time.time()
        with prometheus_url(self.namespace) as url:
            query = SpanMetricsQuery(url)
            before = query.success_rates(*self.baseline_window)
            during = query.success_rates(start, end)
        drops = rate_drops(before, during, min_rate=self.min_expected_requests / (end - start))
        return blast_radius(drops, self.faulty_service, self.threshold), drops


################## Detection Problem ##################
class CascadingFailureDetection(CascadingFailureBaseTask, DetectionTask):
    def __init__(self, faulty_service: str):
        CascadingFailureBaseTask.__init__(self, faulty_service)
        DetectionTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        if isinstance(soln, str) and soln.strip().lower() == "yes":
            self.add_result("Detection Accuracy", "Correct")
        elif isinstance(soln, str):
            self.add_result("Detection Accuracy", "Incorrect")
        else:
            self.add_result("Detection Accuracy", "Invalid Format")
        return super().eval(soln, trace, duration)


################## Localization Problem ##################
class CascadingFailureLocalization(CascadingFailureBaseTask, LocalizationTask):
    def __init__(self, faulty_service: str):
        CascadingFailureBaseTask.__init__(self, faulty_service)
        LocalizationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        if soln is None:
            self.add_result("Localization Accuracy", 0.0)
            self.results["success"] = False
            self.results["is_subset"] = False
            super().eval(soln, trace, duration)
            return self.results

        is_exact = is_exact_match(soln, self.faulty_service)
        is_sub = is_subset([self.faulty_service], soln)
        if is_exact:
            accuracy = 100.0
        elif is_sub:
            accuracy = 100.0 / len(soln)
        else:
            accuracy = 0.0
        self.add_result("Localization Accuracy", accuracy)
        super().eval(soln, trace, duration)
        self.results["success"] = is_exact or (is_sub and len(soln) == 1)
        self.results["is_subset"] = is_sub
        return self.results


################## Blast Radius Problem ##################
class CascadingFailureBlastRadius(CascadingFailureBaseTask, LocalizationTask):
    def __init__(self, faulty_service: str):
        CascadingFailureBaseTask.__init__(self, faulty_service)
        LocalizationTask.__init__(self, self.app)
        self.task_desc = """\
            You are an expert DevOps engineer assigned to assess the impact of a fault in a deployed service.

            Service Details:
            {app_summary}

            One service has failed. Identify every service whose rate of successful requests
            dropped by {percent}% or more while the fault was active, because of the failure.
            You will begin by analyzing the service's state and telemetry, and then submit:
            1. list[str]: the affected services (i.e., service names), including the failed one if you know it
            2. list[]: an empty list if no service was affected
            """

    def get_task_description(self):
        return textwrap.dedent(self.task_desc).format(
            app_summary=self.app_summary, percent=round(self.threshold * 100)
        )

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        truth, drops = self.measure_blast_radius()
        self.results["measured_blast_radius"] = sorted(truth)
        self.results["measured_drops"] = drops
        submitted = soln if isinstance(soln, list) else []
        scores = score_submission(submitted, truth, self.faulty_service)
        self.add_result("Blast Radius Precision", scores["precision"])
        self.add_result("Blast Radius Recall", scores["recall"])
        self.add_result("Blast Radius F1", scores["f1"])
        self.add_result("Blast Radius Jaccard", scores["jaccard"])
        self.add_result("Names Failed Service", scores["names_failed_service"])
        super().eval(soln, trace, duration)
        self.results["success"] = isinstance(soln, list) and scores["exact_match"]
        return self.results

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the mitigation recovery settle-poll (MitigationTask)."""

import unittest
from types import SimpleNamespace as NS

from aiopslab.orchestrator.tasks.mitigation import MitigationTask


def _container(name, ready=True, waiting=None, terminated=None):
    state = NS(
        waiting=NS(reason=waiting) if waiting else None,
        terminated=NS(reason=terminated) if terminated else None,
    )
    return NS(name=name, ready=ready, state=state)


def _pod(*containers, name="pod", phase="Running"):
    return NS(
        metadata=NS(name=name),
        status=NS(phase=phase, container_statuses=list(containers) or None),
    )


def _pod_list(*pods):
    return NS(items=list(pods))


def _workloads(replicas):
    return NS(
        items=[
            NS(metadata=NS(name=name), spec=NS(replicas=count))
            for name, count in replicas.items()
        ]
    )


class FakeKubeCtl:
    """Returns a queued sequence of pod-list snapshots (last one repeats)."""

    def __init__(self, snapshots, deployments=None, stateful_sets=None):
        self.snapshots = snapshots
        self.calls = 0
        self.deployments = dict(deployments or {})
        self.stateful_sets = dict(stateful_sets or {})

    def list_pods(self, namespace):
        snap = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
        self.calls += 1
        return snap

    def list_deployments(self, namespace):
        return _workloads(self.deployments)

    def list_stateful_sets(self, namespace):
        return _workloads(self.stateful_sets)


def _make_task(snapshots, deployments=None, stateful_sets=None):
    task = MitigationTask.__new__(MitigationTask)  # bypass Application-dependent __init__
    task.kubectl = FakeKubeCtl(snapshots, deployments, stateful_sets)
    task.app = NS(namespace="ns")
    return task


class TestMitigationSettle(unittest.TestCase):
    def test_healthy_namespace_has_no_reasons(self):
        task = _make_task([_pod_list(_pod(_container("a")), _pod(_container("b")))])
        self.assertEqual(task.pods_unhealthy_reasons("ns"), [])
        self.assertTrue(task.wait_until_pods_healthy("ns", timeout=0))

    def test_crashloop_terminated_and_notready_are_flagged(self):
        task = _make_task(
            [
                _pod_list(
                    _pod(_container("crash", ready=False, waiting="CrashLoopBackOff")),
                    _pod(_container("term", ready=False, terminated="Error")),
                    _pod(_container("pending", ready=False)),
                    _pod(_container("ok")),
                )
            ]
        )
        reasons = task.pods_unhealthy_reasons("ns")
        self.assertEqual(len(reasons), 3)
        self.assertTrue(any("CrashLoopBackOff" in r for r in reasons))
        self.assertTrue(any("terminated" in r for r in reasons))
        self.assertTrue(any("not ready" in r for r in reasons))

    def test_completed_container_is_not_abnormal(self):
        # Preserves the original eval semantics: a "Completed" termination is not
        # an abnormal termination.
        task = _make_task(
            [_pod_list(_pod(_container("job", ready=True, terminated="Completed")))]
        )
        self.assertEqual(task.pods_unhealthy_reasons("ns"), [])

    def test_pod_without_container_statuses_is_unhealthy_unless_succeeded(self):
        # A pod that cannot be scheduled stays Pending with no container statuses.
        task = _make_task(
            [
                _pod_list(
                    _pod(name="user-service", phase="Pending"),
                    _pod(name="job", phase="Succeeded"),
                )
            ]
        )
        reasons = task.pods_unhealthy_reasons("ns")
        self.assertEqual(len(reasons), 1)
        self.assertIn("user-service", reasons[0])
        self.assertFalse(task.wait_until_pods_healthy("ns", timeout=0))

    def test_waits_until_dependent_pod_recovers(self):
        # First snapshot crash-looping, second healthy -> polling should succeed.
        unhealthy = _pod_list(_pod(_container("rate", ready=False, waiting="CrashLoopBackOff")))
        healthy = _pod_list(_pod(_container("rate")))
        task = _make_task([unhealthy, healthy])
        self.assertTrue(task.wait_until_pods_healthy("ns", timeout=10, poll_interval=0))
        self.assertGreaterEqual(task.kubectl.calls, 2)

    def test_returns_false_when_never_recovers(self):
        task = _make_task([_pod_list(_pod(_container("rate", ready=False, waiting="CrashLoopBackOff")))])
        self.assertFalse(task.wait_until_pods_healthy("ns", timeout=0))


class TestDeployedWorkloads(unittest.TestCase):
    HEALTHY = _pod_list(_pod(_container("frontend")))

    def test_deleted_and_scaled_to_zero_workloads_are_flagged(self):
        # Removing the broken workload leaves no unhealthy pod behind.
        task = _make_task(
            [self.HEALTHY],
            deployments={"frontend": 1, "geo": 1, "rate": 2},
            stateful_sets={"mongodb": 1},
        )
        task.record_deployed_workloads()
        del task.kubectl.deployments["geo"]
        task.kubectl.deployments["rate"] = 0
        reasons = task.workloads_unhealthy_reasons("ns")
        self.assertEqual(len(reasons), 2)
        self.assertTrue(any("geo" in r and "no longer exists" in r for r in reasons))
        self.assertTrue(any("rate" in r and "scaled to zero" in r for r in reasons))
        self.assertEqual(task.pods_unhealthy_reasons("ns"), [])
        self.assertFalse(task.wait_until_pods_healthy("ns", timeout=0))

    def test_restored_workload_passes(self):
        task = _make_task([self.HEALTHY], deployments={"geo": 1})
        task.record_deployed_workloads()
        task.kubectl.deployments["geo"] = 0
        self.assertFalse(task.wait_until_pods_healthy("ns", timeout=0))
        task.kubectl.deployments["geo"] = 1
        self.assertTrue(task.wait_until_pods_healthy("ns", timeout=0))

    def test_workload_deployed_with_zero_replicas_is_not_required(self):
        task = _make_task([self.HEALTHY], deployments={"standby": 0})
        task.record_deployed_workloads()
        self.assertEqual(task.workloads_unhealthy_reasons("ns"), [])

    def test_without_a_record_only_pods_are_checked(self):
        task = _make_task([self.HEALTHY])
        self.assertEqual(task.workloads_unhealthy_reasons("ns"), [])
        self.assertTrue(task.wait_until_pods_healthy("ns", timeout=0))


class TestKafkaQueueProblemsMitigation(unittest.TestCase):
    @staticmethod
    def _task(variant):
        from aiopslab.orchestrator.problems.kafka_queue_problems.kafka_queue_problems import (
            KafkaQueueProblemsMitigation,
        )

        task = KafkaQueueProblemsMitigation.__new__(KafkaQueueProblemsMitigation)
        task.results = {}
        task.namespace = "astronomy-shop"
        task.kubectl = FakeKubeCtl([_pod_list(_pod(_container("kafka")))])
        task.injector = NS(get_flag_variant=lambda flag: variant)
        return task

    def test_flag_still_on_fails_with_healthy_pods(self):
        # The fault is a feature flag that leaves every pod healthy.
        self.assertFalse(self._task("on").eval(None, [], 1.0)["success"])

    def test_flag_off_and_healthy_pods_passes(self):
        self.assertTrue(self._task("off").eval(None, [], 1.0)["success"])


if __name__ == "__main__":
    unittest.main()

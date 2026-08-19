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


def _pod(*containers):
    return NS(status=NS(container_statuses=list(containers) or None))


def _pod_list(*pods):
    return NS(items=list(pods))


class FakeKubeCtl:
    """Returns a queued sequence of pod-list snapshots (last one repeats)."""

    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = 0

    def list_pods(self, namespace):
        snap = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
        self.calls += 1
        return snap


def _make_task(snapshots):
    task = MitigationTask.__new__(MitigationTask)  # bypass Application-dependent __init__
    task.kubectl = FakeKubeCtl(snapshots)
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

    def test_completed_container_and_empty_statuses_are_not_abnormal(self):
        # Preserves the original eval semantics: a "Completed" termination is not
        # an abnormal termination, and pods without container statuses are skipped.
        task = _make_task(
            [_pod_list(_pod(_container("job", ready=True, terminated="Completed")), _pod())]
        )
        self.assertEqual(task.pods_unhealthy_reasons("ns"), [])

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


if __name__ == "__main__":
    unittest.main()

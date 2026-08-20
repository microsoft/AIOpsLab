# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests that TaskActions.exec_shell forwards its timeout to Shell.exec."""

import unittest
from unittest.mock import patch

from aiopslab.orchestrator.actions.base import TaskActions


class TestExecShellTimeoutForwarding(unittest.TestCase):
    def test_default_timeout_is_forwarded(self):
        with patch(
            "aiopslab.orchestrator.actions.base.Shell.exec", return_value="ok"
        ) as mock_exec:
            TaskActions.exec_shell("echo hi")
        mock_exec.assert_called_once_with("echo hi", timeout=30)

    def test_custom_timeout_is_forwarded(self):
        with patch(
            "aiopslab.orchestrator.actions.base.Shell.exec", return_value="ok"
        ) as mock_exec:
            TaskActions.exec_shell("kubectl rollout status deploy/x", timeout=90)
        mock_exec.assert_called_once_with(
            "kubectl rollout status deploy/x", timeout=90
        )

    def test_blocked_command_does_not_reach_shell(self):
        with patch(
            "aiopslab.orchestrator.actions.base.Shell.exec", return_value="ok"
        ) as mock_exec:
            out = TaskActions.exec_shell("kubectl edit svc/foo", timeout=90)
        mock_exec.assert_not_called()
        self.assertIn("kubectl patch", out)


if __name__ == "__main__":
    unittest.main()

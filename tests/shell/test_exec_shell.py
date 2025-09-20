# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import time
from aiopslab.service.shell import Shell


class TestExecShell(unittest.TestCase):
    def test_echo(self):
        command = "echo 'Hello, World!'"
        output = Shell.exec(command)
        self.assertEqual(output, "Hello, World!\n")

    def test_kubectl_pods(self):
        command = "kubectl get pods -n test-social-network"
        output = Shell.exec(command)
        self.assertTrue("compose-post-service" in output)

    def test_kubectl_services(self):
        command = "kubectl get services -n test-social-network"
        output = Shell.exec(command)
        self.assertTrue("compose-post-service" in output)

    def test_patch(self):
        command = 'kubectl patch svc user-service -n test-social-network --type=\'json\' -p=\'[{"op": "replace", "path": "/spec/ports/0/targetPort", "value": 9090}]\''
        output = Shell.exec(command)

        command = "kubectl get svc user-service -n test-social-network -o jsonpath='{.spec.ports[0].targetPort}'"
        output = Shell.exec(command)
        self.assertEqual(output, "9090")

    def test_timeout_parameter(self):
        """Test that timeout parameter is accepted and works for quick commands."""
        command = "echo 'Timeout test'"
        output = Shell.exec(command, timeout=5)
        self.assertEqual(output, "Timeout test\n")

    def test_local_exec_timeout(self):
        """Test that local_exec properly handles timeout for quick commands."""
        command = "echo 'Local exec test'"
        output = Shell.local_exec(command, timeout=5)
        self.assertEqual(output, "Local exec test\n")

    def test_timeout_behavior(self):
        """Test that timeout actually interrupts long-running commands."""
        start_time = time.time()
        with self.assertRaises(RuntimeError):
            # This should timeout after 1 second, not complete after 3 seconds
            Shell.local_exec("sleep 3", timeout=1)
        
        elapsed_time = time.time() - start_time
        # Should timeout around 1 second, allow some margin for system overhead
        self.assertLess(elapsed_time, 2.0, "Command should have timed out within ~1 second")


if __name__ == "__main__":
    unittest.main()

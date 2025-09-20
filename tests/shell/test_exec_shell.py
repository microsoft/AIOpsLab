# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch

from aiopslab.service.shell import Shell


class TestExecShell(unittest.TestCase):
    def test_echo(self):
        command = "echo 'Hello, World!'"
        with patch.object(Shell, "docker_exec", return_value="Hello, World!\n") as mocked_exec:
            output = Shell.exec(command)
            mocked_exec.assert_called_once_with("kind-control-plane", command)
        self.assertEqual(output, "Hello, World!\n")

    def test_kubectl_pods(self):
        command = "kubectl get pods -n test-social-network"
        fake_output = "compose-post-service   1/1     Running"
        with patch.object(Shell, "docker_exec", return_value=fake_output) as mocked_exec:
            output = Shell.exec(command)
            mocked_exec.assert_called_once_with("kind-control-plane", command)
        self.assertTrue("compose-post-service" in output)

    def test_kubectl_services(self):
        command = "kubectl get services -n test-social-network"
        fake_output = "compose-post-service   ClusterIP"
        with patch.object(Shell, "docker_exec", return_value=fake_output) as mocked_exec:
            output = Shell.exec(command)
            mocked_exec.assert_called_once_with("kind-control-plane", command)
        self.assertTrue("compose-post-service" in output)

    def test_patch(self):
        command = 'kubectl patch svc user-service -n test-social-network --type=\'json\' -p=\'[{"op": "replace", "path": "/spec/ports/0/targetPort", "value": 9090}]\''
        with patch.object(Shell, "docker_exec", side_effect=["", "9090"]) as mocked_exec:
            Shell.exec(command)

            command = "kubectl get svc user-service -n test-social-network -o jsonpath='{.spec.ports[0].targetPort}'"
            output = Shell.exec(command)
            self.assertEqual(
                mocked_exec.call_args_list[1][0], ("kind-control-plane", command)
            )
        self.assertEqual(output, "9090")


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest

from aiopslab.service.apps.socialnet import SocialNetwork
from aiopslab.orchestrator.parser import ResponseParser
from aiopslab.orchestrator.tasks.detection import DetectionTask


class TestParser(unittest.TestCase):
    def setUp(self):
        self.app = SocialNetwork()
        self.task = DetectionTask(self.app)
        self.parser = ResponseParser()

    def test_non_shell(self):
        input = """
        Action:
        ```
        get_logs('user-service', 'test-social-network')
        ```
        """
        resp = self.parser.parse(input)
        api = resp["api_name"]
        args = resp["args"]
        self.assertEqual(api, "get_logs")
        self.assertEqual(args, ["user-service", "test-social-network"])
        resp = self.task.perform_action(api, *args)
        self.assertTrue(resp)

    def test_shell(self):
        input = """
        Action:
        ```
        exec_shell('echo "Hello World"')
        ```
        """
        resp = self.parser.parse(input)
        api = resp["api_name"]
        args = resp["args"]
        self.assertEqual(api, "exec_shell")
        self.assertEqual(args, ['echo "Hello World"'])
        resp = self.task.perform_action(api, *args)
        self.assertTrue(resp)


if __name__ == "__main__":
    unittest.main()

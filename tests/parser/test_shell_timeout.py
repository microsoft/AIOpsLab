# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests that exec_shell honors an optional timeout argument."""

import unittest

from aiopslab.orchestrator.parser import ResponseParser


def _block(call: str) -> str:
    return f"Action:\n```\n{call}\n```"


class TestShellTimeout(unittest.TestCase):
    def setUp(self):
        self.parser = ResponseParser()

    def test_shell_without_timeout(self):
        resp = self.parser.parse(_block('exec_shell("kubectl get pods")'))
        self.assertEqual(resp["args"], ["kubectl get pods"])
        self.assertEqual(resp["kwargs"], {})

    def test_shell_positional_timeout(self):
        resp = self.parser.parse(_block('exec_shell("kubectl rollout status deploy/x", 90)'))
        self.assertEqual(resp["args"], ["kubectl rollout status deploy/x"])
        self.assertEqual(resp["kwargs"], {"timeout": 90})

    def test_shell_keyword_timeout(self):
        resp = self.parser.parse(_block('exec_shell("sleep 5", timeout=120)'))
        self.assertEqual(resp["args"], ["sleep 5"])
        self.assertEqual(resp["kwargs"], {"timeout": 120})

    def test_trailing_number_inside_command_is_not_a_timeout(self):
        # The command itself ends in ", 90"; without a closing quote after it,
        # it must stay part of the command, not be parsed as a timeout.
        resp = self.parser.parse(_block('exec_shell("echo a, 90")'))
        self.assertEqual(resp["args"], ["echo a, 90"])
        self.assertEqual(resp["kwargs"], {})

    def test_single_quoted_command_still_parses(self):
        resp = self.parser.parse(_block("exec_shell('echo hello', 30)"))
        self.assertEqual(resp["args"], ["echo hello"])
        self.assertEqual(resp["kwargs"], {"timeout": 30})


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from aiopslab.utils.actions import get_actions


class TestGetActions(unittest.TestCase):
    def test_get_actions(self):
        actions = get_actions("detection")
        self.assertEqual(len(actions), 7)
        self.assertEqual(
            set(actions.keys()),
            {
                "get_logs",
                "get_metrics",
                "read_metrics",
                "get_traces",
                "read_traces",
                "exec_shell",
                "submit",
            },
        )

    def test_get_read_actions(self):
        actions = get_actions("detection", "read")
        self.assertEqual(len(actions), 5)
        self.assertEqual(
            set(actions.keys()),
            {
                "get_logs",
                "get_metrics",
                "read_metrics",
                "get_traces",
                "read_traces",
            },
        )

    def test_get_write_actions(self):
        actions = get_actions("detection", "write")
        self.assertEqual(len(actions), 0)


if __name__ == "__main__":
    unittest.main()

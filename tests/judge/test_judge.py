# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch, MagicMock
from aiopslab.orchestrator.evaluators.qualitative import LLMJudge
from aiopslab.session import SessionItem


class TestLLMJudge(unittest.TestCase):
    def setUp(self):
        self.trace = [
            SessionItem(role="user", content="Hello"),
            SessionItem(role="assistant", content="Hi there!"),
        ]
        self.judge = LLMJudge(trace=self.trace)

    @patch("aiopslab.orchestrator.evaluators.qualitative.OpenAI")
    def test_reasoning_score(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Rating: [[5]]"))
        ]
        result = self.judge.reasoning_score()
        # print(self.judge.prompt)
        self.assertEqual(result, (5, "Rating: [[5]]"))

    def test_format_trace(self):
        expected_trace = "###user:\nHello\n\n###assistant:\nHi there!\n\n"
        self.assertEqual(self.judge.trace, expected_trace)


if __name__ == "__main__":
    unittest.main()

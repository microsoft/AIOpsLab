# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests that submit() stores the same solution for positional and keyword arguments."""

import asyncio
import unittest
from types import SimpleNamespace as NS

from aiopslab.onboarding_evaluator import Evaluator
from aiopslab.orchestrator.actions.analysis import AnalysisActions
from aiopslab.orchestrator.actions.detection import DetectionActions
from aiopslab.orchestrator.actions.localization import LocalizationActions
from aiopslab.orchestrator.actions.mitigation import MitigationActions
from aiopslab.orchestrator.onboarding_eval_parser import EvalParser
from aiopslab.orchestrator.orchestrator import Orchestrator
from aiopslab.orchestrator.parser import ResponseParser
from aiopslab.utils.status import SubmissionStatus

ANALYSIS = {"system_level": "Application", "fault_type": "Misconfiguration"}

CASES = [
    (DetectionActions, 'submit("Yes")', 'submit(has_anomaly="Yes")', "Yes"),
    (
        LocalizationActions,
        'submit(["user-service"])',
        'submit(faulty_components=["user-service"])',
        ["user-service"],
    ),
    (
        AnalysisActions,
        f"submit({ANALYSIS})",
        f"submit(analysis={ANALYSIS})",
        ANALYSIS,
    ),
]


class FakeSession:
    """Records the solution; actions and eval come from the test."""

    def __init__(self, actions, expected=None):
        self.solution = None
        self.history = []
        self.start_time = 0.0
        self.problem = NS(
            actions=actions,
            perform_action=lambda name, *args, **kwargs: getattr(actions, name)(
                *args, **kwargs
            ),
            eval=lambda soln, trace, duration: {"success": soln == expected},
        )

    def set_solution(self, solution):
        self.solution = solution

    def add(self, item):
        self.history.append(item)


def _run(runner_cls, parser, actions, call, expected=None):
    # Bypass the cluster-dependent __init__.
    runner = runner_cls.__new__(runner_cls)
    runner.parser = parser
    runner.session = FakeSession(actions, expected)
    response = asyncio.run(runner.ask_env(f"```\n{call}\n```"))
    return response, runner.session.solution


class TestSubmitSolution(unittest.TestCase):
    def test_orchestrator_stores_positional_and_keyword_calls_alike(self):
        for actions, positional, keyword, expected in CASES:
            for call in (positional, keyword):
                with self.subTest(call=call):
                    response, solution = _run(
                        Orchestrator, ResponseParser(), actions, call
                    )
                    self.assertEqual(response, SubmissionStatus.VALID_SUBMISSION)
                    self.assertEqual(solution, expected)

    def test_orchestrator_mitigation_submit_without_arguments(self):
        response, solution = _run(
            Orchestrator, ResponseParser(), MitigationActions, "submit()"
        )
        self.assertEqual(response, SubmissionStatus.VALID_SUBMISSION)
        self.assertEqual(solution, [])

    def test_orchestrator_unknown_keyword_is_not_taken_as_the_answer(self):
        response, solution = _run(
            Orchestrator, ResponseParser(), DetectionActions, 'submit(answer="Yes")'
        )
        self.assertNotEqual(response, SubmissionStatus.VALID_SUBMISSION)
        self.assertEqual(solution, [])

    def test_onboarding_evaluator_accepts_keyword_calls(self):
        for actions, positional, keyword, expected in CASES:
            for call in (positional, keyword):
                with self.subTest(call=call):
                    response, solution = _run(
                        Evaluator, EvalParser(), actions, call, expected
                    )
                    self.assertEqual(response, SubmissionStatus.VALID_SUBMISSION)
                    self.assertEqual(solution, expected)

    def test_onboarding_evaluator_rejects_unknown_keyword(self):
        # The onboarding evaluator grades the stored solution without calling the
        # action, so a keyword that submit() does not declare must not be accepted.
        response, solution = _run(
            Evaluator, EvalParser(), DetectionActions, 'submit(answer="Yes")', "Yes"
        )
        self.assertEqual(response, SubmissionStatus.INVALID_SUBMISSION)
        self.assertEqual(solution, [])


if __name__ == "__main__":
    unittest.main()

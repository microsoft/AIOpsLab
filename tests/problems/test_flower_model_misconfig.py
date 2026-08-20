import unittest

from aiopslab.orchestrator.problems.flower_model_misconfig.model_misconfig import (
    FlowerModelMisconfigBaseTask,
)


class FlowerModelMisconfigTimeoutTest(unittest.TestCase):
    def test_wait_until_returns_when_predicate_succeeds(self):
        task = object.__new__(FlowerModelMisconfigBaseTask)

        task._wait_until("ready", lambda: True, timeout_seconds=1, interval_seconds=0)

    def test_wait_until_times_out_when_predicate_never_succeeds(self):
        task = object.__new__(FlowerModelMisconfigBaseTask)

        with self.assertRaises(TimeoutError):
            task._wait_until("ready", lambda: False, timeout_seconds=0, interval_seconds=0)

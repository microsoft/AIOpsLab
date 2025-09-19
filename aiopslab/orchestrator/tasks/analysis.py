# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define and query information about an AIOps Root Cause Analysis task."""

import textwrap
from typing import Any

from aiopslab.orchestrator.actions.analysis import AnalysisActions
from aiopslab.orchestrator.evaluators.quantitative import is_exact_match_lower
from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.tasks.variant_task import VariantProblemMixin
from aiopslab.service.apps.base import Application
from aiopslab.session import SessionItem
from aiopslab.utils.actions import get_actions
from aiopslab.utils.status import InvalidActionError


class AnalysisTask(Task):
    """An AIOps root cause analysis task."""

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.helm_configs = self.app.helm_configs
        self.app_summary = self.app.get_app_summary()
        self.actions = AnalysisActions()

        self.task_desc = """\
            You are an expert DevOps engineer assigned to do root cause analysis in a deployed service.

            Service Details:
            {app_summary}

            You will begin by analyzing the service's state and telemetry, and then submit one of two possible solutions:
            1. dict[str, str]: A dictionary with two keys: 'system_level' and 'fault_type'.
                - system_level: The system level at which the fault occurred. Please choose from the following options:
                    - 'Hardware'
                    - 'Operating System'
                    - 'Virtualization'
                    - 'Application'
                - fault_type: The type of fault that occurred. Please choose from the following options:
                    - 'Misconfiguration'
                    - 'Code Defect'
                    - 'Authentication Issue'
                    - 'Network/Storage Issue'
                    - 'Operation Error'
                    - 'Dependency Problem'
            
            2. str: `None` if no faults were detected
            """

        self.instructions = """\
            You will respond with one of the above APIs as your next action.
            Please respond in the following format in a markdown code block:
            ```\n<API_NAME>(<API_PARAM1>, <API_PARAM2> ...)\n```

            For instance, if you want to list files in current directory, your response must be exactly:
            
            ```\nexec_shell("ls -l")\n```

            When submitting your analysis, use the following format:

            ```\nsubmit({"system_level": "your_system_level_analysis", "fault_type": "your_fault_type_analysis"})\n```
            
            Replace "your_system_level_analysis" and "your_fault_type_analysis" with the actual analysis of the system level and fault type.

            Or, if no fault is detected, you should respond with:

            ```\nsubmit()\n```

            Please respond with only a single API call (a.k.a., action) per turn without any additional words, labels, or prefixes.
            """

    def get_task_description(self):
        return textwrap.dedent(self.task_desc).format(app_summary=self.app_summary)

    def get_instructions(self):
        return textwrap.dedent(self.instructions)

    def get_available_actions(self):
        return get_actions(task="analysis")

    def perform_action(self, action_name, *args, **kwargs):
        action_method = getattr(self.actions, action_name, None)

        if action_method is not None and callable(action_method):
            return action_method(*args, **kwargs)
        else:
            raise InvalidActionError(action_name)

    def evaluate_variant_analysis(self, soln: Any):
        """Evaluate the solution against the active variant expectations."""
        if not isinstance(self, VariantProblemMixin):
            return None

        if any(
            key in self.results
            for key in ("system_level_correct", "fault_type_correct", "success")
        ):
            return self.results.get("success")

        expected_system_level = self.get_expected_system_level()
        expected_fault_type = self.get_expected_fault_type()

        if expected_system_level is None and expected_fault_type is None:
            return None

        if not isinstance(soln, dict):
            if expected_system_level is not None:
                self.results["system_level_correct"] = False
            if expected_fault_type is not None:
                self.results["fault_type_correct"] = False
            self.results["success"] = False
            return False

        system_level_correct = True
        fault_type_correct = True

        if expected_system_level is not None:
            system_level_correct = is_exact_match_lower(
                soln.get("system_level", ""), expected_system_level
            )
            self.results["system_level_correct"] = system_level_correct

        if expected_fault_type is not None:
            fault_type_correct = is_exact_match_lower(
                soln.get("fault_type", ""), expected_fault_type
            )
            self.results["fault_type_correct"] = fault_type_correct

        if expected_system_level is not None or expected_fault_type is not None:
            self.results["success"] = system_level_correct and fault_type_correct

        return self.results.get("success")

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self.evaluate_variant_analysis(soln)
        self.add_result("TTA", duration)
        self.common_eval(trace)
        return self.results

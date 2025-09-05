# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define and query information about an AIOps Localization task."""

import textwrap
from typing import Any


from aiopslab.orchestrator.tasks.base import Task
from aiopslab.orchestrator.actions.localization import LocalizationActions
from aiopslab.service.apps.base import Application
from aiopslab.session import SessionItem
from aiopslab.utils.actions import get_actions
from aiopslab.utils.status import InvalidActionError


class LocalizationTask(Task):
    """An AIOps fault localization task."""

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.helm_configs = self.app.helm_configs
        self.app_summary = self.app.get_app_summary()
        self.actions = LocalizationActions()

        self.task_desc = """\
            You are an expert DevOps engineer assigned to localize faults in a deployed service.

            Service Details:
            {app_summary}

            You are requested to identify the service(s) where the root cause of the fault lies.
            You will begin by analyzing the service's state and telemetry, and then submit one of two possible solutions:
            1. list[str]: list of faulty components (i.e., service names)
            2. list[]: an empty list if no faults were detected
            """

        self.instructions = """\
            You will respond with one of the above APIs as your next action.
            Please respond in the following format in a markdown code block:
            ```\n<API_NAME>(<API_PARAM1>, <API_PARAM2> ...)\n```

            For instance, if you want to list files in current directory, your response must be exactly:
            
            ```\nexec_shell("ls -l")\n```

            If there are faulty components to submit:

            ```\nsubmit([\"placeholderA\", \"placeholderB\"])\n```

            (where "placeholderA" and "placeholderB" are placeholders; replace them with the faulty components in your environment)
            
            Or, if no faults are found:

            ```\nsubmit([])\n```

            Please respond with only a single API call (a.k.a., action) per turn without any additional words, labels, or prefixes.
            """

    def get_task_description(self):
        return textwrap.dedent(self.task_desc).format(app_summary=self.app_summary)

    def get_instructions(self):
        return textwrap.dedent(self.instructions)

    def get_available_actions(self):
        return get_actions(task="localization")

    def perform_action(self, action_name, *args, **kwargs):
        action_method = getattr(self.actions, action_name, None)

        if action_method is not None and callable(action_method):
            return action_method(*args, **kwargs)
        else:
            raise InvalidActionError(action_name)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self.add_result("TTL", duration)
        self.common_eval(trace)
        return self.results

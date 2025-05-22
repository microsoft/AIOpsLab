# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Actions for the detection task."""

from aiopslab.orchestrator.actions.base import TaskActions
from aiopslab.utils.actions import action
from aiopslab.utils.status import SubmissionStatus


class DetectionActions(TaskActions):
    """
    Class for detection task's actions.
    """

    @staticmethod
    @action
    def submit(has_anomaly: str) -> SubmissionStatus:
        """
        Submit if anomalies are detected to the orchestrator for evaluation.

        Args:
            has_anomaly (str): "Yes" if anomalies are detected, "No" otherwise.

        Returns:
            SubmissionStatus: The status of the submission.
        """
        # TODO: check if anomalies are in the correct format
        return SubmissionStatus.VALID_SUBMISSION

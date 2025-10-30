# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for task actions."""

import os
import pandas as pd
from datetime import datetime, timedelta
from aiopslab.utils.actions import action, read, write
from aiopslab.service.kubectl import KubeCtl
from aiopslab.service.dock import Docker
from aiopslab.service.shell import Shell

# from aiopslab.observer import initialize_pod_and_service_lists
from aiopslab.observer.metric_api import PrometheusAPI
from aiopslab.observer.trace_api import TraceAPI

from aiopslab.orchestrator.actions.log_deduplication import greedy_compress_lines 

import re

LOG_COMMAND_PATTERN: str = (
    r"\b(?:"
    r"kubectl\s+(?:logs|get\s+events|describe|get\s+\S+\s+-w)"  # logs/events/describe/watch
    r"|docker\s+(?:logs|events)"                               # docker logs/events
    r")\b(?:[^\n]*)"
)

class TaskActions:
    """Base class for task actions."""

    @staticmethod
    @read
    def get_logs(namespace: str, service: str) -> str:
        """
        Collects relevant log data from a pod using Kubectl or from a container with Docker.

        Args:
            namespace (str): The namespace in which the service is running.
            service (str): The name of the service.

        Returns:
            str | dict | list[dicts]: Log data as a structured object or a string.
        """
        if namespace == "docker":
            docker = Docker()
            try:
                logs = docker.get_logs(service)
            except Exception as e:
                return "Error: Your service does not exist. Use docker to check."
        
        else:
            kubectl = KubeCtl()
            try:
                if namespace == "test-social-network":
                    user_service_pod = kubectl.get_pod_name(namespace, f"app={service}")
                elif namespace == "test-hotel-reservation":
                    user_service_pod = kubectl.get_pod_name(
                        namespace, f"io.kompose.service={service}"
                    )
                elif namespace == "astronomy-shop":
                    user_service_pod = kubectl.get_pod_name(
                        namespace, f"app.kubernetes.io/name={service}"
                    )
                elif namespace == "default" and "wrk2-job" in service:
                    user_service_pod = kubectl.get_pod_name(namespace, f"job-name=wrk2-job")
                else:
                        raise Exception
                logs = kubectl.get_pod_logs(user_service_pod, namespace)
            except Exception as e:
                return "Error: Your service/namespace does not exist. Use kubectl to check."

        logs = greedy_compress_lines(logs) 
        print(logs)

        return logs

    @staticmethod
    @action
    def exec_shell(command: str, timeout: int = 30) -> str:
        """
        Execute any shell command in a predefined debugging environment.
        Note: this is NOT A STATEFUL OR INTERACTIVE shell session. So you cannot
        execute commands like "kubectl edit".

        Args:
            command (str): The command to execute.
            timeout (int): Timeout in seconds for the command execution. Default is 30.

        Returns:
            str: The output of the command.
        """
        BLOCK_LIST: dict[str, str] = {
            "kubectl edit": "Error: Cannot use `kubectl edit`. Use `kubectl patch` instead.",
            "edit svc": "Error: Cannot use `kubectl edit`. Use `kubectl patch` instead.",
            "kubectl port-forward": "Error: Cannot use `kubectl port-forward` because it is an interactive command.",
            "docker logs -f": "Error: Cannot use `docker logs -f`. Use `docker logs` instead.",
            "kubectl logs -f": "Error: Cannot use `kubectl logs -f`. Use `kubectl logs` instead.",
        }
        for pattern, error in BLOCK_LIST.items():
            if pattern in command:
                return error

        result = Shell.exec(command) 

        if re.search(LOG_COMMAND_PATTERN, command):
            result = greedy_compress_lines(result)

        print(result)

        return result

    @staticmethod
    @read
    def get_metrics(namespace: str, duration: int = 5) -> str:
        """
        Collects metrics data from the service using Prometheus.

        Args:
            namespace (str): The namespace in which the service is running.
            duration (int): The number of minutes from now to start collecting metrics until now.

        Returns:
            str: Path to the directory where metrics are saved.
        """
        prometheus_url = (
            "http://localhost:32000"  # Replace with your Prometheus server URL
        )
        prometheus_api = PrometheusAPI(prometheus_url, namespace)
        prometheus_api.initialize_pod_and_service_lists(namespace)

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration)
        save_path = os.path.join(os.getcwd(), "metrics_output")

        # Export all metrics and save to the specified path
        save_dir_str = prometheus_api.export_all_metrics(
            start_time=start_time, end_time=end_time, save_path=save_path, step=15
        )

        return save_dir_str
    
    @staticmethod
    @read
    def read_metrics(file_path: str) -> str:
        """
        Reads and returns metrics from a specified CSV file.

        Args:
            file_path (str): Path to the metrics file (CSV format).

        Returns:
            str: The requested metrics or an error message.
        """
        if not os.path.exists(file_path):
            return f"error: Metrics file '{file_path}' not found."

        try:
            df_metrics = pd.read_csv(file_path)

            return df_metrics.to_string(index=False)

        except Exception as e:
            return f"Failed to read metrics: {str(e)}"

    @staticmethod
    @read
    def get_traces(namespace: str, duration: int = 5) -> str:
        """
        Collects trace data from the service using Jaeger.

        Args:
            namespace (str): The namespace in which the service is running.
            duration (int): The number of minutes from now to start collecting traces until now.

        Returns:
            str: Path to the directory where traces are saved.
        """
        # jaeger_url = "http://localhost:16686"
        print(namespace)
        trace_api = TraceAPI(namespace=namespace)

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration)

        traces = trace_api.extract_traces(start_time=start_time, end_time=end_time)
        df_traces = trace_api.process_traces(traces)
        save_path = os.path.join(os.getcwd(), "trace_output")

        return trace_api.save_traces(df_traces, save_path)
        # return f"Trace data exported to: {save_path}"

    @staticmethod
    @read
    def read_traces(file_path: str) -> str:
        """
        Reads and returns traces from a specified CSV file.

        Args:
            file_path (str): Path to the traces file (CSV format).

        Returns:
            str: The requested traces or an error message.
        """
        if not os.path.exists(file_path):
            return f"error: Traces file '{file_path}' not found."

        try:
            df_traces = pd.read_csv(file_path)

            return df_traces.to_string(index=False)

        except Exception as e:
            return f"Failed to read traces: {str(e)}"

    @staticmethod
    # @read
    # NOTE: disabled for now, since seems like a cheat for code changes
    def get_microservice_repo_diff(start: int, end: int, token=None) -> list[dict]:
        pass
        # """
        # Fetch the latest commits and their diffs from a GitHub repository.

        # Args:
        #     start (int): The start timestamp.
        #     end (int): The end timestamp.
        #     token: GitHub personal access token for authenticated requests (optional).

        # Returns:
        #     A list of commit messages and other details.
        # """
        # api_url = f"https://api.github.com/repos/owner/repo/commits"
        # headers = {}

        # if token:
        #     headers["Authorization"] = f"token {token}"

        # try:
        #     response = requests.get(api_url, headers=headers)
        #     response.raise_for_status()  # Raise an error for bad responses
        #     commits = response.json()

        #     for commit in commits:
        #         sha = commit["sha"]
        #         # Fetch diff details for each commit
        #         diff_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        #         diff_response = requests.get(diff_url, headers=headers)
        #         diff_response.raise_for_status()
        #         diff_data = diff_response.json()

        #         # Extract relevant commit information and diff
        #         commit_info = {
        #             "sha": sha,
        #             "message": commit["commit"]["message"],
        #             "author": commit["commit"]["author"]["name"],
        #             "date": commit["commit"]["author"]["date"],
        #             "files": [],
        #         }

        #         # Process the diff data
        #         for file in diff_data["files"]:
        #             commit_info["files"].append(
        #                 {
        #                     "filename": file["filename"],
        #                     "status": file["status"],
        #                     "additions": file["additions"],
        #                     "deletions": file["deletions"],
        #                     "changes": file["changes"],
        #                     "patch": file["patch"],
        #                 }
        #             )

        #         commit_details.append(commit_info)

        #     return commit_details

        # except requests.RequestException as e:
        #     print(f"An error occurred: {e}")
        #     return []


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Session wrapper to manage the an agent's session with the orchestrator."""

import time
import uuid
import json
import wandb
import sys
from io import StringIO
from pydantic import BaseModel

from aiopslab.paths import RESULTS_DIR


class SessionItem(BaseModel):
    role: str  # system / user / assistant
    content: str


class Session:
    def __init__(self, results_dir=None) -> None:
        self.session_id = uuid.uuid4()
        self.pid = None
        self.problem = None
        self.solution = None
        self.results = {}
        self.history: list[SessionItem] = []
        self.start_time = None
        self.end_time = None
        self.agent_name = None
        self.results_dir = results_dir
        self.print_logs = []
        self.original_stdout = None

    def set_problem(self, problem, pid=None):
        """Set the problem instance for the session.

        Args:
            problem (Task): The problem instance to set.
            pid (str): The problem ID.
        """
        self.problem = problem
        self.pid = pid

    def set_solution(self, solution):
        """Set the solution shared by the agent.

        Args:
            solution (Any): The solution instance to set.
        """
        self.solution = solution

    def set_results(self, results):
        """Set the results of the session.

        Args:
            results (Any): The results of the session.
        """
        self.results = results

    def set_agent(self, agent_name):
        """Set the agent name for the session.

        Args:
            agent_name (str): The name of the agent.
        """
        self.agent_name = agent_name

    def add(self, item):
        """Add an item into the session history.

        Args:
            item: The item to inject into the session history.
        """
        if not item:
            return

        if isinstance(item, SessionItem):
            self.history.append(item)
        elif isinstance(item, dict):
            self.history.append(SessionItem.model_validate(item))
        elif isinstance(item, list):
            for sub_item in item:
                self.add(sub_item)
        else:
            raise TypeError("Unsupported type %s" % type(item))

    def clear(self):
        """Clear the session history."""
        self.history = []

    def start(self):
        """Start the session and begin capturing print output."""
        self.start_time = time.time()
        self.start_print_capture()

    def end(self):
        """End the session and stop capturing print output."""
        self.end_time = time.time()
        self.stop_print_capture()
    
    def start_print_capture(self):
        """Start capturing print output to logs."""
        class PrintCapture:
            def __init__(self, session):
                self.session = session
                self.original_stdout = sys.stdout
            
            def write(self, text):
                if text.strip():  # Only capture non-empty lines
                    self.session.print_logs.append(text.rstrip())
                self.original_stdout.write(text)  # Still print to console
            
            def flush(self):
                self.original_stdout.flush()
        
        self.original_stdout = sys.stdout
        sys.stdout = PrintCapture(self)
    
    def stop_print_capture(self):
        """Stop capturing print output."""
        if self.original_stdout:
            sys.stdout = self.original_stdout

    def get_duration(self) -> float:
        """Get the duration of the session."""
        duration = self.end_time - self.start_time
        return duration

    def to_dict(self):
        """Return the session history as a dictionary."""
        summary = {
            "agent": self.agent_name,
            "session_id": str(self.session_id),
            "problem_id": self.pid,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "trace": [item.model_dump() for item in self.history],
            "results": self.results,
        }

        return summary

    def to_json(self):
        """Save the session to a JSON file."""
        from pathlib import Path
        results_dir = Path(self.results_dir) if self.results_dir else RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)

        filename_base = f"{self.session_id}_{self.start_time}"
        
        # Save JSON file
        with open(results_dir / f"{filename_base}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        
        # Save TXT file with print logs
        self.to_txt(filename_base)
    
    def to_txt(self, filename_base):
        """Save the session print logs to a TXT file."""
        from pathlib import Path
        results_dir = Path(self.results_dir) if self.results_dir else RESULTS_DIR
        
        with open(results_dir / f"{filename_base}.txt", "w") as f:
            # Write all captured print outputs
            for log_entry in getattr(self, 'print_logs', []):
                f.write(log_entry + "\n")

    def to_wandb(self):
        """Log the session to Weights & Biases."""
        wandb.log(self.to_dict())

    def from_json(self, filename: str):
        """Load a session from a JSON file."""
        from pathlib import Path
        results_dir = Path(self.results_dir) if self.results_dir else RESULTS_DIR

        with open(results_dir / filename, "r") as f:
            data = json.load(f)

        self.session_id = data.get("session_id")
        self.start_time = data.get("start_time")
        self.end_time = data.get("end_time")
        self.results = data.get("results")
        self.history = [SessionItem.model_validate(item) for item in data.get("trace")]

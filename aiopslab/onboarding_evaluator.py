# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Orchestrator class that interfaces with the agent and the environment."""

from aiopslab.service.helm import Helm
from aiopslab.service.kubectl import KubeCtl
from aiopslab.session import Session
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from aiopslab.orchestrator.onboarding_eval_parser import EvalParser
from aiopslab.utils.status import *
from aiopslab.service.telemetry.prometheus import Prometheus
import time
import inspect
import asyncio


class Evaluator:
    def __init__(self):
        self.agent = None
        self.session = None
        self.parser = EvalParser()
        self.probs = ProblemRegistry()
        self.sprint = SessionPrint()
        self.execution_start_time = None
        self.execution_end_time = None
        self.kubectl = KubeCtl()

    def init_problem(self, problem_id: str):
        """Initialize a problem instance for the agent to solve.

        Args:
            problem_id (str): The problem instance identifier.

        Returns:
            tuple: A tuple containing the problem description, task message, and session object.
        """
        # Start timer
        self.execution_start_time = time.time()

        self.session = Session()
        print(f"Session ID: {self.session.session_id}")
        prob = self.probs.get_problem_instance(problem_id)
        self.session.set_problem(prob, pid=problem_id)
        self.session.set_agent(self.agent_name)

        print("Setting up OpenEBS...")

        command = "kubectl get pods -n openebs"
        result = self.kubectl.exec_command(command)
        if "Running" in result:
            print("OpenEBS is already running. Skipping installation.")
        else:
            self.kubectl.exec_command(
                "kubectl apply -f https://openebs.github.io/charts/openebs-operator.yaml"
            )
            self.kubectl.exec_command(
                "kubectl patch storageclass openebs-hostpath -p '{\"metadata\": {\"annotations\":{\"storageclass.kubernetes.io/is-default-class\":\"true\"}}}'"
            )
            self.kubectl.wait_for_ready("openebs")
            print("OpenEBS setup completed.")

        # Setup and deploy Prometheus
        self.prometheus = Prometheus()
        self.prometheus.deploy()

        # deploy service
        prob.app.delete()
        prob.app.deploy()

        # inject fault
        prob.inject_fault()

        # Check if start_workload is async or sync
        if inspect.iscoroutinefunction(prob.start_workload):
            asyncio.create_task(prob.start_workload())
        else:
            prob.start_workload()

        task_desc = prob.get_task_description()
        instructions = prob.get_instructions()
        actions = prob.get_available_actions()

        return task_desc, instructions, actions

    def register_agent(self, agent, name="agent"):
        """Register the agent for the current session.

        Args:
            agent: The agent to register.
            name: The name of the agent (default: "agent").
        """
        self.agent = agent
        self.agent_name = name

    async def ask_agent(self, input):
        """Ask the agent for the next action given the current context."""
        assert self.session is not None
        assert self.agent is not None

        agent_response = await self.agent.get_action(input)
        self.session.add({"role": "assistant", "content": agent_response})

        return agent_response

    async def ask_env(self, input):
        """Ask the environment for the observation given the current action."""
        assert self.session is not None

        try:
            resp = self.parser.parse(input)
        except ResponseParsingError as e:
            self.session.add({"role": "env", "content": str(e)})
            return str(e)

        api, args, kwargs = resp["api_name"], resp["args"], resp["kwargs"]

        # special handling for submit
        if api == "submit":
            self.session.set_solution(args[0] if len(args) == 1 else args)
            
            # Use the problem's eval method to check if solution is valid
            try:
                # Calculate the current duration manually since session isn't ended yet
                current_time = time.time()
                current_duration = current_time - self.session.start_time
                
                # Create a temporary dict to store results
                temp_results = self.session.problem.eval(
                    self.session.solution, 
                    self.session.history, 
                    current_duration
                )
                
                # Check if the solution is successful based on eval results
                if temp_results.get("success", False):
                    env_response = SubmissionStatus.VALID_SUBMISSION
                else:
                    env_response = SubmissionStatus.INVALID_SUBMISSION
                    
            except Exception as e:
                print(f"Error validating submission: {e}")
                import traceback
                traceback.print_exc()
                env_response = SubmissionStatus.INVALID_SUBMISSION
        else:
            # Regular action handling
            try:
                env_response = self.session.problem.perform_action(api, *args, **kwargs)
            except InvalidActionError as e:
                env_response = str(e)

        self.session.add({"role": "env", "content": env_response})
        return env_response

    async def start_problem(self):
        """Start the task and run until a valid submission is received.

        Returns:
            dict: The final state of the session.
        """
        assert self.session is not None
        action_instr = "Please take the next action"
        action, env_response, results = "", "", {}
        self.session.start()
        self.execution_start_time = time.time()
        
        # Initial environment response
        env_response = await self.ask_env(action)
        
        while env_response != SubmissionStatus.VALID_SUBMISSION:
            action = await self.ask_agent(action_instr)
            self.sprint.agent(action)
            
            env_response = await self.ask_env(action)
            self.sprint.service(env_response)
            
            if env_response == SubmissionStatus.VALID_SUBMISSION:
                print("Submission is correct!")
                break
            elif env_response == SubmissionStatus.INVALID_SUBMISSION:
                print("Your submission was invalid. Please continue working on the problem.")
            else:
                action_instr = env_response
        
        self.session.end()
        
        # Final evaluation with the valid submission
        if env_response == SubmissionStatus.VALID_SUBMISSION:
            results = self.session.problem.eval(
                self.session.solution, self.session.history, self.session.get_duration()
            )
            self.sprint.result(results)
        
        self.session.set_results(results)
        self.session.to_json()
        self.session.problem.recover_fault()
        
        # App cleanup
        self.session.problem.app.cleanup()
        
        self.execution_end_time = time.time()
        total_execution_time = self.execution_end_time - self.execution_start_time
        time_keys = ["TTD", "TTL", "TTA", "TTM"]
        key = next((k for k in time_keys if k in results), None)
        framework_overhead = (
            total_execution_time - (results.get(key, 0) or 0)
        )
        print(f"Framework overhead: {framework_overhead}")
        
        return {
            "history": self.session.history,
            "final_state": env_response,
            "results": results,
            "framework_overhead": framework_overhead,
        }
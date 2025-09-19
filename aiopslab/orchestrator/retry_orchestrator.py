# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Orchestrator with retry capabilities using task variants."""

import asyncio
import inspect
import copy
from typing import Dict, Any, Optional

from aiopslab.orchestrator.orchestrator import Orchestrator
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.session import Session
from aiopslab.utils.status import SubmissionStatus


class RetryOrchestrator:
    """Orchestrator that supports retry with task variants using composition."""
    
    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        max_retries: int = 3,
        enable_variants: bool = True,
        retry_delay: float = 5.0,
    ):
        """
        Initialize retry orchestrator with composition pattern.

        Args:
            orchestrator: Base orchestrator to wrap (creates new if None)
            max_retries: Maximum number of retry attempts
            enable_variants: Whether to use variants for retries
            retry_delay: Delay (in seconds) between retry attempts
        """
        self.orchestrator = orchestrator or Orchestrator()
        self.max_retries = max_retries
        self.enable_variants = enable_variants
        self.retry_count = 0
        self.retry_history = []
        self.retry_delay = retry_delay
        
    def __getattr__(self, name):
        """
        Forward attribute access to the wrapped orchestrator.
        
        This allows accessing properties like session, probs, agent, etc.
        from the base orchestrator transparently.
        """
        return getattr(self.orchestrator, name)
    
    def init_problem(self, problem_id: str):
        """Initialize problem using the wrapped orchestrator."""
        return self.orchestrator.init_problem(problem_id)
    
    def register_agent(self, agent, name="agent"):
        """Register agent using the wrapped orchestrator."""
        self.orchestrator.register_agent(agent, name)
        
    async def start_problem(
        self,
        max_steps: int,
        *,
        max_retries: Optional[int] = None,
        enable_variants: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Start the problem using retry semantics.

        This mirrors :meth:`Orchestrator.start_problem` to keep parity with the
        base orchestrator while still exposing retry overrides.
        """

        return await self.start_problem_with_retry(
            max_steps,
            max_retries=max_retries,
            enable_variants=enable_variants,
        )

    async def start_problem_with_retry(
        self,
        max_steps: int,
        *,
        max_retries: Optional[int] = None,
        enable_variants: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Start problem solving with retry logic.

        Args:
            max_steps: Maximum steps per attempt
            max_retries: Optional override for retry attempts
            enable_variants: Optional override to toggle variant usage

        Returns:
            Final results including retry information
        """


        retries_allowed = self.max_retries if max_retries is None else max_retries
        variants_enabled = (
            self.enable_variants if enable_variants is None else enable_variants
        )
        session = getattr(self.orchestrator, "session", None)
        overall_results = {
            "retry_count": 0,
            "attempts": [],
            "variant_attempts": [],
            "final_success": False,
            "final_results": None,
            "problem_id": getattr(session, "problem_id", None),
            "canonical_problem_id": getattr(session, "canonical_pid", None),
            "variant_mode": self.orchestrator.probs.variant_mode,
            "variants_enabled_for_retries": self.enable_variants,
        }

        session = getattr(self.orchestrator, "session", None)
        if not session or not getattr(session, "problem", None):
            raise RuntimeError(
                "Problem is not initialized. Call init_problem before starting."
            )

        problem = session.problem

        for attempt in range(retries_allowed + 1):
            print(f"\n{'='*60}")
            print(f"Attempt {attempt + 1}/{retries_allowed + 1}")

            if attempt > 0:
                problem = await self._prepare_retry()
                if problem is None:
                    raise RuntimeError(
                        "Unable to prepare retry without a problem instance"
                    )

            variant_info = None
            if isinstance(problem, VariantTask) and attempt > 0:
                problem.reset_to_base()

            if variants_enabled and isinstance(problem, VariantTask):
                if attempt > 0:
                    variant = problem.get_next_variant()
                    if variant:
                        print(f"Applying variant: {variant}")
                        problem.apply_variant(variant)
                    else:
                        print("No more variants available, using base configuration")
                variant_info = problem.current_variant

            if attempt > 0:
                variant_summary = (
                    problem.get_variant_summary()
                    if isinstance(problem, VariantTask)
                    else "same configuration"
                )
                print(f"Retrying with {variant_summary}")

            if attempt > 0:
                await self._deploy_problem(problem)

            try:
                results = await self.orchestrator.start_problem(max_steps)
                overall_results["final_results"] = results
                variant_metadata = self._capture_variant_metadata()

                attempt_info = {
                    "attempt_number": attempt + 1,
                    "problem_id": getattr(self.orchestrator.session, "problem_id", None),
                    "canonical_problem_id": getattr(self.orchestrator.session, "canonical_pid", None),
                    "results": results,
                    "variant": variant_metadata.get("current_variant"),
                    "variant_context": variant_metadata,
                }
                overall_results["attempts"].append(attempt_info)
                overall_results["variant_attempts"].append(
                    {"attempt_number": attempt + 1, **variant_metadata}
                )
                overall_results["problem_id"] = attempt_info["problem_id"]
                overall_results["canonical_problem_id"] = attempt_info["canonical_problem_id"]
                self.retry_history.append(attempt_info)
                
                # Check if successful
                if self._is_successful(results):
                    print(f"✓ Success on attempt {attempt + 1}")
                    overall_results["final_success"] = True
                    overall_results["final_results"] = results
                    self.retry_count = attempt
                    break

                print(f"✗ Failed on attempt {attempt + 1}")
                if attempt < retries_allowed:
                    print("Preparing for retry...")

            except Exception as e:
                print(f"Exception during attempt {attempt + 1}: {e}")

                variant_metadata = self._capture_variant_metadata()
                attempt_info = {
                    "attempt_number": attempt + 1,
                    "problem_id": getattr(self.orchestrator.session, "problem_id", None),
                    "canonical_problem_id": getattr(self.orchestrator.session, "canonical_pid", None),
                    "error": str(e),
                    "variant": variant_metadata.get("current_variant"),
                    "variant_context": variant_metadata,
                }
                overall_results["attempts"].append(attempt_info)
                overall_results["variant_attempts"].append(
                    {"attempt_number": attempt + 1, **variant_metadata}
                )
                overall_results["problem_id"] = attempt_info["problem_id"]
                overall_results["canonical_problem_id"] = attempt_info["canonical_problem_id"]
                self.retry_history.append(attempt_info)

                if attempt < self.max_retries:
                    await self._prepare_retry()
                else:
                    raise

        print(f"\n{'='*60}")
        print("RETRY SUMMARY")
        print(f"Total attempts: {len(overall_results['attempts'])}")
        print(f"Success: {overall_results['final_success']}")

        if overall_results["final_success"]:
            print(f"Succeeded on attempt: {self.retry_count + 1}")
        else:
            print("All attempts failed")
            self.retry_count = min(
                len(overall_results["attempts"]) - 1,
                retries_allowed,
            )

        overall_results["retry_count"] = self.retry_count


        return overall_results

    def _is_successful(self, results: Dict[str, Any]) -> bool:
        """
        Determine if the results indicate success.
        
        Args:
            results: Results from problem execution
            
        Returns:
            True if successful, False otherwise
        """
        # Check various success indicators
        if "results" in results:
            task_results = results["results"]
            
            # Check for explicit success field
            if "success" in task_results:
                return task_results["success"]
                
            # Check for detection accuracy
            if "Detection Accuracy" in task_results:
                return task_results["Detection Accuracy"] == "Correct"
                
            # Check for localization accuracy
            if "Localization Accuracy" in task_results:
                return task_results["Localization Accuracy"] > 50.0
                
            # Check for analysis correctness
            if "system_level_correct" in task_results and "fault_type_correct" in task_results:
                return task_results["system_level_correct"] and task_results["fault_type_correct"]
                
        # Check final state
        if "final_state" in results:
            return results["final_state"] == SubmissionStatus.VALID_SUBMISSION
            
        return False

    def _capture_variant_metadata(self) -> Dict[str, Any]:
        """Capture variant-specific metadata for the current session."""
        session = getattr(self.orchestrator, "session", None)
        problem = getattr(session, "problem", None)
        metadata = {
            "mode": self.orchestrator.probs.variant_mode,
            "supports_variants": isinstance(problem, VariantTask),
            "current_variant": None,
            "variant_history": [],
            "summary": "Base configuration",
            "base_config": None,
            "applied": False,
            "problem_id": getattr(session, "problem_id", None),
            "canonical_problem_id": getattr(session, "canonical_pid", None),
        }

        if isinstance(problem, VariantTask):
            metadata["current_variant"] = copy.deepcopy(problem.current_variant)
            metadata["variant_history"] = copy.deepcopy(problem.variant_history)
            metadata["summary"] = problem.get_variant_summary()
            metadata["applied"] = problem.current_variant is not None
            generator = getattr(problem, "variant_generator", None)
            if generator is not None:
                metadata["base_config"] = copy.deepcopy(
                    getattr(generator, "base_config", None)
                )

        return metadata
    
    async def _prepare_retry(self):
        """Prepare for a retry attempt."""
        session = getattr(self.orchestrator, "session", None)
        problem = getattr(session, "problem", None)

        if problem:
            try:
                problem.recover_fault()
            except Exception as e:
                print(f"Warning: Error during fault recovery: {e}")

            try:
                app = getattr(problem, "app", None)
                if app and hasattr(app, "cleanup"):
                    app.cleanup()
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")

        print(f"Waiting {self.retry_delay} seconds before retry...")
        await asyncio.sleep(self.retry_delay)
        # Re-initialize problem
        problem_id = None
        if hasattr(self.orchestrator.session, 'problem_id'):
            problem_id = self.orchestrator.session.problem_id
            
        if problem_id:
            # Re-initialize the problem
            self.orchestrator.session = Session(results_dir=self.orchestrator.results_dir)
            prob = self.orchestrator.probs.get_problem_instance(problem_id)
            canonical_pid = self.orchestrator.probs.get_canonical_id(problem_id)
            self.orchestrator.session.set_problem(
                prob, pid=problem_id, canonical_pid=canonical_pid
            )
            self.orchestrator.session.set_agent(self.orchestrator.agent_name)
            
            # Re-deploy application
            prob.app.deploy()
            
            # Re-inject fault
            prob.inject_fault()
            
            # Start workload
            if inspect.iscoroutinefunction(prob.start_workload):
                await prob.start_workload()

            else:
                new_session = session_cls()
        except TypeError:
            new_session = session_cls()
            if results_dir is not None and hasattr(new_session, "results_dir"):
                new_session.results_dir = results_dir

        self.orchestrator.session = new_session

        if problem:
            new_session.set_problem(problem, pid=problem_id)

        if hasattr(new_session, "set_agent"):
            new_session.set_agent(self.orchestrator.agent_name)

        return problem

    async def _deploy_problem(self, problem):
        """Deploy application, inject the fault, and start workloads."""

        if not problem:
            return

        try:
            app = getattr(problem, "app", None)
            if app and hasattr(app, "deploy"):
                app.deploy()
        except Exception as e:
            print(f"Warning: Error during app deployment: {e}")

        try:
            if hasattr(problem, "inject_fault"):
                problem.inject_fault()
        except Exception as e:
            print(f"Warning: Error during fault injection: {e}")

        start_workload = getattr(problem, "start_workload", None)
        if start_workload:
            try:
                if inspect.iscoroutinefunction(start_workload):
                    await start_workload()
                else:
                    start_workload()
            except Exception as e:
                print(f"Warning: Error starting workload: {e}")

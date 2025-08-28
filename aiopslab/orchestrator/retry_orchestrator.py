# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Orchestrator with retry capabilities using task variants."""

import time
import asyncio
import inspect
from typing import Dict, Any, Optional
from aiopslab.orchestrator.orchestrator import Orchestrator
from aiopslab.orchestrator.tasks.variant_task import VariantTask
from aiopslab.utils.status import SubmissionStatus
from aiopslab.session import Session


class RetryOrchestrator:
    """Orchestrator that supports retry with task variants using composition."""
    
    def __init__(self, orchestrator: Optional[Orchestrator] = None, 
                 max_retries: int = 3, enable_variants: bool = True):
        """
        Initialize retry orchestrator with composition pattern.
        
        Args:
            orchestrator: Base orchestrator to wrap (creates new if None)
            max_retries: Maximum number of retry attempts
            enable_variants: Whether to use variants for retries
        """
        self.orchestrator = orchestrator or Orchestrator()
        self.max_retries = max_retries
        self.enable_variants = enable_variants
        self.retry_count = 0
        self.retry_history = []
        
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
        
    async def start_problem_with_retry(self, max_steps: int) -> Dict[str, Any]:
        """
        Start problem solving with retry logic.
        
        Args:
            max_steps: Maximum steps per attempt
            
        Returns:
            Final results including retry information
        """
        overall_results = {
            "retry_count": 0,
            "attempts": [],
            "final_success": False,
            "final_results": None
        }
        
        for attempt in range(self.max_retries + 1):
            print(f"\n{'='*60}")
            print(f"Attempt {attempt + 1}/{self.max_retries + 1}")
            
            # Apply variant if this is a retry and variants are enabled
            if attempt > 0 and self.enable_variants:
                if isinstance(self.orchestrator.session.problem, VariantTask):
                    variant = self.orchestrator.session.problem.get_next_variant()
                    if variant:
                        print(f"Applying variant: {variant}")
                        self.orchestrator.session.problem.apply_variant(variant)
                    else:
                        print("No more variants available, using base configuration")
            
            if attempt > 0:
                variant_summary = (
                    self.orchestrator.session.problem.get_variant_summary() 
                    if isinstance(self.orchestrator.session.problem, VariantTask) 
                    else 'same configuration'
                )
                print(f"Retrying with {variant_summary}")
                        
            # Run the problem
            try:
                results = await self.orchestrator.start_problem(max_steps)
                
                # Record attempt
                variant_info = None
                if isinstance(self.orchestrator.session.problem, VariantTask):
                    variant_info = self.orchestrator.session.problem.current_variant
                    
                attempt_info = {
                    "attempt_number": attempt + 1,
                    "results": results,
                    "variant": variant_info
                }
                overall_results["attempts"].append(attempt_info)
                
                # Check if successful
                if self._is_successful(results):
                    print(f"✓ Success on attempt {attempt + 1}")
                    overall_results["final_success"] = True
                    overall_results["final_results"] = results
                    overall_results["retry_count"] = attempt
                    break
                else:
                    print(f"✗ Failed on attempt {attempt + 1}")
                    
                    # If this isn't the last attempt, prepare for retry
                    if attempt < self.max_retries:
                        print(f"Preparing for retry...")
                        # Reset the session for next attempt
                        await self._prepare_retry()
                        
            except Exception as e:
                print(f"Exception during attempt {attempt + 1}: {e}")
                
                variant_info = None
                if isinstance(self.orchestrator.session.problem, VariantTask):
                    variant_info = self.orchestrator.session.problem.current_variant
                    
                attempt_info = {
                    "attempt_number": attempt + 1,
                    "error": str(e),
                    "variant": variant_info
                }
                overall_results["attempts"].append(attempt_info)
                
                if attempt < self.max_retries:
                    await self._prepare_retry()
                else:
                    raise
                    
        # Summary
        print(f"\n{'='*60}")
        print("RETRY SUMMARY")
        print(f"Total attempts: {len(overall_results['attempts'])}")
        print(f"Success: {overall_results['final_success']}")
        
        if overall_results["final_success"]:
            print(f"Succeeded on attempt: {overall_results['retry_count'] + 1}")
        else:
            print("All attempts failed")
            
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
    
    async def _prepare_retry(self):
        """Prepare for a retry attempt."""
        # Clean up current problem state
        if self.orchestrator.session and self.orchestrator.session.problem:
            try:
                self.orchestrator.session.problem.recover_fault()
                self.orchestrator.session.problem.app.cleanup()
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
                
        # Wait before retry
        print("Waiting 5 seconds before retry...")
        await asyncio.sleep(5)
        
        # Re-initialize problem
        problem_id = None
        if hasattr(self.orchestrator.session, 'problem_id'):
            problem_id = self.orchestrator.session.problem_id
            
        if problem_id:
            # Re-initialize the problem
            self.orchestrator.session = Session()
            prob = self.orchestrator.probs.get_problem_instance(problem_id)
            self.orchestrator.session.set_problem(prob, pid=problem_id)
            self.orchestrator.session.set_agent(self.orchestrator.agent_name)
            
            # Re-deploy application
            prob.app.deploy()
            
            # Re-inject fault
            prob.inject_fault()
            
            # Start workload
            if inspect.iscoroutinefunction(prob.start_workload):
                await prob.start_workload()
            else:
                prob.start_workload()
#!/usr/bin/env python3
# Modified Flash agent to run a single scenario for testing

import asyncio
import logging
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
from clients.utils.llm import GPT4Turbo
from aiopslab.orchestrator import Orchestrator

# Import the FlashAgent class from the original file
import sys
sys.path.append('.')
from clients.flash import FlashAgent

if __name__ == "__main__":
    # Test with just one scenario instead of 12
    pid = "k8s_target_port-misconfig-detection-2"  # Single scenario for testing
    
    print(f"Running Flash agent with scenario: {pid}")
    
    flash_agent = FlashAgent()
    orchestrator = Orchestrator()

    orchestrator.register_agent(flash_agent, name="flash")

    problem_desc, instructions, apis = orchestrator.init_problem(pid)
    flash_agent.init_context(problem_desc, instructions, apis)

    print("Starting Flash agent scenario...")
    asyncio.run(orchestrator.start_problem(max_steps=20))
    print("Flash agent scenario completed!")

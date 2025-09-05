import asyncio
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aiopslab.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.registry import AgentRegistry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("aiopslab-service")

# Create FastAPI app with description and version
app = FastAPI(
    title="AIOpsLab API Service",
    description="A service for running AIOps problem simulations",
    version="0.1.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class SimulationRequest(BaseModel):
    problem_id: str
    agent_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_steps: Optional[int] = None
    # vLLM specific parameters
    model: Optional[str] = "Qwen/Qwen2.5-Coder-3B-Instruct"
    repetition_penalty: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024  # Aligned with vLLMAgent default
    
    class Config:
        schema_extra = {
            "example": {
                "problem_id": "misconfig_app_hotel_res-mitigation-1",
                "agent_name": "vllm",
                "max_steps": 10,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

class SimulationResponse(BaseModel):
    agent: str
    session_id: str
    problem_id: str
    start_time: float
    end_time: float
    trace: List[Dict[str, Any]]
    results: Dict[str, Any]

# Get all available problems
@app.get("/problems", 
         response_model=List[str],
         summary="List all available problems",
         description="Returns a list of all problem IDs that can be used for simulation")
def list_problems():
    registry = ProblemRegistry()
    return registry.get_problem_ids()

# Get all available agents
@app.get("/agents", 
         response_model=List[str],
         summary="List all available agents",
         description="Returns a list of all agent implementations that can be used for simulation")
def list_agents():
    registry = AgentRegistry()
    return registry.get_agent_ids()

# Health check endpoint
@app.get("/health", 
         response_model=Dict[str, str],
         summary="Health check",
         description="Simple endpoint to verify the service is running")
def health_check():
    return {"status": "healthy", "service": "AIOpsLab"}

# Main simulation endpoint
@app.post("/simulate", 
          response_model=SimulationResponse,
          summary="Run an AIOps problem simulation",
          description="Takes a problem ID, agent name, and optional parameters to run a simulation and return results")
def simulate(req: SimulationRequest):
    logger.info(f"Starting simulation with problem={req.problem_id}, agent={req.agent_name}, max_steps={req.max_steps}")
    
    # Check if the problem ID is valid
    problem_registry = ProblemRegistry()
    problem = problem_registry.get_problem(req.problem_id)
    if problem is None:
        logger.error(f"Problem {req.problem_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Problem {req.problem_id} not found. Available problems: {problem_registry.get_problem_ids()}"
        )
    pid = req.problem_id

    # Get agent from registry
    agent_registry = AgentRegistry()
    agent_cls = agent_registry.get_agent(req.agent_name)
    if agent_cls is None:
        logger.error(f"Agent {req.agent_name} not registered")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Agent {req.agent_name} not registered. Available agents: {agent_registry.get_agent_ids()}"
        )
    
    # Initialize agent with vLLM-specific parameters if applicable
    if req.agent_name == "vllm":
        # Extract vLLM parameters from request
        vllm_params = {
            "model": req.model,
            "repetition_penalty": req.repetition_penalty,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
        }
        agent = agent_cls(**vllm_params)
    else:
        agent = agent_cls()
    logger.info(f"Created agent: {req.agent_name}")

    # Check if max_steps is provided, else set default
    max_steps = req.max_steps if req.max_steps is not None else 10
    
    # Set up orchestrator
    orchestrator = Orchestrator()
    orchestrator.register_agent(agent, name=f"{req.agent_name}-agent")

    # Run the simulation
    logger.info(f"Starting simulation for problem {pid} with agent {req.agent_name}")
    try:
        problem_desc, instructs, apis = orchestrator.init_problem(pid)
        agent.init_context(problem_desc, instructs, apis)
        asyncio.run(orchestrator.start_problem(max_steps=max_steps))
        
        raw = orchestrator.session.to_dict()
        raw["trace"].insert(0, {"role": "system", "content": agent.system_message})
        raw["trace"].insert(1, {"role": "user", "content": agent.task_message})
        # Remove last message if it's from environment
        if raw["trace"] and raw["trace"][-1].get("role") == "env":
            raw["trace"].pop()
        
        return SimulationResponse(**raw)
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error during simulation: {e}"
        )

# Entry point for running the service
if __name__ == "__main__":
    import uvicorn
    # Load environment variables for host, port, and workers
    host = os.environ.get("SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVICE_PORT", 1818))
    workers = int(os.environ.get("SERVICE_WORKERS", 1))
    
    logger.info(f"Starting AIOpsLab service on host {host} port {port} with {workers} workers")
    uvicorn.run("service:app", host=host, port=port, workers=workers)
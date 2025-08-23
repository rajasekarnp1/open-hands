"""
Pydantic models for the PlanningAgent, including Plan, Step, and request/response types.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility

import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from ..models import HumanInputRequest # Import for type hint

# --- Plan Structure Models ---

class Step(BaseModel):
    """Represents a single step in a plan."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "pending"  # e.g., "pending", "in_progress", "completed", "failed", "requires_human_input"

    # Agent/Tool assignment for execution
    agent_to_use: Optional[str] = None  # e.g., "CodeAgent", "ResearchAgent", "HumanAssistance"
    tool_to_use: Optional[str] = None    # Specific tool name if directly determined by plan
    tool_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict) # Parameters for the tool

    # For CodeAgent, the 'instruction' would be the step description or derived from it.
    # Other agent-specific inputs could be added here or inferred during orchestration.
    expected_output_description: Optional[str] = None # What this step is expected to produce

    # Execution results
    result: Optional[str] = None # Textual summary of this step's execution outcome
    error_details: Optional[str] = None # If the step failed

    # For potential dependencies or sequencing, though simple list order is primary for now
    # depends_on: List[str] = Field(default_factory=list) # List of step_ids

class Plan(BaseModel):
    """Represents a multi-step plan to achieve a goal."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4()}")
    goal: str
    steps: List[Step] = Field(default_factory=list)
    plan_status: str = "generating"  # e.g., "generating", "ready", "in_progress", "completed", "failed", "requires_human_input"

    # Overall plan execution result/summary
    final_output: Optional[str] = None
    error_details: Optional[str] = None

# --- Agent Request/Response Models ---

class PlanningAgentRequest(BaseModel):
    goal: str
    project_directory: Optional[str] = None  # Context for planning, enables filesystem tools for planner if needed
    thread_id: Optional[str] = None          # For stateful planning sessions & resuming plans
    # Potentially add:
    # existing_plan_id: Optional[str] = None # To modify or update an existing plan
    # preferred_agents: Optional[List[str]] = None # Hints for agent selection per step
    # user_context: Optional[str] = None # Additional context from user for planning

class PlanningAgentResponse(BaseModel):
    plan: Optional[Plan] = None
    agent_status: str  # e.g., "plan_generated", "plan_updated", "error", "plan_retrieved_existing", "executing_step", "plan_completed", "plan_failed", "paused_for_human_input"
    error_details: Optional[str] = None
    thread_id: Optional[str] = None # Return thread_id for client to continue session
    human_input_request: Optional[HumanInputRequest] = None # If a sub-agent (like CodeAgent) pauses for HITL

```
